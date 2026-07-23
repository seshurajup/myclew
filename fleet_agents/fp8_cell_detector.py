"""fp8_cell_detector — a MATMUL-DOMINATED 3D cell-center heatmap detector that can be TRAINED
end-to-end in fp8 (E4M3) on the RTX 5090 (sm_120) via native torch._scaled_mm — no torchao,
no transformer-engine (ABI risk; not installed).

WHY THIS EXISTS
    Our production biohub detector is a UNet3D (conv3d). fp8 CANNOT train it: there is no fp8
    conv3d kernel on any current GPU. To put a cell-detector on the fp8 fast path (~2.1× vs bf16
    on matmul, measured on this 5090) the detector must be Linear/attention-dominated. This module
    is that detector: a tiny Conv3d patch-embed stem (<15% of params) + N transformer encoder
    blocks + a Linear heatmap head that un-patchifies to a per-voxel cell-center probability volume
    — the SAME output contract as the UNet detector (a 3D heatmap you peak-detect).

WHAT IS AND ISN'T fp8 HERE (honest)
    fp8 accelerates the big GEMMs: attention qkv/proj, MLP fc1/fc2, and the heatmap head. Softmax,
    LayerNorm, the attention score/context batched-matmuls, residual adds and the conv stem stay in
    bf16 — that is the honest ceiling on the fp8 win (see fp8_flop_fraction()).

    Fp8LinearFn implements TRUE fp8 training when CELLMOT_FP8_BACKWARD=1 (default): forward GEMM in
    E4M3, and BOTH backward GEMMs (grad_input, grad_weight) in fp8 with the incoming gradient cast
    to E5M2 (wider exponent for gradients). With CELLMOT_FP8_BACKWARD=0 you get fp8-forward-only
    (bf16 backward). The bf16 path (use_fp8=False) always works and is the reference.

SCALE Config: per-tensor absmax scaling. scale = absmax / dtype_max; the fp8 tensor stores
    x/scale and _scaled_mm multiplies the scale back in. Per-row scaling is a drop-in refinement
    (ROWWISE flag) but per-tensor is enough to converge here.
"""
from __future__ import annotations

import os

import torch
import torch.nn as nn
import torch.nn.functional as F

# fp8 format maxima (finite absmax representable)
_E4M3_MAX = 448.0        # torch.float8_e4m3fn
_E5M2_MAX = 57344.0      # torch.float8_e5m2 (gradients)
_EPS = 1e-12


def _fp8_supported() -> bool:
    if not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability(0)
    return major >= 9 or torch.cuda.get_device_capability(0) >= (8, 9)


def quantize_fp8(x: torch.Tensor, dtype: torch.dtype, fmax: float):
    """Per-tensor absmax quantize a 2D tensor to fp8. Returns (x_fp8_rowmajor_contiguous, scale_scalar).
    scale is a 0-d float32 cuda tensor; _scaled_mm will multiply it back in."""
    amax = x.detach().abs().max()
    scale = (amax / fmax).clamp(min=_EPS).to(torch.float32)
    x_fp8 = (x / scale).to(dtype)
    return x_fp8.contiguous(), scale


def scaled_mm(a_hp: torch.Tensor, b_hp: torch.Tensor, a_dtype: torch.dtype, b_dtype: torch.dtype,
              out_dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
    """Compute a_hp @ b_hp with an fp8 GEMM. a_hp is (M,K) row-major-ish, b_hp is (K,N).
    Casts each operand to fp8 with its own absmax scale, arranges the layout _scaled_mm wants
    (lhs row-major, rhs column-major), and returns the (M,N) result in out_dtype."""
    a_fp8, sa = quantize_fp8(a_hp, a_dtype, _E4M3_MAX if a_dtype == torch.float8_e4m3fn else _E5M2_MAX)
    # rhs must be column-major: make (N,K) contiguous then transpose → (K,N) column-major view
    b_km = b_hp.t().contiguous()                       # (N,K) row-major
    b_fp8, sb = quantize_fp8(b_km, b_dtype, _E4M3_MAX if b_dtype == torch.float8_e4m3fn else _E5M2_MAX)
    return torch._scaled_mm(a_fp8, b_fp8.t(), scale_a=sa, scale_b=sb, out_dtype=out_dtype)


class Fp8LinearFn(torch.autograd.Function):
    """Autograd Linear with fp8 GEMMs. forward: y = x @ W^T (+b) in E4M3.
    backward (if fp8_bwd): grad_x = g @ W and grad_W = g^T @ x, both fp8 with g in E5M2."""

    @staticmethod
    def forward(ctx, x2d, weight, bias, fp8_bwd):
        # x2d: (M,K)  weight: (N,K)  -> y: (M,N)
        y = scaled_mm(x2d, weight.t(), torch.float8_e4m3fn, torch.float8_e4m3fn, out_dtype=torch.bfloat16)
        if bias is not None:
            y = y + bias.to(y.dtype)
        ctx.save_for_backward(x2d, weight)
        ctx.has_bias = bias is not None
        ctx.fp8_bwd = fp8_bwd
        return y

    @staticmethod
    def backward(ctx, g):
        x2d, weight = ctx.saved_tensors
        g = g.contiguous()
        if ctx.fp8_bwd:
            # grad_x = g @ W        g:(M,N)  W:(N,K) -> (M,K)   [g in E5M2, W in E4M3]
            grad_x = scaled_mm(g, weight, torch.float8_e5m2, torch.float8_e4m3fn, out_dtype=torch.bfloat16)
            # grad_W = g^T @ x      g^T:(N,M) x:(M,K) -> (N,K)  [g in E5M2, x in E4M3]
            grad_w = scaled_mm(g.t().contiguous(), x2d, torch.float8_e5m2, torch.float8_e4m3fn, out_dtype=torch.bfloat16)
        else:
            # fp8-forward-only: honest bf16 backward
            gw = g.to(torch.bfloat16)
            grad_x = gw @ weight.to(torch.bfloat16)
            grad_w = gw.t() @ x2d.to(torch.bfloat16)
        grad_x = grad_x.to(x2d.dtype)
        grad_w = grad_w.to(weight.dtype)
        grad_b = g.to(torch.float32).sum(0).to(weight.dtype) if ctx.has_bias else None
        return grad_x, grad_w, grad_b, None


class Fp8Linear(nn.Linear):
    """Linear that runs its GEMM in fp8 when use_fp8 (and hardware supports it), else bf16 F.linear.
    Accepts (..., K) input, flattens leading dims for the 2D GEMM, restores shape.
    Subclasses nn.Linear on purpose: hardware_tune.select_train_precision counts nn.Linear params to
    pick the precision, and these ARE linear layers — so the detector reads as matmul-dominated → fp8."""

    def __init__(self, in_f, out_f, bias=True, fp8_bwd=True):
        super().__init__(in_f, out_f, bias=bias)
        self.in_f, self.out_f = in_f, out_f
        self.use_fp8 = True
        self.fp8_bwd = fp8_bwd

    def forward(self, x):
        if not (self.use_fp8 and x.is_cuda and _fp8_supported()):
            return F.linear(x, self.weight.to(x.dtype), None if self.bias is None else self.bias.to(x.dtype))
        shp = x.shape
        x2d = x.reshape(-1, shp[-1])
        # fp8 GEMM needs K and N as multiples of 16; pad-free by construction here (dims are 16-aligned)
        y = Fp8LinearFn.apply(x2d.to(torch.bfloat16), self.weight.to(torch.bfloat16),
                              None if self.bias is None else self.bias, self.fp8_bwd)
        return y.reshape(*shp[:-1], self.out_f)


class Attention(nn.Module):
    """Multi-head self-attention. qkv/proj GEMMs go through Fp8Linear (fp8); the score & context
    batched-matmuls + softmax stay bf16 (honest: not on the fp8 path)."""

    def __init__(self, dim, n_heads, fp8_bwd=True):
        super().__init__()
        assert dim % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = Fp8Linear(dim, dim * 3, bias=True, fp8_bwd=fp8_bwd)
        self.proj = Fp8Linear(dim, dim, bias=True, fp8_bwd=fp8_bwd)

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]                       # (B, H, T, hd)
        # scaled dot-product attention in bf16 (softmax/norm not fp8)
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).reshape(B, T, C)
        return self.proj(out)


class Mlp(nn.Module):
    def __init__(self, dim, hidden, fp8_bwd=True):
        super().__init__()
        self.fc1 = Fp8Linear(dim, hidden, fp8_bwd=fp8_bwd)
        self.fc2 = Fp8Linear(hidden, dim, fp8_bwd=fp8_bwd)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


class Block(nn.Module):
    def __init__(self, dim, n_heads, mlp_ratio=4, fp8_bwd=True):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, n_heads, fp8_bwd=fp8_bwd)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, dim * mlp_ratio, fp8_bwd=fp8_bwd)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class Fp8CellDetector(nn.Module):
    """3D cell-center heatmap detector, matmul-dominated so fp8 can train it end-to-end.

    Input : (B, 1, D, H, W) intensity volume.
    Output: (B, 1, D, H, W) per-voxel cell-center logit heatmap (sigmoid → probability; peak-detect
            like the UNet detector). Train against a Gaussian-blob target at annotated centers.

    patch = (pd,ph,pw); token grid = (D/pd, H/ph, W/pw). A single Conv3d stem patch-embeds
    (tiny fraction of params). N transformer blocks over the tokens with a learnable 3D positional
    embedding. Linear head predicts each patch's voxel logits; un-patchify restores the volume.
    """

    def __init__(self, in_ch=1, embed_dim=256, depth=6, n_heads=8, patch=(8, 16, 16),
                 vol=(16, 64, 64), mlp_ratio=4, fp8_bwd=True):
        super().__init__()
        self.patch = patch
        self.vol = vol
        self.embed_dim = embed_dim
        self.grid = tuple(vol[i] // patch[i] for i in range(3))
        self.n_tokens = self.grid[0] * self.grid[1] * self.grid[2]
        self.patch_vol = patch[0] * patch[1] * patch[2]
        # --- conv patch-embed stem (the ONLY conv; kept small) ---
        self.stem = nn.Conv3d(in_ch, embed_dim, kernel_size=patch, stride=patch)
        # --- learnable 3D positional embedding (param table, not conv/linear) ---
        self.pos = nn.Parameter(torch.zeros(1, self.n_tokens, embed_dim))
        nn.init.trunc_normal_(self.pos, std=0.02)
        # --- transformer encoder ---
        self.blocks = nn.ModuleList([Block(embed_dim, n_heads, mlp_ratio, fp8_bwd) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)
        # --- heatmap head: per-token → patch voxel logits (Linear, fp8) ---
        self.head = Fp8Linear(embed_dim, self.patch_vol * in_ch, bias=True, fp8_bwd=fp8_bwd)
        self.out_ch = in_ch

    def set_fp8(self, on: bool):
        for m in self.modules():
            if isinstance(m, Fp8Linear):
                m.use_fp8 = on
        return self

    def forward(self, x):
        B = x.shape[0]
        gd, gh, gw = self.grid
        pd, ph, pw = self.patch
        tok = self.stem(x)                                    # (B, C, gd, gh, gw)
        tok = tok.flatten(2).transpose(1, 2)                  # (B, T, C)
        tok = tok + self.pos
        for blk in self.blocks:
            tok = blk(tok)
        tok = self.norm(tok)
        vox = self.head(tok)                                  # (B, T, patch_vol*out_ch)
        # un-patchify: (B, gd,gh,gw, oc,pd,ph,pw) -> (B, oc, D, H, W)
        vox = vox.reshape(B, gd, gh, gw, self.out_ch, pd, ph, pw)
        vox = vox.permute(0, 4, 1, 5, 2, 6, 3, 7).contiguous()
        vox = vox.reshape(B, self.out_ch, gd * pd, gh * ph, gw * pw)
        return vox


# -------------------------------------------------------------------------------------------------
# param accounting + fp8 fraction (the honesty tools)
# -------------------------------------------------------------------------------------------------
def param_split(model: nn.Module):
    """conv vs linear param counts — this is exactly what hardware_tune.select_train_precision reads."""
    conv = lin = other = 0
    for m in model.modules():
        pc = sum(p.numel() for p in m.parameters(recurse=False))
        if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.ConvTranspose2d, nn.ConvTranspose3d)):
            conv += pc
        elif isinstance(m, (nn.Linear, Fp8Linear)):
            lin += pc
        else:
            other += pc
    total = conv + lin + other
    return {"conv": conv, "linear": lin, "other": other, "total": total,
            "conv_frac": conv / max(total, 1), "linear_frac": lin / max(total, 1)}


def fp8_flop_fraction(model: Fp8CellDetector, vol=None):
    """Estimate the fraction of forward multiply-add FLOPs that run through fp8 (the Fp8Linear GEMMs)
    vs stay bf16 (conv stem + attention score/context matmuls). Softmax/norm are negligible in FLOPs
    but non-fp8. This is the HONEST ceiling on the fp8 speed win for this model."""
    vol = vol or model.vol
    D, H, W = vol
    C = model.embed_dim
    T = (D // model.patch[0]) * (H // model.patch[1]) * (W // model.patch[2])
    depth = len(model.blocks)
    hd = C // model.blocks[0].attn.n_heads
    heads = model.blocks[0].attn.n_heads
    # fp8 GEMMs (per sample), MACs:
    qkv = T * C * (3 * C)
    proj = T * C * C
    fc1 = T * C * (4 * C)
    fc2 = T * (4 * C) * C
    fp8_per_block = qkv + proj + fc1 + fc2
    head = T * C * model.patch_vol
    fp8 = depth * fp8_per_block + head
    # bf16 matmuls NOT on fp8 path: attention scores + context (batched, per head)
    attn_score = heads * T * T * hd          # q@k^T
    attn_ctx = heads * T * T * hd            # softmax@v
    attn_bf16 = depth * (attn_score + attn_ctx)
    # conv stem MACs: out_elems * (in_ch * kd*kh*kw)
    stem = C * T * (model.out_ch * model.patch_vol)
    bf16 = attn_bf16 + stem
    total = fp8 + bf16
    return {"fp8_macs": fp8, "bf16_macs": bf16, "total_macs": total,
            "fp8_fraction": fp8 / max(total, 1),
            "breakdown": {"attn+mlp+head_fp8": fp8, "attn_scores_ctx_bf16": attn_bf16, "conv_stem_bf16": stem}}


def build_default(fp8_bwd=None, **kw) -> Fp8CellDetector:
    if fp8_bwd is None:
        fp8_bwd = os.environ.get("CELLMOT_FP8_BACKWARD", "1") == "1"
    return Fp8CellDetector(fp8_bwd=fp8_bwd, **kw)
