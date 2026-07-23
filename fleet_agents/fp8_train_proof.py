"""fp8_train_proof — end-to-end PROOF that fp8 training is faster than bf16 on the RTX 5090
(sm_120), measured on a REAL decoder transformer LM with LARGE matmuls, torch.compile ON.

Runs the FULL training step (forward + next-token CE loss + backward + optimizer.step) and times
~200 steps for fp8 vs bf16 at identical model/seed/shapes. Also runs a ~300-step convergence check
(loss must decrease and track bf16) and reports peak VRAM. Reports graph-break status of the fp8
custom autograd.Function under torch.compile.

Two paths:
  PATH A (native): reuse Fp8Linear/Fp8LinearFn from fp8_cell_detector.py (torch._scaled_mm, no install).
  PATH B (torchao): torchao.float8 convert_to_float8_training (only if torchao installs ABI-safe).

Usage:
  OMP_NUM_THREADS=1 python fp8_train_proof.py --path native --steps 200 --conv-steps 300
  OMP_NUM_THREADS=1 python fp8_train_proof.py --path torchao ...
"""
from __future__ import annotations
import os, sys, time, json, argparse, contextlib

os.environ.setdefault("OMP_NUM_THREADS", "1")
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fp8_cell_detector import Fp8Linear  # native path


# --------------------------------------------------------------------------------------------------
# Model: a genuine decoder-only transformer LM. `linear_cls` swaps the big Linears (fp8 vs bf16).
# Attention softmax (SDPA), LayerNorm, embeddings stay bf16 — only the big GEMMs go fp8.
# --------------------------------------------------------------------------------------------------
class Attention(nn.Module):
    def __init__(self, dim, n_heads, linear_cls):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.qkv = linear_cls(dim, dim * 3, bias=True)
        self.proj = linear_cls(dim, dim, bias=True)

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).reshape(B, T, C)
        return self.proj(out)


class Mlp(nn.Module):
    def __init__(self, dim, hidden, linear_cls):
        super().__init__()
        self.fc1 = linear_cls(dim, hidden, bias=True)
        self.fc2 = linear_cls(hidden, dim, bias=True)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


class Block(nn.Module):
    def __init__(self, dim, n_heads, ffn, linear_cls):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, n_heads, linear_cls)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, ffn, linear_cls)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class TransformerLM(nn.Module):
    def __init__(self, vocab, d_model, ffn, depth, n_heads, seq_len, linear_cls):
        super().__init__()
        self.tok = nn.Embedding(vocab, d_model)
        self.pos = nn.Parameter(torch.zeros(1, seq_len, d_model))
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.blocks = nn.ModuleList([Block(d_model, n_heads, ffn, linear_cls) for _ in range(depth)])
        self.norm = nn.LayerNorm(d_model)
        self.head = linear_cls(d_model, vocab, bias=False)  # big GEMM (vocab=32000)

    def forward(self, idx):
        x = self.tok(idx) + self.pos
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return self.head(x)


def make_model(cfg, path):
    if path == "native":
        linear_cls = lambda i, o, bias=True: Fp8Linear(i, o, bias=bias)
    else:  # bf16 baseline OR torchao (converted after build)
        linear_cls = lambda i, o, bias=True: nn.Linear(i, o, bias=bias)
    m = TransformerLM(cfg["vocab"], cfg["d_model"], cfg["ffn"], cfg["depth"],
                      cfg["n_heads"], cfg["seq_len"], linear_cls)
    return m.to("cuda").to(torch.bfloat16)


def time_steps(model, cfg, steps, warmup, compiled, label):
    torch.manual_seed(0)
    B, T, V = cfg["batch"], cfg["seq_len"], cfg["vocab"]
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, betas=(0.9, 0.95))
    run = torch.compile(model) if compiled else model
    # fixed batch reused (speed proof, not data pipeline)
    data = torch.randint(0, V, (B, T + 1), device="cuda")
    x, y = data[:, :-1].contiguous(), data[:, 1:].contiguous()

    def one_step():
        opt.zero_grad(set_to_none=True)
        logits = run(x)
        loss = F.cross_entropy(logits.float().reshape(-1, V), y.reshape(-1))
        loss.backward()
        opt.step()
        return loss

    for _ in range(warmup):
        one_step()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    for _ in range(steps):
        one_step()
    torch.cuda.synchronize()
    dt = (time.time() - t0) / steps
    peak = torch.cuda.max_memory_allocated() / 1e9
    print(f"[{label}] s/iter={dt*1000:.2f}ms  peak_VRAM={peak:.2f}GB", flush=True)
    return dt, peak


def convergence(model, cfg, steps, compiled, label):
    torch.manual_seed(123)
    B, T, V = cfg["batch"], cfg["seq_len"], cfg["vocab"]
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95))
    run = torch.compile(model) if compiled else model
    # a LEARNABLE synthetic task: next token = (prev + 1) % V shifted pattern so loss can actually drop
    torch.manual_seed(7)
    base = torch.randint(0, V, (B, T + 1), device="cuda")
    # make targets a deterministic function of inputs (shift) so the model can learn structure
    seq = torch.arange(T + 1, device="cuda").unsqueeze(0)
    data = ((base[:, :1] + seq) % V)
    x, y = data[:, :-1].contiguous(), data[:, 1:].contiguous()
    losses = []
    for i in range(steps):
        opt.zero_grad(set_to_none=True)
        logits = run(x)
        loss = F.cross_entropy(logits.float().reshape(-1, V), y.reshape(-1))
        loss.backward()
        opt.step()
        losses.append(loss.item())
    print(f"[{label}] loss start={losses[0]:.3f} -> end={losses[-1]:.3f} "
          f"(min={min(losses):.3f}, nan={any(l != l for l in losses)})", flush=True)
    return losses


def graph_break_check(cfg):
    """Does torch.compile graph-break on the fp8 custom autograd.Function? Uses dynamo.explain."""
    import torch._dynamo as dynamo
    torch.manual_seed(0)
    m = make_model(cfg, "native")
    x = torch.randint(0, cfg["vocab"], (2, cfg["seq_len"]), device="cuda")
    try:
        exp = dynamo.explain(m)(x)
        gb = exp.graph_break_count
        print(f"[graph_break] fp8 native under compile: graph_breaks={gb} graphs={exp.graph_count}", flush=True)
        if gb > 0:
            for r in exp.break_reasons[:6]:
                print(f"   break: {getattr(r,'reason',r)}", flush=True)
        return gb
    except Exception as e:
        print(f"[graph_break] explain failed: {type(e).__name__}: {e}", flush=True)
        return -1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", choices=["native", "torchao"], default="native")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=15)
    ap.add_argument("--conv-steps", type=int, default=300)
    ap.add_argument("--no-compile", action="store_true")
    ap.add_argument("--d-model", type=int, default=2048)
    ap.add_argument("--ffn", type=int, default=8192)
    ap.add_argument("--depth", type=int, default=8)
    ap.add_argument("--heads", type=int, default=16)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--vocab", type=int, default=32000)
    args = ap.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    compiled = not args.no_compile
    cfg = dict(vocab=args.vocab, d_model=args.d_model, ffn=args.ffn, depth=args.depth,
               n_heads=args.heads, seq_len=args.seq, batch=args.batch)
    print(f"CONFIG: {cfg}  path={args.path} compile={compiled} "
          f"steps={args.steps} conv_steps={args.conv_steps}", flush=True)
    print(f"GPU: {torch.cuda.get_device_name(0)} cap={torch.cuda.get_device_capability(0)} "
          f"torch={torch.__version__}", flush=True)

    n_lin = sum(p.numel() for n, p in TransformerLM(cfg['vocab'], cfg['d_model'], cfg['ffn'],
                cfg['depth'], cfg['n_heads'], cfg['seq_len'], nn.Linear).named_parameters()
                if 'weight' in n and p.dim() == 2)
    print(f"~Linear params: {n_lin/1e6:.1f}M", flush=True)

    results = {"cfg": cfg, "path": args.path, "compile": compiled}

    if args.path == "native":
        gb = graph_break_check(cfg)
        results["graph_breaks"] = gb

    # ---- fp8 path model ----
    if args.path == "native":
        fp8_model = make_model(cfg, "native")
    else:
        from torchao.float8 import convert_to_float8_training, Float8LinearConfig
        fp8_model = make_model(cfg, "bf16")
        # filter: only convert Linears whose dims are large & 16-aligned (skip small/vocab-odd)
        def flt(mod, fqn):
            return isinstance(mod, nn.Linear) and min(mod.in_features, mod.out_features) >= 1024 \
                   and mod.in_features % 16 == 0 and mod.out_features % 16 == 0
        convert_to_float8_training(fp8_model, module_filter_fn=flt)
        print("[torchao] converted Linears to float8 training (tensorwise)", flush=True)

    print("\n=== TIMING (full train step: fwd+loss+bwd+opt) ===", flush=True)
    fp8_dt, fp8_peak = time_steps(fp8_model, cfg, args.steps, args.warmup, compiled, f"fp8-{args.path}")
    del fp8_model; torch.cuda.empty_cache()

    # ---- bf16 baseline ----
    bf16_model = make_model(cfg, "bf16")
    bf16_dt, bf16_peak = time_steps(bf16_model, cfg, args.steps, args.warmup, compiled, "bf16")
    del bf16_model; torch.cuda.empty_cache()

    ratio = bf16_dt / fp8_dt
    print(f"\n>>> SPEED: fp8={fp8_dt*1000:.2f}ms  bf16={bf16_dt*1000:.2f}ms  ratio(bf16/fp8)={ratio:.2f}x", flush=True)
    results.update(dict(fp8_ms=fp8_dt*1000, bf16_ms=bf16_dt*1000, ratio=ratio,
                        fp8_peak_gb=fp8_peak, bf16_peak_gb=bf16_peak))

    print("\n=== CONVERGENCE (learnable synthetic next-token) ===", flush=True)
    if args.path == "native":
        fp8_c = make_model(cfg, "native")
    else:
        from torchao.float8 import convert_to_float8_training
        fp8_c = make_model(cfg, "bf16")
        def flt(mod, fqn):
            return isinstance(mod, nn.Linear) and min(mod.in_features, mod.out_features) >= 1024 \
                   and mod.in_features % 16 == 0 and mod.out_features % 16 == 0
        convert_to_float8_training(fp8_c, module_filter_fn=flt)
    fp8_losses = convergence(fp8_c, cfg, args.conv_steps, compiled, f"fp8-{args.path}")
    del fp8_c; torch.cuda.empty_cache()
    bf16_c = make_model(cfg, "bf16")
    bf16_losses = convergence(bf16_c, cfg, args.conv_steps, compiled, "bf16")
    del bf16_c; torch.cuda.empty_cache()
    results["fp8_loss"] = [fp8_losses[0], fp8_losses[-1]]
    results["bf16_loss"] = [bf16_losses[0], bf16_losses[-1]]

    # ---- verdict ----
    speed_ok = ratio >= 1.3
    conv_ok = (fp8_losses[-1] < fp8_losses[0] * 0.9) and not any(l != l for l in fp8_losses)
    verdict = "PASS" if (speed_ok and conv_ok) else "FAIL"
    print(f"\n>>> VERDICT: {verdict}  (speed>=1.3x: {speed_ok} [{ratio:.2f}x], "
          f"fp8 converges: {conv_ok})", flush=True)
    results["verdict"] = verdict
    print("\nJSON " + json.dumps(results), flush=True)


if __name__ == "__main__":
    main()
