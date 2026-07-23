"""lowbit_qat_test — data-wise verifier for the ternary / low-bit QAT primitives.

Proves the load-bearing properties (grounded in BitNet b1.58 + the Bonsai ternary format):
  1. ternary_quantize gives values in {-1,0,+1}·scale, preserves sign, and the absmean scale is exact.
  2. int4/int8 fake-quant round-trips within a bounded tolerance and improves with more bits.
  3. the STE passes gradients: grad through the quantizer is finite and non-zero (a plain round() is not
     differentiable — the whole point of the STE).
  4. wrap_qat swaps ONLY Linears (norms/embeddings/head kept fp) and a tiny wrapped model still trains
     (loss falls) under quantized weights.
  5. effective_bits matches the Bonsai group-wise accounting (ternary g128 ≈ 1.71, 1-bit g128 ≈ 1.125).
  6. agent contract (run → done, learns).
"""
import os, sys
COMP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, COMP); sys.path.insert(0, os.path.join(COMP, "tools", "researchpapers"))
import torch
import torch.nn as nn
from fleet_agents import lowbit_qat as L


def _run():
    print("=== LOWBIT-QAT VERIFIER ===")
    torch.manual_seed(0)
    checks = {}

    # 1. ternary quantize — values in {-1,0,1}, sign preserved, absmean scale
    W = torch.randn(32, 48)
    t, scale = L.ternary_quantize(W)
    uniq = set(torch.unique(t).tolist())
    checks["ternary_values"] = uniq.issubset({-1.0, 0.0, 1.0})
    checks["ternary_has_zero"] = 0.0 in uniq          # the expressive zero state must be used
    checks["ternary_scale_absmean"] = torch.allclose(scale, W.abs().mean(), atol=1e-6)
    # sign preserved wherever the ternary code is non-zero
    nz = t != 0
    checks["ternary_sign_preserved"] = bool((torch.sign(t[nz]) == torch.sign(W[nz])).all())
    # per-channel path: one scale per output row
    tpc, spc = L.ternary_quantize(W, per_channel=True)
    checks["ternary_perchannel_shape"] = tuple(spc.shape) == (32, 1)
    print(f"  -> ternary uniq={sorted(uniq)}  scale={float(scale):.4f}  perch_scale_shape={tuple(spc.shape)}")

    # 2. int fake-quant round-trip within tolerance + monotone in bits
    x = torch.randn(64, 64)
    errs = {}
    for b in (4, 8):
        xq, sc = L.int_fake_quant(x, bits=b, per_channel=True)
        errs[b] = float((xq - x).abs().max())
        # every value must land on a grid multiple of its (per-row) scale
        grid_ok = torch.allclose(xq, torch.round(xq / sc) * sc, atol=1e-5)
        checks[f"int{b}_on_grid"] = grid_ok
    checks["int4_roundtrip_tol"] = errs[4] < 0.5 * x.abs().max().item()
    checks["more_bits_lower_err"] = errs[8] < errs[4]
    print(f"  -> int4 maxerr={errs[4]:.4f}  int8 maxerr={errs[8]:.4f}")

    # 3. STE passes gradients (round() alone would give zero grad)
    Wg = torch.randn(8, 8, requires_grad=True)
    out = L.ste_ternary(Wg).pow(2).sum()
    out.backward()
    g = Wg.grad
    checks["ste_grad_finite"] = bool(torch.isfinite(g).all())
    checks["ste_grad_nonzero"] = bool(g.abs().sum() > 0)
    # int STE too
    Wg2 = torch.randn(8, 8, requires_grad=True)
    L.ste_quant(Wg2, bits=4).pow(2).sum().backward()
    checks["ste_int_grad_nonzero"] = bool(Wg2.grad.abs().sum() > 0)
    print(f"  -> STE grad |sum|={float(g.abs().sum()):.3f}  finite={checks['ste_grad_finite']}")

    # 4. wrap_qat swaps ONLY Linears, keeps norm/embed/head fp, model still trains
    model = nn.Sequential()
    model.add_module("embed", nn.Embedding(10, 16))          # must NOT be swapped (not Linear anyway)
    model.add_module("fc1", nn.Linear(16, 32))
    model.add_module("norm", nn.LayerNorm(32))               # has an internal? no — but name-skip guard
    model.add_module("fc2", nn.Linear(32, 8))
    model.add_module("lm_head", nn.Linear(8, 4))             # name-skipped → stays fp
    n_lin_before = sum(isinstance(m, nn.Linear) for m in model.modules())
    _, n_swapped = L.wrap_qat(model, bits=4, scheme="ternary")
    # fc1, fc2 swapped (2); lm_head skipped by name
    checks["wrap_swapped_two"] = n_swapped == 2
    checks["lm_head_kept_fp"] = isinstance(model.lm_head, nn.Linear) and not isinstance(model.lm_head, L.QuantLinear)
    checks["embed_kept"] = isinstance(model.embed, nn.Embedding)
    checks["fc1_is_quant"] = isinstance(model.fc1, L.QuantLinear)
    # norm untouched
    checks["norm_kept"] = isinstance(model.norm, nn.LayerNorm)
    print(f"  -> wrap_qat: {n_lin_before} Linears, swapped {n_swapped} (lm_head kept fp)")

    # tiny end-to-end train: a wrapped MLP regression should reduce loss under ternary weights
    torch.manual_seed(1)
    net = nn.Sequential(nn.Linear(12, 48), nn.ReLU(), nn.Linear(48, 1))
    net, ns = L.wrap_qat(net, bits=4, scheme="ternary")
    X = torch.randn(256, 12); Wt = torch.randn(12, 1); Y = X @ Wt
    losses = L.qat_finetune(net, X, Y, steps=200, lr=5e-3)
    checks["qat_swapped_all"] = ns == 2
    checks["qat_loss_falls"] = losses[-1] < 0.6 * losses[0]
    print(f"  -> ternary QAT train: {losses[0]:.3f} -> {losses[-1]:.3f}")

    # int4 QAT also learns
    torch.manual_seed(1)
    net2 = nn.Sequential(nn.Linear(12, 48), nn.ReLU(), nn.Linear(48, 1))
    L.wrap_qat(net2, bits=4, scheme="int4", a_bits=8)
    losses2 = L.qat_finetune(net2, X, Y, steps=200, lr=5e-3)
    checks["int4_qat_loss_falls"] = losses2[-1] < 0.6 * losses2[0]
    print(f"  -> int4/a8 QAT train: {losses2[0]:.3f} -> {losses2[-1]:.3f}")

    # 5. effective-bits accounting (Bonsai group-wise)
    checks["ebits_ternary"] = abs(L.effective_bits("ternary", 128) - 1.71) < 0.02
    checks["ebits_onebit"] = abs(L.effective_bits("onebit", 128) - 1.125) < 0.01
    checks["ebits_int4"] = L.effective_bits("int4") == 4.0
    print(f"  -> effective bits: ternary g128={L.effective_bits('ternary',128)}  "
          f"1-bit g128={L.effective_bits('onebit',128)}")

    # 5b. Kimi-K3 MX (microscaling) formats — MXFP4 weights + MXFP8 activations
    xk = torch.randn(4, 128) * 3.0
    qe = {f: (L.mxfp_quantize(xk, fmt=f)[0] - xk).abs().mean().item() for f in ("e2m1", "e4m3", "e5m2")}
    checks["mx_shared_scale_shape"] = tuple(L.mxfp_quantize(xk, fmt="e2m1", block_size=32)[1].shape) == (4, 4)
    checks["mxfp8_beats_mxfp4"] = qe["e4m3"] < qe["e2m1"]            # 8-bit element < 4-bit element error
    onblk = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6.]).repeat(1, 4)  # exact E2M1 grid, one block
    checks["mxfp4_grid_exact"] = torch.allclose(L.mxfp_quantize(onblk, fmt="e2m1", block_size=32)[0], onblk, atol=1e-5)
    checks["mx_ebits_fp4"] = abs(L.mx_effective_bits("e2m1", 32) - 4.25) < 1e-6
    checks["mx_ebits_fp8"] = abs(L.mx_effective_bits("e4m3", 32) - 8.25) < 1e-6
    # STE trains through MX fake-quant (K3 QAT cell): MXFP4 weights + MXFP8 acts, loss must fall
    torch.manual_seed(1)
    net = nn.Sequential(L.QuantLinear(16, 32, scheme="mxfp4", act_mx="e4m3"),
                        nn.ReLU(), L.QuantLinear(32, 1, scheme="mxfp4", act_mx="e4m3"))
    Xd = torch.randn(128, 16); Yd = torch.randn(128, 1)
    opt = torch.optim.Adam(net.parameters(), lr=5e-3); l_mx = []
    for _ in range(120):
        opt.zero_grad(); L_ = ((net(Xd) - Yd) ** 2).mean(); L_.backward(); opt.step(); l_mx.append(float(L_))
    print(f"  -> MX: mxfp4 err={qe['e2m1']:.3f} mxfp8 err={qe['e4m3']:.3f}  MXQAT loss {l_mx[0]:.3f}->{l_mx[-1]:.3f}")
    checks["mx_qat_loss_falls"] = l_mx[-1] < 0.6 * l_mx[0]

    # 5c. NVIDIA NVFP4 (Blackwell / RTX 5090 native 4-bit; Unsloth Gemma-4 quants)
    xn = torch.randn(4, 128) * 3.0
    qn, bscale, glob = L.nvfp4_quantize(xn, block_size=16)
    nv_err = (qn - xn).abs().mean().item() / xn.abs().mean().item()
    mx_err = (L.mxfp_quantize(xn, fmt="e2m1", block_size=32)[0] - xn).abs().mean().item() / xn.abs().mean().item()
    checks["nvfp4_block16_scale_shape"] = tuple(bscale.shape) == (4, 8)     # 128/16 = 8 blocks
    checks["nvfp4_global_scalar"] = glob.ndim == 0 and float(glob) > 0
    checks["nvfp4_beats_mxfp4"] = nv_err < mx_err                          # finer 16-blk + E4M3 scale + global
    checks["nvfp4_ebits"] = abs(L.nvfp4_effective_bits(16) - 4.5) < 1e-6
    # NVFP4 STE trains (QAT finetune cell for the 5090)
    torch.manual_seed(2)
    net = nn.Sequential(L.QuantLinear(16, 32, scheme="nvfp4"), nn.ReLU(), L.QuantLinear(32, 1, scheme="nvfp4"))
    Xn = torch.randn(128, 16); Yn = torch.randn(128, 1)
    opt = torch.optim.Adam(net.parameters(), lr=5e-3); ln = []
    for _ in range(120):
        opt.zero_grad(); Ln = ((net(Xn) - Yn) ** 2).mean(); Ln.backward(); opt.step(); ln.append(float(Ln))
    print(f"  -> NVFP4: err={nv_err:.3f} (MXFP4 {mx_err:.3f}) bits={L.nvfp4_effective_bits(16)}  QAT {ln[0]:.3f}->{ln[-1]:.3f}")
    checks["nvfp4_qat_loss_falls"] = ln[-1] < 0.6 * ln[0]

    # 5d. capability-gated 4-bit default (adopt NVFP4 natively where supported, degrade on T4)
    checks["pref_blackwell_nvfp4"] = L.preferred_4bit_scheme((12, 0)) == "nvfp4"   # RTX 5090 sm_120
    checks["pref_b200_nvfp4"] = L.preferred_4bit_scheme((10, 0)) == "nvfp4"        # B200 sm_100
    checks["pref_ada_mxfp4"] = L.preferred_4bit_scheme((8, 9)) == "mxfp4"          # 4090/H100 emulated
    checks["pref_t4_int8"] = L.preferred_4bit_scheme((7, 5)) == "int8"             # Kaggle T4 sm_75 — NO fp4
    checks["pref_none_int8"] = L.preferred_4bit_scheme(None) in ("int8", L.preferred_4bit_scheme())
    print(f"  -> preferred 4-bit: 5090→{L.preferred_4bit_scheme((12,0))} T4→{L.preferred_4bit_scheme((7,5))}")

    # 6. agent contract
    st, dta, to, msg = L.run({"spec": {"scheme": "ternary", "steps": 150, "gpu": False}}, "t")
    checks["agent_done"] = st == "done"
    checks["agent_learns"] = bool(dta.get("learned"))
    checks["agent_reports_bits"] = dta.get("effective_bits", 0) > 1.0

    # =====================================================================================================
    # APPEND-ONLY (2026-07-16): checks for the finished end-to-end low-bit TRAINING capability.
    # =====================================================================================================

    # 7. GROUP-WISE quantization (Bonsai/BitNet g128 block scaling)
    torch.manual_seed(3)
    Wg = torch.randn(4, 256)                                  # last dim 256 = 2 groups of 128
    tg, sg = L.ternary_quantize(Wg, group_size=128)
    checks["grp_ternary_on_grid"] = set(torch.unique(tg).tolist()).issubset({-1.0, 0.0, 1.0})
    checks["grp_ternary_scale_shape"] = tuple(sg.shape) == (4, 2)   # one scale per (row, group)
    nzg = tg != 0
    checks["grp_ternary_sign"] = bool((torch.sign(tg[nzg]) == torch.sign(Wg[nzg])).all())
    # scales differ across groups (block scaling really is per-group)
    checks["grp_ternary_scales_differ"] = bool((sg[:, 0] != sg[:, 1]).any())
    # grouped dequant round-trips shape and stays on the ternary·scale grid
    dqg = L.dequant_ternary(tg, sg, 128)
    checks["grp_dequant_shape"] = tuple(dqg.shape) == tuple(Wg.shape)
    # grouped int: per-group scales differ + values land on each group's grid
    qi, si = L.int_fake_quant(Wg, bits=4, group_size=128)
    checks["grp_int_scale_shape"] = tuple(si.shape) == (4, 2)
    checks["grp_int_scales_differ"] = bool((si[:, 0] != si[:, 1]).any())
    qi_b = qi.reshape(4, 2, 128)
    on_grid = torch.allclose(qi_b, torch.round(qi_b / si.unsqueeze(-1)) * si.unsqueeze(-1), atol=1e-5)
    checks["grp_int_on_grid"] = on_grid
    print(f"  -> group-wise: ternary scales={sg[0].tolist()} int scales={si[0].tolist()}")

    # 8. ACTIVATION quant — round-trip within tol + STE grad flows
    xa = torch.randn(8, 32, requires_grad=True)
    ya = L.act_fake_quant(xa, bits=8, per_token=True)
    checks["act_roundtrip_tol"] = float((ya - xa).abs().max()) < 0.1 * xa.abs().max().item()
    L.act_fake_quant(xa, bits=8).pow(2).sum().backward()
    checks["act_ste_grad"] = bool(xa.grad is not None and xa.grad.abs().sum() > 0 and torch.isfinite(xa.grad).all())
    print(f"  -> act int8 maxerr={float((ya - xa).abs().max()):.4f}  ste_grad={checks['act_ste_grad']}")

    # 9. QuantLinear with act_bits still trains (W+A quant path)
    torch.manual_seed(1)
    netwa = nn.Sequential(nn.Linear(12, 48), nn.ReLU(), nn.Linear(48, 1))
    L.wrap_qat(netwa, scheme="ternary", act_bits=8, group_size=128)
    Xwa = torch.randn(256, 12); Ywa = Xwa @ torch.randn(12, 1)
    lwa = L.qat_finetune(netwa, Xwa, Ywa, steps=200, lr=5e-3)
    checks["wa_quant_trains"] = lwa[-1] < 0.6 * lwa[0]
    print(f"  -> W-ternary/A8 QAT: {lwa[0]:.3f} -> {lwa[-1]:.3f}")

    # 10. LowBitAdam trains (>30% drop) AND packed state < fp32 Adam state
    torch.manual_seed(1)
    netlb = nn.Sequential(nn.Linear(12, 64), nn.ReLU(), nn.Linear(64, 1))
    Xlb = torch.randn(256, 12); Ylb = Xlb @ torch.randn(12, 1)
    opt = L.LowBitAdam(netlb.parameters(), lr=5e-3, state_bits=8, block=128)
    lf = nn.MSELoss(); l0lb = None
    for i in range(300):
        opt.zero_grad(); loss = lf(netlb(Xlb), Ylb); loss.backward(); opt.step()
        if i == 0:
            l0lb = float(loss)
    l1lb = float(loss)
    checks["lowbitadam_trains"] = l1lb < 0.7 * l0lb
    checks["lowbitadam_state_smaller"] = opt.state_bytes() < opt.fp32_state_bytes()
    print(f"  -> LowBitAdam: {l0lb:.3f} -> {l1lb:.3f}  state {opt.state_bytes()}B < fp32 {opt.fp32_state_bytes()}B")

    # 11. lowbit_finetune end-to-end: final<initial, effective_bits<4 for ternary, amp reflects hardware
    torch.manual_seed(2)
    modelft = nn.Sequential(nn.Linear(16, 64), nn.ReLU(), nn.Linear(64, 1))
    Wft = torch.randn(16, 1)
    batches = [(lambda x: (x, x @ Wft))(torch.randn(128, 16)) for _ in range(8)]
    res = L.lowbit_finetune(modelft, batches, scheme="ternary", group_size=128, optimizer="lowbit", epochs=5)
    checks["finetune_loss_falls"] = res["final_loss"] < res["initial_loss"]
    checks["finetune_ebits_sub4"] = res["effective_bits"] < 4.0
    checks["finetune_layers"] = res["quantized_layers"] == 2
    if torch.cuda.is_available():
        import fleet_agents.hardware_tune as HW
        want = (HW.load_config() or {}).get("amp_dtype", "fp32")
        modelc = nn.Sequential(nn.Linear(16, 64), nn.ReLU(), nn.Linear(64, 1)).cuda()
        Wc = torch.randn(16, 1, device="cuda")
        bc = [(lambda x: (x, x @ Wc))(torch.randn(128, 16, device="cuda")) for _ in range(6)]
        rc = L.lowbit_finetune(modelc, bc, scheme="ternary", epochs=3)
        checks["finetune_amp_hw"] = rc["amp_dtype_used"] in (want, "fp32")   # matches hw cfg (or fp32 if cfg not bf16)
    else:
        checks["finetune_amp_hw"] = res["amp_dtype_used"] == "fp32"
    print(f"  -> lowbit_finetune: {res['initial_loss']:.3f} -> {res['final_loss']:.3f}  "
          f"ebits={res['effective_bits']}  amp={res['amp_dtype_used']}")

    # 12. REAL packing round-trip EXACT + memory ratio + KV quant round-trip
    q3 = torch.randint(-1, 2, (3, 101)).float()
    pk3 = L.pack_ternary(q3, 128); un3 = L.unpack_ternary(pk3)
    checks["pack_ternary_exact"] = bool((un3 == q3).all())
    codes = torch.randint(-8, 8, (5, 33)).float()
    pk4 = L.pack_int4(codes); un4 = L.unpack_int4(pk4)
    checks["pack_int4_exact"] = bool((un4 == codes).all())
    mem = L.effective_memory_bytes((4096, 4096), "ternary", 128)
    checks["mem_ratio_over4x"] = mem["ratio_vs_fp16"] > 4.0
    memi = L.effective_memory_bytes((4096, 4096), "int4", 128)
    checks["mem_int4_ratio"] = 3.5 < memi["ratio_vs_fp16"] < 4.5      # int4 ~4x vs fp16
    kk = torch.randn(2, 4, 10, 32); vv = torch.randn(2, 4, 10, 32)
    kq, vq = L.quantize_kv(kk, vv, bits=4); kd, vd = L.dequantize_kv(kq, vq)
    checks["kv_roundtrip_tol"] = (float((kd - kk).abs().max()) < 0.3 * kk.abs().max().item()
                                  and float((vd - vv).abs().max()) < 0.3 * vv.abs().max().item())
    print(f"  -> pack ternary bytes={pk3['packed'].numel()} (~1.6bpw)  mem ternary={mem['ratio_vs_fp16']}x "
          f"int4={memi['ratio_vs_fp16']}x  kv int4 err={float((kd - kk).abs().max()):.3f}")

    # 13. agent run modes: finetune -> done+falling loss; memory -> done+ratio; empty -> done (unchanged)
    stf, df, _, _ = L.run({"spec": {"mode": "finetune", "gpu": False, "epochs": 3}}, "t")
    checks["agent_mode_finetune"] = stf == "done" and bool(df.get("loss_fell"))
    stm, dm, _, _ = L.run({"spec": {"mode": "memory", "shape": [2048, 2048], "scheme": "ternary"}}, "t")
    checks["agent_mode_memory"] = stm == "done" and dm.get("ratio_vs_fp16", 0) > 4.0
    ste, de, _, _ = L.run({}, "t")
    checks["agent_mode_empty"] = ste == "done" and bool(de.get("learned"))
    print(f"  -> agent modes: finetune fell={df.get('loss_fell')}  memory ratio={dm.get('ratio_vs_fp16')}  "
          f"empty learned={de.get('learned')}")

    # Gemma-4 QAT format footprint extension (arXiv 2607.02770, Table 3): bf16=16b, int8=8b, Q4_0≈4.5b,
    # mobile int2/int4 mix ≈3.5b; footprint shrinks and compression-vs-bf16 grows with fewer bits.
    checks["g4_bf16_16bit"] = abs(L.gemma4_format_bits("bf16") - 16.0) < 1e-9
    checks["g4_q40_bits"] = abs(L.gemma4_format_bits("q4_0", 32) - (4.0 + 16.0/32)) < 1e-6
    checks["g4_mobile_below_q40"] = L.gemma4_format_bits("mobile", 32) < L.gemma4_format_bits("q4_0", 32)
    fp = L.gemma4_quant_footprint(31.0, "q4_0")      # 31B in billions
    checks["g4_footprint_below_bf16"] = fp["gb"] < fp["bf16_gb"]
    checks["g4_compression_gt3"] = fp["compression_vs_bf16"] > 3.0
    print(f"  -> Gemma-4 31B Q4_0 footprint {fp['gb']:.1f} GB vs bf16 {fp['bf16_gb']:.1f} GB "
          f"({fp['compression_vs_bf16']:.2f}× )")

    for k, v in checks.items():
        print(f"  {'OK' if v else 'X'} {k}")
    ok = all(checks.values())
    print(f"=== lowbit-qat: {'PASS' if ok else 'FAIL'} ({sum(checks.values())}/{len(checks)}) ===")
    return ok


if __name__ == "__main__":
    sys.exit(0 if _run() else 1)
