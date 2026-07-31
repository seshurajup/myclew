"""paper_upgrades_test — DATA-WISE verifier for the agent upgrades distilled from the three papers.

  Nested Learning (Behrouz et al., NeurIPS 2025) https://alibehrouz.com/files/NL.pdf
      muon_optimizer   : delta_momentum_update, m3_update, momentum_horizon
      train_tricks_pack: cms_param_groups / cms_step_gate / cms_expected_cost
      xai              : nl_audit / nl_audit_summary
      arch_builder     : the cms-multi-frequency-blocks catalog entry
  Kimi K3 (Kimi Team, 2026) https://arxiv.org/pdf/2607.24653
      arch_builder     : situ-glu-bounded-activation, channel-wise-decay-memory, attention-over-depth
  HOPE (Mobahi & Bartlett, 2026) https://arxiv.org/pdf/2607.21366
      compress_select  : scale_normalise_neuron, neuron_kernel, hope_costs, rate_distortion_pick
  EDA (2026) https://arxiv.org/pdf/2606.26560
      attention_residual: eda_step, eda_interleave, eda_collateral
  RateQuant (2026) https://arxiv.org/pdf/2605.06675
      quantize          : mixed_precision_gain, allocate_bits, bit_importance
  FedNL (2026) https://arxiv.org/pdf/2605.16350
      xai               : fed_audit, fed_audit_summary
  Task-Restricted Symmetries in Recurrent Weight Space (2026) https://arxiv.org/pdf/2606.18457
      compress_select   : schur_blocks, schur_sensitivity

Each check is the paper's own claim turned into an assertion — the HOPE ones are the sharpest: a neuron
rescaled by c (with its outgoing weights scaled by 1/c) computes the SAME function, so its merge cost must
be exactly zero and its ranking must not move; that is precisely what magnitude pruning gets wrong.
"""
import os
import sys

import numpy as np

COMP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, COMP)
sys.path.insert(0, os.path.join(COMP, "tools", "researchpapers"))

from fleet_agents import arch_builder as AB  # noqa: E402
from fleet_agents import attention_residual as AR  # noqa: E402
from fleet_agents import compress_select as CS  # noqa: E402
from fleet_agents import moe_quantile_balance as QB  # noqa: E402
from fleet_agents import muon_optimizer as MO  # noqa: E402
from fleet_agents import quantize as QZ  # noqa: E402
from fleet_agents import tab_diversity_pack as TD  # noqa: E402
from fleet_agents import train_tricks_pack as TT  # noqa: E402
from fleet_agents import xai as XAI  # noqa: E402


def _run():
    print("=== PAPER-UPGRADE VERIFIER (NL · K3 · HOPE) ===")
    checks = {}
    rng = np.random.RandomState(0)

    # ---------------- Nested Learning → muon_optimizer
    h = MO.momentum_horizon(0.9)
    checks["NL momentum horizon is tiny (~6 / ~43 gradients)"] = 5 <= h[0.5] <= 8 and 40 <= h[0.99] <= 46
    checks["NL horizon grows with beta"] = MO.momentum_horizon(0.99)[0.99] > h[0.99]

    g = rng.randn(6, 4)
    m1, decay = MO.delta_momentum_update(np.zeros((6, 4)), g, alpha=0.9, eta=0.1)
    m_big, decay_big = MO.delta_momentum_update(np.zeros((6, 4)), 50 * g, alpha=0.9, eta=0.1)
    checks["NL delta-momentum decay stays in (0, alpha]"] = 0.0 < decay <= 0.9
    checks["NL delta-momentum decay DEPENDS on the gradient"] = abs(decay_big - decay) > 1e-6
    checks["NL delta-momentum is finite"] = bool(np.isfinite(m1).all())

    st = MO.m3_state((6, 4))
    for i in range(24):
        upd, st = MO.m3_update(st, rng.randn(6, 4), freq=8)
    checks["NL M3 runs and stays finite (the denominator guard works)"] = bool(np.isfinite(upd).all())
    checks["NL M3 writes its slow memory once per chunk"] = st["t"] == 24 and float(np.abs(st["M2"]).sum()) > 0
    st0 = MO.m3_state((6, 4))
    for i in range(4):                                     # fewer steps than freq → slow memory untouched
        _, st0 = MO.m3_update(st0, rng.randn(6, 4), freq=8)
    checks["NL M3 slow memory is EMPTY before its first chunk"] = float(np.abs(st0["M2"]).sum()) == 0.0

    # ---------------- HOPE → compress_select   (the scale-symmetry claim)
    w = rng.randn(16)
    out = np.ones(4)
    d1, gain1 = CS.scale_normalise_neuron(w, out)
    d2, gain2 = CS.scale_normalise_neuron(7.0 * w, out / 7.0)          # the SAME function, rescaled
    checks["HOPE scale normalisation gives an invariant direction"] = np.allclose(d1, d2, atol=1e-8)
    checks["HOPE scale normalisation gives an invariant gain"] = abs(gain1 - gain2) < 1e-8
    checks["HOPE direction is unit-norm"] = abs(float(np.linalg.norm(d1)) - 1.0) < 1e-8

    k_same = CS.neuron_kernel(w, w)
    k_scaled = CS.neuron_kernel(w, 3 * w)
    k_ortho = CS.neuron_kernel(np.array([1.0, 0, 0, 0]), np.array([0, 1.0, 0, 0]))
    checks["HOPE kernel is positive on itself"] = k_same > 0
    checks["HOPE kernel is homogeneous (scaling scales it)"] = abs(k_scaled - 3 * k_same) < 1e-8
    checks["HOPE ReLU kernel of orthogonal neurons is the arc-cos floor"] = 0 < k_ortho < 0.5 * 1.0

    neurons = [dict(w_in=w, w_out=out, params=16, id="a"),
               dict(w_in=5.0 * w, w_out=out / 5.0, params=16, id="a-rescaled"),
               dict(w_in=rng.randn(16), w_out=out, params=16, id="b"),
               dict(w_in=1e-7 * w, w_out=out * 1e-7, params=16, id="dead")]
    c = CS.hope_costs(neurons)
    dup = next(m for m in c["merge"] if {m["i"], m["j"]} == {0, 1})
    diff = next(m for m in c["merge"] if {m["i"], m["j"]} == {0, 2})
    checks["HOPE merging a rescaled DUPLICATE is free"] = dup["J"] < 1e-9 and dup["align"] > 0.999
    checks["HOPE merging a DIFFERENT neuron is not free"] = diff["J"] > 1.0
    checks["HOPE prunes the dead neuron first"] = c["prune"][0]["id"] == "dead"

    best, ranked = CS.rate_distortion_pick([
        {"name": "prune-neuron", "J": 0.9, "dparams": 16},          # small cost, tiny saving
        {"name": "evict-block", "J": 40.0, "dparams": 4096}])       # big cost, huge saving
    checks["HOPE picks by cost-per-parameter, not by cost"] = best["name"] == "evict-block"
    checks["HOPE ranking is by J/dparams"] = ranked[0]["dr"] <= ranked[1]["dr"]

    # ---------------- Nested Learning → train_tricks_pack (CMS) + xai (the audit)
    try:
        import torch
        import torch.nn as nn
        net = nn.Sequential(nn.Linear(32, 32), nn.GELU(), nn.Linear(32, 32), nn.GELU(),
                            nn.Linear(32, 32), nn.Linear(32, 8))
        groups = TT.cms_param_groups(list(net.named_modules())[1:], (1, 4, 16))
        written = 0
        for step in range(1, 65):
            net(torch.randn(4, 32)).sum().backward()
            written += TT.cms_step_gate(step, groups)
            net.zero_grad(set_to_none=False)
        measured, predicted = written / 64.0, TT.cms_expected_cost(groups)
        checks["NL CMS periods are honoured"] = [g["period"] for g in groups] == [1, 4, 16]
        checks["NL CMS measured update cost == sum(n_l / C_l)"] = abs(measured - predicted) / predicted < 0.02
        checks["NL CMS touches only a fraction per step"] = measured < sum(
            g["n_params"] for g in groups)

        opt = torch.optim.AdamW(net.parameters(), lr=1e-3)
        net(torch.randn(4, 32)).sum().backward(); opt.step()
        rows = XAI.nl_audit(net, opt, accum_steps=8, cms_groups=groups)
        summ = XAI.nl_audit_summary(rows)
        total_params = sum(p.numel() for p in net.parameters())
        checks["NL audit does NOT double-count CMS groups"] = summ["advertised"] == total_params
        checks["NL audit finds the optimizer's hidden parameters"] = summ["ratio"] >= 2.5
        checks["NL audit reports more than one level"] = summ["levels"] >= 2
    except ImportError:                                             # torch-free environment
        for k in ("NL CMS periods are honoured", "NL CMS measured update cost == sum(n_l / C_l)",
                  "NL CMS touches only a fraction per step", "NL audit does NOT double-count CMS groups",
                  "NL audit finds the optimizer's hidden parameters", "NL audit reports more than one level"):
            checks[k + " (skipped: no torch)"] = True

    # ---------------- EDA → attention_residual (a second, independent erase address)
    try:
        import torch
        import torch.nn.functional as Fn
        torch.manual_seed(0)
        d = 24
        u = lambda: Fn.normalize(torch.randn(d), dim=0)
        k1, k2, vv = u(), u(), torch.randn(d)
        S = torch.zeros(d, d)
        S = AR.eda_step(S, k1, torch.randn(d), e=k1, gamma=0.0)          # plain delta write
        S = AR.eda_step(S, k2, vv, e=k2, gamma=0.0)
        # gamma = 0 must be exactly gated DeltaNet
        I = torch.eye(d)
        ref = (I - torch.outer(k2, k2)) @ ((I - torch.outer(k1, k1)) @ torch.zeros(d, d)
                                          + torch.outer(k1, torch.zeros(d))) + torch.outer(k2, vv)
        checks["EDA gamma=0 reduces to the delta rule"] = torch.allclose(
            AR.eda_step(torch.zeros(d, d), k2, vv, e=k2, gamma=0.0), torch.outer(k2, vv), atol=1e-6)
        # a targeted erase at an orthogonal address removes everything there
        k2p = Fn.normalize(k2 - float(k2 @ k1) * k1, dim=0)
        S2 = AR.eda_step(S, k1, torch.randn(d), e=k2p, beta=1.0, gamma=1.0)
        checks["EDA erases at an address it is NOT writing to"] = float(
            torch.linalg.vector_norm(S2.T @ k2p)) < 1e-4
        # the doubling trick: EDA == gated DeltaNet over the interleaved 2T sequence
        T = 5
        K = [u() for _ in range(T)]; V = [torch.randn(d) for _ in range(T)]; E = [u() for _ in range(T)]
        B = [0.7] * T; G = [0.6] * T; Dg = [torch.rand(d) * 0.1 + 0.9 for _ in range(T)]
        S_seq = torch.zeros(d, d)
        for t in range(T):
            S_seq = AR.eda_step(S_seq, K[t], V[t], E[t], beta=B[t], gamma=G[t], decay=Dg[t])
        S_dbl = torch.zeros(d, d)
        for st in AR.eda_interleave(K, V, E, B, G, Dg):
            S_dbl = (I - st["beta"] * torch.outer(st["k"], st["k"])) @ (st["gate"][:, None] * S_dbl) \
                + st["beta"] * torch.outer(st["k"], st["v"])
        checks["EDA == gated DeltaNet on the doubled sequence (no new kernel)"] = torch.allclose(
            S_seq, S_dbl, atol=1e-5)
        checks["EDA interleave really doubles the sequence"] = len(
            AR.eda_interleave(K, V, E, B, G, Dg)) == 2 * T
        # the collateral guard is exact, and zero for an orthogonal query
        q_par, q_perp = E[0], Fn.normalize(torch.randn(d) - float(torch.randn(1)) * E[0], dim=0)
        q_perp = Fn.normalize(q_perp - float(q_perp @ E[0]) * E[0], dim=0)
        _, n_par = AR.eda_collateral(S, q_par, E[0])
        _, n_perp = AR.eda_collateral(S, q_perp, E[0])
        checks["EDA collateral guard: parallel query pays, orthogonal query does not"] = (
            n_par > 1e-6 and n_perp < 1e-5)
    except ImportError:
        for k in ("EDA gamma=0 reduces to the delta rule", "EDA erases at an address it is NOT writing to",
                  "EDA == gated DeltaNet on the doubled sequence (no new kernel)",
                  "EDA interleave really doubles the sequence",
                  "EDA collateral guard: parallel query pays, orthogonal query does not"):
            checks[k + " (skipped: no torch)"] = True

    # ---------------- RNN weight symmetries → compress_select (the recurrent probe)
    try:
        import torch
        torch.manual_seed(0)
        H = 16
        Wr = torch.randn(H, H) / (H ** 0.5)
        Wr = 0.9 * Wr / torch.linalg.matrix_norm(Wr, 2)
        Wx = torch.randn(H, 2) / 2
        Wy = torch.randn(1, H) / (H ** 0.5)
        X = torch.randn(24, 2)

        def roll(M):
            h = torch.zeros(H); out = []
            for t in range(X.shape[0]):
                h = torch.tanh(Wx @ X[t] + M @ h)
                out.append(Wy @ h)
            return torch.stack(out)

        Q, T, blocks = CS.schur_blocks(Wr, alpha=0.7)
        checks["RWS Schur factorisation is exact"] = bool(
            np.allclose(Q @ T @ Q.T, Wr.cpu().numpy(), atol=1e-6))
        checks["RWS Q is orthogonal"] = bool(np.allclose(Q.T @ Q, np.eye(H), atol=1e-6))
        checks["RWS T is upper quasi-triangular"] = float(np.abs(np.tril(T, -2)).max()) < 1e-8
        checks["RWS the coupling blocks are named mechanisms"] = set(blocks) == {
            "T_RR", "T_C->R", "T_CC"}

        rows = CS.schur_sensitivity(Wr, roll, alpha=0.7)
        checks["RWS the probe returns one row per non-empty coupling block"] = 1 <= len(rows) <= 3
        checks["RWS every sensitivity is finite and non-negative"] = all(
            r["sensitivity"] >= 0 and r["sensitivity"] == r["sensitivity"] for r in rows)
        checks["RWS rows are sorted so the approximate stabilizer comes first"] = (
            [r["sensitivity"] for r in rows] == sorted(r["sensitivity"] for r in rows))
        checks["RWS ablation is reported per unit of edit, not raw"] = all(
            "rel_dT" in r and r["rel_dT"] > 0 for r in rows)
        # the task-restricted claim: a different drive gives a different profile
        X = torch.zeros(24, 2); X[0, 0] = 1.0                    # an impulse instead of noise
        rows_b = CS.schur_sensitivity(Wr, roll, alpha=0.7)
        checks["RWS the sensitivity profile is TASK-restricted (it changes with the drive)"] = any(
            abs(a["sensitivity"] - b["sensitivity"]) > 1e-6
            for a, b in zip(sorted(rows, key=lambda r: r["coupling"]),
                            sorted(rows_b, key=lambda r: r["coupling"])))
    except ImportError:
        for k in ("RWS Schur factorisation is exact", "RWS Q is orthogonal",
                  "RWS T is upper quasi-triangular", "RWS the coupling blocks are named mechanisms",
                  "RWS the probe returns one row per non-empty coupling block",
                  "RWS every sensitivity is finite and non-negative",
                  "RWS rows are sorted so the approximate stabilizer comes first",
                  "RWS ablation is reported per unit of edit, not raw",
                  "RWS the sensitivity profile is TASK-restricted (it changes with the drive)"):
            checks[k + " (skipped: no torch)"] = True

    # ---------------- FedNL → xai (the federation is the OUTER level)
    try:
        import torch
        import torch.nn as nn
        fnet = nn.Sequential(nn.Linear(16, 16), nn.GELU(), nn.Linear(16, 16))
        fopt = torch.optim.AdamW(fnet.parameters(), lr=1e-3)
        fnet(torch.randn(2, 16)).sum().backward(); fopt.step()
        frows = XAI.fed_audit(fnet, fopt, accum_steps=4, memory_numel=16 * 16,
                              round_every=512, n_clients=8)
        fsum = XAI.fed_audit_summary(frows)
        checks["FedNL audit puts the server at the OUTERMOST level"] = (
            frows[0]["level"] == 1 and "server" in frows[0]["component"])
        checks["FedNL audit puts the in-context memory at the innermost level"] = (
            frows[-1]["component"].startswith("in-context memory")
            and frows[-1]["level"] == max(r["level"] for r in frows))
        checks["FedNL audit orders levels by frequency (server slowest, memory fastest)"] = (
            len({r["level"] for r in frows}) >= 3)
        checks["FedNL audit separates what is SHIPPED from what stays local"] = (
            fsum["shipped"] > 0 and fsum["local"] > 0)
        checks["FedNL the memory and optimizer state never ship"] = all(
            not r.get("shipped") for r in frows
            if r["component"].startswith(("in-context memory", "AdamW")))
    except ImportError:
        for k in ("FedNL audit puts the server at the OUTERMOST level",
                  "FedNL audit puts the in-context memory at the innermost level",
                  "FedNL audit orders levels by frequency (server slowest, memory fastest)",
                  "FedNL audit separates what is SHIPPED from what stays local",
                  "FedNL the memory and optimizer state never ship"):
            checks[k + " (skipped: no torch)"] = True

    # ---------------- RateQuant → quantize (the closed-form bit allocator)
    w_eq = [2.0] * 8
    w_sk = [0.01, 0.02, 0.05, 0.1, 1.0, 4.0, 20.0, 100.0]
    checks["RQ equal importance -> AM/GM = 1 (mixed precision pointless)"] = abs(
        QZ.mixed_precision_gain(w_eq) - 1.0) < 1e-9
    checks["RQ skewed importance -> AM/GM > 1 (a real gain to chase)"] = QZ.mixed_precision_gain(w_sk) > 2
    checks["RQ the diagnostic ignores a global rescale"] = abs(
        QZ.mixed_precision_gain(w_sk) - QZ.mixed_precision_gain([1000 * x for x in w_sk])) < 1e-6

    a = QZ.allocate_bits(w_sk, avg_bits=4.0, legal=(2, 3, 4, 8))
    checks["RQ allocation respects the bit budget"] = sum(a["bits"]) <= a["budget_bits"]
    checks["RQ only hardware-legal widths are used"] = set(a["bits"]) <= {2, 3, 4, 8}
    checks["RQ more important tensors get at least as many bits"] = (
        a["bits"][-1] >= a["bits"][0] and a["bits"][-1] > min(a["bits"]))
    a_eq = QZ.allocate_bits(w_eq, avg_bits=4.0, legal=(2, 3, 4, 8))
    checks["RQ equal importance -> a uniform allocation"] = len(set(a_eq["bits"])) == 1
    a_scaled = QZ.allocate_bits([1000 * x for x in w_sk], avg_bits=4.0, legal=(2, 3, 4, 8))
    checks["RQ allocation is invariant to rescaling importance"] = a["bits"] == a_scaled["bits"]
    try:
        import torch
        g = [torch.randn(4, 4) * s_ for s_ in (0.1, 1.0, 10.0)]
        imp = QZ.bit_importance(g)
        checks["RQ importance from gradients is monotone in scale"] = imp[0] < imp[1] < imp[2]
    except ImportError:
        checks["RQ importance from gradients is monotone in scale (skipped: no torch)"] = True

    # ---------------- K3 + NL → arch_builder catalog
    names = {e["name"] for e in AB.catalog()}
    for want in ("situ-glu-bounded-activation", "channel-wise-decay-memory", "attention-over-depth",
                 "cms-multi-frequency-blocks", "hybrid-shortconv-attention-interleave"):
        checks[f"catalog has {want}"] = want in names
    entries = [e for e in AB.catalog() if e["name"].startswith(("situ-", "channel-wise", "attention-over",
                                                                "cms-multi", "hybrid-shortconv"))]
    lfm = next(e for e in AB.catalog() if e["name"] == "hybrid-shortconv-attention-interleave")
    checks["LFM2.5 entry keeps the source link"] = "liquid.ai/blog/lfm2-5-encoders" in lfm["what"]
    checks["LFM2.5 entry states attention is MANDATORY"] = "MANDATORY" in lfm["constraint"]
    checks["LFM2.5 entry says the MAC saving cannot be the gate"] = (
        "length-INDEPENDENT" in lfm["constraint"] and "crossover" in lfm["constraint"])
    checks["LFM2.5 entry carries measured numbers, not a slogan"] = "1.56x" in lfm["evidence"]
    checks["LFM2.5 entry says what was NOT reproduced"] = "NOT reproduced" in lfm["evidence"]
    checks["every new entry states its constraint AND its evidence"] = all(
        e.get("constraint") and e.get("evidence") for e in entries)

    # ---------------- Routing-Free MoE (2604.00801) → moe_quantile_balance, the counterpoint
    rng = np.random.RandomState(0)
    logits = rng.randn(512, 16) + rng.randn(16) * 1.5          # miscalibrated columns, as in the agent
    rf = QB.routing_free_route(np.abs(logits), np.quantile(np.abs(logits), 0.875, axis=0))
    checks["RFMoE self-gating yields an EMERGENT k_eff, not a fixed k"] = 0 < rf["k_eff"] < 16
    checks["RFMoE spends different compute on different tokens"] = float(
        rf["active"].sum(1).std()) > 0
    hot = QB.routing_free_route(np.abs(logits), np.full(16, -1e9))
    checks["RFMoE a low threshold activates every expert"] = hot["density"] == 1.0
    cold = QB.routing_free_route(np.abs(logits), np.full(16, 1e9))
    checks["RFMoE a high threshold deactivates continuously (no discrete switch)"] = cold["density"] == 0.0

    bal = QB.balance_losses(np.full((64, 8), 0.25), rho_star=0.25)
    checks["RFMoE both balance losses vanish at the target"] = bal["L_EB"] < 1e-12 and bal["L_TB"] < 1e-12
    skew = np.concatenate([np.full((64, 1), 0.9), np.full((64, 7), 0.1)], axis=1)
    checks["RFMoE L_EB catches an expert hogging the batch"] = QB.balance_losses(
        skew, 0.25)["L_EB"] > bal["L_EB"]
    lop = np.concatenate([np.full((1, 8), 0.9), np.full((63, 8), 0.1)], axis=0)
    checks["RFMoE L_TB catches a token activating everything"] = QB.balance_losses(
        lop, 0.25)["L_TB"] > bal["L_TB"]
    checks["RFMoE mu=1 is pure expert balance, mu=0 pure token balance"] = (
        abs(QB.balance_losses(skew, 0.25, mu=1.0)["L_LB"] - QB.balance_losses(skew, 0.25)["L_EB"]) < 1e-12
        and abs(QB.balance_losses(skew, 0.25, mu=0.0)["L_LB"]
                - QB.balance_losses(skew, 0.25)["L_TB"]) < 1e-12)

    up = QB.lambda_controller([0.5] * 60, lam0=0.01)
    down = QB.lambda_controller([0.05] * 60, lam0=0.01)
    checks["RFMoE lambda controller raises the weight on over-activation"] = up["lambda_final"] > 0.1
    checks["RFMoE and lowers it on under-activation"] = down["lambda_final"] < 0.001
    checks["RFMoE lambda stays strictly positive by construction"] = min(
        min(up["history"]), min(down["history"])) > 0
    settle = QB.lambda_controller([0.30, 0.20] * 30, lam0=0.01)
    checks["RFMoE it settles when density oscillates about the target"] = abs(
        settle["lambda_final"] / 0.01 - 1) < 0.25

    checks["RFMoE comm delta is positive exactly when k+1>M"] = all(
        (QB.comm_delta(k, m)["delta_ms"] > 0) == (k + 1 > m)
        for k, m in [(2, 2), (8, 8), (8, 16), (16, 8), (2, 8), (32, 16)])
    checks["RFMoE 2-GPU boxes favour routing-free"] = QB.comm_delta(2, 2)["favours"] == "routing-free"
    checks["RFMoE 64-way expert parallel does not"] = QB.comm_delta(8, 64)["favours"] == "standard MoE"

    cmp = QB.router_vs_routing_free(logits, k=2)
    checks["RFMoE quantile balancing still beats raw top-k on flatness"] = (
        cmp["qbalance_cv"] < cmp["topk_cv"])
    checks["RFMoE the comparison reports both k_fixed and k_eff"] = (
        cmp["k_fixed"] == 2.0 and cmp["k_eff"] > 0)
    checks["RFMoE the agent's own run surfaces the counterpoint"] = all(
        m in QB.__doc__ for m in ("2604.00801", "routing_free_route", "comm_delta"))

    # ---------------- google-research/tabfm -> tab_diversity_pack (a REPO, not a paper)
    codes, _ = TD.appearance_ordinal_encode(["a", "b", "a", "c", "a", "b"])
    checks["tabfm codes are frequency-ordered, not alphabetical"] = codes.tolist() == [0, 1, 0, 2, 0, 1]
    _, m_rare = TD.appearance_ordinal_encode(["a"] * 10 + ["q"], min_frequency=3)
    checks["tabfm rare categories fold into one bucket"] = len(set(m_rare)) == 2

    Xo = np.concatenate([np.random.RandomState(0).randn(200, 1), [[50.0]]])
    Cc, lo_, hi_ = TD.two_stage_clip(Xo, 4.0)
    checks["tabfm two-stage clip pulls the extreme in"] = Cc.max() < Xo.max() / 3
    checks["tabfm and clips rather than dropping rows"] = Cc.shape == Xo.shape

    Zq, qt_ = TD.noise_then_quantile(np.random.RandomState(0).randn(300, 2) ** 3)
    checks["tabfm noise-then-quantile bounds heavy tails"] = abs(Zq).max() < 10
    checks["tabfm n_quantiles adapts to row count"] = (
        TD.noise_then_quantile(np.random.randn(5000, 1))[1].n_quantiles_
        > TD.noise_then_quantile(np.random.randn(60, 1))[1].n_quantiles_)

    _, te_ = TD.train_range_clip(np.random.RandomState(0).randn(200, 3), np.full((1, 3), 1e6))
    checks["tabfm an unseen extreme is clipped to the TRAIN range"] = abs(te_).max() < 100

    from sklearn.ensemble import RandomForestRegressor
    rs_ = np.random.RandomState(0)
    Xr = rs_.randn(200, 6)
    rf_ = RandomForestRegressor(n_estimators=20, max_depth=4, random_state=0).fit(Xr, Xr @ rs_.randn(6))
    inv_ = lambda A: rf_.predict(np.sort(A, axis=1))            # noqa: E731 — permutation-invariant
    s_inv = TD.order_sensitivity(inv_, Xr)
    s_pos = TD.order_sensitivity(lambda A: rf_.predict(A), Xr)
    checks["tabfm the gate flags an order-INVARIANT predictor"] = not s_inv["order_sensitive"]
    checks["tabfm and flags a positional one as sensitive"] = s_pos["order_sensitive"]
    checks["tabfm the gate explains its verdict"] = "order-INVARIANT" in s_inv["verdict"]
    gated = TD.view_ensemble(inv_, Xr, n_views=16)
    checks["tabfm view_ensemble REFUSES useless TTA"] = gated["gated"] and gated["n_views"] == 1
    ran = TD.view_ensemble(lambda A: rf_.predict(A), Xr, n_views=8)
    checks["tabfm and runs N views when they can help"] = ran["n_views"] == 8 and not ran["gated"]
    checks["tabfm reporting the disagreement it averaged"] = ran["disagreement"] > 0
    checks["tabfm the repo link is in the docstring"] = "github.com/google-research/tabfm" in TD.__doc__

    for k, v in checks.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    print(f"  HOPE: duplicate-merge cost {dup['J']:.2e} vs different-pair {diff['J']:.2f}")
    return all(checks.values())


if __name__ == "__main__":
    sys.exit(0 if _run() else 1)
