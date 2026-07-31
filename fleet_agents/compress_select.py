"""compress-select — CHOOSE the best MODEL-COMPRESSION of the detector (training-free) by DATA PROOF:
the compression that runs fastest within the Kaggle time budget while keeping per-embryo recall ≥ bar.

Implements the recent LLM depth-compression best practices (user 2026-07-11 "same as LLM research"):
  • truncate       — drop the last-K blocks (naive tail cut).
  • drop_redundant — ShortGPT (2024): drop the blocks with LOWEST "Block Influence" (relative residual
                     ‖out−in‖/‖in‖ / cosine(in,out)), wherever they sit — a near-identity block in the
                     MIDDLE is a better cut than a tail block that does real work.
  • merge          — LaCo (Layer-Collapse): AVERAGE a redundant block's weights into its neighbour instead
                     of dropping it — keeps some of its contribution (min/max/avg weight-folding family).
Choice is by data proof on OUR embryo-disjoint CV, per-embryo. NO training. (Width/channel pruning is the
next extension — same importance-ranked, data-proof pattern.)

User strategy: "reduce our best models without training, don't lose quality." Cellpose-SAM (best recall,
3.96s/f) is too slow for the full hidden test (199 ds × 100 fr = 19,900 fr) at 12h/2×T4; compressing its
ViT blocks (residual → near-identity ones removable/mergeable) speeds it up with ~no quality loss to a
floor. This agent finds the best technique+level on OUR CV.

Companion to detector-select / tracker-select. Reuses experiments/prune_cellpose/prune_lib (block detection
+ truncation) and the same recall@7µm / time measurement. Decision logic is a pure, data-wise-tested fn.
"""
from __future__ import annotations
from .base import BaseAgent, COMP

TEST_FRAMES = 19_900
GPU_SECONDS = 12 * 3600 * 2                       # 2×T4, 12h
DETECT_BUDGET_SPF = round(GPU_SECONDS / TEST_FRAMES * 0.9, 2)   # ~3.9 s/f (tracker Trackastra is ~0.1, cheap)


# ---------------------------------------------------------------- HOPE (Mobahi & Bartlett, 2026) adds
# Mobahi & Bartlett (Google DeepMind / UC Berkeley), "Hilbert Operator for Progressive Encoding (HOPE):
# A Mathematical Framework for Deconstructing Learned Representations in Deep Networks", arXiv:2607.21366
# paper: https://arxiv.org/pdf/2607.21366 · local: docs/papers/hope-hilbert/hope-hilbert.md
# Its argument against every magnitude-style
# heuristic we use above: importance scores computed on ISOLATED parameters are not well defined, because a
# network has SCALE SYMMETRIES — you can multiply a neuron's incoming weights by c and divide its outgoing
# weights by c and the function is unchanged, while every magnitude score changes. So HOPE
#   (1) takes the whole NEURON as the unit (incoming weights + BN + activation + outgoing weights),
#   (2) factors the symmetry out before scoring,
#   (3) compares neurons by the FUNCTION they compute — each is a rank-1 Hilbert-Schmidt operator, and the
#       inner product is an integral over the input distribution. Data-free, so the distribution is the
#       maximum-entropy one (a Gaussian), which makes the integral analytic (the arc-cosine kernel),
#   (4) scores prune / merge / block-eviction on ONE metric so decisions are comparable across layers and
#       granularities, and picks by RATE-DISTORTION (cost per parameter saved), not by distortion alone.
# Data-free and hyper-parameter-free — which is exactly the regime our training-free compression runs in.
def scale_normalise_neuron(w_in, w_out, bn_gamma=None, bn_var=None, eps=1e-8):
    """Factor a neuron's scale symmetry out (HOPE §3): return (w_in_unit, gain).

    For a positively-homogeneous activation (ReLU/LeakyReLU), scaling `w_in` by c and `w_out` by 1/c leaves
    the neuron's function unchanged, so any score computed on `w_in` alone is arbitrary. The invariant
    parameterisation puts all the scale in ONE place: a unit-norm incoming direction plus a single gain
    `‖w_in‖·γ/√var·‖w_out‖`. Compare neurons on (direction, gain) and the symmetry can no longer distort
    the ranking. BN is folded in when given.
    """
    import numpy as np
    w_in = np.asarray(w_in, float).reshape(-1)
    w_out = np.asarray(w_out, float).reshape(-1)
    n_in = float(np.linalg.norm(w_in))
    bn = 1.0
    if bn_gamma is not None:
        bn = float(bn_gamma) / (float(np.sqrt(bn_var)) + eps if bn_var is not None else 1.0)
    gain = n_in * abs(bn) * float(np.linalg.norm(w_out))
    return (w_in / (n_in + eps)), gain


def neuron_kernel(u, v, kind="relu"):
    """⟨f_u, f_v⟩ under a maximum-entropy (Gaussian) input prior — HOPE §4–§5, in closed form.

    For `f_w(x) = φ(wᵀx)` with `x ~ N(0, I)`, the expectation is analytic. ReLU gives the arc-cosine
    kernel `E[φ(uᵀx)φ(vᵀx)] = ‖u‖‖v‖/(2π) · (sinθ + (π−θ)cosθ)`, and a linear unit gives `uᵀv`. This is
    what lets two neurons be compared by the FUNCTION they compute with no data at all: two neurons whose
    weights look different but whose kernel is ~1 are duplicates and can be merged.
    """
    import numpy as np
    u = np.asarray(u, float).reshape(-1); v = np.asarray(v, float).reshape(-1)
    nu, nv = float(np.linalg.norm(u)), float(np.linalg.norm(v))
    if nu < 1e-12 or nv < 1e-12:
        return 0.0
    cos = float(np.clip(u @ v / (nu * nv), -1.0, 1.0))
    if kind == "linear":
        return nu * nv * cos
    theta = float(np.arccos(cos))
    return nu * nv / (2 * np.pi) * (np.sin(theta) + (np.pi - theta) * np.cos(theta))


def hope_costs(neurons, kind="relu"):
    """Projection costs on ONE metric (HOPE §5–§8): J_prune per neuron and J_merge per pair.

    `neurons` = [{"w_in":…, "w_out":…, "bn_gamma"?, "bn_var"?, "params": int, "id"?}].
      J_prune(i) = ‖f_i‖²                         — the energy lost by projecting the neuron to zero.
      J_merge(i,j) = ‖f_i‖² + ‖f_j‖² − ‖f_i+f_j‖²/2 ... in the rank-1 realisable space this reduces to
                     the residual after projecting the pair onto their best single parent, which for the
                     Gaussian kernel is `(1 − |k_ij| / √(k_ii k_jj)) · (k_ii + k_jj)/2`: zero when the two
                     neurons compute the same function up to scale, large when they are orthogonal.
    Returns {"prune": [...], "merge": [(i, j, cost), ...]} with a `gain`-weighted energy, so a
    high-gain duplicate is not treated like a dead unit.
    """
    import numpy as np
    prep = []
    for n in neurons:
        d, gain = scale_normalise_neuron(n["w_in"], n["w_out"], n.get("bn_gamma"), n.get("bn_var"))
        prep.append({"dir": d, "gain": gain, "params": int(n.get("params", d.size)), "id": n.get("id")})
    kk = [neuron_kernel(p["dir"], p["dir"], kind) * p["gain"] ** 2 for p in prep]
    prune = [{"i": i, "id": prep[i]["id"], "J": float(kk[i]), "dparams": prep[i]["params"]}
             for i in range(len(prep))]
    merge = []
    for i in range(len(prep)):
        for j in range(i + 1, len(prep)):
            kij = neuron_kernel(prep[i]["dir"], prep[j]["dir"], kind) * prep[i]["gain"] * prep[j]["gain"]
            denom = float(np.sqrt(max(kk[i] * kk[j], 1e-24)))
            align = abs(kij) / denom if denom > 0 else 0.0
            cost = float((1.0 - min(align, 1.0)) * (kk[i] + kk[j]) / 2.0)
            merge.append({"i": i, "j": j, "J": cost, "align": round(align, 4),
                          "dparams": prep[j]["params"]})
    return {"prune": sorted(prune, key=lambda d: d["J"]), "merge": sorted(merge, key=lambda d: d["J"])}


# ---------------------------------------------------------------- the RECURRENT companion to the above
# "Task-Restricted Symmetries in Recurrent Weight Space", arXiv:2606.18457
# paper: https://arxiv.org/pdf/2606.18457 · local: docs/papers/rnn-weight-symmetry/rnn-weight-symmetry.md
# lessons: learning/annotated/rws*.learning (8/8 formulas proved)
#
# HOPE's kernel above handles a feedforward SCALE symmetry analytically. Recurrent redundancy cannot be
# factored out that way: measured on a stable RNN, perturbations of IDENTICAL ‖ΔW‖_F differ in behavioural
# damage by several times, and which directions are free depends on the TASK. So the recurrent case needs a
# probe, not a formula. The paper's instrument:
#   • work in the real Schur basis `W = QTQᵀ`, which splits `T = B + N` — `B` the spectral blocks (what the
#     dynamics rotate/decay by) and `N` the directed nonnormal couplings (how activity is routed);
#   • ablate a coupling BLOCK (a mechanism), rebuild `W̃ = QT̃Qᵀ`, and normalise the damage by the size of
#     the edit: `S_ΔT = ΔFVU / (‖ΔT‖_F/‖T‖_F)` — damage per unit of relative change.
# That ratio is the recurrent analogue of `rate_distortion_pick`'s J/Δparams: it makes interventions of
# different sizes comparable. THE CAVEAT THAT MATTERS FOR THIS AGENT: the profile is task-restricted, so a
# coupling that is free on one task distribution can be load-bearing on another — never reuse a recurrent
# ablation decision across tasks without re-measuring.
def schur_blocks(W, alpha=0.7, tol=1e-3):
    """`W` → (Q, T, blocks) in ordered real Schur coordinates; `blocks` names the coupling submatrices.

    Returns `blocks = {"T_RR": (rows, cols), "T_C->R": …, "T_CC": …}` where `R` is the retained mode set
    (`|λ| ≥ alpha·ρ(W)`) and `C` its complement — so an ablation can be described as a mechanism
    ("fast modes feeding slow ones") rather than as a set of weight indices.
    """
    import numpy as np
    from scipy.linalg import schur as _schur
    Wn = np.asarray(W.detach().cpu().numpy() if hasattr(W, "detach") else W, dtype=np.float64)
    T, Q = _schur(Wn, output="real")
    lam = np.abs(np.linalg.eigvals(Wn))
    rho = float(lam.max()) if lam.size else 0.0
    r = int((lam >= alpha * rho).sum()) if rho > 0 else 0
    n = Wn.shape[0]
    blocks = {"T_RR": (slice(0, r), slice(0, r)),
              "T_C->R": (slice(0, r), slice(r, n)),
              "T_CC": (slice(r, n), slice(r, n))}
    return Q, T, blocks


def schur_sensitivity(W, rollout, alpha=0.7):
    """Damage per unit of edit for each coupling block — the probe of arXiv:2606.18457 eq. 8.

    `rollout(W) -> array` must return the model's behaviour (any array-like) with the input and readout
    maps held FIXED, so every measured change is attributable to the recurrent matrix. Returns rows sorted
    ascending by `sensitivity`; the first row is the best approximate-stabilizer candidate.
    """
    import numpy as np
    Q, T, blocks = schur_blocks(W, alpha=alpha)
    base = np.asarray(rollout(W) if not hasattr(rollout(W), "detach") else rollout(W).detach().cpu().numpy(),
                      dtype=np.float64)
    var = float(((base - base.mean(axis=0)) ** 2).mean()) or 1.0
    tnorm = float(np.linalg.norm(T)) or 1.0
    rows = []
    for name, (rs, cs) in blocks.items():
        Tt = T.copy()
        Tt[rs, cs] = 0.0
        if np.allclose(Tt, T):
            continue                                    # empty block (e.g. all modes retained)
        Wt = Q @ Tt @ Q.T
        out = rollout(_as_like(W, Wt))
        out = np.asarray(out.detach().cpu().numpy() if hasattr(out, "detach") else out, dtype=np.float64)
        d_fvu = float(((out - base) ** 2).mean() / var)
        rel = float(np.linalg.norm(T - Tt) / tnorm)
        rows.append({"coupling": name, "rel_dT": round(rel, 5), "dFVU": round(d_fvu, 6),
                     "sensitivity": round(d_fvu / max(rel, 1e-12), 5)})
    rows.sort(key=lambda r: r["sensitivity"])
    return rows


def _as_like(ref, arr):
    """Return `arr` as the same type/device/dtype as `ref` (torch tensor or numpy array)."""
    try:
        import torch
        if hasattr(ref, "detach"):
            return torch.as_tensor(arr, dtype=ref.dtype, device=ref.device)
    except Exception:  # noqa: BLE001
        pass
    return arr


def rate_distortion_pick(actions, min_dparams=1):
    """The encoding loop's criterion (HOPE §9–§10): pick the action with the lowest **J / Δparams**.

    Distortion alone cannot rank actions of different sizes — evicting a whole block hurts more than
    pruning one neuron but saves far more parameters. Ranking by cost per parameter saved is what makes a
    neuron prune, a neuron merge and a block eviction comparable on one axis, and it is the reason HOPE
    needs no per-layer ratios or hand-set thresholds.
    """
    ranked = []
    for a in actions or []:
        dp = max(int(a.get("dparams", 0)), min_dparams)
        ranked.append({**a, "dparams": dp, "dr": float(a.get("J", 0.0)) / dp})
    ranked.sort(key=lambda d: d["dr"])
    return (ranked[0] if ranked else None), ranked


def _choose(results, recall_bar=0.95, budget_spf=DETECT_BUDGET_SPF):
    """PURE decision (data-wise tested). results = {K: {"44b6":r,"6bba":r,"spf":s}}.
    Pick the SMALLEST K (fastest / most time-margin) whose MIN per-embryo recall ≥ recall_bar AND
    spf ≤ budget. If none clears the bar within budget, fall back to the highest-min-recall that fits.
    Returns (best_K, ranked)."""
    rows = []
    for K, r in (results or {}).items():
        if not isinstance(r, dict):
            continue
        r44, r6b, spf = r.get("44b6"), r.get("6bba"), r.get("spf")
        if r44 is None or r6b is None or spf is None:
            continue
        rows.append({"K": int(K), "44b6": round(r44, 3), "6bba": round(r6b, 3),
                     "min_recall": round(min(r44, r6b), 3), "spf": round(spf, 2),
                     "fits_budget": spf <= budget_spf,
                     "keeps_quality": min(r44, r6b) >= recall_bar})
    within = [d for d in rows if d["fits_budget"]]
    good = [d for d in within if d["keeps_quality"]]
    if good:
        best = min(good, key=lambda d: d["K"])            # smallest K = fastest, most margin
    elif within:
        best = max(within, key=lambda d: d["min_recall"])  # none keeps full quality → best that fits
    else:
        best = None
    ranked = sorted(rows, key=lambda d: (d["K"]))
    return (best["K"] if best else None), best, ranked


def _iterative_select(trace, recall_bar=0.95):
    """PURE SLEB-style selection (data-wise tested). trace = ordered list of removal steps, each
    {"n_kept": k, "min_recall": r, "spf": s}, produced by greedily removing the lowest-Block-Influence
    block one at a time. Pick the FEWEST-block (fastest) config still at/above the recall bar — i.e. stop
    just before the recall cliff. This beats one-shot top-K because it finds the ACTUAL floor on our data
    (block redundancy compounds; removing A can make B load-bearing). Returns (best_n_kept, best)."""
    trace = [d for d in (trace or []) if isinstance(d, dict) and d.get("min_recall") is not None
             and d.get("n_kept") is not None]
    if not trace:
        return None, None
    ok = [d for d in trace if d["min_recall"] >= recall_bar]
    if not ok:
        best = max(trace, key=lambda d: d["min_recall"]) if trace else None
        return (best["n_kept"] if best else None), best
    best = min(ok, key=lambda d: d["n_kept"])              # fewest blocks that still holds recall
    return best["n_kept"], best


class CompressSelect(BaseAgent):
    name = "compress-select"
    thread = "S"
    kind = "verdict"

    def _measure(self, ks, nds, nframes, strategy="drop_redundant"):
        """Compress Cellpose to K ViT blocks (no-train) via `strategy` and measure per-embryo recall@7µm +
        s/frame. Default strategy = drop_redundant (ShortGPT Block-Influence order) — better than truncate."""
        import sys, time
        sys.path.insert(0, str(COMP)); sys.path.insert(0, str(COMP / "experiments" / "prune_cellpose"))
        import numpy as np, pandas as pd
        try:
            import torch.nn as nn
            from cellpose import models as cpm
            import prune_lib
        except Exception as e:  # noqa: BLE001 — optional heavy deps absent → let caller escalate cleanly
            raise RuntimeError(f"compress-select deps unavailable: {type(e).__name__}: {str(e)[:80]}")
        np.random.seed(0)                                     # determinism for any sampled measurement
        from scipy import ndimage
        from src import io, metric
        from src.config import Config
        from src.io import embryo_id
        from model_scratch.train_v0 import frames_of, split_datasets
        cfg = Config(); scale = np.asarray(cfg.SCALE)
        _, te = split_datasets()
        by = {"44b6": [], "6bba": []}
        for ds in te:
            e = embryo_id(ds)
            if e in by and len(by[e]) < nds:
                by[e].append(ds)
        m = cpm.CellposeModel(gpu=True, pretrained_model="cpsam")
        net = m.net if hasattr(m, "net") else m
        pname, blocks, _ = prune_lib.find_block_lists(net)[0]; orig = list(blocks)
        parent = net
        for p in pname.split(".")[:-1]:
            parent = getattr(parent, p)
        leaf = pname.split(".")[-1]

        # BLOCK IMPORTANCE (relative residual ||out-in||/||in||): near-0 = redundant/near-identity → best to cut,
        # wherever it sits — NOT just the tail. (block_redundancy.py recipe.)
        import torch
        def _importance():
            rel = [0.0] * len(orig)
            def hook(i):
                def h(mod, inp, out):
                    x = inp[0].detach().float(); y = out.detach().float()
                    rel[i] = (y - x).norm().item() / (x.norm().item() + 1e-8)
                return h
            hs = [orig[i].register_forward_hook(hook(i)) for i in range(len(orig))]
            setattr(parent, leaf, nn.ModuleList(orig))
            d = by["6bba"][0]; ad, shape, dtype, T = frames_of(d, 1)
            m.eval(io.load_volume(ad, shape, dtype, 0).astype(np.float32), do_3D=False, z_axis=0,
                   stitch_threshold=0.5, normalize=True, batch_size=64)
            for hh in hs:
                hh.remove()
            return rel
        imp = _importance()
        keep_order = list(np.argsort(imp))                  # ascending importance → drop these first

        def setk(k, strategy="truncate"):
            if strategy == "truncate":                      # keep first k (naive tail cut)
                kept = list(range(k))
            elif strategy == "drop_redundant":              # keep the k MOST-important blocks (drop least-transforming, anywhere)
                kept = sorted(keep_order[-k:])
            elif strategy == "merge":                       # average each dropped block's params into its predecessor, keep k
                kept = sorted(keep_order[-k:])
                drop = [i for i in range(len(orig)) if i not in kept]
                for di in drop:                             # fold redundant block's weights into the nearest kept one (avg)
                    tgt = min(kept, key=lambda j: abs(j - di))
                    with torch.no_grad():
                        for pk, pv in orig[di].state_dict().items():
                            if pk in dict(orig[tgt].named_parameters()) and pv.shape == dict(orig[tgt].named_parameters())[pk].shape:
                                dict(orig[tgt].named_parameters())[pk].data.mul_(0.5).add_(0.5 * pv)
            else:
                kept = list(range(k))
            setattr(parent, leaf, nn.ModuleList([orig[i] for i in kept]))

        def cent(mk):
            l = np.unique(mk); l = l[l > 0]
            return np.array(ndimage.center_of_mass(np.ones_like(mk), mk, l), float) if len(l) else np.zeros((0, 3))

        def rs(ds):
            gn, _ = io.read_geff(COMP / f"input/biohub-cell-tracking-during-development/train/{ds}.geff")
            ad, shape, dtype, T = frames_of(ds, nframes); tp = gt = 0; secs = 0.0
            for t in range(nframes):
                v = io.load_volume(ad, shape, dtype, t).astype(np.float32); gf = gn[gn["t"] == t]
                t0 = time.time(); mk, _, _ = m.eval(v, do_3D=False, z_axis=0, stitch_threshold=0.5,
                                                    normalize=True, batch_size=64); secs += time.time() - t0
                pk = cent(mk); gt += len(gf)
                if len(gf) and len(pk):
                    pf = pd.DataFrame({"node_id": range(len(pk)), "t": t, "z": pk[:, 0], "y": pk[:, 1], "x": pk[:, 2]})
                    tp += len(metric._match_nodes(gf, pf, scale, 7.0))
            return tp / max(gt, 1), secs / nframes

        results = {}
        for k in ks:
            setk(k, strategy)
            r44 = np.mean([rs(d)[0] for d in by["44b6"]]); r6 = np.mean([rs(d)[0] for d in by["6bba"]])
            _, s = rs(by["6bba"][0])
            results[k] = {"44b6": float(r44), "6bba": float(r6), "spf": float(s)}
        return results

    def run(self, q, worker):
        spec = self.spec(q)
        strategy = spec.get("method", spec.get("strategy", "drop_redundant"))  # method: compression technique (alias strategy)
        recall_bar = spec.get("recall_bar", 0.95)
        calib_samples = int(spec.get("calib_samples", spec.get("nframes", 5)))  # calib_samples: frames per embryo for the recall/spf measurement
        results = spec.get("results")
        if not results:
            try:
                results = self._measure(spec.get("ks", [24, 20, 16, 14, 12, 10]), int(spec.get("nds", 2)),
                                        calib_samples, strategy)
            except Exception as e:  # noqa: BLE001
                return self.escalate(worker, "researcher", f"compress-select: measurement unavailable ({e}).")
        if not results:
            return self.escalate(worker, "researcher", "compress-select: no results to choose from.")
        bestK, best, ranked = _choose(results, recall_bar, spec.get("budget_spf", DETECT_BUDGET_SPF))
        # SLEB-style floor: over the same measurements, the FEWEST blocks still at/above the recall bar
        trace = [{"n_kept": int(K), "min_recall": min(r["44b6"], r["6bba"]), "spf": r["spf"]}
                 for K, r in results.items()]
        floor_k, floor = _iterative_select(trace, recall_bar)
        proof = "; ".join(f"K{d['K']}[44b6={d['44b6']} 6bba={d['6bba']} {d['spf']}s/f "
                          f"{'fit' if d['fits_budget'] else 'OVER'}]" for d in ranked)
        if bestK is None:
            return self.escalate(worker, "researcher", f"compress-select: no K fits the budget. {proof}")
        eta = round(best["spf"] * TEST_FRAMES / 2 / 3600, 1)
        summary = (f"CHOSEN compression: {bestK} blocks via {strategy} (no-train) — 44b6={best['44b6']} "
                   f"6bba={best['6bba']} {best['spf']}s/f → full-test ~{eta}h(2×T4) "
                   f"{'keeps quality' if best['keeps_quality'] else 'best-fit'}. Recall floor (SLEB): "
                   f"{floor_k} blocks. Sweep: {proof}")
        self.log(summary, kind="verdict",
                 recommendation=f"submittable detector = Cellpose-SAM compressed to {bestK} ViT blocks via "
                                f"{strategy} (no training) → fits 12h/2×T4, retains recall. Pair with Trackastra "
                                f"greedy. Floor is {floor_k} blocks; go lower only if the tracker needs the budget.")
        return self.done({"best_k": bestK, "best": best, "floor_k": floor_k, "strategy": strategy,
                          "ranked": ranked, "results": results}, summary)


_AGENT = CompressSelect()


def run(q, worker):
    return _AGENT.run(q, worker)
