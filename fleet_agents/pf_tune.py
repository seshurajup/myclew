"""pf_tune — ARCHITECTURE-CORRECT PF knob governor. Instead of re-running the ~68-min cupy PF per knob value,
we compute the expensive part ONCE and cache the per-seed trajectories, then sweep every seed-combination knob
(scale/softmax-temp, seed weighting, seed selection, blend) as an INSTANT re-weighting of the cache — zero PF
re-runs. Only true DYNAMICS knobs (gs, process noise, n_particles) need a fresh pass, and those batch as a
tensor dim. math_master.knob_vertex governs the resulting 1-D curves.

  build_seed_cache(gs_scale=1.30)      # one PF pass over the tuning subset, saves per-seed res+log_lik+truth
  sweep_scale([2,5,8,12,20])           # INSTANT: pooled RMSE per scale off the cache + governed optimum
"""
from __future__ import annotations
import os, sys, types, time, pickle
import numpy as np, pandas as pd

FA = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(FA)
CACHE = os.path.join(ROOT, "results", "pf_seed_cache.pkl")

def _trackB(gs_scale=1.0):
    src = open(os.path.join(FA, "geology_trackB.py")).read()
    if gs_scale != 1.0:
        src = src.replace('gs = float(np.clip(np.nanstd(kn["GR"].fillna(0).values - tw_at_k), 10.0, 60.0))',
                          f'gs = float(np.clip(np.nanstd(kn["GR"].fillna(0).values - tw_at_k), 10.0, 60.0))*{gs_scale}')
    if "fleet_agents" not in sys.modules:
        pkg = types.ModuleType("fleet_agents"); pkg.__path__ = [FA]; sys.modules["fleet_agents"] = pkg
    m = types.ModuleType("fleet_agents.geology_trackB"); m.__file__ = os.path.join(FA, "geology_trackB.py")
    m.__dict__["__name__"] = "fleet_agents.geology_trackB"; sys.modules["fleet_agents.geology_trackB"] = m
    exec(compile(src, m.__file__, "exec"), m.__dict__); return m

def build_seed_cache(gs_scale=1.30, n_seeds=128, n_particles=500):
    wells = sorted(pd.read_csv(os.path.join(ROOT, "config/pf_tune_subset.csv")).well)
    tb = _trackB(gs_scale); store = {}; t0 = time.time()
    for i, w in enumerate(wells):
        try:
            hw = pd.read_csv(f"{ROOT}/input/train/{w}__horizontal_well.csv")
            tw = pd.read_csv(f"{ROOT}/input/train/{w}__typewell.csv")
        except Exception: continue
        pm = hw["TVT_input"].isna().to_numpy() & hw["TVT"].notna().to_numpy()
        if pm.sum() == 0: continue
        ps = int(np.argmax(hw["TVT_input"].isna().to_numpy())); tvtps = float(hw["TVT"].iloc[ps-1])
        _, _, per = tb.run_pf_ensemble_gpu(hw, tw, n_particles=n_particles, n_seeds=n_seeds, scale=5.0, return_per_seed=True)
        # res rows follow list(ev.index); keep only eval-and-true rows (pm)
        evidx = per["idx"]; keep = np.isin(evidx, np.where(pm)[0])
        store[w] = {"res": per["res"][:, keep].astype(np.float32), "ll": per["log_lik"].astype(np.float64),
                    "true": hw["TVT"].to_numpy()[evidx[keep]].astype(np.float32)}
        if i % 50 == 0: print(f"  cached {i}/{len(wells)} ({time.time()-t0:.0f}s)", flush=True)
    pickle.dump({"gs_scale": gs_scale, "n_seeds": n_seeds, "wells": store}, open(CACHE, "wb"))
    print(f"seed cache: {len(store)} wells, gs={gs_scale} -> {CACHE} ({time.time()-t0:.0f}s)")
    return CACHE

def _combine(ll, res, scale, sel=None):
    l = ll - ll.max(); wv = np.exp(l / scale)
    if sel is not None: wv = wv * sel
    wv = wv / max(wv.sum(), 1e-300)
    return (wv[:, None] * res).sum(0)

def sweep_scale(scales, cache=CACHE):
    d = pickle.load(open(cache, "rb")); W = d["wells"]
    pts = []
    for sc in scales:
        P, T = [], []
        for w, c in W.items():
            P.append(_combine(c["ll"], c["res"], sc)); T.append(c["true"])
        P = np.concatenate(P); T = np.concatenate(T)
        pts.append((float(sc), float(np.sqrt(np.mean((P - T) ** 2)))))
    import importlib.util
    spec = importlib.util.spec_from_file_location("fleet_agents.math_master", os.path.join(FA, "math_master.py"))
    mm = importlib.util.module_from_spec(spec); sys.modules["fleet_agents.math_master"] = mm
    try: spec.loader.exec_module(mm); opt = mm.knob_vertex(pts)
    except Exception as e: opt = {"argmin": min(pts, key=lambda p: p[1]), "note": str(e)}
    return {"points": pts, "optimum": opt, "gs_scale": d["gs_scale"]}


def tune_dynamics(pn_scales=(0.5, 1.0, 2.0, 4.0), base_gs=1.30, n_seeds=48, n_wells=120):
    """Govern the PF PROCESS-NOISE (position noise PN) — a DYNAMICS knob, so it needs fresh passes (the seed
    cache cannot be reused). Runs a reduced-seed subset per PN scale, then math_master.knob_vertex governs the
    curve. Kept small (n_wells) for speed; confirm the winner at full scale. Reusable for any dynamics scalar."""
    wells = sorted(pd.read_csv(os.path.join(ROOT, "config/pf_tune_subset.csv")).well)[:n_wells]
    import importlib.util
    pts = []; t0 = time.time()
    for sc in pn_scales:
        src = open(os.path.join(FA, "geology_trackB.py")).read()
        src = src.replace('gs = float(np.clip(np.nanstd(kn["GR"].fillna(0).values - tw_at_k), 10.0, 60.0))',
                          f'gs = float(np.clip(np.nanstd(kn["GR"].fillna(0).values - tw_at_k), 10.0, 60.0))*{base_gs}')
        src = src.replace("MOM, VN, PN, RP, RR, RESAMP = 0.998, 0.002, 0.005, 0.1, 0.001, 0.5",
                          f"MOM, VN, PN, RP, RR, RESAMP = 0.998, 0.002, {0.005*sc}, 0.1, 0.001, 0.5")
        m = types.ModuleType("fleet_agents.geology_trackB"); m.__file__ = os.path.join(FA, "geology_trackB.py")
        m.__dict__["__name__"] = "fleet_agents.geology_trackB"; sys.modules["fleet_agents.geology_trackB"] = m
        if "fleet_agents" not in sys.modules:
            pkg = types.ModuleType("fleet_agents"); pkg.__path__ = [FA]; sys.modules["fleet_agents"] = pkg
        exec(compile(src, m.__file__, "exec"), m.__dict__)
        P, T = [], []
        for w in wells:
            try:
                hw = pd.read_csv(f"{ROOT}/input/train/{w}__horizontal_well.csv")
                tw = pd.read_csv(f"{ROOT}/input/train/{w}__typewell.csv")
            except Exception: continue
            pm = hw["TVT_input"].isna().to_numpy() & hw["TVT"].notna().to_numpy()
            if pm.sum() == 0: continue
            idx = np.where(pm)[0]
            full, _ = m.run_pf_ensemble_gpu(hw, tw, n_particles=500, n_seeds=n_seeds, scale=5.0)
            P.append(full[idx]); T.append(hw["TVT"].to_numpy()[idx])
        P = np.concatenate(P); T = np.concatenate(T)
        pts.append((float(sc), float(np.sqrt(np.mean((P - T) ** 2)))))
        print(f"  PN_scale={sc}: subset PF={pts[-1][1]:.3f} ({time.time()-t0:.0f}s)", flush=True)
    spec = importlib.util.spec_from_file_location("fleet_agents.math_master", os.path.join(FA, "math_master.py"))
    mm = importlib.util.module_from_spec(spec); sys.modules["fleet_agents.math_master"] = mm
    try: spec.loader.exec_module(mm); opt = mm.knob_vertex(pts)
    except Exception as e: opt = {"argmin": min(pts, key=lambda x: x[1]), "note": str(e)}
    return {"knob": "PN_scale", "points": pts, "optimum": opt}

if __name__ == "__main__":
    import sys as _s
    if len(_s.argv) > 1 and _s.argv[1] == "build":
        build_seed_cache(gs_scale=1.30)
    else:
        import json; print(json.dumps(sweep_scale([2, 3.5, 5, 7, 10, 14]), default=str, indent=2))
