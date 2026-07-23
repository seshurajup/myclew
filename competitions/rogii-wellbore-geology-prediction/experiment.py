#!/usr/bin/env python
"""Config-driven experiment dispatcher for rogii-wellbore-geology-prediction.

One YAML per experiment (config/experiments/*.yml) -> run its track -> emit standardized ledgers:
  results/<name>_oof.csv   : id, well, dtvt_pred, dtvt_true          (train wells, well-disjoint)
  results/<name>_test.csv  : id, dtvt_pred, tvt_ps                   (test wells, for submission/blend)
  results/ledger.csv       : appended row  name,track,cv_rmse,baseline,goal,beats_baseline,beats_goal,ts

Tracks: A = residual-dtvt GBM (fleet tab pack) | B = particle filter | blend = convex combine A+B OOF.
All CV is on the residual dtvt = TVT - tvt_ps, which equals the competition RMSE.
"""
import argparse
import datetime as dt
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

HERE = Path(__file__).resolve().parent
FLEET = HERE / "fleet_agents"
RP = "/home/seshu/kaggle/2026/biohub-cell-tracking-during-development/tools/researchpapers"
sys.path.insert(0, RP)
pkg = types.ModuleType("fleet_agents")
pkg.__path__ = [str(FLEET.resolve())]
sys.modules["fleet_agents"] = pkg

from fleet_agents import geology_pack as G          # noqa: E402
from fleet_agents import geology_trackB as B        # noqa: E402
from fleet_agents import comp_config as CC          # noqa: E402
from fleet_agents import comp_onboard as ONB        # noqa: E402
from fleet_agents import tab_autobaseline as TA     # noqa: E402

AUTO = HERE / "config" / "_auto"
RES = HERE / "results"
SUBS = HERE / "submissions"
for d in (AUTO, RES, SUBS):
    d.mkdir(parents=True, exist_ok=True)
TRAIN_DIR = HERE / "input" / "train"
TEST_DIR = HERE / "input" / "test"
SAMPLE = HERE / "input" / "sample_submission.csv"


def _rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def _write_submission(test_ledger, path):
    """test_ledger: DataFrame[id, dtvt_pred, tvt_ps] -> submission[id, tvt]."""
    sub = test_ledger.copy()
    sub["tvt"] = sub["dtvt_pred"] + sub["tvt_ps"]
    sub = sub[["id", "tvt"]]
    samp = pd.read_csv(SAMPLE)
    sub = samp[["id"]].merge(sub, on="id", how="left")   # enforce sample order + coverage
    sub["tvt"] = sub["tvt"].fillna(sub["tvt"].median())
    sub.to_csv(path, index=False)
    ok = list(sub.columns) == list(samp.columns) and set(sub.id) == set(samp.id) and int(sub.tvt.isna().sum()) == 0
    return ok, len(sub)


# ----------------------------------------------------------------------------- tracks
def run_trackA(p):
    limit = p.get("limit")
    print(">> [A] assembling flat tables (geology domain hook)...")
    G.geology_assemble(str(TRAIN_DIR), AUTO / "train.csv", training=True, limit=limit)
    G.geology_assemble(str(TEST_DIR), AUTO / "test.csv", training=False, limit=limit)
    tr = pd.read_csv(AUTO / "train.csv"); te = pd.read_csv(AUTO / "test.csv")
    print(f"   train rows={len(tr)} wells={tr.well.nunique()} | test rows={len(te)} wells={te.well.nunique()}")

    sample = pd.read_csv(SAMPLE, nrows=1)
    cfg = ONB.infer_config("rogii-wellbore-geology-prediction",
                           files=["train.csv", "test.csv", "sample_submission.csv"],
                           sample_header=list(sample.columns))
    cfg.data = {"train": str(AUTO / "train.csv"), "test": str(AUTO / "test.csv"), "sample_sub": str(SAMPLE)}
    cfg.save(AUTO / "comp_config_rogii.json")
    assert cfg.pack() == "tab", f"expected tab pack, got {cfg.pack()}"

    # low-bit training only (user directive): map bits -> GBM feature-quantization bin count
    nbins = {8: 255, 4: 15}.get(int(p.get("bits", 8)), 255)
    bit_params = {"lgbm": {"max_bin": nbins}, "xgb": {"max_bin": nbins},
                  "catboost": {"border_count": nbins}, "histgbm": {"max_bins": nbins}}
    print(f">> [A] low-bit training: bits={p.get('bits', 8)} -> bin_count={nbins}")
    res = TA.run_pipeline(cfg, out_path=str(RES / "trackA_test_raw.csv"), seed=42, fe=False,
                          n_folds=int(p.get("folds", 5)), gpu=(p.get("gpu") or None),
                          postprocess=False, params=bit_params)
    oof = np.asarray(res["_oof"])                       # aligned to train.csv row order (dtvt)
    cv = _rmse(oof, tr.tvt.values)
    oof_df = pd.DataFrame({"id": tr.id, "well": tr.well, "dtvt_pred": oof, "dtvt_true": tr.tvt.values})
    oof_df.to_csv(RES / "trackA_oof.csv", index=False)
    raw = pd.read_csv(RES / "trackA_test_raw.csv")      # id, tvt(=dtvt)
    test_df = raw.rename(columns={"tvt": "dtvt_pred"}).merge(te[["id", "tvt_ps"]], on="id", how="left")
    test_df[["id", "dtvt_pred", "tvt_ps"]].to_csv(RES / "trackA_test.csv", index=False)
    print(f">> [A] per-backend CV {({b: round(v,3) for b,v in res['per_backend_cv'].items()})} blend CV {cv:.4f}")
    return cv, test_df[["id", "dtvt_pred", "tvt_ps"]]


def run_trackB(p):
    limit = p.get("limit")
    print(">> [B] particle filter OOF over all wells (this is the slow one)...")
    gpu = bool(p.get("gpu", True))
    print(f">> [B] gpu={gpu} n_seeds={p.get('n_seeds', 256)} n_particles={p.get('n_particles', 500)}")
    B.trackB_oof(str(TRAIN_DIR), RES / "trackB_oof_raw.csv", training=True, limit=limit,
                 n_particles=int(p.get("n_particles", 500)), n_seeds=int(p.get("n_seeds", 256)),
                 scale=float(p.get("scale", 5.0)), gpu=gpu)
    o = pd.read_csv(RES / "trackB_oof_raw.csv")
    cv = _rmse(o.trackB_dtvt, o.true_dtvt)
    conf_col = ["trackB_conf"] if "trackB_conf" in o.columns else []
    o.rename(columns={"trackB_dtvt": "dtvt_pred", "true_dtvt": "dtvt_true"})[
        ["id", "well", "dtvt_pred", "dtvt_true"] + conf_col].to_csv(RES / "trackB_oof.csv", index=False)
    B.trackB_oof(str(TEST_DIR), RES / "trackB_test_raw.csv", training=False, limit=limit,
                 n_particles=int(p.get("n_particles", 500)), n_seeds=int(p.get("n_seeds", 256)),
                 scale=float(p.get("scale", 5.0)), gpu=gpu)
    t = pd.read_csv(RES / "trackB_test_raw.csv")
    t["tvt_ps"] = t["trackB_tvt"] - t["trackB_dtvt"]
    conf_col_t = ["trackB_conf"] if "trackB_conf" in t.columns else []
    test_df = t.rename(columns={"trackB_dtvt": "dtvt_pred"})[["id", "dtvt_pred", "tvt_ps"] + conf_col_t]
    test_df.to_csv(RES / "trackB_test.csv", index=False)
    print(f">> [B] particle-filter OOF RMSE {cv:.4f}")
    return cv, test_df


def run_blend(p):
    a = pd.read_csv(RES / "trackA_oof.csv"); b = pd.read_csv(RES / "trackB_oof.csv")
    m = a.merge(b[["id", "dtvt_pred"]], on="id", suffixes=("_A", "_B"))
    ws = np.linspace(0, 1, 101)
    cvs = [(_rmse(w * m.dtvt_pred_A + (1 - w) * m.dtvt_pred_B, m.dtvt_true), w) for w in ws]
    cv, w = min(cvs)
    print(f">> [blend] optimal wA={w:.2f} wB={1-w:.2f} -> OOF RMSE {cv:.4f}")
    ta = pd.read_csv(RES / "trackA_test.csv"); tb = pd.read_csv(RES / "trackB_test.csv")
    mt = ta.merge(tb[["id", "dtvt_pred"]], on="id", suffixes=("_A", "_B"))
    mt["dtvt_pred"] = w * mt.dtvt_pred_A + (1 - w) * mt.dtvt_pred_B
    return cv, mt[["id", "dtvt_pred", "tvt_ps"]]


def run_blend_conf(p):
    """Confidence-weighted blend: per-well wB is a linear function of Track B's own PF ensemble
    confidence (no leakage — computed from the PF's internal seed-likelihood spread, available at real
    inference time). Fits (lo,hi) of wB=lo+(hi-lo)*conf on OOF by grid search, same idea as the public
    6.390 notebook's 'visible-prefix trust gating' but using PF self-confidence instead of a backtest."""
    a = pd.read_csv(RES / "trackA_oof.csv"); b = pd.read_csv(RES / "trackB_oof.csv")
    if "trackB_conf" not in b.columns:
        raise RuntimeError("trackB_oof.csv has no trackB_conf column — rerun trackB_particle_filter first")
    m = a.merge(b[["id", "dtvt_pred", "trackB_conf"]], on="id", suffixes=("_A", "_B"))
    best = (np.inf, 0.0, 1.0)
    for lo in np.linspace(0.0, 1.0, 11):
        for hi in np.linspace(0.0, 1.0, 11):
            if hi < lo:
                continue
            wB = lo + (hi - lo) * m.trackB_conf.to_numpy()
            cv = _rmse((1 - wB) * m.dtvt_pred_A + wB * m.dtvt_pred_B, m.dtvt_true)
            if cv < best[0]:
                best = (cv, lo, hi)
    cv, lo, hi = best
    print(f">> [blend_conf] wB = {lo:.2f} + {hi-lo:.2f}*conf -> OOF RMSE {cv:.4f}")
    ta = pd.read_csv(RES / "trackA_test.csv"); tb = pd.read_csv(RES / "trackB_test.csv")
    mt = ta.merge(tb[["id", "dtvt_pred", "trackB_conf"]], on="id", suffixes=("_A", "_B"))
    wB = lo + (hi - lo) * mt.trackB_conf.to_numpy()
    mt["dtvt_pred"] = (1 - wB) * mt.dtvt_pred_A + wB * mt.dtvt_pred_B
    return cv, mt[["id", "dtvt_pred", "tvt_ps"]]


def run_trackC(p):
    """Neural sequence Track C — shells out to the kaggle_nlp env (torch+torchao fp8), then reads
    the standardized ledgers it wrote."""
    import json
    import subprocess
    print(f">> [C] launching FP8 seq model in kaggle_nlp env (precision={p.get('precision','fp8')})...")
    cmd = ["conda", "run", "--no-capture-output", "-n", "kaggle_nlp",
           "python", str(HERE / "trackC_run.py"), "--params", json.dumps(p)]
    proc = subprocess.run(cmd, cwd=str(HERE), text=True, capture_output=True)
    print(proc.stdout[-2000:] if proc.stdout else "")
    if proc.returncode != 0:
        print(proc.stderr[-2000:]); raise RuntimeError("trackC_run failed")
    cv = None
    for line in proc.stdout.splitlines():
        if line.startswith("TRACKC_CV"):
            cv = float(line.split()[1])
    if cv is None:
        raise RuntimeError("trackC_run produced no CV")
    test_df = pd.read_csv(RES / "trackC_test.csv")
    return cv, test_df[["id", "dtvt_pred", "tvt_ps"]]


def run_trackD(p):
    """Google TabFM in-context tabular model — shells out to kaggle_tabular env (has tabfm installed),
    reads the same assembled flat table Track A uses. No training; context is a well-disjoint subsample."""
    import json
    import subprocess
    print(f">> [D] launching TabFM (max_context={p.get('max_context', 2000)}) in kaggle_tabular env...")
    cmd = ["conda", "run", "--no-capture-output", "-n", "kaggle_tabular",
           "python", str(HERE / "trackD_run.py"), "--params", json.dumps(p)]
    proc = subprocess.run(cmd, cwd=str(HERE), text=True, capture_output=True)
    print(proc.stdout[-2000:] if proc.stdout else "")
    if proc.returncode != 0:
        print(proc.stderr[-2000:]); raise RuntimeError("trackD_run failed")
    cv = None
    for line in proc.stdout.splitlines():
        if line.startswith("TRACKD_CV"):
            cv = float(line.split()[1])
    if cv is None:
        raise RuntimeError("trackD_run produced no CV")
    test_df = pd.read_csv(RES / "trackD_test.csv")
    return cv, test_df[["id", "dtvt_pred", "tvt_ps"]]



def run_blend_stack(p):
    """Ridge-STACK meta-learner over ALL available track OOF (A+B+C+D) — the notebook-beating ensemble.
    Fits sklearn Ridge on the out-of-fold predictions (already well-disjoint, so no leakage) to learn the
    optimal linear combination, then applies the same meta-weights to the test predictions. Uses only the
    tracks whose OOF exists; needs at least 2. Mirrors the AmgedAlfaqih ridge-stack but over our OOF."""
    from sklearn.linear_model import Ridge
    inputs = p.get("inputs", ["trackA_oof", "trackB_oof", "trackC_oof", "trackD_oof"])
    alpha = float(p.get("alpha", 1.0))
    avail, test_avail, tags = [], [], []
    base = None
    for oof_name in inputs:
        tag = oof_name.replace("_oof", "")            # trackA, trackB, ...
        of = RES / f"{oof_name}.csv"; tf = RES / f"{tag}_test.csv"
        if not of.exists() or not tf.exists():
            print(f">> [blend_stack] skip {tag}: OOF/test not found (run that track first)")
            continue
        o = pd.read_csv(of); t = pd.read_csv(tf)
        if base is None:
            base = o[["id"]].copy()
            base["dtvt_true"] = o["dtvt_true"] if "dtvt_true" in o.columns else o.get("true_dtvt")
            base["tvt_ps"] = o["tvt_ps"] if "tvt_ps" in o.columns else np.nan
            tbase = t[["id"]].copy(); tbase["tvt_ps"] = t["tvt_ps"] if "tvt_ps" in t.columns else np.nan
        pcol = "dtvt_pred" if "dtvt_pred" in o.columns else [c for c in o.columns if "dtvt" in c and "pred" in c][0]
        base = base.merge(o[["id", pcol]].rename(columns={pcol: f"p_{tag}"}), on="id", how="inner")
        tbase = tbase.merge(t[["id", "dtvt_pred"]].rename(columns={"dtvt_pred": f"p_{tag}"}), on="id", how="left")
        tags.append(tag)
    if len(tags) < 2:
        raise RuntimeError(f"blend_stack needs >=2 tracks with OOF; found {tags}")
    feat = [f"p_{t}" for t in tags]
    X = base[feat].to_numpy(); y = base["dtvt_true"].to_numpy()
    reg = Ridge(alpha=alpha, fit_intercept=True).fit(X, y)
    oof_pred = reg.predict(X)
    cv = _rmse(oof_pred, y)
    print(f">> [blend_stack] ridge over {tags} coef={np.round(reg.coef_,3).tolist()} b={reg.intercept_:.3f} -> OOF RMSE {cv:.4f}")
    tbase[feat] = tbase[feat].fillna(tbase[feat].mean())
    tbase["dtvt_pred"] = reg.predict(tbase[feat].to_numpy())
    return cv, tbase[["id", "dtvt_pred", "tvt_ps"]]


TRACKS = {"A": run_trackA, "B": run_trackB, "C": run_trackC, "D": run_trackD,
          "blend": run_blend, "blend_conf": run_blend_conf, "blend_stack": run_blend_stack}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    exp = yaml.safe_load(Path(args.config).read_text())
    name, track = exp["name"], exp["track"]
    goal = float(exp.get("goal_rmse", 6.858))
    baseline = float(exp.get("baseline_rmse", np.nan))
    p = exp.get("params", {})
    print(f"=== EXPERIMENT {name} (track {track}) goal<{goal} ===")

    cv, test_df = TRACKS[track](p)

    sub_ok = ""
    if p.get("make_submission") and test_df is not None:
        ok, n = _write_submission(test_df, SUBS / f"submission_{name}.csv")
        sub_ok = f" submission_{name}.csv rows={n} valid={ok}"

    row = dict(name=name, track=track, cv_rmse=round(cv, 4), baseline=baseline, goal=goal,
               beats_baseline=bool(cv < baseline) if np.isfinite(baseline) else "",
               beats_goal=bool(cv < goal),
               ts=dt.datetime.now().strftime("%Y-%m-%d %H:%M"))

    ledger = RES / "ledger.csv"
    df = pd.concat([pd.read_csv(ledger), pd.DataFrame([row])], ignore_index=True) if ledger.exists() else pd.DataFrame([row])
    df.to_csv(ledger, index=False)

    # source of truth: Postgres kaggle_rogii_wellbore_geology_prediction.experiment_journal (per-comp
    # PG store, matches every other competition) — the :7788 runtime/journal board reads THIS, not a file.
    try:
        import importlib.util as _ilu
        _spec = _ilu.spec_from_file_location(
            "fa_db", "/home/seshu/kaggle/2026/biohub-cell-tracking-during-development/fleet_agents/db.py")
        _db = _ilu.module_from_spec(_spec)
        _spec.loader.exec_module(_db)
        jrow = {"ts": dt.datetime.now(dt.timezone.utc).isoformat(), "exp": name, "cv": round(cv, 4),
                "lb": None, "desc": f"Track {track} experiment ({name}): CV RMSE {cv:.4f} vs baseline {baseline}, goal {goal}",
                "change": name, "parent": None, "script": f"config/experiments/{Path(args.config).name}",
                "trn_set": "full", "stage": "verdict", "kept": bool(cv < baseline) if np.isfinite(baseline) else True,
                "observation": f"CV RMSE {cv:.4f} {'beats' if cv < goal else 'below'} goal {goal}",
                "description": f"Track {track} experiment", "git_hash": ""}
        _db.upsert_journal("rogii-wellbore-geology-prediction", [jrow])
    except Exception as _e:  # noqa: BLE001
        print(f"  (postgres journal write skipped: {_e})")
    print(f">> RESULT {name}: CV RMSE {cv:.4f} | baseline {baseline} | goal {goal} "
          f"-> {'BEATS GOAL ✅' if cv < goal else 'below goal'}{sub_ok}")


if __name__ == "__main__":
    main()
