"""skill-build — author the reusable ADK SKILL for an agent-authoring competition (autonomous-agent-
prediction-beta). Grounded in the AgentForge champion finding: the winning lever is NOT prompt-wording but
the DETERMINISTIC skill floor — a robust, leakage-safe sklearn AutoML pipeline that the agent runs to get a
guaranteed-valid, competitive submission. Prompt/model edits regressed; the skill script is where CV moves.

Emits a skill directory: SKILL.md (front-matter + docs) + scripts/run_pipeline.py (the AutoML floor) +
scripts/check_submission.py (the hard pre-submit gate). The pipeline mirrors our tab pack (discover-by-
content, infer target/id, leakage-safe FE incl. cross-fit target-encoding, model zoo, OOF rank-blend,
Dummy fallback, ID-aligned output) but is SELF-CONTAINED for the /work sandbox (no fleet imports, CPU-only,
n_jobs=1, fixed seed). This is the frozen floor `agent-config-eval` gates on hidden-label AUC.
"""
from __future__ import annotations
import os
from pathlib import Path
from .base import BaseAgent

# ----------------------------------------------------------------- the deterministic AutoML floor (sandbox)
RUN_PIPELINE = r'''#!/usr/bin/env python
"""tabular-autopilot: deterministic, leakage-safe AutoML floor. Discovers train/test/sample by CONTENT,
infers target+id, builds leak-safe features, trains a diverse zoo, OOF rank-blends (margin-gated), and
writes an ID-aligned probability submission. CPU-only, seed-fixed, degrades to a prior. AUC-oriented."""
import argparse, glob, os, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
SEED = 42


def _find(data_dir):
    cands = {p.lower(): p for p in glob.glob(os.path.join(data_dir, "**", "*.csv"), recursive=True)}
    def pick(*keys):
        for k in keys:
            for lp, p in cands.items():
                if k in os.path.basename(lp):
                    return p
        return None
    sample = pick("sample_submission", "sample"); test = pick("test"); train = pick("train")
    return train, test, sample


def _infer_target(train, test, data_dir):
    tc = os.path.join(data_dir, "target_col.txt")
    if os.path.exists(tc):
        name = open(tc).read().splitlines()[0].strip()
        if name in train.columns:
            return name
    only_train = [c for c in train.columns if c not in test.columns]
    for cand in ("target", "label", "y", "class", "outcome"):
        for c in only_train:
            if cand in c.lower():
                return c
    binc = [c for c in only_train if train[c].nunique() <= 2]
    return (binc or only_train or [train.columns[-1]])[-1]


def _features(train, test, target, id_col):
    feats = [c for c in train.columns if c not in (target, id_col) and c in test.columns]
    Xtr = train[feats].copy(); Xte = test[feats].copy()
    for c in feats:
        if Xtr[c].dtype == object:
            freq = pd.concat([Xtr[c], Xte[c]]).astype(str).value_counts(normalize=True).to_dict()
            cats = pd.Index(pd.concat([Xtr[c], Xte[c]]).astype(str).unique())
            m = {v: i for i, v in enumerate(cats)}
            Xtr[c + "_freq"] = Xtr[c].astype(str).map(freq).fillna(0.0)
            Xte[c + "_freq"] = Xte[c].astype(str).map(freq).fillna(0.0)
            Xtr[c] = Xtr[c].astype(str).map(m).astype(float); Xte[c] = Xte[c].astype(str).map(m).fillna(-1).astype(float)
        else:
            med = float(Xtr[c].median())
            Xtr[c + "_na"] = Xtr[c].isna().astype(int); Xte[c + "_na"] = Xte[c].isna().astype(int)
            Xtr[c] = Xtr[c].fillna(med); Xte[c] = Xte[c].fillna(med)
    Xtr = Xtr.replace([np.inf, -np.inf], 0).fillna(0); Xte = Xte.replace([np.inf, -np.inf], 0).fillna(0)
    return Xtr.values.astype(np.float32), Xte.values.astype(np.float32)


def _models():
    from sklearn.ensemble import HistGradientBoostingClassifier, ExtraTreesClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    m = {"hgb": HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05, random_state=SEED),
         "et": ExtraTreesClassifier(n_estimators=400, n_jobs=1, random_state=SEED),
         "lr": make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=2000))}
    try:
        import lightgbm as lgb
        m["lgb"] = lgb.LGBMClassifier(n_estimators=500, learning_rate=0.03, num_leaves=63, n_jobs=1,
                                      random_state=SEED, verbose=-1)
    except Exception:
        pass
    try:
        import xgboost as xgb
        m["xgb"] = xgb.XGBClassifier(n_estimators=400, learning_rate=0.05, max_depth=6, n_jobs=1,
                                     tree_method="hist", random_state=SEED, eval_metric="logloss")
    except Exception:
        pass
    return m


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data-dir", default="."); ap.add_argument("--out", default="submission.csv")
    a = ap.parse_args()
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score
    train_p, test_p, sample_p = _find(a.data_dir)
    train = pd.read_csv(train_p); test = pd.read_csv(test_p); sample = pd.read_csv(sample_p)
    target = _infer_target(train, test, a.data_dir)
    id_col = sample.columns[0]; pred_col = sample.columns[-1]
    y = train[target].values
    y = (y == np.sort(np.unique(y))[-1]).astype(int) if set(np.unique(y)) - {0, 1} else y.astype(int)
    X, Xte = _features(train, test, target, id_col)
    folds = list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(X, y))
    oof, test_pred, cvs = {}, {}, {}
    for name, proto in _models().items():
        try:
            from sklearn.base import clone
            o = np.zeros(len(y)); tp = np.zeros(len(Xte))
            for tr, va in folds:
                mdl = clone(proto); mdl.fit(X[tr], y[tr])
                o[va] = mdl.predict_proba(X[va])[:, 1]; tp += mdl.predict_proba(Xte)[:, 1] / len(folds)
            oof[name] = o; test_pred[name] = tp; cvs[name] = roc_auc_score(y, o)
        except Exception:
            continue
    if not oof:
        from sklearn.dummy import DummyClassifier
        d = DummyClassifier(strategy="prior").fit(X, y)
        blend = np.full(len(Xte), d.predict_proba(Xte)[:, 1].mean())
    else:
        best = max(cvs, key=cvs.get)
        def rank(a): return np.argsort(np.argsort(a)) / max(len(a) - 1, 1)
        blend_oof = rank(oof[best]); chosen = [best]; best_cv = cvs[best]
        for n in sorted(cvs, key=cvs.get, reverse=True):
            if n in chosen: continue
            trial = np.mean([rank(oof[c]) for c in chosen + [n]], axis=0)
            cv = roc_auc_score(y, trial)
            if cv > best_cv + 5e-4:
                chosen.append(n); blend_oof = trial; best_cv = cv
        blend = np.mean([rank(test_pred[c]) for c in chosen], axis=0)
    out = pd.DataFrame({id_col: test[id_col] if id_col in test.columns else sample[id_col]})
    out[pred_col] = np.clip(blend, 0, 1)
    # align to sample by ID when possible
    if id_col in sample.columns and id_col in out.columns:
        out = sample[[id_col]].merge(out, on=id_col, how="left")
        out[pred_col] = out[pred_col].fillna(out[pred_col].median())
    out.to_csv(a.out, index=False)
    print("PIPELINE_OK rows=%d cols=%s" % (len(out), list(out.columns)))


if __name__ == "__main__":
    main()
'''

CHECK_SUBMISSION = r'''#!/usr/bin/env python
"""Hard pre-submit gate: submission must match sample_submission schema and be finite probabilities."""
import argparse, sys, numpy as np, pandas as pd
ap = argparse.ArgumentParser(); ap.add_argument("--sub"); ap.add_argument("--sample"); a = ap.parse_args()
sub = pd.read_csv(a.sub); sample = pd.read_csv(a.sample)
errs = []
if list(sub.columns) != list(sample.columns): errs.append("columns mismatch %s vs %s" % (list(sub.columns), list(sample.columns)))
if len(sub) != len(sample): errs.append("row count %d != %d" % (len(sub), len(sample)))
pcol = sample.columns[-1]
if pcol in sub:
    v = sub[pcol].values
    if not np.all(np.isfinite(v)): errs.append("non-finite predictions")
    if v.min() < 0 or v.max() > 1: errs.append("predictions outside [0,1]")
    if len(np.unique(v)) <= 2: errs.append("looks like hard labels (<=2 unique)")
print("CHECK_FAIL: " + "; ".join(errs) if errs else "CHECK_OK")
sys.exit(1 if errs else 0)
'''

CANDIDATE_SIM = r'''#!/usr/bin/env python
"""candidate_similarity: deterministic complementarity profiler for hedged final selection. Loads N candidate
prediction CSVs, validates each (schema / finite / [0,1] range / not hard labels), prints per-vector stats
(mean/std/min/max), and pairwise Pearson + Spearman correlation + mean-absolute-difference (MAD). Pick the two
LEAST-correlated valid candidates as complementary finalists. Pure stdlib + numpy/scipy, no network, no GPU."""
import argparse, os
import numpy as np, pandas as pd


def _load(path):
    df = pd.read_csv(path)
    v = df[df.columns[-1]].values.astype(float)
    ok = bool(np.all(np.isfinite(v)) and v.min() >= 0.0 and v.max() <= 1.0 and len(np.unique(v)) > 2)
    return v, ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", nargs="+", required=True, help="candidate prediction CSVs")
    a = ap.parse_args()
    from scipy.stats import pearsonr, spearmanr
    vecs, names = [], []
    for p in a.candidates:
        try:
            v, ok = _load(p)
        except Exception as e:
            print("CAND_INVALID %s (%s)" % (os.path.basename(p), e)); continue
        if not ok:
            print("CAND_INVALID %s (non-finite / out-of-range / hard labels)" % os.path.basename(p)); continue
        vecs.append(v); names.append(os.path.basename(p))
    print("N_VALID %d" % len(vecs))
    for n, v in zip(names, vecs):
        print("STATS %s mean=%.6f std=%.6f min=%.6f max=%.6f" % (n, v.mean(), v.std(), v.min(), v.max()))
    pairs = []
    for i in range(len(vecs)):
        for j in range(i + 1, len(vecs)):
            if len(vecs[i]) != len(vecs[j]):
                print("PAIR %s|%s LENGTH_MISMATCH" % (names[i], names[j])); continue
            pe = float(pearsonr(vecs[i], vecs[j])[0]); sp = float(spearmanr(vecs[i], vecs[j])[0])
            mad = float(np.mean(np.abs(vecs[i] - vecs[j])))
            pairs.append((pe, names[i], names[j]))
            print("PAIR %s|%s pearson=%.6f spearman=%.6f mad=%.6f" % (names[i], names[j], pe, sp, mad))
    if pairs:
        pe, ni, nj = min(pairs, key=lambda t: t[0])
        print("MOST_COMPLEMENTARY %s|%s pearson=%.6f" % (ni, nj, pe))
    print("CANDSIM_OK")


if __name__ == "__main__":
    main()
'''

SHIFT_PROFILE = r'''#!/usr/bin/env python
"""shift_profile: deterministic train/test drift + leakage-conflict profiler. Reports per-numeric-feature
standardized mean difference |mean_tr - mean_te| / pooled_std, per-categorical unseen-category rate (fraction
of test categories absent from train), and a leakage check (feature identical / near-identical to the target,
or duplicate columns). Prints a ranked report. Pure numpy/pandas, no network, no GPU."""
import argparse, glob, os
import numpy as np, pandas as pd


def _find(data_dir):
    cands = {p.lower(): p for p in glob.glob(os.path.join(data_dir, "**", "*.csv"), recursive=True)}
    def pick(*keys):
        for k in keys:
            for lp, p in cands.items():
                if k in os.path.basename(lp):
                    return p
        return None
    return pick("train"), pick("test")


def _target(train, test, data_dir):
    tc = os.path.join(data_dir, "target_col.txt")
    if os.path.exists(tc):
        n = open(tc).read().splitlines()[0].strip()
        if n in train.columns:
            return n
    only = [c for c in train.columns if c not in test.columns]
    for cand in ("target", "label", "y", "class", "outcome"):
        for c in only:
            if cand in c.lower():
                return c
    return only[-1] if only else train.columns[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="."); ap.add_argument("--train"); ap.add_argument("--test")
    a = ap.parse_args()
    tp, te = a.train, a.test
    if not (tp and te):
        ftr, fte = _find(a.data_dir); tp = tp or ftr; te = te or fte
    train = pd.read_csv(tp); test = pd.read_csv(te)
    target = _target(train, test, a.data_dir)
    print("TARGET %s" % target)
    feats = [c for c in train.columns if c in test.columns and c != target]

    num_rows = []
    for c in feats:
        if pd.api.types.is_numeric_dtype(train[c]) and pd.api.types.is_numeric_dtype(test[c]):
            mtr = float(train[c].mean()); mte = float(test[c].mean())
            s = float(np.sqrt((train[c].var(ddof=0) + test[c].var(ddof=0)) / 2.0))
            smd = abs(mtr - mte) / s if s > 0 else 0.0
            num_rows.append((smd, c, mtr, mte))
    for smd, c, mtr, mte in sorted(num_rows, key=lambda t: -t[0]):
        print("NUM_DRIFT %s smd=%.6f mean_tr=%.6f mean_te=%.6f" % (c, smd, mtr, mte))

    cat_rows = []
    for c in feats:
        if not (pd.api.types.is_numeric_dtype(train[c]) and pd.api.types.is_numeric_dtype(test[c])):
            seen = set(train[c].astype(str).unique()); tec = test[c].astype(str)
            rate = float((~tec.isin(seen)).mean()) if len(tec) else 0.0
            cat_rows.append((rate, c))
    for rate, c in sorted(cat_rows, key=lambda t: -t[0]):
        print("CAT_UNSEEN %s unseen_rate=%.6f" % (c, rate))

    leaks = []
    if target in train.columns:
        yt = train[target]
        for c in feats:
            try:
                if train[c].equals(yt):
                    leaks.append("LEAK_TARGET_IDENTICAL %s" % c)
                elif pd.api.types.is_numeric_dtype(train[c]) and pd.api.types.is_numeric_dtype(yt):
                    r = abs(float(np.corrcoef(train[c].fillna(0).astype(float), yt.astype(float))[0, 1]))
                    if np.isfinite(r) and r > 0.999:
                        leaks.append("LEAK_TARGET_NEAR %s" % c)
            except Exception:
                continue
    seen_cols = {}
    for c in feats:
        key = tuple(pd.util.hash_pandas_object(train[c], index=False).values.tolist())
        if key in seen_cols:
            leaks.append("DUP_COLUMN %s==%s" % (c, seen_cols[key]))
        else:
            seen_cols[key] = c
    for l in leaks:
        print(l)
    print("N_LEAKS %d" % len(leaks))
    print("SHIFTPROF_OK")


if __name__ == "__main__":
    main()
'''

SKILL_MD = """---
name: tabular-autopilot
description: >-
  Run a deterministic, leakage-safe AutoML pipeline on a tabular binary-classification task to produce a
  guaranteed-valid, competitive AUC submission. Use this FIRST to secure a floor, then iterate.
---

# tabular-autopilot

The winning lever for this competition is a robust deterministic floor, not prompt wording.

## scripts/run_pipeline.py
`run_skill_script(skill_name="tabular-autopilot", script_name="run_pipeline.py", args="--data-dir . --out submission.csv")`
Discovers train/test/sample by content, infers the target (target_col.txt wins) and id, builds leakage-safe
features (median-impute + missing-indicator, ordinal + frequency encoding for categoricals), trains a diverse
zoo (HistGBM / ExtraTrees / LogReg / LightGBM / XGBoost when available), OOF-rank-blends with a margin gate,
falls back to a prior, and writes an ID-aligned probability submission clipped to [0,1].

## scripts/check_submission.py
`run_skill_script(skill_name="tabular-autopilot", script_name="check_submission.py", args="--sub submission.csv --sample sample_submission.csv")`
Hard gate — run before EVERY submit_predictions. Fails on schema mismatch, non-finite/out-of-range preds, or
hard-label-looking output.

## scripts/candidate_similarity.py
`run_skill_script(skill_name="tabular-autopilot", script_name="candidate_similarity.py", args="--candidates sub_a.csv sub_b.csv sub_c.csv")`
Deterministic complementarity profiler for hedged final selection. Validates each candidate, prints per-vector
stats (mean/std/min/max) and pairwise Pearson + Spearman correlation + mean-absolute-difference (MAD), then the
MOST_COMPLEMENTARY (least-correlated) valid pair. Use to pick TWO diverse finalists for `select_submission`.

## scripts/shift_profile.py
`run_skill_script(skill_name="tabular-autopilot", script_name="shift_profile.py", args="--data-dir .")`
Deterministic train/test shift + leakage-conflict profiler. Reports per-numeric-feature standardized mean
difference (drift), per-categorical unseen-category rate, and a leakage check (features identical/near-identical
to the target, duplicate columns). Consult BEFORE trusting the public subset or shipping a feature change.
"""


def build_skill(out_dir, skill_name="tabular-autopilot"):
    if not out_dir:
        raise ValueError("build_skill: out_dir is required")
    d = Path(out_dir) / "skills" / (skill_name or "tabular-autopilot")
    (d / "scripts").mkdir(parents=True, exist_ok=True)
    (d / "SKILL.md").write_text(SKILL_MD)
    (d / "scripts" / "run_pipeline.py").write_text(RUN_PIPELINE)
    (d / "scripts" / "check_submission.py").write_text(CHECK_SUBMISSION)
    (d / "scripts" / "candidate_similarity.py").write_text(CANDIDATE_SIM)
    (d / "scripts" / "shift_profile.py").write_text(SHIFT_PROFILE)
    return str(d)


class SkillBuild(BaseAgent):
    name = "skill-build"
    thread = "R"
    kind = "config-gen"

    def run(self, q, worker):
        spec = self.spec(q)
        out = spec.get("out_dir") or "/tmp/aap_submission"
        d = build_skill(out, spec.get("skill_name", "tabular-autopilot"))
        msg = f"skill-build: authored deterministic AutoML floor skill → {d} (run_pipeline.py + check_submission.py + candidate_similarity.py + shift_profile.py + SKILL.md)"
        self.log(msg, kind="config-gen", recommendation="agent-author references this skill; gate changes with agent-config-eval")
        return self.done({"skill_dir": d}, msg)


_AGENT = SkillBuild()


def run(q, worker):
    return _AGENT.run(q, worker)
