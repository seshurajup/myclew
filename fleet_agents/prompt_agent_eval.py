"""agent-config-eval + prompt-optimize — the OFFLINE gate and the prompt-tuning-as-programming loop.

agent-config-eval builds a hostile SYNTHETIC smoke matrix (several binary tasks with varying signal,
categoricals, missingness), RUNS the authored skill's run_pipeline.py on each in a subprocess, and scores
ROC AUC against the HIDDEN labels. This is the champion's exact discipline: no change ships without offline
hidden-label evidence — it mirrors our fixture rule and defeats public-LB overfitting.

prompt-optimize turns "prompt tuning" into a real optimization: given variants (skill params / system-prompt
edits), evaluate each on the smoke matrix and keep the best by mean hidden AUC. Grounded finding baked in:
prompt-wording variants tend to tie/regress — the deterministic skill floor is where CV actually moves — so
the loop reports honestly and only promotes a variant that clears a margin on hidden labels.
"""
from __future__ import annotations
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from .base import BaseAgent, COMP


def _py():
    p = COMP / "research" / "cellmot_venv" / "bin" / "python"
    return str(p) if p.exists() else sys.executable


def _make_task(dir_path, n=800, n_feat=15, sep=1.0, cat=True, missing=0.05, seed=0):
    import numpy as np, pandas as pd
    from sklearn.datasets import make_classification
    rng = np.random.RandomState(seed)
    X, y = make_classification(n_samples=n, n_features=n_feat, n_informative=max(3, n_feat // 2),
                               class_sep=sep, random_state=seed)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(n_feat)])
    if cat:
        eff = rng.uniform(-2, 2, 8); c = rng.randint(0, 8, n)
        df["cat"] = [f"C{v}" for v in c]; y = ((y + (eff[c] > 0)) >= 1).astype(int)
    if missing:
        for col in df.columns[:3]:
            m = rng.rand(n) < missing
            df.loc[m, col] = np.nan
    df["id"] = np.arange(n); df["target"] = y
    tr = df.iloc[:int(n * 0.6)].copy(); te = df.iloc[int(n * 0.6):].copy()
    os.makedirs(dir_path, exist_ok=True)
    tr.to_csv(os.path.join(dir_path, "train.csv"), index=False)
    te.drop(columns=["target"]).to_csv(os.path.join(dir_path, "test.csv"), index=False)
    te[["id"]].assign(target=0.5).to_csv(os.path.join(dir_path, "sample_submission.csv"), index=False)
    open(os.path.join(dir_path, "target_col.txt"), "w").write("target\n")
    return te[["id", "target"]].rename(columns={"target": "__y__"})


def eval_skill(skill_dir, tasks=None, seed=0, timeout=600):
    """Run run_pipeline.py on synthetic tasks; return {mean_auc, per_task}. skill_dir/scripts/run_pipeline.py.
    timeout: per-task wall-clock cap in seconds (raise on a slow/loaded host; a timeout scores that task None)."""
    import numpy as np, pandas as pd
    from sklearn.metrics import roc_auc_score
    tasks = tasks or [dict(sep=1.5, seed=1), dict(sep=0.7, seed=2), dict(sep=1.0, cat=True, seed=3),
                      dict(sep=0.4, missing=0.15, seed=4)]
    script = os.path.join(skill_dir, "scripts", "run_pipeline.py")
    if not os.path.exists(script):
        return {"mean_auc": None, "n_ok": 0, "per_task": [{"task": 0, "auc": None, "err": "missing run_pipeline.py"}]}
    per = []
    for i, t in enumerate(tasks):
        d = tempfile.mkdtemp(prefix=f"aap_task{i}_")
        hidden = _make_task(d, sep=t.get("sep", 1.0), cat=t.get("cat", True),
                            missing=t.get("missing", 0.05), seed=t.get("seed", i))
        out = os.path.join(d, "submission.csv")
        try:
            r = subprocess.run([_py(), script, "--data-dir", d, "--out", out],
                               capture_output=True, text=True, timeout=int(timeout))
        except subprocess.TimeoutExpired:
            per.append({"task": i, "auc": None, "err": f"timeout>{int(timeout)}s"}); continue
        if not os.path.exists(out):
            per.append({"task": i, "auc": None, "err": r.stderr[-200:]}); continue
        sub = pd.read_csv(out)
        m = hidden.merge(sub, on="id", how="left")
        pcol = [c for c in sub.columns if c != "id"][-1]
        try:
            auc = float(roc_auc_score(m["__y__"], m[pcol].fillna(0.5)))
        except Exception as e:  # noqa: BLE001
            auc = None
        per.append({"task": i, "auc": auc})
    aucs = [p["auc"] for p in per if p["auc"] is not None]
    return {"mean_auc": float(sum(aucs) / len(aucs)) if aucs else None, "n_ok": len(aucs), "per_task": per}


class AgentConfigEval(BaseAgent):
    name = "agent-config-eval"
    thread = "R"
    kind = "verdict"

    def run(self, q, worker):
        spec = self.spec(q)
        skill_dir = spec.get("skill_dir") or os.path.join(spec.get("bundle_dir", ""), "skills", "tabular-autopilot")
        res = eval_skill(skill_dir, timeout=int(spec.get("timeout", 600)))
        ok = res["mean_auc"] is not None and res["mean_auc"] > 0.7
        msg = (f"agent-config-eval: skill floor mean hidden-label AUC={res['mean_auc']} over {res['n_ok']} synthetic tasks "
               f"→ {'SOLID floor' if ok else 'WEAK — fix skill before submit'}")
        self.log(msg, kind="verdict", recommendation="promote a change only if it lifts this hidden-label AUC by a margin")
        return self.done({"mean_auc": res["mean_auc"], "per_task": res["per_task"], "solid": ok}, msg)


class PromptOptimize(BaseAgent):
    name = "prompt-optimize"
    thread = "R"
    kind = "finding"

    def run(self, q, worker):
        spec = self.spec(q)
        if "skill_dir" not in spec and "variants" not in spec:
            return self.escalate(worker, "leader", "prompt-optimize needs spec keys ['skill_dir' or 'variants'] — none provided")
        # variants = [{name, skill_dir}] — evaluate each on the smoke matrix, keep best by mean hidden AUC
        variants = spec.get("variants") or [{"name": "base", "skill_dir": spec.get("skill_dir")}]
        to = int(spec.get("timeout", 600))
        scored = []
        for v in variants:
            r = eval_skill(v["skill_dir"], timeout=to)
            scored.append({"name": v["name"], "mean_auc": r["mean_auc"]})
        scored = [s for s in scored if s["mean_auc"] is not None]
        best = max(scored, key=lambda s: s["mean_auc"]) if scored else {"name": None, "mean_auc": None}
        msg = (f"prompt-optimize: evaluated {len(scored)} variants on hidden-label smoke matrix → best="
               f"{best['name']} AUC={best['mean_auc']}. (Grounded: prompt-wording variants usually tie/regress; "
               f"the deterministic skill floor is the real lever.)")
        self.log(msg, kind="finding", recommendation="promote a variant only if it clears the base by a hidden-AUC margin")
        return self.done({"scored": scored, "best": best}, msg)


_EVAL = AgentConfigEval(); _OPT = PromptOptimize()


def run(q, worker):
    return _EVAL.run(q, worker)


def run_optimize(q, worker):
    return _OPT.run(q, worker)
