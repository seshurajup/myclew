"""lora-validate — the honest generalization test for a trained LoRA adapter.

Loads the pilkwang base + the complete adapter (LoRA weights + detect_head via modules_to_save),
predicts + ILP-links + scores on ALL 199 training datasets, and reports the delta vs the
full-CV baseline. This is the ONLY signal that decides whether a LoRA run is a real, both-embryo
generalizing win (not a golden-12 / frame-cap mirage). Wraps ``research/lora_finetune/full_cv_lora.py``
VERBATIM through argv+env — no scoring or PEFT-load logic is reimplemented here.

Spec: {adapter (path to adapter_best), baseline (default 0.8675), max_datasets (0=all 199),
       out_dir, timeout}.
A BaseAgent subclass with a data-wise test (test_fleet_agents/lora_validate_test.py) that loads a
real adapter and scores a 1-dataset slice, confirming a real score + a parsed delta.
"""
from __future__ import annotations
import os
import re
import subprocess
import sys
from pathlib import Path
from .base import BaseAgent

COMP = Path(__file__).resolve().parent.parent
_REPO = COMP / "research" / "pilkwang_support_pack" / "repo"
_SCRIPT = COMP / "research" / "lora_finetune" / "full_cv_lora.py"
_DEFAULT_ADAPTER = COMP / "research" / "lora_finetune" / "runs" / "det_v3" / "adapter_best"


def _int(v, default=0):
    """Best-effort int coercion (never raises on junk spec values)."""
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


def _env(spec):
    """`device`: optional device hint; 'cpu' forces a CUDA-free full-CV run (CPU fallback)."""
    env = dict(os.environ)
    pp = os.pathsep.join([str(_REPO / "src"), str(_REPO / "scripts"), str(COMP)])
    env["PYTHONPATH"] = pp + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    if _int(spec.get("max_datasets", 0)):
        env["CELLMOT_MAX_DATASETS"] = str(_int(spec["max_datasets"]))
    if spec.get("out_dir"):
        env["CELLMOT_FULLCV_OUT"] = str(spec["out_dir"])
    if spec.get("baseline") is not None:
        env["CELLMOT_BASELINE"] = str(spec["baseline"])
    if str(spec.get("device", "") or "").lower() in ("cpu", "-1"):
        env["CUDA_VISIBLE_DEVICES"] = ""
    return env


def _py():
    v = COMP / "research" / "cellmot_venv" / "bin" / "python"
    return str(v) if v.exists() else sys.executable


def _parse(out):
    sc = re.search(r"LoRA FULL-CV:\s*score=(\S+)\s*edge=(\S+)\s*div=(\S+)"
                   r"\s*44b6=(\S+)\s*6bba=(\S+)\s*\(n=(\d+)\)", out)
    dl = re.search(r"vs BASE full-CV ([0-9.]+)\s*→\s*DELTA\s*([+\-][0-9.]+)", out)

    def _f(x):
        try:
            return float(x)
        except Exception:  # noqa: BLE001
            return None
    res = {}
    if sc:
        res.update(score=_f(sc.group(1)), edge=_f(sc.group(2)), div=_f(sc.group(3)),
                   cv_44b6=_f(sc.group(4)), cv_6bba=_f(sc.group(5)), n=int(sc.group(6)))
    if dl:
        res.update(baseline=float(dl.group(1)), delta=float(dl.group(2)),
                   improved=float(dl.group(2)) > 0)
    return res or None


class LoraValidate(BaseAgent):
    name = "lora-validate"
    thread = "S"

    def run(self, q, worker):
        spec = self.spec(q)
        if not _SCRIPT.exists():
            return self.escalate(worker, "researcher", f"[{worker}] lora-validate: script missing at {_SCRIPT}")
        adapter = Path(spec.get("adapter") or _DEFAULT_ADAPTER)
        if not adapter.exists():
            return self.escalate(worker, "researcher", f"[{worker}] lora-validate: adapter not found at {adapter}")
        timeout = _int(spec.get("timeout", 60 * 60 * 11), 60 * 60 * 11)
        argv = [_py(), "-u", str(_SCRIPT), str(adapter)]
        try:
            r = subprocess.run(argv, capture_output=True, text=True, cwd=str(COMP),
                               env=_env(spec), timeout=timeout)
        except subprocess.TimeoutExpired:
            return self.escalate(worker, "researcher", f"[{worker}] lora-validate: timed out after {timeout}s.")
        except Exception as e:  # noqa: BLE001 — launch failure (bad python/OSError) → clean escalate
            return self.escalate(worker, "researcher", f"[{worker}] lora-validate: could not launch — {type(e).__name__}: {str(e)[:120]}")
        out = (r.stdout or "") + "\n" + (r.stderr or "")
        tail = out.strip().splitlines()[-25:]
        if r.returncode != 0:
            self.log(summary=f"lora-validate FAILED (rc={r.returncode})", detail="\n".join(tail),
                     kind="finding", recommendation="check adapter path + PYTHONPATH + weights")
            return self.escalate(worker, "researcher",
                                 f"[{worker}] lora-validate FAILED rc={r.returncode}: {tail[-1] if tail else ''}")
        p = _parse(out)
        if not p or p.get("score") is None:
            return self.escalate(worker, "researcher",
                                 f"[{worker}] lora-validate ran OK but no score parsed. tail={tail[-1] if tail else ''}")
        improved = bool(p.get("improved"))
        result = {"status": "ok", "returncode": 0, "cv": p["score"], "score": p["score"],
                  "edge": p.get("edge"), "div": p.get("div"), "baseline": p.get("baseline"),
                  "delta": p.get("delta"), "improved": improved, "cv_44b6": p.get("cv_44b6"),
                  "cv_6bba": p.get("cv_6bba"), "n_datasets": p.get("n"), "adapter": str(adapter),
                  "argv": " ".join(argv)}
        verdict = "✅ REAL WIN" if improved else "❌ did not generalize"
        self.save_state({"adapter": str(adapter), "cv": p["score"], "delta": p.get("delta"),
                         "improved": improved})
        self.log(summary=f"lora-validate: {adapter.name} full-CV={p['score']:.4f} "
                         f"Δ={p.get('delta'):+.4f} {verdict}",
                 detail=str(p), kind="verdict",
                 recommendation="adopt adapter ONLY if improved on full-199 (both embryos)")
        # honest ledger: record the measured result whether it won OR lost — negative results count
        if p.get("n", 0) >= 150:
            self.record(change=f"LoRA validate: {adapter.name}", cv=p["score"], script="lora-validate",
                        train_set="full199", kept=improved,
                        observation=f"Δ{p.get('delta'):+.4f} vs {p.get('baseline')} · "
                                    f"44b6={p.get('cv_44b6')} 6bba={p.get('cv_6bba')} · {verdict}")
        msg = (f"[{worker}] **LORA-VALIDATE** · `{adapter.name}` on {p.get('n')} datasets\n"
               f"full-CV **{p['score']:.4f}** vs base {p.get('baseline')} → **Δ{p.get('delta'):+.4f}** {verdict} "
               f"(44b6 {p.get('cv_44b6')} / 6bba {p.get('cv_6bba')})")
        self.post(worker, "leader", msg, routine=False, kind="verdict")
        return self.done(result, msg, to="leader")


_AGENT = LoraValidate()


def run(q, worker):
    return _AGENT.run(q, worker)
