"""lora-train — LoRA/PEFT warm-started fine-tune of pilkwang's UNetNodeTransformer (detector + graph-
transformer edge/division predictor) on external Zebrahub + competition data, to lift BOTH edge_jaccard
and division_jaccard WITHOUT regressing the strong base (pilkwang raw: edge 0.8527, div 0.000).

Wraps research/lora_finetune/train_lora.py (the engine, which reuses pilkwang's own model/data/losses):
  recipe = LoRA + rsLoRA + LoRA+ (B-LR 8×), r16 α32 drop0.08 fp32 lr2e-4 cosine; adapt the transformer
  nn.Linear layers + detect_head, FREEZE the Conv3d U-Net encoder + BatchNorm; division lever = up-weight
  division rows (--div-weight); anti-overfit = early-stop on the REAL competition metric (edge+0.1·div)
  on a held-out embryo; ZSNS003 dim-collapse fix = --strong-intensity-aug + ZSNS003 in train.

Spec-driven: {r, alpha, dropout, lora_plus_ratio, div_weight, lr, max_epochs, eval_every, patience,
eval_frames, batch_size, splits, use_dora, target_unet_encoder, strong_intensity_aug, eval_ilp, seed,
dry_run, timeout}. A BaseAgent subclass with a data-wise self-test (engine --dry-run must be GREEN).
"""
from __future__ import annotations
import os
import re
import subprocess
import sys
from pathlib import Path
from .base import BaseAgent

COMP = Path(__file__).resolve().parent.parent
ENGINE = COMP / "research" / "lora_finetune" / "train_lora.py"
PY = COMP / "research" / "cellmot_venv" / "bin" / "python"

_VAL_FLAGS = {  # spec key -> CLI flag (value-carrying)
    "r": "--r", "alpha": "--alpha", "dropout": "--dropout", "lora_plus_ratio": "--lora-plus-ratio",
    "div_weight": "--div-weight", "lr": "--lr", "max_epochs": "--max-epochs", "eval_every": "--eval-every",
    "patience": "--patience", "eval_frames": "--eval-frames", "batch_size": "--batch-size",
    "splits": "--splits", "data_dir": "--data-dir", "out": "--out", "seed": "--seed", "rslora": "--rslora",
    "num_workers": "--num-workers", "max_iters": "--max-iters", "ckpt": "--ckpt", "bf16": "--bf16",
    "data_source": "--data-source", "max_train_datasets": "--max-train-datasets", "optimizer": "--optimizer",
    "hard_neg": "--hard-neg",
}
_BOOL_FLAGS = {"use_dora": "--use-dora", "target_unet_encoder": "--target-unet-encoder",
               "strong_intensity_aug": "--strong-intensity-aug", "eval_ilp": "--eval-ilp"}


class LoraTrain(BaseAgent):
    name = "lora-train"
    thread = "B"
    kind = "verdict"

    def _cmd(self, spec, dry=False):
        py = str(PY) if PY.exists() else sys.executable
        cmd = [py, str(ENGINE)]
        if dry:
            cmd.append("--dry-run")
        for k, f in _VAL_FLAGS.items():
            if spec.get(k) is not None:
                cmd += [f, str(spec[k])]
        # `epochs`: convenience alias for --max-epochs when max_epochs was not given directly.
        if spec.get("epochs") is not None and spec.get("max_epochs") is None:
            cmd += ["--max-epochs", str(spec["epochs"])]
        for k, f in _BOOL_FLAGS.items():
            if spec.get(k):
                cmd.append(f)
        return cmd

    def _env(self, spec):
        """`device`: optional compute-device hint; 'cpu' forces a CUDA-free run (CPU fallback)."""
        env = dict(os.environ)
        dev = str(spec.get("device", "") or "").lower()
        if dev in ("cpu", "-1"):
            env["CUDA_VISIBLE_DEVICES"] = ""
        return env

    def run(self, q, worker):
        spec = self.spec(q)
        if not ENGINE.exists():
            return self.done({"error": f"engine missing at {ENGINE}"},
                             f"[{worker}] lora-train: engine script missing at {ENGINE} — cannot train.")
        cmd = self._cmd(spec, dry=bool(spec.get("dry_run")))
        try:
            timeout = int(spec.get("timeout", 36000))
        except (TypeError, ValueError):
            timeout = 36000
        try:
            r = subprocess.run(cmd, cwd=str(COMP), capture_output=True, text=True,
                               timeout=timeout, env=self._env(spec))
            out = (r.stdout or "") + "\n" + (r.stderr or "")
        except Exception as e:  # noqa: BLE001
            return self.done({"error": str(e)[:200]}, f"[{worker}] lora-train FAILED: {str(e)[:120]}")

        # SUCCESS GATE (user 2026-07-12): lora-train counts ONLY if it beats the pilkwang baseline.
        # Engine prints:  RESULT: baseline=B  best_lora=L  delta=+D  → SUCCESS/NO-IMPROVEMENT
        best = baseline = delta = None; improved = False
        mr = re.search(r"RESULT:\s*baseline=([0-9.]+)\s+best_lora=([0-9.]+)\s+delta=([+\-][0-9.]+)", out)
        if mr:
            baseline = float(mr.group(1)); best = float(mr.group(2)); delta = float(mr.group(3))
            improved = "SUCCESS" in out and delta > 1e-4
        else:  # fallback to the older marker
            m = re.search(r"best (?:held-out )?CV=([0-9.]+)", out)
            best = float(m.group(1)) if m else None
        rows = re.findall(r"\|\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+(\d+)/(\d+)/(\d+)", out)
        edge = float(rows[-1][1]) if rows else None
        div = float(rows[-1][2]) if rows else None
        r_int = int(spec.get("r", 16)); dw = spec.get("div_weight", 3.0)
        verdict = "IMPROVED — adopt adapter" if improved else "NO IMPROVEMENT — keep pristine base, discard adapter"
        self.save_state({"best_cv": best, "baseline_cv": baseline, "delta": delta, "improved": improved,
                         "edge": edge, "div": div, "spec": spec})
        self.log(summary=f"lora-train r={r_int} div_w={dw}: baseline {baseline} → best {best} (Δ{delta}) — {verdict}",
                 detail="LoRA+rsLoRA+LoRA+ warm-start fine-tune; early-stop on real metric; adapter-only (base pristine)",
                 kind="verdict",
                 recommendation=("ADOPT the adapter — CV improved over pilkwang base" if improved
                                 else "DISCARD — no CV gain over pilkwang base; keep the base"))
        # Only stamp the ledger as a kept result when it actually improved (else it's a negative result, not an adoption)
        if improved and best is not None:
            self.record(change=f"lora-finetune r={r_int} div_w={dw}", cv=best,
                        description="pilkwang warm-start + LoRA on external+comp (IMPROVED over base)", script="train_lora.py")
        emoji = "✅" if improved else "❌"
        msg = (f"[{worker}] **LORA-TRAIN** {emoji} · warm-start + LoRA (r={r_int}, div_weight={dw})\n"
               f"• baseline (pilkwang) CV **{baseline}** → best LoRA CV **{best}**  (Δ **{delta}**)\n"
               f"• edge {edge} / div {div}\n"
               f"→ {verdict}. (adapter-only; base never modified)")
        self.post(worker, "leader", msg, routine=False, kind="verdict")
        return self.done({"best_cv": best, "baseline_cv": baseline, "delta": delta, "improved": improved,
                          "edge": edge, "div": div, "cmd": " ".join(cmd)}, msg, to="leader")


_AGENT = LoraTrain()


def run(q, worker):
    return _AGENT.run(q, worker)
