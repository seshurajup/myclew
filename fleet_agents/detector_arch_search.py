"""detector-arch-search — build the from-scratch UNet3D DETECTOR architecture FROM BASICS, one axis at a time,
every choice decided PURELY on the competition 2-CV (embryo-disjoint node_recall on 44b6 + 6bba).

The discipline (user, biohub_external_detector_program): start from the SIMPLEST value on every CNN axis
(padding → stem_kernel → kernel → channels → blocks → norm → activation → dilated), change ONE axis at a time,
and KEEP a change only if it TRAINS to a better 2-CV. Each candidate config is a deep-merge of the axis override
into the base YAML; training reuses the verified `model_scratch/train_v0.py` (--config, --hold-embryo = LOEO
embryo-disjoint eval that mirrors the hidden Kaggle test) on a SHARED peak-cache (data fixed, only arch varies).
Score = mean node_recall over the two held-out competition embryos. Every step is RECORDED to a NEW journal
section (train_set="detector_arch"). A BaseAgent subclass with its own data-wise test.
"""
from __future__ import annotations
import json
import re
import subprocess
from pathlib import Path
from .base import BaseAgent

COMP = Path(__file__).resolve().parent.parent
PY = str(COMP / "research" / "cellmot_venv" / "bin" / "python")     # env with torch+MONAI for train_v0
TRAIN = str(COMP / "model_scratch" / "train_v0.py")
BASE_CFG = str(COMP / "model_scratch" / "config" / "exp_det_v0.yml")

# FROM-BASICS search axes — simplest value FIRST (that is the baseline). Formats match the config schema
# (norm = {type,num_groups} dict; dilated = {enabled,dilation} dict; activation ∈ gelu/relu/silu; channels = 4-stage
# to match the 3 downsample strides). Each maps to a model.backbone.* key.
AXES = [
    ("padding_mode", ["zeros", "reflect", "replicate"]),
    ("stem_kernel", [[3, 3, 3], [5, 5, 5], [7, 7, 7]]),
    ("kernel", [[3, 3, 3], [5, 5, 5]]),
    ("channels_per_stage", [[16, 32, 64, 128], [24, 48, 96, 192], [32, 64, 128, 256]]),
    ("blocks_per_stage", [1, 2, 3]),
    ("norm", [{"type": "group", "num_groups": 8}, {"type": "instance"}, {"type": "batch"}]),
    ("activation", ["gelu", "relu", "silu"]),
    ("dilated_bottleneck", [{"enabled": False}, {"enabled": True, "dilation": 2}]),
]


def _deep_set(d, dotted, val):
    keys = dotted.split("."); cur = d
    for k in keys[:-1]:
        cur = cur.setdefault(k, {})
    cur[keys[-1]] = val


class DetectorArchSearch(BaseAgent):
    name = "detector-arch-search"
    thread = "A"
    kind = "verdict"

    def _write_cfg(self, base, overrides, path, yaml):
        cfg = json.loads(json.dumps(base))                      # deep copy
        for k, v in overrides.items():
            _deep_set(cfg, k if "." in k else f"model.backbone.{k}", v)   # dotted key = full path (e.g. loss.detection.type)
        Path(path).write_text(yaml.safe_dump(cfg))
        return path

    def _train_one(self, cfg_path, hold, spec, init_from=None, outdir=None):
        """Run train_v0 on one held-out embryo; parse best node_recall + the saved best.pt path. WARM-STARTS from
        `init_from` (previous best of the SAME embryo) so shared layers reuse proven weights. Returns (recall, ckpt)."""
        cache = spec.get("cache_dir", str(COMP / "model_scratch" / "results" / "cache_arch"))
        cmd = [PY, TRAIN, "--config", cfg_path, "--hold-embryo", hold, "--peak-cache", "--cache-dir", cache,
               "--epochs", str(int(spec.get("epochs", 10))),
               "--subset-per-embryo", str(int(spec.get("subset_per_embryo", 12))),
               "--eval-ds", str(int(spec.get("eval_ds", 4)))]
        if outdir:
            cmd += ["--out", outdir]
        if init_from and Path(init_from).exists():             # warm-start (unless the axis changed shape → auto-skipped)
            cmd += ["--init-from", init_from]
        try:
            out = subprocess.run(cmd, cwd=str(COMP), capture_output=True, text=True,
                                 timeout=int(spec.get("timeout", 1800))).stdout
        except Exception:  # noqa: BLE001
            return None, None
        m = re.findall(r"best node_recall=([0-9.]+)", out)
        ckpt = str(Path(outdir) / "best.pt") if outdir else None
        return (float(m[-1]) if m else None), ckpt

    def _score_2cv(self, cfg_path, spec, tag, init_ckpts=None):
        """Competition 2-CV = mean embryo-disjoint node_recall over 44b6 and 6bba; warm-start each embryo from its
        own previous-best checkpoint. Returns (mean_cv, per_embryo_recall, per_embryo_ckpt)."""
        init_ckpts = init_ckpts or {}
        recs, ckpts = {}, {}
        scratch = Path(spec.get("scratch", "/tmp/det_arch"))
        for emb in spec.get("embryos", ["44b6", "6bba"]):
            od = str(scratch / f"{tag}_{emb}")
            r, ck = self._train_one(cfg_path, emb, spec, init_from=init_ckpts.get(emb), outdir=od)
            if r is None:
                return None, recs, ckpts
            recs[emb] = round(r, 4); ckpts[emb] = ck
        return round(sum(recs.values()) / len(recs), 4), recs, ckpts

    def run(self, q, worker):
        import yaml
        spec = self.spec(q)
        bcfg = Path(spec.get("base_cfg", BASE_CFG))
        if not bcfg.is_absolute():                                 # fleet worker cwd = tools/researchpapers → resolve vs comp root
            bcfg = COMP / bcfg
        try:
            base = yaml.safe_load(bcfg.read_text())
        except Exception as e:  # noqa: BLE001
            return self.escalate(worker, "researcher", f"[{worker}] detector-arch-search: cannot read base config {bcfg}: {str(e)[:80]}")
        if not isinstance(base, dict):
            return self.escalate(worker, "researcher", f"[{worker}] detector-arch-search: base config {bcfg} is not a mapping.")
        for k, v in (spec.get("base_overrides") or {}).items():    # bake ALREADY-DECIDED axes into the base (continuation)
            _deep_set(base, k if "." in k else f"model.backbone.{k}", v)
        scratch = Path(spec.get("scratch", "/tmp/det_arch")); scratch.mkdir(parents=True, exist_ok=True)
        axes = [(k, v) for k, v in AXES if k in (spec.get("axes") or [a[0] for a in AXES])]
        axes += [tuple(a) for a in (spec.get("extra_axes") or [])]  # extra config-path axes, e.g. ["loss.detection.type", [...]]
        jset = spec.get("journal_set", "detector_arch")

        # 1) BASIC baseline = simplest value on every axis (first in each list). No prior weights → random init.
        best = {k: v[0] for k, v in axes}
        cfg = self._write_cfg(base, best, str(scratch / "baseline.yml"), yaml)
        base_cv, base_recs, best_ckpts = self._score_2cv(cfg, spec, "baseline", init_ckpts=spec.get("baseline_init"))
        trail = [{"step": "baseline", "cfg": dict(best), "cv": base_cv, "per_embryo": base_recs}]
        self._rec("baseline", best, base_cv, base_recs, jset)
        if base_cv is None:
            return self.done({"error": "baseline train failed"}, f"[{worker}] detector-arch-search: baseline train failed (check cache/env).")
        best_cv = base_cv

        # 2) grow ONE axis at a time; WARM-START each candidate from the current best's per-embryo checkpoints
        #    (shared layers reuse proven weights; the changed-axis layers auto-reinit). Keep a value only if 2-CV improves.
        for k, vals in axes:
            for v in vals[1:]:
                cand = dict(best); cand[k] = v
                tag = re.sub(r"[^A-Za-z0-9]+", "", f"{k}_{v}")[:40]   # filesystem-safe (dicts/lists → alnum)
                cfg = self._write_cfg(base, cand, str(scratch / f"{tag}.yml"), yaml)
                cv, recs, ckpts = self._score_2cv(cfg, spec, tag, init_ckpts=best_ckpts)
                trail.append({"step": f"{k}={v}", "cfg": dict(cand), "cv": cv, "per_embryo": recs, "warm_started": True})
                self._rec(f"{k}={v}", cand, cv, recs, jset)
                if cv is not None and cv > best_cv:
                    best, best_cv, best_ckpts = cand, cv, ckpts  # keep the improvement AND its weights for the next warm-start
        self.save_state({"best_cfg": best, "best_cv": best_cv, "trail": trail})
        line = " → ".join(f"{t['step']}:{t['cv']}" for t in trail)
        # GROUNDED modern-technique proposal for THIS detector's deployment target (default = Kaggle T4-offline):
        # the CNN axes above tune the backbone; the catalog tells you how to SHIP it (int8 not FP4, graft, gate).
        prop_note = ""
        try:
            from . import arch_builder
            tgt = spec.get("target_profile") or {"hardware": "t4", "data_regime": "sparse_label", "bit_budget": 8}
            prop = arch_builder.propose(tgt)
            names = ", ".join(p["name"] for p in prop["recommended"][:5])
            exc = "; ".join(f"{e['name']} ({e['reason']})" for e in prop["excluded"])
            prop_note = (f"\n🧩 modern-technique catalog ({prop['target']['_label']}): {names}"
                         + (f" · ⛔ {exc}" if exc else "") + f" · GATE {prop['gate']['name']}")
        except Exception:  # noqa: BLE001
            pass
        msg = (f"[{worker}] **DETECTOR-ARCH-SEARCH** · from-basics CNN, decided on competition 2-CV (node_recall)\n"
               f"baseline {base_cv} → best **{best_cv}**\n{line}\nbest arch: {best}\n"
               f"→ each choice kept only if it TRAINED to a better embryo-disjoint 2-CV (journal `{jset}`)." + prop_note)
        self.post(worker, "leader", msg, routine=False, kind="verdict")
        return self.done({"best_cfg": best, "best_cv": best_cv, "trail": trail}, msg, to="leader")

    def _rec(self, step, cfg, cv, recs, jset):
        # LEAD the description with BOTH CVs (our CV is the embryo-disjoint 2-CV) so the journal shows both,
        # not just the mean — cv column = mean, description opens with 44b6 + 6bba node_recall.
        two = " ".join(f"{e}={recs.get(e)}" for e in ("44b6", "6bba") if recs)
        try:
            self.record(change=f"det_arch_{step}".replace("=", "_").replace(" ", "")[:60],
                        script="fleet_dispatch detector-arch-search", cv=cv, train_set=jset,
                        description=f"2-CV[{two}] mean={cv} — detector arch {step}: {cfg}")
        except Exception:  # noqa: BLE001
            pass


_AGENT = DetectorArchSearch()


def run(q, worker):
    return _AGENT.run(q, worker)
