"""comp-onboard — THE FRONT DOOR. Given any competition slug, fingerprint it into a `CompConfig` and
route it to the pack that handles it (or emit an honest "unknown-comp" report for a cold-start type).

This is the generalization of biohub's `orchestrate`/`journey` to ANY competition. Flow:

  1. PULL (via the Kaggle CLI, reusing the kaggle-scout mechanism): the file manifest, the Evaluation page
     (→ metric), and the Overview page (→ modality/paradigm/task keywords). Reads official pages FIRST
     (`kaggle competitions pages <slug> --content --page-name Evaluation`) per the standing rule.
  2. FINGERPRINT (pure, offline-testable `infer_config`): file extensions + eval/overview keywords →
     data-modality × paradigm × task × metric × cv-scheme × submission-schema.
  3. ROUTE: comp_config.route() → 'tab'/'img'/'vid'/'pc'/'biohub'/'llm'/'agent'/'agent/sec'/'reason', OR
     'unknown' → emit a gap report (closest pack + the ONE capability to build + a proposed CompConfig)
     rather than crash. This is the neurogolf-class cold-start path.

KNOWN_COMPS gives a high-confidence override for the example comps we already understand; everything else
is inferred generically so a brand-new comp still onboards. The fingerprinter is data-wise tested with
synthetic manifests (no network) to prove the cold-start inference, not just the memorized table.
"""
from __future__ import annotations
import os
import subprocess
from pathlib import Path
from .base import BaseAgent, COMP
from . import comp_config as CC

KAGGLE = os.environ.get("KAGGLE_BIN", "/home/seshu/miniconda3/envs/llm/bin/kaggle")

# ------------------------------------------------------------------ high-confidence table for known comps
# (we already understand these — used as an override so onboard is exact on them; generic inference still
#  runs and is what makes an UNKNOWN comp work.)
KNOWN_COMPS: dict = {
    "playground-series-s6e7": dict(modality="tabular", paradigm="predictive", task="classification",
                                   metric="roc_auc", cv_scheme="stratified"),
    "rogii-wellbore-geology-prediction": dict(modality="sequence", paradigm="predictive",
                                              task="regression", metric="rmse",
                                              cv_scheme="grouped-sequence", domain="geology", group_col="well"),
    "biohub-cell-tracking-during-development": dict(modality="volume-time", paradigm="predictive",
                                                    task="tracking", metric="edge_jaccard",
                                                    cv_scheme="leave-one-group-out", group_col="embryo"),
    "arc-prize-2026-arc-agi-3": dict(modality="grid-reasoning", paradigm="reasoning",
                                     task="program-synthesis", metric="exact_match", cv_scheme="none"),
    "ai-agent-security-multi-step-tool-attacks": dict(modality="agent-env", paradigm="agentic",
                                                      task="attack", metric="unknown", cv_scheme="none"),
    "pokemon-tcg-ai-battle": dict(modality="agent-env", paradigm="agentic", task="policy",
                                  metric="unknown", cv_scheme="none"),
    "autonomous-agent-prediction-beta": dict(modality="agent-config", paradigm="prompt-program",
                                             task="classification", metric="roc_auc", cv_scheme="holdout"),
}

# prompt-program signals: the submission is an authored agent bundle (ADK), scored by running the agent.
_AGENTCFG_FILE_KEYS = ("agent.yaml", "skill.md", "skills/", "prompts/system", "subagent", "submission.zip")
_AGENTCFG_TEXT_KEYS = ("agent config", "adk", "system prompt", "custom tools", "custom skills",
                       "agent.yaml", "compiled into", "authored agent", "prompt", "sandbox")

# ------------------------------------------------------------------ keyword tables (generic inference)
_EXT_MODALITY = [
    ((".geff", ".zarr"), "volume-time"),
    ((".nii", ".nii.gz", ".mha", ".ply", ".pcd"), "pointcloud"),
    ((".mp4", ".avi", ".mov", ".webm"), "video"),
    ((".png", ".jpg", ".jpeg", ".tif", ".tiff", ".dcm", ".bmp"), "image"),
    ((".txt", ".jsonl"), "text"),
    ((".parquet", ".csv", ".tsv", ".feather"), "tabular"),
]
_METRIC_KEYWORDS = [
    ("roc_auc", ("area under the roc", "roc curve", "roc-auc", "auc")),
    ("logloss", ("log loss", "logloss", "cross entropy", "cross-entropy")),
    ("quadratic_weighted_kappa", ("quadratic weighted kappa", "cohen", "kappa")),
    ("rmsle", ("root mean squared log", "rmsle")),
    ("rmse", ("root mean squared error", "rmse")),
    ("mae", ("mean absolute error", "mae")),
    ("r2", ("r2", "r-squared", "coefficient of determination")),
    ("f1", ("f1 score", "f1-score", "f-measure")),
    ("mcc", ("matthews", "mcc")),
    ("exact_match", ("exact match", "percentage of correct", "exactly correct")),
    ("edge_jaccard", ("edge jaccard", "lineage", "tracking accuracy")),
    ("accuracy", ("accuracy", "categorization accuracy", "classification accuracy")),
]
_AGENTIC_KEYWORDS = ("environment", "simulator", "agent", "episode", "reward", "opponent", "battle",
                     "self-play", "tool call", "policy", "action space")
_REASONING_KEYWORDS = ("reasoning", "program synthesis", "abstraction", "puzzle", "grid", "arc-agi",
                       "few-shot generalization", "novel task")
_ATTACK_KEYWORDS = ("attack", "exfil", "jailbreak", "adversarial prompt", "security", "red team", "tool-attack")


def _pick_modality(files, text):
    exts = {os.path.splitext(f.lower())[1] for f in files}
    # multi-ext files like .nii.gz
    joined = " ".join(f.lower() for f in files)
    t = text.lower()
    # agent-config FIRST: the submission is an authored agent bundle (ADK), even though the DATA folder holds
    # csv/etc. Key on the submission format (eval text) + agent-bundle filenames, not the data files.
    if any(k in joined for k in _AGENTCFG_FILE_KEYS) or \
       (("agent.yaml" in t or "adk" in t or ("agent config" in t)) and ("skill" in t or "prompt" in t or "tool" in t)):
        return "agent-config"
    for group, mod in _EXT_MODALITY:
        if any(e in exts for e in group) or any(g in joined for g in group):
            return mod
    t = text.lower()
    if any(k in t for k in _REASONING_KEYWORDS):
        return "grid-reasoning"
    if any(k in t for k in _AGENTIC_KEYWORDS):
        return "agent-env"
    return "unknown"


def _pick_metric(text):
    t = text.lower()
    for metric, kws in _METRIC_KEYWORDS:
        if any(k in t for k in kws):
            return metric
    return "unknown"


def _pick_paradigm(text, modality):
    t = text.lower()
    if modality == "agent-config":
        return "prompt-program"
    if modality in ("agent-env",) or any(k in t for k in _AGENTIC_KEYWORDS):
        if any(k in t for k in _REASONING_KEYWORDS):
            return "reasoning"
        return "agentic"
    if modality == "grid-reasoning" or any(k in t for k in _REASONING_KEYWORDS):
        return "reasoning"
    return "predictive"


def _pick_task(text, metric, modality, paradigm):
    t = text.lower()
    if any(k in t for k in _ATTACK_KEYWORDS):
        return "attack"
    if paradigm == "reasoning":
        return "program-synthesis"
    if paradigm == "agentic":
        return "policy"
    if modality == "volume-time":
        return "tracking"
    if "segment" in t or "mask" in t:
        return "segmentation"
    if "detect" in t or "bounding box" in t:
        return "detection"
    if metric in ("rmse", "rmsle", "mae", "r2"):
        return "regression"
    if metric == "quadratic_weighted_kappa":
        return "ordinal"
    if metric in ("roc_auc", "logloss", "accuracy", "f1", "mcc"):
        return "classification"
    return "unknown"


def _submission_schema(sample_header):
    """Parse the sample_submission header (a list of column names) → (id_col, target_cols)."""
    if not sample_header:
        return None, []
    try:
        header = list(sample_header)
    except TypeError:
        return None, []
    if not header:
        return None, []
    id_col = header[0]
    return id_col, list(header[1:])


def _kaggle_cached_modality(slug):
    """The Kaggle-tag-derived modality from the cached map (docs/kaggle_modality_map.json), if it is a real
    modality in our taxonomy. Read-only, never raises (offline). Late import → no import cycle."""
    try:
        from . import kaggle_modality as KM
        m = KM.cached_modality(slug)
        return m if m in CC.MODALITIES and m != "unknown" else ""
    except Exception:  # noqa: BLE001
        return ""


def infer_config(slug, files=None, eval_text="", overview_text="", sample_header=None, hints=None,
                 validate=False):
    """PURE fingerprinter (data-wise tested, no network). Returns a CompConfig.
    files = filenames in the comp; eval_text/overview_text = official page text; sample_header = sample_sub cols.
    validate: coerce any invalid/missing enum field to a safe default before returning (default off)."""
    files = list(files) if files else []
    hints = dict(hints or {})
    text = f"{eval_text or ''}\n{overview_text or ''}"
    base = dict(KNOWN_COMPS.get(slug, {}))          # known override (may be partial/empty)
    # AUGMENT KNOWN_COMPS with Kaggle's OWN tag-derived modality (cached, offline): fills the gap for any
    # active comp we don't hand-table (birdclef→audio, deep-past→text, nemotron→text). KNOWN_COMPS still wins.
    kag_mod = "" if base.get("modality") else _kaggle_cached_modality(slug)
    modality = base.get("modality") or kag_mod or _pick_modality(files, text)
    metric = base.get("metric") or _pick_metric(text)
    paradigm = base.get("paradigm") or _pick_paradigm(text, modality)
    task = base.get("task") or _pick_task(text, metric, modality, paradigm)
    id_col, target_cols = _submission_schema(sample_header)
    cfg = CC.CompConfig(
        slug=slug,
        modality=modality,
        paradigm=paradigm,
        task=task,
        metric=metric,
        metric_direction=CC.metric_spec(metric)["direction"],
        cv_scheme=base.get("cv_scheme") or ("stratified" if task in ("classification", "ordinal") else "kfold"),
        group_col=base.get("group_col") or hints.get("group_col"),
        time_col=base.get("time_col") or hints.get("time_col"),
        id_col=id_col or hints.get("id_col"),
        target_cols=target_cols or hints.get("target_cols", []),
        domain=base.get("domain") or hints.get("domain"),
        n_folds=int(hints.get("n_folds", 5)),
        extra={"known": slug in KNOWN_COMPS, "files_seen": len(files)},
    )
    # allow explicit spec hints to override any inferred field (human-in-the-loop)
    for k in ("modality", "paradigm", "task", "metric", "cv_scheme"):
        if k in hints:
            setattr(cfg, k, hints[k])
    cfg.metric_direction = CC.metric_spec(cfg.metric)["direction"]
    if validate or hints.get("validate"):
        cfg = cfg.validated()
    return cfg


def _gap_report(cfg):
    """For an unknown route: closest pack by modality-only, and the ONE capability to build."""
    by_mod = {m: p for (m, _), p in CC.PACK_ROUTES.items()}
    closest = by_mod.get(cfg.modality, "tab")
    return {
        "verdict": "UNKNOWN-COMP — no pack matches; onboard is reporting the gap (did NOT crash).",
        "closest_pack": closest,
        "build_next": f"a '{cfg.modality}/{cfg.paradigm}' handler (task={cfg.task}, metric={cfg.metric})",
        "proposed_config": cfg.to_dict(),
    }


# ------------------------------------------------------------------ Kaggle CLI pull (best-effort, online)
def _cli(args, timeout=45):
    try:
        r = subprocess.run([KAGGLE, *args], capture_output=True, text=True, timeout=timeout, env={**os.environ})
        return r.stdout if r.returncode == 0 else ""
    except Exception:  # noqa: BLE001
        return ""


def _pull(slug):
    """Best-effort: file manifest + Evaluation/Overview page text + sample-submission header. Empty on offline."""
    files_raw = _cli(["competitions", "files", slug, "--csv"])
    files = [ln.split(",")[0].strip().strip('"') for ln in files_raw.splitlines()[1:] if ln.strip()]
    eval_text = _cli(["competitions", "pages", slug, "--content", "--page-name", "Evaluation"])
    overview = _cli(["competitions", "pages", slug, "--content", "--page-name", "Overview"])
    return files, eval_text, overview


class CompOnboard(BaseAgent):
    name = "comp-onboard"
    thread = "S"
    kind = "config-gen"

    def run(self, q, worker):
        spec = self.spec(q)
        slug = spec.get("slug") or os.environ.get("KAGGLE_COMP_SLUG", "")
        if not slug:
            return self.escalate(worker, "leader", "comp-onboard needs a competition slug (spec.slug).")
        # offline/test path: caller may pass the manifest+text directly (no network)
        files = spec.get("files")
        eval_text = spec.get("eval_text", "")
        overview = spec.get("overview_text", "")
        sample_header = spec.get("sample_header")
        if files is None and not eval_text and not spec.get("offline"):
            files, eval_text, overview = _pull(slug)
        cfg = infer_config(slug, files=files or [], eval_text=eval_text, overview_text=overview,
                           sample_header=sample_header, hints=spec.get("hints"))
        pack = cfg.pack()
        # persist the config for downstream agents
        out = COMP / "config" / "_auto" / f"comp_config_{slug}.json"
        cfg.save(out)
        data = {"config": cfg.to_dict(), "pack": pack, "config_file": str(out)}
        if pack == "unknown":
            report = _gap_report(cfg)
            data["gap_report"] = report
            msg = (f"comp-onboard {slug}: {report['verdict']} closest={report['closest_pack']}; "
                   f"BUILD NEXT: {report['build_next']}. Config saved → {out}")
            self.log(msg, kind="config-gen", recommendation=report["build_next"])
            # escalate to a human, but CARRY the gap report + config in the return (self.escalate drops data)
            self.post(worker, "leader", msg, routine=False, kind="reason")
            return ("escalated", data, "leader", msg)
        msg = (f"comp-onboard {slug}: modality={cfg.modality} paradigm={cfg.paradigm} task={cfg.task} "
               f"metric={cfg.metric}({cfg.metric_direction}) cv={cfg.cv_scheme} → PACK '{pack}'. Config → {out}")
        self.log(msg, kind="config-gen",
                 recommendation=f"route to pack '{pack}'; downstream agents read {out.name}")
        return self.done(data, msg)


_AGENT = CompOnboard()


def run(q, worker):
    return _AGENT.run(q, worker)
