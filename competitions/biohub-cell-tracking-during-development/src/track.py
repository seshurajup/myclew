"""Lightweight MLflow logging for CV experiments (and later, training runs).

All runs log to one experiment (kaggle-biohub-cell-tracking-during-development) so classical
config CVs and future model-training runs share one comparable table. Logging is best-effort:
if the MLflow server is down, it warns and continues (CV still computed).
"""
from __future__ import annotations

EXPERIMENT = "kaggle-biohub-cell-tracking-during-development"
URI = "http://localhost:5000"


def log_cv(run_name: str, params: dict, metrics: dict, tags: dict | None = None,
           kind: str = "classical_config", known_lb: float | None = None,
           kaggle_lb: float | None = None) -> bool:
    """Log one entry to the per-competition table. `kind` in {classical_config, public_repro,
    our_submission, trained_model}. known_lb = a public notebook's LB; kaggle_lb = our actual LB
    after submitting. So the same experiment accumulates the CV<->LB table over time."""
    try:
        import mlflow
        mlflow.set_tracking_uri(URI)
        mlflow.set_experiment(EXPERIMENT)
        with mlflow.start_run(run_name=run_name):
            mlflow.set_tags({"phase": "cv", "kind": kind, **(tags or {})})
            mlflow.log_params({k: (list(v) if isinstance(v, tuple) else v) for k, v in params.items()})
            m = dict(metrics)
            if known_lb is not None:
                m["known_lb"] = known_lb
            if kaggle_lb is not None:
                m["kaggle_lb"] = kaggle_lb
            mlflow.log_metrics({k: float(v) for k, v in m.items() if v is not None})
        return True
    except Exception as e:
        print(f"[mlflow] skip ({type(e).__name__}: {str(e)[:60]})")
        return False


# ---- TRAINING runs: ONE mlflow run, per-epoch log_metrics(step=ep) (extends log_cv for model training) ----
_TRAIN_ACTIVE = False


def start_training_run(run_name: str, params: dict, tags: dict | None = None) -> bool:
    """Open ONE persistent mlflow run for a training job. Best-effort (warns + continues if server down)."""
    global _TRAIN_ACTIVE
    try:
        import mlflow
        mlflow.set_tracking_uri(URI)
        mlflow.set_experiment(EXPERIMENT)
        mlflow.start_run(run_name=run_name)
        mlflow.set_tags({"phase": "train", "kind": "trained_model", **(tags or {})})
        mlflow.log_params({k: (list(v) if isinstance(v, tuple) else v) for k, v in params.items()})
        _TRAIN_ACTIVE = True
        print(f"[mlflow] training run '{run_name}' → {URI} / {EXPERIMENT}")
        return True
    except Exception as e:  # noqa: BLE001
        print(f"[mlflow] train-run skip ({type(e).__name__}: {str(e)[:60]})")
        return False


def log_step(metrics: dict, step: int) -> None:
    """Log per-epoch metrics at `step`. Skips None/NaN. No-op if the run failed to start."""
    if not _TRAIN_ACTIVE:
        return
    try:
        import mlflow
        clean = {k: float(v) for k, v in metrics.items() if v is not None and float(v) == float(v)}
        mlflow.log_metrics(clean, step=step)
    except Exception:  # noqa: BLE001
        pass


def end_training_run(final: dict | None = None) -> None:
    """Log final summary metrics and close the training run."""
    global _TRAIN_ACTIVE
    if not _TRAIN_ACTIVE:
        return
    try:
        import mlflow
        if final:
            mlflow.log_metrics({k: float(v) for k, v in final.items() if v is not None and float(v) == float(v)})
        mlflow.end_run()
    except Exception:  # noqa: BLE001
        pass
    _TRAIN_ACTIVE = False
