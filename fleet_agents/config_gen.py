"""config-gen — deterministic YAML config author (takes this off the researcher).

Clones a base config and applies ONE change (an isolated aug, an aug MIX, or a param), always with the
correct split + method/run_name wired. So configs are correct-by-construction (one change, right split,
no confounding) — the researcher is not needed to hand-write them.

  make(base="config/aug_ablation/00_no_aug.yml", name="auto_contrast",
       augment=[{"name":"contrast","p":0.5,"range":0.2}], split="splits_screen_matched.json") -> path
"""
from __future__ import annotations

from pathlib import Path

COMP = Path(__file__).resolve().parent.parent
AUTO = COMP / "config" / "_auto"
FT = "learning/ensemble_work/finetune"


def _load(base: str) -> dict:
    import yaml
    return yaml.safe_load((COMP / base).read_text()) or {}


def make(base: str, name: str, augment=None, params=None, split=None, purpose="", num_workers=0) -> str:
    """Write config/_auto/<name>.yml = base + one change. Returns the repo-relative path.
    num_workers: DataLoader worker count (default 0 = hang-safe; raise only if you accept the deadlock risk)."""
    import yaml
    cfg = _load(base)
    cfg["name"] = name
    cfg.setdefault("train", {})["method"] = name
    # HANG-SAFE: force num_workers=0 in auto-generated configs — num_workers>0 deadlocks the DataLoader
    # (blur/noise augs hung 100+ min at iter 0). 0 = no worker processes = no deadlock (fine on the mini screen).
    cfg["train"]["num_workers"] = max(0, int(num_workers))
    cfg.setdefault("mlflow", {})["run_name"] = name
    if augment is not None:
        cfg["augment"] = augment                       # the ONE change (isolated aug or a mix)
    if params:
        cfg["train"].update(params)                    # or a param change
    if split:
        cfg.setdefault("paths", {})["splits"] = f"{FT}/{split}" if "/" not in split else split
    if purpose:
        cfg["purpose"] = purpose
    AUTO.mkdir(parents=True, exist_ok=True)
    out = AUTO / f"{name}.yml"
    out.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return str(out.relative_to(COMP))


def generate(q, worker):
    """Fleet handler — build a config from a spec and return its path (for the orchestrator/leader)."""
    s = q.get("spec") or {}
    base = s.get("base", "config/aug_ablation/00_no_aug.yml")
    name = s.get("name")
    if not name or not (COMP / base).exists():
        return ("escalated", {"base": base, "name": name}, "researcher",
                f"[{worker}] CONFIG-GEN: need a valid base + name (base='{base}', name='{name}').")
    try:
        rel = make(base, name, augment=s.get("augment"), params=s.get("params"),
                   split=s.get("split"), purpose=s.get("purpose", ""),
                   num_workers=s.get("num_workers", 0))
    except Exception as exc:  # noqa: BLE001
        return ("escalated", {"error": str(exc)}, "researcher", f"[{worker}] CONFIG-GEN failed: {exc}")
    return ("done", {"config": rel, "name": name}, "all",
            f"[{worker}] CONFIG-GEN: wrote {rel} (base={Path(base).name}, one change) — ready to enqueue.")
