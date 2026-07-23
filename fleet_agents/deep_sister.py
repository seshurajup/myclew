"""deep-sister (Part B) — deep sister model. Final form: reuse PRETRAINED detector features (researcher).
Runs under cellmot_venv (torch+zarr). Until the pretrained-feature extractor is wired, escalates cleanly
(the forest div-model, AUC~0.82, is the working secondary model)."""
from __future__ import annotations
from pathlib import Path
COMP = Path(__file__).resolve().parent.parent


def train(q, worker):
    return ("escalated", {"approach": "reuse pretrained detector features"}, "researcher",
            f"[{worker}] deep-sister: use the PRETRAINED detector (pilkwang ft_divW / model_scratch) as a frozen "
            f"feature extractor + a small head — researcher build. Forest div-model is the working baseline meanwhile.")
