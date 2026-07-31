"""Central config for the biohub cell-tracking pipeline.

Defaults follow the best public notebooks (pilkwang EDA + V2 sub-voxel + V3 motion prior).
Voxel scale and the 7µm matching gate come from the competition data contract.
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Tuple
import yaml


@dataclass
class Config:
    # --- data contract ---
    SCALE: Tuple[float, float, float] = (1.625, 0.40625, 0.40625)  # z,y,x µm/voxel
    MATCH_GATE_UM: float = 7.0          # metric node-matching radius (physical µm)

    # --- detection ---
    XY_DS: int = 4                      # XY block-mean -> isotropic ~1.625µm grid
    SMOOTH_SIGMA: float = 1.0
    MIN_PEAK_DIST: int = 2              # min separation on the working (downsampled) grid
    THRESH_REL: float = 0.20           # threshold = max(Otsu, P50 + REL*(P99.8-P50))
    USE_SUBVOXEL: bool = True          # intensity-weighted CoM on the raw volume
    REFINE_RZ: int = 2
    REFINE_RYX: int = 5
    USE_BORDER_FILTER: bool = True
    BORDER_KEEP_QUANTILE: float = 0.0  # drop weak peaks within border margin; 0 disables
    BORDER_MARGIN_VOX: int = 4
    NMS_RADIUS_UM: float = 0.0         # physical NMS to remove duplicate detections; 0 disables

    # --- linking ---
    MAX_LINK_DIST_UM: float = 11.0     # gate for frame T->T+1 Hungarian assignment
    USE_VELOCITY_PRIOR: bool = True    # V3: predict pos = pos + INERTIA*velocity
    VELOCITY_INERTIA: float = 1.0
    USE_GAP_CLOSING: bool = True       # V3: heal T->T+2 dropouts
    GAP_CLOSE_DIST_UM: float = 11.0
    GAP_CLOSE_MAX_SKIP: int = 1        # k=2 measured WORSE (false edges → precision loss); keep k=1 + GMC-in-gapclose
    # global-motion compensation (research #1, ultrack +0.18; CONFIRMED +8%, flips focal above DoG) — LOCKED ON
    USE_GLOBAL_MOTION_COMP: bool = True
    GMC_GATE_MULT: float = 4.0         # coarse-match gate = 4x the link gate (catch big setup jumps)
    GMC_MIN_MATCHES: int = 5           # need >=5 coarse matches to estimate the global shift
    GMC_MIN_SHIFT_UM: float = 2.0      # only compensate a shift bigger than this (a real jump, not jitter)
    GMC_LOCAL_FLOW: bool = True        # research #1: per-REGION flow (divergent gastrulation motion) not one global vector
    GMC_FLOW_K: int = 8                # per-node shift = median displacement of its K nearest coarse-matched neighbours

    # --- divisions (mitosis) ---
    DETECT_DIVISIONS: bool = True
    DIV_PARENT_DIST_UM: float = 8.75
    DIV_SISTER_DIST_UM: float = 6.25
    DIV_MIN_COUNT_GAIN: int = 1

    # --- post-processing ---
    PRUNE_ISOLATED_NODES: bool = True  # remove nodes that never participate in an edge

    # --- runtime ---
    run_name: str = "baseline"

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        with open(path) as f:
            d = yaml.safe_load(f) or {}
        # tuples come back as lists from yaml
        if "SCALE" in d and isinstance(d["SCALE"], list):
            d["SCALE"] = tuple(d["SCALE"])
        known = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**known)

    def to_dict(self) -> dict:
        return asdict(self)
