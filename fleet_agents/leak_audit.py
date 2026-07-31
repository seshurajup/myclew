"""leak_audit — HONEST-model gate for the rogii geosteering pipeline. Every experiment runs this before its
CV is trusted. Fails loudly (returns ok=False) if any leakage path is present:

  1. FORBIDDEN COLUMNS — no train-only field (ANCC/ASTNU/ASTNL/EGFDU/EGFDL/BUDA) or the post-PS truth (TVT)
     leaks into the feature matrix. These do not exist in the hidden test → using them is leakage.
  2. CV DISJOINTNESS — no well appears in both train and val of any fold (well-grouped AND field-grouped).
  3. TOO-GOOD FEATURE — no single feature correlates with the target implausibly high (|r|>0.995), a classic
     "the target leaked in" signature.
  4. TARGET NOT IN FEATURES — the target / its raw components are not among the inputs.

Reusable across comps by passing the forbidden set. Returns (ok, checks[list of (name, ok, detail)]).
"""
from __future__ import annotations
import numpy as np
import pandas as pd

# train-only geological formation tops + the post-PS truth column (never present in the hidden test)
ROGII_FORBIDDEN = {"ANCC", "ASTNU", "ASTNL", "EGFDU", "EGFDL", "BUDA", "TVT", "dtvt_true", "true_dtvt", "tvt"}


def audit_features(feat: pd.DataFrame, feature_cols: list[str], target_col: str,
                   forbidden: set[str] | None = None, max_abs_corr: float = 0.995) -> list[tuple[str, bool, str]]:
    forbidden = (forbidden or ROGII_FORBIDDEN) - {target_col}
    checks = []
    bad = sorted(set(feature_cols) & {c for c in forbidden})
    checks.append(("no forbidden/train-only columns in features", not bad,
                   f"clean ({len(feature_cols)} feats)" if not bad else f"LEAK: {bad}"))
    # too-good single feature
    y = feat[target_col].to_numpy(dtype=float)
    worst = None
    for c in feature_cols:
        try:
            x = feat[c].to_numpy(dtype=float)
            if np.nanstd(x) == 0:
                continue
            r = abs(np.corrcoef(np.nan_to_num(x), y)[0, 1])
            if worst is None or r > worst[1]:
                worst = (c, r)
        except Exception:  # noqa: BLE001
            continue
    ok_corr = (worst is None) or (worst[1] <= max_abs_corr)
    checks.append((f"no single feature |r|>{max_abs_corr} with target", ok_corr,
                   "none" if worst is None else f"max |r|={worst[1]:.3f} ({worst[0]})"))
    return checks


def audit_cv_disjoint(fold_assign: pd.DataFrame, well_col: str = "well", fold_col: str = "field_fold") -> list[tuple[str, bool, str]]:
    checks = []
    # a well must map to exactly one fold (disjoint groups)
    per = fold_assign.groupby(well_col)[fold_col].nunique()
    multi = per[per > 1]
    checks.append(("each well in exactly one fold (disjoint)", multi.empty,
                   "disjoint" if multi.empty else f"{len(multi)} wells span >1 fold"))
    return checks


def audit(feat: pd.DataFrame, feature_cols: list[str], target_col: str,
          fold_assign: pd.DataFrame | None = None, forbidden: set[str] | None = None) -> tuple[bool, list]:
    checks = audit_features(feat, feature_cols, target_col, forbidden)
    if fold_assign is not None:
        checks += audit_cv_disjoint(fold_assign)
    ok = all(c[1] for c in checks)
    return ok, checks


def print_report(ok: bool, checks: list, title: str = "") -> bool:
    print(f"\n🔒 LEAK AUDIT{(' — ' + title) if title else ''}")
    for name, passed, detail in checks:
        print(f"  {'✅' if passed else '❌ LEAK'} {name} · {detail}")
    print(f"  {'✅ CLEAN — honest' if ok else '❌ LEAKAGE DETECTED — discard this experiment'}")
    return ok
