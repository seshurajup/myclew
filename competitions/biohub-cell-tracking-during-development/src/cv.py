"""Embryo-disjoint cross-validation splitting.

The hidden test is embryo-disjoint, so local validation must group datasets by embryo
(`<embryo>_<hash>` -> embryo = prefix) and never let the same embryo appear in both folds.
"""
from __future__ import annotations
from collections import defaultdict
from typing import Dict, List, Tuple
from .io import embryo_id


def group_by_embryo(dataset_ids: List[str]) -> Dict[str, List[str]]:
    groups: Dict[str, List[str]] = defaultdict(list)
    for d in dataset_ids:
        groups[embryo_id(d)].append(d)
    return dict(groups)


def kfold_embryo(dataset_ids: List[str], k: int = 5) -> List[Tuple[List[str], List[str]]]:
    """Return k (train_ids, val_ids) splits with embryos kept whole and disjoint."""
    groups = group_by_embryo(dataset_ids)
    embryos = sorted(groups)
    k = min(k, len(embryos))
    folds: List[List[str]] = [[] for _ in range(k)]
    # round-robin embryos across folds (balances dataset counts)
    for i, emb in enumerate(embryos):
        folds[i % k].extend(groups[emb])
    splits = []
    for i in range(k):
        val = folds[i]
        train = [d for j in range(k) if j != i for d in folds[j]]
        splits.append((train, val))
    return splits
