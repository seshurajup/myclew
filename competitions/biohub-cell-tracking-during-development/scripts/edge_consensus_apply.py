"""Apply the shipped edge-error consensus artifact to prediction geffs (NO ground truth).

Remove-only + division-protected: an edge is removed iff the second-architecture model predicts
P(true edge) < tau AND its source out-degree < 2 (never break a division). Features are computed
exactly as in training (edge_consensus.build_features, with_labels=False). Used by the v7 notebook
(per dataset) and as a CLI for local verification (writes pruned geffs to --out-dir).
"""
import argparse, sys, glob, os, json
from pathlib import Path
import numpy as np
COMP = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(COMP / "scripts"))
sys.path.insert(0, str(COMP / "learning/ensemble_work"))
import importlib.util
_spec = importlib.util.spec_from_file_location("ec", str(COMP / "scripts/edge_consensus.py"))
ec = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(ec)
from score_pilkwang import geff_to_dicts
from score_golden12_official import write_geff
import joblib


def removal_keys_for_geff(geff_path, ds, model, tau):
    """Return set of (source_id,target_id) to remove for one prediction geff (no GT)."""
    feat, src, tgt, prob, _, _, s_out = ec.build_features(Path(geff_path), ds, with_labels=False)
    proba = model.predict_proba(feat)[:, 1]
    rem = (proba < tau) & (s_out < 2)
    return {(int(src[i]), int(tgt[i])) for i in np.where(rem)[0]}


def prune_geff(geff_path, ds, model, tau, out_path):
    nbi, edges = geff_to_dicts(geff_path)
    keys = removal_keys_for_geff(geff_path, ds, model, tau)
    kept = [e for e in edges if (int(e["source_id"]), int(e["target_id"])) not in keys]
    write_geff(out_path, nbi, kept)
    return len(edges) - len(kept)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", default="scratchpad/base_stacked")
    ap.add_argument("--out-dir", default="scratchpad/ec_artifact_out")
    ap.add_argument("--model-dir", default="scratchpad/ec_artifact")
    args = ap.parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    model = joblib.load(Path(args.model_dir) / "ec_model.joblib")
    meta = json.loads((Path(args.model_dir) / "ec_meta.json").read_text())
    tau = meta["tau"]
    tot = 0
    for g in sorted(glob.glob(str(Path(args.in_dir) / "*.geff"))):
        ds = os.path.basename(g)[:-5]
        if ds[:4] not in ("44b6", "6bba"):
            continue
        n = prune_geff(g, ds, model, tau, out / f"{ds}.geff")
        tot += n
        print(f"{ds}: removed {n} edges", file=sys.stderr)
    print(f"total removed {tot} (tau={tau})")


if __name__ == "__main__":
    main()
