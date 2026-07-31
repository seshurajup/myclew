"""Part A — run pilkwang predict_video on golden-12, capture the FULL candidate
edge pool (all edges >0.5 prob, BEFORE ILP collapses it) + the ILP-solved base graph.
Cache to npz so division recovery can sweep thresholds without re-running the GPU.
"""
import os, sys, json
from pathlib import Path
import numpy as np

ROOT = Path("/home/seshu/kaggle/2026/biohub-cell-tracking-during-development")
REPO = ROOT / "research/pilkwang_support_pack/repo"
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
os.environ.setdefault("BIOHUB_DET_THRESHOLD", "0.99")

import torch
import tracksdata as td
import predict_unet_transformer as PUT

TRAIN = ROOT / "input/biohub-cell-tracking-during-development/train"
OUT = ROOT / "learning/ensemble_work/pool_cache"
OUT.mkdir(exist_ok=True)
GOLDEN12 = ["44b6_0113de3b", "44b6_0b24845f", "44b6_0c582fdc", "44b6_0db75fae",
            "44b6_12dfb391", "44b6_144b256d", "6bba_05b6850b", "6bba_05db0fb1",
            "6bba_062c8d37", "6bba_07477033", "6bba_07e24132", "6bba_085bf656"]

WEIGHTS = ROOT / "research/pilkwang_support_pack/weights/unet_transformer/split_0/edge_predictor_best.pth"


def main():
    device = torch.device("cuda")
    model, window_size, downsample = PUT.load_model(WEIGHTS, device)
    cfg = PUT.PredictConfig(det_threshold=0.99, use_ilp=True)
    print(f"cfg: det_thr={cfg.det_threshold} edge_thr={cfg.threshold} act={cfg.edge_activation} "
          f"ilp={cfg.use_ilp} maxchild={cfg.max_children_per_node}", flush=True)
    for name in GOLDEN12:
        cache = OUT / f"{name}.npz"
        if cache.exists():
            print(f"  skip {name} (cached)", flush=True); continue
        coords, edges = PUT.predict_video(model, TRAIN / f"{name}.zarr", device, cfg=cfg,
                                          window_size=window_size, downsample=downsample)
        e = np.array([(s, t, p, d) for s, t, p, d in edges], dtype=np.float64) if edges else np.zeros((0, 4))
        # ILP-solved base graph edges (source,target)
        graph = PUT.build_graph(coords, edges)
        if cfg.use_ilp and graph.num_edges() > 0:
            solver = td.solvers.ILPSolver(
                edge_weight=cfg.ilp_edge_weight * td.EdgeAttr("edge_prob"),
                appearance_weight=cfg.ilp_appearance_weight,
                disappearance_weight=cfg.ilp_disappearance_weight,
                division_weight=cfg.ilp_division_weight)
            with PUT.suppress_output():
                graph = solver.solve(graph)
        ea = graph.edge_attrs()
        base_edges = np.array([(int(r["source_id"]), int(r["target_id"]))
                               for r in ea.iter_rows(named=True)], dtype=np.int64) if graph.num_edges() else np.zeros((0, 2), np.int64)
        np.savez(cache, coords=coords.astype(np.float32), pool=e, base_edges=base_edges)
        ndiv = 0
        if len(base_edges):
            from collections import Counter
            ndiv = sum(1 for v in Counter(base_edges[:, 0]).values() if v >= 2)
        print(f"  {name}: nodes={len(coords)} pool_edges={len(e)} base_edges={len(base_edges)} base_div={ndiv}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
