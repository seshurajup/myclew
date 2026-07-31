"""THREAD 2 exp#1 — LEARNED LINKER (Trackastra ctc, coord_dim=3) on pilkwang's FIXED detections.

Approved by leader (2026-07-05). Tests the BINDING L1 bucket: swap pilkwang's geometric ILP linker
for a pretrained learned linker WITHOUT retraining the detector. Nodes stay FIXED = pilkwang's solution
detections (node recall saturated, L4); only the edges are re-derived by Trackastra, then re-scored with
the OFFICIAL metric on golden-12 vs the 0.8708 anchor.

Pretrained model = **ctc** (config coord_dim=3, trained on a CTC mix incl. 3D Fluo-C3DH-*); general_2d
is coord_dim=2 (2D-only) and is NOT used. Trackastra's ctc is division-aware, so mode=greedy/ilp also
yields a FREE golden-12 division-recall signal (low-power: only 8 GT divisions — see ledger L9 / exp#3).

ADAPTER: Trackastra.track(imgs, masks) consumes label-mask volumes + raw images, not points. We paint a
small anisotropic labelled blob at each fixed pilkwang detection into a (T,Z,Y,X) uint32 mask (label L at
time t <-> pilkwang node_id), load the raw (T,Z,Y,X) image, run track(), then map the output DiGraph
links (node attrs time+label) back to pilkwang node_ids to form predicted edges.

KNOWN APPROXIMATIONS (documented; a NEGATIVE here is confounded by these, a WIN is strong):
  * shape features come from synthetic blobs (pilkwang gives points, not instance segmentations);
  * coords are fed in VOXEL units (anisotropy 1.625/0.406/0.406 not folded into the model's positional
    bias); intensity features ARE real (sampled from the raw volume at the detection).
A clean version = fine-tune a 3D Trackastra on our tracks (exp#4, gated behind this).

ABSOLUTE paths throughout so the same invocation is GREEN from any cwd (trainer POSTs from a different
cwd). Run with the repo-root cellmot venv:
  /home/seshu/kaggle/2026/biohub-cell-tracking-during-development/research/cellmot_venv/bin/python \
    /home/seshu/kaggle/2026/biohub-cell-tracking-during-development/tools/researchpapers/eda/thread2/exp1_trackastra_link.py \
    --mode greedy            # full golden-12
  ... --dry-run              # 1 embryo, startup + wiring validation
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# ---- ABSOLUTE paths (repo root, not the runtime cwd tools/researchpapers) ----
ROOT = Path("/home/seshu/kaggle/2026/biohub-cell-tracking-during-development")
WORKDIR = ROOT / "tools/researchpapers"
ENSEMBLE = ROOT / "learning/ensemble_work"
TRAIN = ROOT / "input/biohub-cell-tracking-during-development/train"
GEFF_DIR = ROOT / "research/pilkwang_support_pack/repo/predictions/seshu/unet_transformer/split_0"
OUT_DEFAULT = WORKDIR / "output/thread2_exp1_trackastra"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ENSEMBLE))

from src import io, metric  # noqa: E402
from src.config import Config  # noqa: E402
import pilk_post as P  # noqa: E402

SRC = Config()
SCALE = np.asarray(SRC.SCALE)            # (z,y,x) um/voxel: (1.625, 0.40625, 0.40625)
GATE = SRC.MATCH_GATE_UM                 # 7.0 um
ANCHOR = 0.8708                          # pilkwang post-proc production adj_edge_jaccard

GOLDEN12 = ["44b6_0113de3b", "44b6_0b24845f", "44b6_0c582fdc", "44b6_0db75fae",
            "44b6_12dfb391", "44b6_144b256d", "6bba_05b6850b", "6bba_05db0fb1",
            "6bba_062c8d37", "6bba_07477033", "6bba_07e24132", "6bba_085bf656"]
# smallest first so --dry-run touches the cheapest embryo
DRY_ORDER = ["44b6_0113de3b", "44b6_0b24845f"]


def _pilk_geff_path(ds: str) -> Path:
    """pilkwang preds are named <ds>.zarr.geff (some tooling writes <ds>.geff)."""
    for name in (f"{ds}.zarr.geff", f"{ds}.geff"):
        p = GEFF_DIR / name
        if p.exists():
            return p
    raise FileNotFoundError(f"no pilkwang geff for {ds} under {GEFF_DIR}")


def load_pilk_nodes(ds: str) -> pd.DataFrame:
    """pilkwang FIXED solution detections (node_id, t, z, y, x) in voxel coords."""
    g = P.graph_from_geff(_pilk_geff_path(ds))
    nd = g.node_attrs().to_pandas()
    if "solution" in nd.columns:
        nd = nd[nd["solution"] == True]  # noqa: E712
    return nd[["node_id", "t", "z", "y", "x"]].astype(
        {"node_id": int, "t": int, "z": float, "y": float, "x": float}).reset_index(drop=True)


def paint_masks(nodes: pd.DataFrame, shape, rz: int, rxy: int):
    """Build (T,Z,Y,X) uint32 label mask + per-frame {local_label: node_id} map.

    One small anisotropic blob per detection; labels are frame-local (1..N_t)."""
    T, Z, Y, X = shape
    masks = np.zeros((T, Z, Y, X), dtype=np.uint32)
    # anisotropic ellipsoid offsets (rz in z, rxy in y/x)
    zz, yy, xx = np.mgrid[-rz:rz + 1, -rxy:rxy + 1, -rxy:rxy + 1]
    ball = (zz / max(rz, 1)) ** 2 + (yy / max(rxy, 1)) ** 2 + (xx / max(rxy, 1)) ** 2 <= 1.0
    dz, dy, dx = np.where(ball)
    dz, dy, dx = dz - rz, dy - rxy, dx - rxy
    label_map = {}  # t -> {local_label: node_id}
    for t, sub in nodes.groupby("t"):
        lm = {}
        for local, (_, r) in enumerate(sub.iterrows(), start=1):
            cz, cy, cx = int(round(r.z)), int(round(r.y)), int(round(r.x))
            zc, yc, xc = np.clip(cz + dz, 0, Z - 1), np.clip(cy + dy, 0, Y - 1), np.clip(cx + dx, 0, X - 1)
            masks[t, zc, yc, xc] = local
            lm[local] = int(r.node_id)
        label_map[int(t)] = lm
    return masks, label_map


def load_image(ds: str, shape) -> np.ndarray:
    """Raw (T,Z,Y,X) intensity volume (uint16)."""
    array_dir, ashape, dtype = io.read_array_meta(TRAIN / f"{ds}.zarr")
    T = shape[0]
    vol = np.empty(shape, dtype=dtype)
    for t in range(T):
        vol[t] = io.load_volume(array_dir, ashape, dtype, t)
    return vol


# coord-scale de-confound (exp#1b): scale regionprops centroid coords (z,y,x voxel) BEFORE the model's
# positional bias. Trackastra ctc reads centroids in VOXEL units with NO anisotropy correction, so our
# scale=(1.625,0.406,0.406) anisotropy compresses z ~4x vs xy -- the prime suspect for the dense collapse.
#   voxel = as-fed in exp#1 (baseline, reproduces 0.6465)
#   iso_z = (4.0,1,1): correct z-compression, KEEP xy in the model's expected voxel range (0-256)
#   um    = physical µm (1.625,0.406,0.406): isotropic in µm but shrinks xy (0-104) -> probes cutoff
COORD_SCALES = {
    "voxel": None,
    "iso_z": (float(SRC.SCALE[0] / SRC.SCALE[2]), 1.0, 1.0),
    "um": tuple(float(s) for s in SRC.SCALE),
}


def track_scaled(model, imgs, masks, scale_vec, mode, prog):
    """Trackastra track with an optional coord-scaling hook after feature extraction.

    Replicates model._predict() but multiplies each WRFeatures.coords by scale_vec before build_windows,
    then runs the same _track_from_predictions linker. scale_vec=None == stock track()."""
    from trackastra.model.model_api import get_features, build_windows, predict_windows, normalize
    imn = normalize(imgs)
    model.transformer.eval()
    feats = get_features(detections=masks, imgs=imn, features=model.train_args["features"],
                         feature_extractor=model.feature_extractor,
                         ndim=model.transformer.config["coord_dim"], n_workers=0, progbar_class=prog)
    if scale_vec is not None:
        sv = np.asarray(scale_vec, np.float32)
        for f in feats:
            f.coords = f.coords * sv[None, :]
    win = build_windows(feats, window_size=model.transformer.config["window"],
                        progbar_class=prog, as_torch=True)
    pred = predict_windows(windows=win, features=feats, model=model.transformer,
                           edge_threshold=0.05, spatial_dim=masks.ndim - 1,
                           progbar_class=prog, batch_size=model.batch_size)
    return model._track_from_predictions(pred, mode=mode)


def graph_to_edges(graph, label_map) -> pd.DataFrame:
    """Map Trackastra DiGraph links (node attrs time+label) back to pilkwang node_id edges."""
    def to_nid(n):
        d = graph.nodes[n]
        return label_map.get(int(d["time"]), {}).get(int(d["label"]))
    rows = []
    for u, v in graph.edges():
        a, b = to_nid(u), to_nid(v)
        if a is not None and b is not None:
            rows.append((a, b))
    return pd.DataFrame(rows, columns=["source_id", "target_id"]) if rows \
        else pd.DataFrame(columns=["source_id", "target_id"])


def run_embryo(model, ds: str, mode: str, rz: int, rxy: int, scale_vec, log) -> dict:
    t0 = time.time()
    nodes = load_pilk_nodes(ds)
    gn, ge = io.read_geff(TRAIN / f"{ds}.geff")
    estN = io.geff_estimated_nodes(TRAIN / f"{ds}.geff")
    array_dir, shape, _ = io.read_array_meta(TRAIN / f"{ds}.zarr")
    masks, label_map = paint_masks(nodes, shape, rz, rxy)
    imgs = load_image(ds, shape)
    graph = track_scaled(model, imgs, masks, scale_vec, mode, _NoProg)
    pe = graph_to_edges(graph, label_map)
    c = metric.official_counts(gn, ge, nodes, pe, SCALE, GATE, t_true=estN)
    c["dataset"] = ds
    c["embryo"] = ds.split("_")[0]
    c["n_pilk_nodes"] = len(nodes)
    c["n_track_edges"] = len(pe)
    log(f"  [{ds}] nodes={len(nodes)} track_edges={len(pe)} "
        f"edgeTP/FP/FN={c['edge_tp']}/{c['edge_fp']}/{c['edge_fn']} "
        f"adjJ={c['adj_jaccard']:.4f} div(tp/fp/fn)={c['div_tp']}/{c['div_fp']}/{c['div_fn']} "
        f"({time.time()-t0:.1f}s)")
    return c


class _NoProg:
    """Silence Trackastra tqdm bars (we emit our own per-embryo progress)."""
    def __init__(self, *a, **k):
        self.it = k.get("iterable", a[0] if a else [])
    def __iter__(self):
        return iter(self.it)
    def update(self, *a, **k):
        pass
    def close(self):
        pass
    def set_description(self, *a, **k):
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["greedy", "greedy_nodiv", "ilp"], default="greedy")
    ap.add_argument("--coord-scale", choices=list(COORD_SCALES), default="voxel",
                    help="exp#1b de-confound knob: voxel(baseline)|iso_z(z*4,xy voxel)|um(physical)")
    ap.add_argument("--model", default="ctc")
    ap.add_argument("--dry-run", action="store_true", help="1 embryo, wiring validation")
    ap.add_argument("--limit", type=int, default=0, help="cap #embryos (0=all golden-12)")
    ap.add_argument("--blob-rz", type=int, default=1)
    ap.add_argument("--blob-rxy", type=int, default=2)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default=str(OUT_DEFAULT))
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    order = DRY_ORDER[:1] if args.dry_run else GOLDEN12
    if args.limit:
        order = order[:args.limit]
    scale_vec = COORD_SCALES[args.coord_scale]
    tag = f"{args.mode}_{args.coord_scale}"

    import torch
    from trackastra.model import Trackastra
    dev = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device

    def log(m):
        print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

    # ---- STARTUP LOG (so leader/trainer can confirm the job truly started) ----
    log("=== THREAD-2 exp#1: Trackastra learned linker on FIXED pilkwang detections ===")
    log(f"  task        : swap geometric ILP -> learned ctc linker; nodes FIXED; re-score official")
    log(f"  model       : Trackastra pretrained '{args.model}' (coord_dim=3, 3D) mode={args.mode}")
    log(f"  coord_scale : {args.coord_scale} -> scale_vec(z,y,x)={scale_vec}  [exp#1b de-confound knob]")
    log(f"  device      : {dev}  (cuda_avail={torch.cuda.is_available()})")
    log(f"  workdir     : {WORKDIR}")
    log(f"  geff_dir    : {GEFF_DIR}")
    log(f"  train_data  : {TRAIN}")
    log(f"  out_dir     : {out}")
    log(f"  blob (rz,rxy)=({args.blob_rz},{args.blob_rxy})  scale={tuple(SCALE)}  gate={GATE}um  anchor={ANCHOR}")
    log(f"  embryos     : {len(order)}{' (DRY-RUN)' if args.dry_run else ''} -> {order}")

    model = Trackastra.from_pretrained(args.model, device=dev)
    log(f"  model loaded OK from cache; starting linking loop")

    rows = []
    for i, ds in enumerate(order, 1):
        log(f"--- embryo {i}/{len(order)} ---")
        try:
            rows.append(run_embryo(model, ds, args.mode, args.blob_rz, args.blob_rxy, scale_vec, log))
        except Exception as e:
            log(f"  [{ds}] ERROR: {type(e).__name__}: {e}")
            raise

    df = pd.DataFrame(rows)
    df.to_csv(out / f"exp1_trackastra_{tag}_percell.csv", index=False)
    agg = metric.official_score(rows)
    # dense-tail = embryos with the most fixed detections (where exp#1 collapsed); judge these, not just W-mean
    dense = df.sort_values("n_pilk_nodes", ascending=False).head(5)
    dense_adjJ = float((dense.w * dense.adj_jaccard).sum() / dense.w.sum()) if dense.w.sum() else float("nan")
    summary = dict(mode=args.mode, coord_scale=args.coord_scale, scale_vec=scale_vec, model=args.model,
                   n_embryos=len(rows), adj_edge_jaccard=agg["adj_edge_jaccard"],
                   dense_tail_adj_jaccard=dense_adjJ,
                   dense_tail_embryos=dense.dataset.tolist(),
                   division_jaccard=agg["division_jaccard"], score=agg["score"],
                   anchor_adj_edge_jaccard=ANCHOR,
                   delta_vs_anchor=agg["adj_edge_jaccard"] - ANCHOR,
                   div_tp=int(df.div_tp.sum()), div_fp=int(df.div_fp.sum()), div_fn=int(df.div_fn.sum()),
                   dry_run=args.dry_run)
    (out / f"exp1_trackastra_{tag}_summary.json").write_text(json.dumps(summary, indent=2))

    log("=== RESULT ===")
    log(f"  coord_scale={args.coord_scale}  scale_vec={scale_vec}")
    log(f"  adj_edge_jaccard = {agg['adj_edge_jaccard']:.4f}  (anchor {ANCHOR:.4f}, "
        f"Δ={agg['adj_edge_jaccard']-ANCHOR:+.4f})")
    log(f"  DENSE-TAIL adjJ  = {dense_adjJ:.4f}  (top-5 by #detections: {dense.dataset.tolist()})")
    log(f"  division_jaccard = {agg['division_jaccard']}  div tp/fp/fn="
        f"{int(df.div_tp.sum())}/{int(df.div_fp.sum())}/{int(df.div_fn.sum())}")
    log(f"  score (adj + 0.1*div) = {agg['score']:.4f}")
    log(f"  wrote {out}/exp1_trackastra_{tag}_percell.csv + _summary.json")
    if args.dry_run:
        log("  DRY-RUN GREEN: startup + adapter + track + back-map + official score all wired.")


if __name__ == "__main__":
    main()
