"""Run canqiang ZebraCellTrace (LB 0.866) on golden-12, score with our golden CV, save nodes."""
import sys, time, importlib.util
from pathlib import Path
import numpy as np, pandas as pd
import torch

ROOT = Path("/home/seshu/kaggle/2026/biohub-cell-tracking-during-development")
sys.path.insert(0, str(ROOT))
from src import golden_cv as gcv
from src.config import Config as SrcCfg
from src import metric, io
SRC = SrcCfg()
TRAIN = ROOT / "input/biohub-cell-tracking-during-development/train"
OUT = ROOT / "learning/ensemble_work"
(OUT / "canqiang_nodes").mkdir(parents=True, exist_ok=True)
(OUT / "canqiang_tracks").mkdir(parents=True, exist_ok=True)

# load canqiang script as module
CQ = ROOT / "learning/public_pull/canqiang_zebracelltrace-full-frame-submit/zebracelltrace-full-frame-submit.py"
spec = importlib.util.spec_from_file_location("cq", CQ)
cq = importlib.util.module_from_spec(spec); sys.modules["cq"] = cq; spec.loader.exec_module(cq)

CKPT = ROOT / "learning/public_pull/data/canqiang_ckpt/best.pt"
GOLDEN12 = ["44b6_0113de3b","44b6_0b24845f","44b6_0c582fdc","44b6_0db75fae","44b6_12dfb391",
            "44b6_144b256d","6bba_05b6850b","6bba_05db0fb1","6bba_062c8d37","6bba_07477033",
            "6bba_07e24132","6bba_085bf656"]
DS = GOLDEN12[:int(sys.argv[1])] if len(sys.argv) > 1 else GOLDEN12

device = torch.device("cuda")
ckpt = torch.load(CKPT, map_location=device, weights_only=False)
cfg = cq.FullFrameTrainingConfig(**ckpt["config"])
model = cq.DeepCenterUNet3D(base_channels=cfg.base_channels)
model.load_state_dict(ckpt["model_state"]); model.to(device).eval()
baseline_cfg = cq.public_reference_baseline_config()
print(f"canqiang: epoch {ckpt.get('epoch')} best {ckpt.get('best_score'):.4f} | {len(DS)} datasets", flush=True)

def score_ds(ds, pn, pe):
    gn, ge = io.read_geff(TRAIN / f"{ds}.geff"); tt = io.geff_estimated_nodes(TRAIN / f"{ds}.geff")
    r = metric.official_counts(gn, ge, pn, pe, SRC.SCALE, SRC.MATCH_GATE_UM, t_true=tt)
    r["dataset"] = ds; r["embryo"] = ds.split("_")[0]
    return r

rows = []
for ds in DS:
    t0 = time.time()
    srows, stats = cq.process_dataset_full_frame(
        TRAIN, ds, model, cfg, baseline_cfg, device,
        peak_threshold=0.20, min_peak_distance=1)
    sub = pd.DataFrame(srows)
    nd = sub[sub.row_type == "node"][["node_id","t","z","y","x"]].reset_index(drop=True)
    ed = sub[sub.row_type == "edge"][["source_id","target_id"]].reset_index(drop=True)
    nd.to_csv(OUT / "canqiang_nodes" / f"{ds}.csv", index=False)
    ed.to_csv(OUT / "canqiang_tracks" / f"{ds}_edges.csv", index=False)
    r = score_ds(ds, nd, ed); rows.append(r)
    print(f"  {ds} {r['embryo']} adjJ={r['adj_jaccard']:.3f} n_pred={r['t_pred']} "
          f"t_true={r['t_true']:.0f} edges={len(ed)} {time.time()-t0:.0f}s", flush=True)

df = pd.DataFrame(rows)
cv = gcv.golden_cv(df)
print(f"\n=== canqiang golden CV ({len(DS)} ds) = {cv['golden_cv']:.4f}  (est LB = {cv['golden_cv']+0.11:.3f}) ===")
print(df[["dataset","embryo","adj_jaccard","t_pred","t_true"]].to_string(index=False))
df.to_csv(OUT / "canqiang_scores.csv", index=False)
