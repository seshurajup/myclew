"""Live golden-CV leaderboard: score every full (199-dataset) result CSV with the
frozen harness, so public notebooks and our configs are ranked apples-to-apples.
Calibration (frozen doc): LB ~= golden_CV + 0.11."""
import sys
from pathlib import Path
ROOT = Path("/home/seshu/kaggle/2026/biohub-cell-tracking-during-development")
sys.path.insert(0, str(ROOT))
import pandas as pd, numpy as np
from src import golden_cv as gcv

rows = []
for f in sorted((ROOT / "results").glob("official_*.csv")):
    d = pd.read_csv(f)
    if d.get("dataset") is None or d.dataset.nunique() < 199:
        continue
    if "embryo" not in d.columns:
        d["embryo"] = d.dataset.str.split("_").str[0]
    cv = gcv.golden_cv(d)["golden_cv"]
    emb = d.groupby("embryo").apply(
        lambda g: (g.w * g.adj_jaccard).sum() / g.w.sum(), include_groups=False)
    rows.append({
        "config": f.stem.replace("official_", ""),
        "golden_cv": round(cv, 4),
        "est_LB": round(cv + 0.11, 3),
        "e44b6": round(float(emb.get("44b6", np.nan)), 3),
        "e6bba": round(float(emb.get("6bba", np.nan)), 3),
    })

df = pd.DataFrame(rows).sort_values("golden_cv", ascending=False)
pub = {"pilkwang_0687", "xiaoleilian_0720", "lucifer_v11_0707", "kojimar_0641",
       "yusuke_0628", "lucifer_0618", "romanrozen_0581",
       "pavloivanin_metric_0637", "pavloivanin_v3_0611"}
df["type"] = np.where(df.config.isin(pub), "PUBLIC", "ours")
pd.set_option("display.width", 130)
print(df.to_string(index=False))

best_pub = df[df.type == "PUBLIC"].golden_cv.max()
print(f"\nBest PUBLIC golden_cv = {best_pub:.4f}  (isaka/pilkwang DoG 0.827 LB = 0.7500, no CSV — hardcoded)")
print(f"Best OURS   golden_cv = {df[df.type=='ours'].golden_cv.max():.4f}")
