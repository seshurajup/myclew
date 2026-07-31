"""Working code for me02 — the division metric (+0.1 term). Running it finds real
divisions in a real .geff and scores them, writing me02_division_metric.learning.
    research/cellmot_venv/bin/python learning/annotated/me02_division_metric.py
"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lessonkit import build_lesson

ROOT = Path("/home/seshu/kaggle/2026/biohub-cell-tracking-during-development")
TRAIN = ROOT / "input/biohub-cell-tracking-during-development/train"
REPO = ROOT / "research/pilkwang_support_pack/repo"

META = dict(id="me02", order=21, title="The division metric (+0.1 term)",
            subtitle="extract_divisions + score_divisions — the term that is our whole lever",
            source="research/pilkwang_support_pack/repo/src/biohub_tracking/division_metrics.py")

CELLS = [
    dict(note="""## The other half of the score
The competition metric is **adjusted edge-Jaccard + 0.1 · division-Jaccard**. me01 covered the
edge part; this is the **division** part — and it's the term where all our score headroom lives
(rs03/rs04). A *division* is a cell splitting into two: parent → divider → **two** children that
both continue. We compute it on real data below."""),

    dict(note="""### Find the real division events
`extract_divisions` scans a graph for **dividing nodes** (out-degree ≥ 2) and pulls out each
event's subgraph (parent → divider → 2 children → grandchildren). Divisions are **rare** — that
rarity is exactly why the model struggles to learn them (rs04).""",
         code="""import sys                                              # to reach the repo package
sys.path.insert(0, f"{REPO}/src")                          # biohub_tracking on the path
import tracksdata as td                                    # the graph library
from biohub_tracking.division_metrics import extract_divisions  # the REAL extractor
gt = td.graph.IndexedRXGraph.from_geff(f"{TRAIN}/6bba_05db0fb1.geff")  # a real embryo's GT graph
gt = gt[0] if isinstance(gt, tuple) else gt                # unwrap if a tuple
divs = extract_divisions(gt)                               # dict: divider node -> event subgraph
{"nodes": gt.num_nodes(), "edges": gt.num_edges(),         # graph size
 "division events": len(divs)}                             # how many real divisions"""),

    dict(note="""### Look at one real division's topology
Each event is a small subgraph. The divider has **two** outgoing edges (the two daughters) — that
2-child topology is exactly what the metric checks for (within 7 µm).""",
         code="""div_node = next(iter(divs))                            # pick one real division
sub = divs[div_node]                                       # its subgraph
children = gt.successors(div_node)                         # the divider's daughters
{"divider node": int(div_node),                            # the dividing cell's id
 "n daughters": len(children),                             # should be 2
 "event subgraph nodes": sub.num_nodes()}                  # parent+divider+children+grandchildren"""),

    dict(note="""### Score the divisions (the +0.1 term)
`score_divisions(pred, gt)` returns 1 for each GT division the prediction recovers (correct
divider + two matched continuing children, single component, 7 µm). Scored against **itself** it
recovers all — the ceiling. **[Lever]** a real detector recovers almost **none** of these (rs03:
0 TP); training the model to find them is our path (rs04).""",
         code="""from biohub_tracking.division_metrics import score_divisions  # the REAL scorer
scale = (1.625, 0.40625, 0.40625)                          # physical voxel size (z,y,x)
perfect = score_divisions(gt, gt, scale=scale)             # GT vs itself = the ceiling
tp = sum(perfect.values())                                 # divisions recovered
{"GT divisions": len(perfect),                             # total real divisions
 "recovered (GT vs GT)": tp,                               # perfect recovery = the ceiling
 "division_jaccard": round(tp / len(perfect), 2)}          # 1.0 when all recovered"""),

    dict(note="""**[Recap]** Divisions are rare, need exact 2-child topology within 7 µm, and are
worth 0.1× in the score. The metric can reach 1.0 in principle (GT vs GT), but real predictions
recover ~none — which is why the division term is our entire remaining headroom.

**Next → the research journey (rs01…)**, where we chase exactly this term."""),
]

if __name__ == "__main__":
    build_lesson(META, CELLS, Path(__file__).with_suffix(".learning"),
                 {"TRAIN": TRAIN, "REPO": REPO})
