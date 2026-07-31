"""Working code for io01 — writing the submission. Running it reads a real .geff graph
and shows the exact node/edge rows the submission.csv needs.
    research/cellmot_venv/bin/python learning/annotated/io01_submission.py
"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lessonkit import build_lesson

ROOT = Path("/home/seshu/kaggle/2026/biohub-cell-tracking-during-development")
TRAIN = ROOT / "input/biohub-cell-tracking-during-development/train"

META = dict(id="io01", order=20, title="Writing the submission",
            subtitle="From a tracking graph to submission.csv — node rows + edge rows",
            source="research/pilkwang_support_pack/repo/src/biohub_tracking/io.py")

CELLS = [
    dict(note="""## The final output
Everything ends here: the predicted **tracking graph** (nodes = cells, edges = links across
frames) is written to `submission.csv`. Two row types: **node** rows carry a cell's `(t,z,y,x)`,
**edge** rows carry `(source_id → target_id)`. We read a real `.geff` and show both."""),

    dict(note="""### The node rows — where every cell is
`node_attrs()` gives each node's frame and coordinates. In the submission these become
`row_type="node"` rows with `id, dataset, node_id, t, z, y, x` (source/target = −1).""",
         code="""import tracksdata as td                                 # the graph library
g = td.graph.IndexedRXGraph.from_geff(f"{TRAIN}/6bba_062c8d37.geff")  # a real embryo graph
g = g[0] if isinstance(g, tuple) else g                    # unwrap if a tuple
nodes = g.node_attrs().to_pandas()                         # nodes as a table (t, node_id, z, y, x)
nodes.head(4)                                              # the first 4 real cell nodes"""),

    dict(note="""### The edge rows — which cell becomes which
`edge_attrs()` gives the links: `source_id` (a cell at t) → `target_id` (the same cell at t+1, or
a daughter). In the submission these are `row_type="edge"` rows.""",
         code="""edges = g.edge_attrs().to_pandas()                     # edges as a table
edges[["source_id", "target_id"]].head(4)                  # the first 4 real links"""),

    dict(note="""### The submission format
`submission.csv` columns: `id, dataset, row_type, node_id, t, z, y, x, source_id, target_id`.
Node rows fill the coords (source/target = −1); edge rows fill source/target (coords = −1). We
build the real rows for this embryo and count them.""",
         code="""rows = []                                              # collect submission rows
for r in nodes.itertuples():                               # one row per cell node
    rows.append(dict(dataset="6bba_062c8d37", row_type="node", node_id=r.node_id,
                     t=r.t, z=r.z, y=r.y, x=r.x, source_id=-1, target_id=-1))
for e in edges.itertuples():                               # one row per link edge
    rows.append(dict(dataset="6bba_062c8d37", row_type="edge", node_id=-1,
                     t=-1, z=-1, y=-1, x=-1, source_id=e.source_id, target_id=e.target_id))
import pandas as pd                                         # to tabulate
sub = pd.DataFrame(rows)                                    # the submission rows for this embryo
sub["row_type"].value_counts().rename_axis("row_type").rename("count").reset_index()  # node vs edge counts"""),

    dict(note="""**[Recap]** A submission is just the tracking graph flattened: one `node` row per
detected cell (with its `t,z,y,x`) and one `edge` row per link. Repeat for every test embryo,
concatenate, write `submission.csv`. **Next → mc01: the modeling contract** — the math tying the
whole pipeline together."""),
]

if __name__ == "__main__":
    build_lesson(META, CELLS, Path(__file__).with_suffix(".learning"), {"TRAIN": TRAIN})
