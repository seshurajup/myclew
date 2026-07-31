"""Stack mtl10/gap5.5 post-proc on the FRESH re-detect geffs (leader real-candidate, CPU, reuse GPU preds).

Reads the fresh det=0.99 re-detect geffs, applies pilk_post.filter_output_graph (env-driven:
BIOHUB_OUTPUT_FILTER_SHORT_TRACKS=1 OUTPUT_MIN_TRACK_LEN=10 GAP_CLOSE_UM=5.5), writes spec-compliant
geffs to a persistent dir, and prints per-ds predN/estN. Then score the OUT dir canonically via
score_golden12_official.py --pred-dir <OUT>.

Usage:
  BIOHUB_OUTPUT_FILTER_SHORT_TRACKS=1 BIOHUB_OUTPUT_MIN_TRACK_LEN=10 BIOHUB_GAP_CLOSE_UM=5.5 \
    research/cellmot_venv/bin/python scripts/stack_postproc_on_fresh.py <pred_dir> <out_dir>
"""
import sys, glob, os
from pathlib import Path
COMP = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(COMP / "learning/ensemble_work"))
sys.path.insert(0, str(COMP / "tools/researchpapers/baseline/postproc"))
sys.path.insert(0, str(COMP / "scripts"))
from score_pilkwang import geff_to_dicts
import pilk_post as P
from score_golden12_official import write_geff, _estimated_n, TRAIN

PRED = Path(sys.argv[1]); OUT = Path(sys.argv[2]); OUT.mkdir(parents=True, exist_ok=True)
print(f"{'dataset':<16}{'bare_predN':>11}{'stacked_predN':>14}{'estN':>8}{'stacked/estN':>13}")
for g in sorted(glob.glob(str(PRED / "*.geff"))):
    ds = os.path.basename(g).replace(".zarr.geff", "").replace(".geff", "")
    nbi, edges = geff_to_dicts(g)
    bare = len(nbi)
    nbi2, e2, _ = P.filter_output_graph(dict(nbi), list(edges), dataset=ds)
    write_geff(OUT / f"{ds}.geff", nbi2, e2)
    est = _estimated_n(TRAIN / f"{ds}.geff")
    print(f"{ds:<16}{bare:>11}{len(nbi2):>14}{int(est):>8}{len(nbi2)/est:>13.3f}")
print(f"stacked geffs -> {OUT}")
