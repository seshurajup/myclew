"""
Kaggle inference notebook for biohub-cell-tracking-during-development (OFFLINE, <=12h).
Reproduces our best NO-TRAIN pipeline (local CV 0.8392 on fold-0):
  detect (UNet, det=0.998, pool=3.0µm, flip-TTA) → link (transformer edges, edge_thr=0.3)
  → ILP (appear/disappear=0.3, division=1.0) → tracking graph → submission.csv

Attach as Kaggle inputs:
  - Dataset `cellmot-inference-bundle` (repo.tar = tracking_cellmot+scripts+our submission.py, wheels.tar, weights.tar)
  - Competition data `biohub-cell-tracking-during-development`
Robust to tar/zip packaging: it extracts archives and locates files by glob.
"""
import os, sys, json, glob, shutil, tarfile, subprocess
from pathlib import Path

WORK = Path("/kaggle/working")
STAGE = WORK / "bundle"


def discover_test_dir():
    """Find the competition test dir: a 'test' folder containing *.zarr, anywhere under /kaggle/input.
    (Kaggle may mount as /kaggle/input/competitions/<comp>/test, not /kaggle/input/<comp>/test.)"""
    root = Path("/kaggle/input")
    cands = [d for d in root.rglob("test") if d.is_dir() and any(d.glob("*.zarr"))]
    # prefer the one NOT inside our code bundle (bundle has no *.zarr, so cands are competition dirs)
    if not cands:
        # fallback: any dir with *.zarr that isn't the bundle
        zdirs = {p.parent for p in root.rglob("*.zarr")}
        cands = [d for d in zdirs if "repo" not in d.parts]
    assert cands, f"no test .zarr found under {root}: {[p.name for p in root.iterdir()] if root.exists() else 'none'}"
    print("test dir detected:", cands[0], flush=True)
    return cands[0]


def discover_bundle():
    """Find the code+wheels bundle among /kaggle/input/* by content (robust to mount name)."""
    root = Path("/kaggle/input")
    inputs = sorted(root.iterdir()) if root.exists() else []
    print("INPUTS under /kaggle/input:", [p.name for p in inputs], flush=True)
    for p in inputs:
        try:
            if list(p.rglob("predict_unet_transformer.py")) or list(p.rglob("*.whl")):
                print("bundle detected at:", p, flush=True)
                return p
        except Exception:
            pass
    raise FileNotFoundError(f"code/wheels bundle not found among inputs: {[p.name for p in inputs]}")


def sh(cmd, **kw):
    print("+", " ".join(map(str, cmd)), flush=True)
    return subprocess.run(cmd, check=True, **kw)


def find(root, pattern):
    hits = glob.glob(str(Path(root) / "**" / pattern), recursive=True)
    return hits


def main():
    TEST_DIR = discover_test_dir()
    BUNDLE_IN = discover_bundle()
    # locate pieces IN PLACE (read-only input) — do NOT copy 308MB into /kaggle/working
    predict_py = find(BUNDLE_IN, "predict_unet_transformer.py")
    assert predict_py, f"predict_unet_transformer.py not found under {BUNDLE_IN}"
    repo_in = Path(predict_py[0]).parent.parent      # repo/ = parent of scripts/
    wheels = find(BUNDLE_IN, "*.whl")
    weights = find(BUNDLE_IN, "edge_predictor_best.pth")
    assert weights, "edge_predictor_best.pth not found"
    weights = Path(weights[0])
    print(f"repo_in={repo_in} | wheels={len(wheels)} | weights={weights}", flush=True)

    # 1) offline deps — install ONLY the missing ILP/IO stack (tracksdata, geff) from local wheels.
    # Do NOT install all 62 wheels: that clobbers Kaggle's numpy/scipy and can break torch's ABI.
    # pip resolves tracksdata's deps (ilpy, pyscipopt, polars, ...) from find-links, keeping Kaggle's
    # already-satisfied numpy/scipy/torch/zarr.
    if wheels:
        sh([sys.executable, "-m", "pip", "install", "--no-index", "--no-build-isolation",
            "--find-links", str(Path(wheels[0]).parent), "tracksdata", "geff"])

    # 2) copy ONLY the small repo (code, ~0.5MB) to a writable location; predictions write there
    repo = WORK / "repo"; shutil.rmtree(repo, ignore_errors=True)
    shutil.copytree(repo_in, repo)
    scripts_dir, src_dir = repo / "scripts", repo / "src"
    (repo / "predictions").mkdir(exist_ok=True)

    # 3) splits.json = every test dataset
    test_names = sorted(p.name for p in TEST_DIR.glob("*.zarr"))
    print(f"test datasets: {len(test_names)}", flush=True)
    (repo / "infer_splits.json").write_text(json.dumps([{"split": 0, "train": [], "test": test_names}]))

    # 4) run the pipeline at our 0.8392 settings
    env = {**os.environ, "PYTHONPATH": "src", "USER": "kaggle",
           "USE_SELF_ATTN": "0", "POOL_KERNEL_UM": "3.0", "EDGE_THRESHOLD": "0.3",
           "CELLMOT_DATA_DIR": str(TEST_DIR)}
    sh([sys.executable, str(scripts_dir / "predict_unet_transformer.py"),
        "--data-dir", str(TEST_DIR), "--splits", "infer_splits.json", "--split", "0",
        "--weights", str(weights), "--method", "submission", "--unet-batch-size", "4",
        "--det-threshold", "0.998", "--ilp-edge-weight", "-1.0",
        "--ilp-appearance-weight", "0.3", "--ilp-disappearance-weight", "0.3",
        "--ilp-division-weight", "1.0", "--use-ilp"],
       cwd=repo, env=env)

    # 5) geff -> submission.csv
    sys.path.insert(0, str(src_dir))
    from submission import from_geff_dir
    pred_dir = repo / "predictions" / "kaggle" / "submission" / "split_0"
    datasets = [n[:-5] for n in test_names]
    df = from_geff_dir(pred_dir, datasets, str(WORK / "submission.csv"))
    print(f"submission.csv: {len(df)} rows, {df.dataset.nunique()} datasets "
          f"({(df.row_type=='node').sum()} nodes / {(df.row_type=='edge').sum()} edges)", flush=True)

    # keep the output lean: submission.csv only (remove the working repo copy + geffs)
    shutil.rmtree(repo, ignore_errors=True)
    shutil.rmtree(STAGE, ignore_errors=True)
    print("output cleaned — submission.csv is the sole large output", flush=True)


if __name__ == "__main__":
    main()
