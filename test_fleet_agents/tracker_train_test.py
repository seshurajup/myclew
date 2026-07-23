"""tracker_train_test — data-wise verifier for the tracker-train agent.

A full train is far too slow for a self-test, so this asserts the WRAPPER is correct:
  1. the wrapped script imports cleanly (its deps resolve on the built PYTHONPATH),
  2. `python train_unet_transformer.py --help` succeeds (rc==0),
  3. the agent builds a well-formed argv carrying every requested hyperparameter.
"""
import os, subprocess, sys
COMP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, COMP); sys.path.insert(0, os.path.join(COMP, "tools", "researchpapers"))
from fleet_agents import tracker_train as A


def _run():
    print("=== TRACKER-TRAIN DATA-WISE VERIFIER ===")
    env = A._env()
    checks = {}

    argv = A.TrackerTrain().build_argv({"epochs": 3, "lr": 2e-4, "batch_size": 8, "split": "0",
                                        "unet_weights": "/tmp/w.pth", "max_iters": 2,
                                        "data_dir": "/tmp/data"})
    j = " ".join(argv)
    checks["script_in_argv"] = str(A._SCRIPT) in j
    checks["epochs_passed"] = "--epochs 3" in j
    checks["lr_passed"] = "--lr 0.0002" in j
    checks["batch_passed"] = "--batch-size 8" in j
    checks["warmstart_passed"] = "--unet-weights /tmp/w.pth" in j
    checks["maxiters_passed"] = "--max-iters 2" in j

    imp = subprocess.run([A._py(), "-c", "import train_unet_transformer"],
                         capture_output=True, text=True, cwd=str(A._REPO / "scripts"), env=env, timeout=180)
    checks["script_imports"] = imp.returncode == 0
    if imp.returncode != 0:
        print("  import stderr:", imp.stderr.strip().splitlines()[-3:])

    hp = subprocess.run(argv[:2] + ["--help"], capture_output=True, text=True,
                        cwd=str(A._REPO), env=env, timeout=180)
    checks["help_ok"] = hp.returncode == 0

    for k, v in checks.items():
        print(f"  {'OK' if v else 'X'} {k}")
    ok = all(checks.values())
    print(f"=== tracker-train: {'PASS' if ok else 'FAIL'} ===")
    return ok


if __name__ == "__main__":
    try:
        sys.exit(0 if _run() else 1)
    except Exception as e:  # noqa: BLE001
        print(f"  X ERROR: {type(e).__name__}: {e}"); sys.exit(1)
