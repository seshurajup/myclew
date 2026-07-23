"""center_train_test — data-wise verifier for the center-train agent.

Full training is too slow for a self-test, so this asserts the WRAPPER:
  1. the wrapped center-detector script imports cleanly,
  2. `--help` succeeds (rc==0),
  3. the agent builds a well-formed argv (incl. --resume) with the requested hyperparameters.
"""
import os, subprocess, sys
COMP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, COMP); sys.path.insert(0, os.path.join(COMP, "tools", "researchpapers"))
from fleet_agents import center_train as A


def _run():
    print("=== CENTER-TRAIN DATA-WISE VERIFIER ===")
    env = A._env()
    checks = {}

    argv = A.CenterTrain().build_argv({"data_dir": "/tmp/data", "output_dir": "/tmp/out",
                                       "epochs": 4, "movie_limit": 2, "resume": True,
                                       "base_channels": 24, "batch_size": 4})
    j = " ".join(argv)
    checks["script_in_argv"] = str(A._SCRIPT) in j
    checks["data_dir_passed"] = "--data-dir /tmp/data" in j
    checks["output_dir_passed"] = "--output-dir /tmp/out" in j
    checks["epochs_passed"] = "--epochs 4" in j
    checks["movie_limit_passed"] = "--movie-limit 2" in j
    checks["resume_passed"] = "--resume" in j

    imp = subprocess.run([A._py(), "-c", "import train_full_frame_center_detector"],
                         capture_output=True, text=True, cwd=str(A._SCRIPT.parent), env=env, timeout=180)
    checks["script_imports"] = imp.returncode == 0
    if imp.returncode != 0:
        print("  import stderr:", imp.stderr.strip().splitlines()[-3:])

    hp = subprocess.run([A._py(), str(A._SCRIPT), "--help"], capture_output=True, text=True,
                        cwd=str(A._SCRIPT.parent), env=env, timeout=180)
    checks["help_ok"] = hp.returncode == 0

    for k, v in checks.items():
        print(f"  {'OK' if v else 'X'} {k}")
    ok = all(checks.values())
    print(f"=== center-train: {'PASS' if ok else 'FAIL'} ===")
    return ok


if __name__ == "__main__":
    try:
        sys.exit(0 if _run() else 1)
    except Exception as e:  # noqa: BLE001
        print(f"  X ERROR: {type(e).__name__}: {e}"); sys.exit(1)
