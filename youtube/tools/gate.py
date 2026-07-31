"""Pre-build gate for one video folder — exit 0 = OK to build, exit 1 = rejected.

Called by build_youtube.sh before any GPU work. Wraps tools/prebuild_check.py (structure +
pace-lock) which in turn enforces tools/retention_rules.py (hook, cold open, tail).

    python tools/gate.py <video_dir>
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prebuild_check import prebuild_check  # noqa: E402

d = Path(sys.argv[1])
probs = prebuild_check(d)
if probs:
    print(f"[gate] {d.parent.name}/{d.name} REJECTED:")
    for p in probs:
        print(f"    - {p}")
    sys.exit(1)
sys.exit(0)
