"""shorts_builder data-wise test — GROUND TRUTH: the 3-input spec (code, transcript, language)
must yield a real video file with >0 frames, timed caption segments, and correct prop assembly.
Uses the PIL fallback (SHORTS_FORCE_PIL) so the test is offline/Node-free; the Remotion path is
exercised manually (needs Chromium download)."""
import os
import sys
import tempfile
from pathlib import Path

COMP = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(COMP)); sys.path.insert(0, str(COMP / "tools" / "researchpapers"))
os.environ["SHORTS_FORCE_PIL"] = "1"

from fleet_agents.shorts_builder import build_props, parse_transcript, run_shorts  # noqa: E402

CODE = 'fn main() {\n    let x: i32 = 42; // answer\n    println!("{}", x);\n}\n'


def test_transcript_parsing():
    timed = parse_transcript("0|2|first line\n2|4.5|second line")
    assert timed == [{"start": 0.0, "end": 2.0, "text": "first line"},
                     {"start": 2.0, "end": 4.5, "text": "second line"}]
    plain = build_props("x" * 60, "a\nb\nc", "python", cps=30)["segments"]  # 60ch/30cps+2s tail = 4s
    assert len(plain) == 3 and plain[0]["start"] == 0.0 and abs(plain[-1]["end"] - 4.0) < 0.1
    assert plain[1]["start"] == plain[0]["end"]                     # contiguous, no caption gaps


def test_props():
    p = build_props(CODE, "0|1|Rust entry point", "Rust", cps=25)
    assert p["language"] == "rust" and p["title"].endswith(".rs") and p["cps"] == 25.0
    assert p["code"] == CODE and p["segments"][0]["text"] == "Rust entry point"


def test_render_end_to_end():
    with tempfile.TemporaryDirectory() as td:
        out = str(Path(td) / "short.mp4")
        status, data, _, msg = run_shorts(
            {"spec": {"code": CODE, "transcript": "0|1|Rust entry point\n1|2|Print the answer",
                      "language": "rust", "out": out, "fps": 8, "size": "180x320", "cps": 60}},
            "test")
        assert status == "done", msg
        p = Path(data["path"])
        assert p.exists() and p.stat().st_size > 1000, f"empty render: {p}"
        assert data["renderer"] == "pil"                            # forced offline path
        assert len(data["props"]["segments"]) == 2


if __name__ == "__main__":
    test_transcript_parsing(); test_props(); test_render_end_to_end()
    print("OK shorts_builder")
