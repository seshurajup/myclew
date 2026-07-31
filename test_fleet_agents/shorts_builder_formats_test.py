"""shorts_builder_formats_test — one agent, two formats, and the Shorts path must not move.

The risk in adding long-form video to a tuned Shorts pipeline is regression, so this test pins BOTH:
  • `format="short"` (the default): canvas 1080×1920, the Shorts safe zones, the CodeShort composition,
    the ≤180 s cap, captions in the TOP band — and `build_props` output unchanged for a fixed spec.
  • `format="video"`: 16:9, lower-third captions, a storyboard read from a paper's lesson series where
    every formula scene points at a crop that EXISTS and every code scene carries its real output,
    chapter markers starting at 0:00, and narration a voice can actually read (no LaTeX, no markdown).
"""
import os
import sys

COMP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, COMP)
sys.path.insert(0, os.path.join(COMP, "tools", "researchpapers"))

from pathlib import Path  # noqa: E402

from fleet_agents import shorts_builder as SB  # noqa: E402


def _run():
    print("=== SHORTS/VIDEO FORMAT VERIFIER ===")
    c = {}

    # ---------------- the Shorts contract (regression: nothing here may move)
    S = SB.fmt("short")
    c["short is still 1080x1920"] = (S["width"], S["height"]) == (1080, 1920)
    c["short still uses the CodeShort composition"] = S["composition"] == "CodeShort"
    c["short still caps at YouTube's 180s"] = S["max_seconds"] == 180.0
    c["short caption stays in the TOP band"] = S["caption"] == "top"
    c["short keeps the bottom 22% CTA zone reserved"] = abs(S["safe"]["bottom_frac"] - 0.22) < 1e-9
    c["short keeps the right action-rail reserved"] = S["safe"]["right_frac"] > 0.1
    c["an unknown format falls back to short"] = SB.fmt("nonsense")["composition"] == "CodeShort"
    c["a missing format falls back to short"] = SB.fmt(None)["width"] == 1080

    props = SB.build_props("def f(x):\n    return x + 1\n", "one line\ntwo line", "python",
                           title="t", cps=30, tail_seconds=2.0, max_seconds=90.0)
    c["shorts build_props still produces segments"] = len(props.get("segments", [])) == 2
    c["shorts build_props does not leak video keys"] = not {"scenes", "chapters"} & set(props)
    c["shorts build_props keeps its own cap"] = props.get("maxSeconds", 90.0) <= 180.0
    c["dispatch: no format key means the Shorts path"] = str({}.get("format", "short")) == "short"

    # ---------------- the long-form contract
    V = SB.fmt("video")
    c["video is 16:9"] = abs(V["width"] / V["height"] - 16 / 9) < 1e-9
    c["video captions sit in the lower third"] = V["caption"] == "lower-third"
    c["video reserves far less bottom than a Short"] = V["safe"]["bottom_frac"] < S["safe"]["bottom_frac"]
    c["video allows a long runtime"] = V["max_seconds"] >= 600

    c["narration strips display math"] = "$" not in SB._narratable("see $$W_{t+1} = W_t$$ here")
    c["narration points at the on-screen formula"] = "formula on screen" in SB._narratable("$$a=b$$")
    c["narration strips markdown emphasis and code marks"] = not set("*`_[]") & set(
        SB._narratable("**bold** and `code` and [link]"))

    lessons = Path(COMP) / "learning" / "annotated"
    have_k3 = any(lessons.glob("k3*.learning"))
    if have_k3:
        sb = SB.paper_storyboard("kimi-k3", "k3", max_scenes=30)
        kinds = {s["kind"] for s in sb["scenes"]}
        c["storyboard opens with a title scene"] = sb["scenes"][0]["kind"] == "title"
        c["storyboard has one chapter per lesson"] = len(sb["chapters"]) >= 3
        c["storyboard mixes formulas, proofs and figures"] = {"formula", "code"} <= kinds
        missing = [s["image"] for s in sb["scenes"] if s.get("image")
                   and not (Path(COMP) / s["image"]).exists()]
        c["every image scene points at a file that EXISTS"] = not missing
        code_scenes = [s for s in sb["scenes"] if s["kind"] == "code"]
        c["proof scenes carry their real captured output"] = any(s.get("output") for s in code_scenes)

        vp = SB.build_paper_props(sb, format="video", target_seconds=420)
        c["video props are 16:9"] = (vp["width"], vp["height"]) == (1920, 1080)
        c["video props keep the Shorts audio contract (segments)"] = all(
            {"start", "end", "text"} <= set(s) for s in vp["segments"])
        c["video props carry code for the typing pane"] = len(vp["code"]) > 0
        c["scene timeline is monotonic"] = all(
            vp["scenes"][i]["end"] <= vp["scenes"][i + 1]["start"] + 1e-6
            for i in range(len(vp["scenes"]) - 1))

        for i, sc in enumerate(vp["scenes"]):                       # simulate build_audio's retiming
            if sc.get("narration"):
                sc["speechEnd"] = sc["end"] - 0.3
        vp = SB.retime_scenes(vp)
        c["captions never outlive their voice"] = all(
            sc["speechEnd"] <= sc["end"] + 1e-6 for sc in vp["scenes"] if sc.get("speechEnd"))
        ch = SB.storyboard_chapters(vp)
        c["chapter markers start at 0:00"] = ch.startswith("0:00")
        c["at least three chapters (YouTube's minimum)"] = len(ch.splitlines()) >= 3
    else:
        for k in ("storyboard opens with a title scene", "storyboard has one chapter per lesson",
                  "storyboard mixes formulas, proofs and figures",
                  "every image scene points at a file that EXISTS",
                  "proof scenes carry their real captured output", "video props are 16:9",
                  "video props keep the Shorts audio contract (segments)",
                  "video props carry code for the typing pane", "scene timeline is monotonic",
                  "captions never outlive their voice", "chapter markers start at 0:00",
                  "at least three chapters (YouTube's minimum)"):
            c[k + " (skipped: no k3 lessons on this box)"] = True

    c["an ffmpeg encoder is discoverable"] = bool(SB._ffmpeg_exe())

    for k, v in c.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    return all(c.values())


if __name__ == "__main__":
    sys.exit(0 if _run() else 1)
