"""Contact sheet for the diagram library — render every artifact type on one page.

The artifact library is meant to compound: each video adds or sharpens a visual, and later videos
reuse it. That only works if the whole set can be reviewed at a glance and regressions are obvious,
so this renders every type with representative data and stacks them into one reviewable image.

    python tools/artifact_gallery.py            # -> gallery/_artifacts/contact_sheet.png
    python tools/artifact_gallery.py --check    # exit 1 if any type warns (aspect/legibility)

Add a diagram type to diagrams.py, add one entry to SAMPLES here, and it is covered from then on.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import diagrams as D  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "gallery" / "_artifacts"


def build(out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    made = []

    made.append(D.cells(out_dir / "01_cells.png", ["apple", "banana", "cherry", "date", "fig"],
                        title="slicing — start included, end excluded",
                        highlight=range(1, 4), labels_neg=True,
                        note="fruits[1:4] -> ['banana', 'cherry', 'date']"))

    made.append(D.variable_box(out_dir / "02_variable_box.png", "name", "Python",
                               title="a variable is a labeled box",
                               note="put a value in once, reuse it by name"))

    made.append(D.substitution(out_dir / "03_substitution.png", 'f"Welcome to {name}"',
                               "{name}", "Python", "Welcome to Python",
                               title="the f-string swap",
                               note="the braces are replaced the moment the line runs"))

    made.append(D.mapping(out_dir / "04_mapping.png",
                          [("red", 3), ("blue", 2), ("green", 1)],
                          title="Counter tallies each word",
                          key_label="word", val_label="count"))

    made.append(D.venn(out_dir / "05_venn.png", {1, 2, 3, 4}, {3, 4, 5, 6},
                       title="a & b — intersection", op="a & b  ->  {3, 4}", result={3, 4}))

    made.append(D.stack(out_dir / "06_stack.png",
                        ["fact(4)", "fact(3)", "fact(2)", "fact(1)"],
                        title="recursion builds a stack of frames",
                        note="each call waits for the one above it"))

    made.append(D.tree(out_dir / "07_tree.png",
                       [("Animal", "Dog"), ("Animal", "Cat"), ("Dog", "Puppy")], "Animal",
                       title="inheritance — Dog and Cat both extend Animal"))

    made.append(D.bars(out_dir / "08_bars.png",
                       ["loop", "comprehension", "map"], [0.91, 0.52, 0.60],
                       title="which is actually faster", highlight=1,
                       xlabel="seconds for 10k runs"))

    tb = "\n".join([
        "Traceback (most recent call last):",
        '  File "prices.py", line 7, in <module>',
        "    apply_discount(100, 150)",
        "AssertionError: bad percent: 150",
    ])
    made.append(D.traceback_shot(out_dir / "09_traceback.png", tb))
    return made


def main():
    made = build(OUT)
    from PIL import Image
    ims = [Image.open(p) for p in made]
    w = max(i.width for i in ims)
    h = sum(i.height for i in ims) + 12 * (len(ims) - 1)
    sheet = Image.new("RGB", (w, h), D.BG)
    y = 0
    for i in ims:
        sheet.paste(i, ((w - i.width) // 2, y))
        y += i.height + 12
    path = OUT / "contact_sheet.png"
    sheet.save(path)
    print(f"\n{len(made)} artifact types -> {path}  ({sheet.size[0]}x{sheet.size[1]})")
    if D._WARNED:
        print(f"{len(D._WARNED)} legibility warning(s)")
        if "--check" in sys.argv:
            sys.exit(1)
    else:
        print("no legibility warnings — every type fits its zone")


if __name__ == "__main__":
    main()
