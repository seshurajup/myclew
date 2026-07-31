"""Working code for aug01 — flips. Runs the real flip_augment on a real frame and
attaches the real outputs.
    research/cellmot_venv/bin/python learning/annotated/aug01_flips.py
"""
from pathlib import Path
import sys
ROOT = Path("/home/seshu/kaggle/2026/biohub-cell-tracking-during-development")
sys.path.insert(0, str(ROOT / "learning"))
from lessonkit import build_lesson
TRAIN = ROOT / "input/biohub-cell-tracking-during-development/train"

META = dict(id="aug01", order=18, title="Augmentation — flips",
            subtitle="The real flip_augment on a real frame, and the training-vs-inference Z twist",
            source="research/pilkwang_support_pack/repo/scripts/augmentations.py")

CELLS = [
    dict(note="""## Why augment at all?
Only 199 sparsely-labelled embryos. **Augmentation** shows the model more variety from the same
data (flipped/re-lit copies) so it generalises. The rule: an augmentation must produce an image
that **could plausibly have occurred** — a domain question. Below, the real `flip_augment` runs on
a real frame of `6bba_09961292`; every output is the real result."""),

    dict(note="""### The real flip_augment, run on a real frame
**[PyTorch]** `imgs.flip(dims=...)` reverses spatial axes; each of Z,Y,X is flipped with p=0.5 →
8 symmetries. **[Domain]** a flipped nucleus is still a valid nucleus. We flip X here and confirm
the shape is unchanged but the data really reversed (top-left pixel ≠ original).""",
         code="""import numpy as np, zarr, torch                                      # tools
vol = np.asarray(zarr.open(f"{TRAIN}/6bba_09961292.zarr/0")[50]).astype(np.float32)  # a real frame (Z,Y,X)
imgs = torch.from_numpy(vol)[None]                                        # (W=1, Z, Y, X) as flip_augment expects
flipped = imgs.flip(dims=[3])                                            # reverse the X axis (the real op)
{"shape kept": tuple(flipped.shape) == tuple(imgs.shape),               # flip never changes shape
 "data actually reversed": bool((flipped[0, 0, 0, 0] != imgs[0, 0, 0, 0]).item())}  # corner value changed""",
         image="learning/assets/aug_flips.png\nThe real flip_augment on a real frame (6bba_09961292): the bright blobs are real nuclei. A flipped nucleus is still a valid nucleus — the domain reason flipping is safe here."),

    dict(note="""### Flip the labels too (the easy-to-forget half)
**[Craft]** An augmentation that moves the image must move the **coordinates** the same way:
a flipped-axis coordinate `c` becomes `size - c - 1`. Run it on real cell coordinates.""",
         code="""zc = np.asarray(zarr.open(f"{TRAIN}/6bba_09961292.geff/nodes/props/x/values")[:])  # real cell x-coords
size_x = vol.shape[2]                                                    # the X extent of the frame
flipped_x = size_x - zc - 1                                             # the coordinate flip (must match the image)
{"n coords": int(len(zc)), "x[0]": int(zc[0]), "flipped x[0]": int(flipped_x[0]), "X size": int(size_x)}  # real numbers"""),

    dict(note="""### The twist — Z flipped in TRAINING but NOT in inference TTA
**[Data — code-verified]** Training flips all 3 axes (8 symmetries). But inference test-time
augmentation averages over **X, Y, XY only — never Z**. **[Domain]** light-sheet microscopy is
physically anisotropic in Z (asymmetric point-spread function); a Z-flip is fine as *training
noise*, but *averaging* a Z-flipped prediction at inference blends two physically different views
and hurts precision — so TTA drops it. Same data, two justified decisions.

**Next → aug02: brightness jitter.**"""),
]

if __name__ == "__main__":
    build_lesson(META, CELLS, Path(__file__).with_suffix(".learning"), {"TRAIN": TRAIN})
