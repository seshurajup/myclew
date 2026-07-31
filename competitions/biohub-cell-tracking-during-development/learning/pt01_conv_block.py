"""
pt01 — nn.Module & the conv block (the FIRST building block of our detector).

This is NOT an abstract tutorial. It takes the EXACT `_conv_block` from pilkwang's
`temporal_unet.py`, runs it on a real competition volume, and writes a detailed
lesson (`pt01_conv_block.md`) explaining every line across four threads:
  [PyTorch]  what the construct does
  [Data]     why this choice, shown on THIS data
  [Craft]    how a grandmaster writes it (bestfitting-style)
  [Domain]   the cell-biology / microscopy reason it exists

Run:  research/cellmot_venv/bin/python learning/pt01_conv_block.py
"""
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import zarr

ROOT = Path("/home/seshu/kaggle/2026/biohub-cell-tracking-during-development")
TRAIN = ROOT / "input/biohub-cell-tracking-during-development/train"
DS = "6bba_062c8d37"
OUT = Path(__file__).parent / "pt01_conv_block.md"

md = []
def w(s=""): md.append(s)


# ---- THE REAL CODE (verbatim from research/.../models/temporal_unet.py) ----
def conv_block(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm3d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm3d(out_channels),
        nn.ReLU(inplace=True),
    )


def main():
    # --- run the real block on a real (small) volume so we can quote true shapes ---
    z = zarr.open(str(TRAIN / f"{DS}.zarr" / "0"), mode="r")     # (T,Z,Y,X)
    sub = np.asarray(z[0, :, 96:128, 96:128]).astype(np.float32)  # (Z,32,32) crop
    lo, hi = np.quantile(sub, 0.01), np.quantile(sub, 0.99)
    sub = np.clip((sub - lo) / (hi - lo + 1e-6), 0, 1)
    x = torch.from_numpy(sub)[None, None]        # -> (batch=1, channels=1, Z, 32, 32)
    block = conv_block(1, 32).eval()
    with torch.no_grad():
        y = block(x)
    n_params = sum(p.numel() for p in block.parameters())

    # ---------------- write the lesson ----------------
    w("# pt01 — `nn.Module` & the conv block")
    w("")
    w("> The detector is a stack of **conv blocks**. Master this one block and the whole "
      "U-Net is just this repeated. We use the *exact* `_conv_block` from "
      "`temporal_unet.py`.")
    w("")
    w("## The real code")
    w("```python")
    w("def _conv_block(in_channels, out_channels):")
    w("    return nn.Sequential(")
    w("        nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),")
    w("        nn.BatchNorm3d(out_channels),")
    w("        nn.ReLU(inplace=True),")
    w("        nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),")
    w("        nn.BatchNorm3d(out_channels),")
    w("        nn.ReLU(inplace=True),")
    w("    )")
    w("```")
    w("")
    w("It's **Conv → Norm → ReLU**, twice. That triplet is the atom of almost every vision "
      "network. Let's take it apart.")
    w("")
    w("---")
    w("## `nn.Sequential` — a container that chains layers")
    w("**[PyTorch]** `nn.Sequential(a, b, c)` makes one module that runs `c(b(a(x)))`. "
      "Calling `block(x)` feeds `x` through all six layers in order.")
    w("")
    w("**[Craft]** Notice it's wrapped in a *function* `_conv_block(in, out)`. This block is "
      "reused ~6 times (every encoder + decoder stage). A grandmaster never copy-pastes the "
      "six lines six times — one factory function, called with different channel counts. "
      "DRY (Don't Repeat Yourself), one place to change.")
    w("")
    w("---")
    w("## `nn.Conv3d` — the 3-D convolution (the core)")
    w("**[PyTorch]** A conv slides a small learned filter over the input and, at each "
      "position, computes a weighted sum of the neighbourhood. `Conv3d(1, 32, kernel_size=3)` "
      "means: input has **1 channel**, output has **32 channels** (32 different filters), each "
      "filter is **3×3×3**.")
    w("")
    w("**[Domain]** Why **3-D** (not 2-D)? A cell nucleus is a ~5–10 µm **blob that spans "
      "several z-planes** in the light-sheet stack. A 2-D filter would only see one slice and "
      "miss the 3-D shape. A 3×3×3 filter looks at a small cube — exactly the scale of a "
      "local piece of a nucleus — so the network learns 'bright compact blob = cell'.")
    w("")
    w("**[Data]** Why `kernel_size=3, padding=1`? `padding=1` adds a 1-voxel border so the "
      "output is the **same spatial size** as the input. We can see it on a real crop:")
    w("```")
    w(f"input  x.shape = {tuple(x.shape)}   # (batch, channels=1, Z, Y, X)")
    w(f"output y.shape = {tuple(y.shape)}   # (batch, channels=32, Z, Y, X)")
    w("```")
    w(f"- The spatial dims **{tuple(x.shape[2:])} stayed {tuple(y.shape[2:])}** — padding "
      f"preserved them. **[Domain]** we need a prediction at *every voxel* (dense detection "
      f"of every cell), so we must not shrink the volume.")
    w(f"- Channels went **1 → 32**: the block now describes each voxel with 32 learned "
      f"features instead of 1 raw intensity.")
    w("")
    w("**[Craft]** Why `bias=False`? Each `Conv3d` is immediately followed by `BatchNorm3d`, "
      "which subtracts the mean — so a bias term would be cancelled out and is redundant. "
      "Dropping it saves parameters. This is a standard, deliberate Conv+BN idiom; seeing "
      "`bias=False` tells an expert 'a BatchNorm follows'.")
    w("")
    w("---")
    w("## `nn.BatchNorm3d` — normalise activations")
    w("**[PyTorch]** BatchNorm rescales each channel to ~zero-mean/unit-variance across the "
      "batch, then applies a learned scale+shift. It makes training faster and more stable.")
    w("")
    w("**[Data/Domain]** Why it helps *here*: different embryos and developmental stages have "
      "different brightness and contrast (you'll measure this in the augmentation lesson). "
      "Even after input normalisation, activations drift. BatchNorm keeps the signal in a "
      "sane range at every layer, so the detector doesn't waste capacity on brightness "
      "differences and can focus on *shape* (is this a nucleus?).")
    w("")
    w("---")
    w("## `nn.ReLU` — the nonlinearity")
    w("**[PyTorch]** `ReLU(x) = max(x, 0)`. Without a nonlinearity between convs, stacking "
      "them would collapse to a single linear map — no matter how deep. ReLU is what lets the "
      "network learn non-trivial patterns.")
    w("")
    w("**[Craft]** `inplace=True` overwrites the input tensor instead of allocating a new "
      "one — a small memory saving that matters when volumes are big (3-D data is heavy).")
    w("")
    w("---")
    w("## Why the block appears TWICE (Conv-BN-ReLU ×2)")
    w("**[PyTorch/Domain]** Two 3×3×3 convs in a row give each output voxel a **5×5×5 "
      "receptive field** (it 'sees' further) and add a second nonlinearity, so the block can "
      "represent richer shapes than a single conv — while using smaller, cheaper 3×3 filters "
      "than one big 5×5. Stacking small convs is a core deep-learning trick.")
    w("")
    w(f"**[Craft]** This tiny block already has **{n_params:,} learnable parameters** "
      f"(the filter weights + BN scale/shift). The full U-Net just stacks blocks like this "
      f"with pooling between — which is the next lesson.")
    w("")
    w("---")
    w("## `nn.Module` — the base class behind all of it")
    w("**[PyTorch]** Every layer above (`Conv3d`, `BatchNorm3d`, …) and `nn.Sequential` "
      "itself is an `nn.Module`. A Module (1) registers its **parameters** so the optimiser "
      "can find them, and (2) defines a **`forward(x)`** so you can just call `block(x)`. When "
      "we build the full `TemporalUNet3D(nn.Module)` next, we're subclassing this.")
    w("")
    w("```python")
    w("block = conv_block(1, 32)          # build it")
    w("y = block(x)                       # calls forward(): runs the 6 layers")
    w("sum(p.numel() for p in block.parameters())  # -> the weights the optimiser trains")
    w("```")
    w("")
    w("---")
    w("## Recap")
    w("| layer | [PyTorch] role | key 'why' |")
    w("|---|---|---|")
    w("| `nn.Sequential` | chain layers | factored into `_conv_block()` — DRY [Craft] |")
    w("| `nn.Conv3d(k=3,pad=1,bias=False)` | learn local 3-D filters | 3-D = nuclei span z; pad keeps size; no bias before BN |")
    w("| `nn.BatchNorm3d` | stabilise activations | robust to per-embryo brightness [Domain] |")
    w("| `nn.ReLU(inplace=True)` | nonlinearity | lets depth actually help; inplace saves memory |")
    w("| ×2 | bigger receptive field | 5×5×5 view from cheap 3×3 convs |")
    w("")
    w("### Try it yourself")
    w("- Change `conv_block(1, 32)` → `conv_block(1, 8)` and re-print `y.shape` — which dim changes?")
    w("- Remove `padding=1` (set `padding=0`) — by how much does each spatial dim shrink, and why?")
    w("- Feed a bigger crop (e.g. `64:192`) and confirm output spatial dims still match input.")
    w("")
    w("**Next → pt02: the U-Net shape** — `MaxPool3d` (down), `Upsample`/`F.interpolate` (up), "
      "and `torch.cat` (skip connections): how stacking these blocks with pooling turns the "
      "volume into a per-voxel cell heat-map.")

    OUT.write_text("\n".join(md) + "\n")
    print(f"input {tuple(x.shape)} -> output {tuple(y.shape)}, params={n_params:,}")
    print(f"wrote detailed lesson -> {OUT}  ({len(md)} lines)")


if __name__ == "__main__":
    main()
