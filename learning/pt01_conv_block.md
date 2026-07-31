# pt01 — `nn.Module` & the conv block

> The detector is a stack of **conv blocks**. Master this one block and the whole U-Net is just this repeated. We use the *exact* `_conv_block` from `temporal_unet.py`.

## The real code
```python
def _conv_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm3d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm3d(out_channels),
        nn.ReLU(inplace=True),
    )
```

It's **Conv → Norm → ReLU**, twice. That triplet is the atom of almost every vision network. Let's take it apart.

---
## `nn.Sequential` — a container that chains layers
**[PyTorch]** `nn.Sequential(a, b, c)` makes one module that runs `c(b(a(x)))`. Calling `block(x)` feeds `x` through all six layers in order.

**[Craft]** Notice it's wrapped in a *function* `_conv_block(in, out)`. This block is reused ~6 times (every encoder + decoder stage). A grandmaster never copy-pastes the six lines six times — one factory function, called with different channel counts. DRY (Don't Repeat Yourself), one place to change.

---
## `nn.Conv3d` — the 3-D convolution (the core)
**[PyTorch]** A conv slides a small learned filter over the input and, at each position, computes a weighted sum of the neighbourhood. `Conv3d(1, 32, kernel_size=3)` means: input has **1 channel**, output has **32 channels** (32 different filters), each filter is **3×3×3**.

**[Domain]** Why **3-D** (not 2-D)? A cell nucleus is a ~5–10 µm **blob that spans several z-planes** in the light-sheet stack. A 2-D filter would only see one slice and miss the 3-D shape. A 3×3×3 filter looks at a small cube — exactly the scale of a local piece of a nucleus — so the network learns 'bright compact blob = cell'.

**[Data]** Why `kernel_size=3, padding=1`? `padding=1` adds a 1-voxel border so the output is the **same spatial size** as the input. We can see it on a real crop:
```
input  x.shape = (1, 1, 64, 32, 32)   # (batch, channels=1, Z, Y, X)
output y.shape = (1, 32, 64, 32, 32)   # (batch, channels=32, Z, Y, X)
```
- The spatial dims **(64, 32, 32) stayed (64, 32, 32)** — padding preserved them. **[Domain]** we need a prediction at *every voxel* (dense detection of every cell), so we must not shrink the volume.
- Channels went **1 → 32**: the block now describes each voxel with 32 learned features instead of 1 raw intensity.

**[Craft]** Why `bias=False`? Each `Conv3d` is immediately followed by `BatchNorm3d`, which subtracts the mean — so a bias term would be cancelled out and is redundant. Dropping it saves parameters. This is a standard, deliberate Conv+BN idiom; seeing `bias=False` tells an expert 'a BatchNorm follows'.

---
## `nn.BatchNorm3d` — normalise activations
**[PyTorch]** BatchNorm rescales each channel to ~zero-mean/unit-variance across the batch, then applies a learned scale+shift. It makes training faster and more stable.

**[Data/Domain]** Why it helps *here*: different embryos and developmental stages have different brightness and contrast (you'll measure this in the augmentation lesson). Even after input normalisation, activations drift. BatchNorm keeps the signal in a sane range at every layer, so the detector doesn't waste capacity on brightness differences and can focus on *shape* (is this a nucleus?).

---
## `nn.ReLU` — the nonlinearity
**[PyTorch]** `ReLU(x) = max(x, 0)`. Without a nonlinearity between convs, stacking them would collapse to a single linear map — no matter how deep. ReLU is what lets the network learn non-trivial patterns.

**[Craft]** `inplace=True` overwrites the input tensor instead of allocating a new one — a small memory saving that matters when volumes are big (3-D data is heavy).

---
## Why the block appears TWICE (Conv-BN-ReLU ×2)
**[PyTorch/Domain]** Two 3×3×3 convs in a row give each output voxel a **5×5×5 receptive field** (it 'sees' further) and add a second nonlinearity, so the block can represent richer shapes than a single conv — while using smaller, cheaper 3×3 filters than one big 5×5. Stacking small convs is a core deep-learning trick.

**[Craft]** This tiny block already has **28,640 learnable parameters** (the filter weights + BN scale/shift). The full U-Net just stacks blocks like this with pooling between — which is the next lesson.

---
## `nn.Module` — the base class behind all of it
**[PyTorch]** Every layer above (`Conv3d`, `BatchNorm3d`, …) and `nn.Sequential` itself is an `nn.Module`. A Module (1) registers its **parameters** so the optimiser can find them, and (2) defines a **`forward(x)`** so you can just call `block(x)`. When we build the full `TemporalUNet3D(nn.Module)` next, we're subclassing this.

```python
block = conv_block(1, 32)          # build it
y = block(x)                       # calls forward(): runs the 6 layers
sum(p.numel() for p in block.parameters())  # -> the weights the optimiser trains
```

---
## Recap
| layer | [PyTorch] role | key 'why' |
|---|---|---|
| `nn.Sequential` | chain layers | factored into `_conv_block()` — DRY [Craft] |
| `nn.Conv3d(k=3,pad=1,bias=False)` | learn local 3-D filters | 3-D = nuclei span z; pad keeps size; no bias before BN |
| `nn.BatchNorm3d` | stabilise activations | robust to per-embryo brightness [Domain] |
| `nn.ReLU(inplace=True)` | nonlinearity | lets depth actually help; inplace saves memory |
| ×2 | bigger receptive field | 5×5×5 view from cheap 3×3 convs |

### Try it yourself
- Change `conv_block(1, 32)` → `conv_block(1, 8)` and re-print `y.shape` — which dim changes?
- Remove `padding=1` (set `padding=0`) — by how much does each spatial dim shrink, and why?
- Feed a bigger crop (e.g. `64:192`) and confirm output spatial dims still match input.

**Next → pt02: the U-Net shape** — `MaxPool3d` (down), `Upsample`/`F.interpolate` (up), and `torch.cat` (skip connections): how stacking these blocks with pooling turns the volume into a per-voxel cell heat-map.
