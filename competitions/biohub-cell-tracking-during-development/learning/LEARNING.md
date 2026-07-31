# Learning Log — Biohub Cell Tracking

A running notebook of what we learn about the data, one step at a time.
Each step has a small Python script + its output, kept in this `learning/` folder.

- Data: `input/biohub-cell-tracking-during-development/train/` — **199 embryos**,
  each stored as a `.geff` tracking graph (+ a `.zarr` image volume).
- In a `.geff`, every **node = one annotated cell at one frame**; node property
  `t` is the frame index (0..99). So **cells-in-frame-`t` = number of nodes with that `t`.**

---

## Step 01 — How many cells per frame?

**Script:** `01_cells_per_frame.py` → **Table:** `01_cells_per_frame_per_dataset.csv`

Two clearly different groups (the 4-char prefix of each filename):

| group | # embryos | cells/frame (max) | cells/frame (mean) | total cells / embryo (median) | character |
|-------|-----------|-------------------|--------------------|-------------------------------|-----------|
| `44b6` | 71  | 16 | 3.0 | 214 | **sparse** group |
| `6bba` | 128 | 33 | 9.0 | 827 | **dense** group |

**Overall ranges (all 199 embryos):**
- frames per embryo: **40 – 100** (median 100)
- total cells per embryo: **50 – 1950** (median 659)
- **cells in a single frame: 1 – 33** (densest = `6bba_09961292`)

**Global histogram — how many of the 18,933 frames fall in each range:**

```
   1-10 cells/frame : 14820 frames  (78%)  ##################################################
  11-25 cells/frame :  4045 frames  (21%)  #############
  26-50 cells/frame :    68 frames  (0.4%)
    >50 cells/frame :     0 frames
```

> ⚠️ **Caveat:** these are **annotated ground-truth graph nodes**, not the true
> biological cell count. `44b6` embryos showing "1–2 cells/frame" have many more
> real cells in the image — the `.geff` only sparsely labels a subset of tracks.
> This tells us the **density of the graph we're scored on**, which is the right
> thing for the metric — just don't read it as "cells physically in the frame."

---

## Step 02 — Follow TIME: the timeline heatmap

**Script:** `02_timeline_heatmap.py` → **Image:** `02_timeline_heatmap.png`

![timeline heatmap](02_timeline_heatmap.png)

**How to read it**
- **x-axis = frame `t` (0→99)** — the timeline.
- **color = cells in that frame** (dark = few, green/yellow = many).
- **grey = no data at that `t`** — the start timing / cropped window, made visible.
- **rows sorted sparse (top) → dense (bottom)** by mean cells-per-frame.
- **left strip = group**: red = `44b6`, blue = `6bba`.

**What the timeline reveals**
1. **Groups separate by density.** Red `44b6` clusters at the top (sparse, 1–2 cells);
   blue `6bba` fills the middle/bottom (dense, 10–30). Density is a group property.
2. **Cells grow over time.** Dense rows brighten left→right — embryos start with
   fewer cells and accumulate them through development (division).
3. **Start timing (grey):** **170/199 start at t=0**; **29 start later** (t_min up to **46**).
   Scattered single-pixel greys mid-track = **missing frames** inside otherwise
   continuous tracks (a real data quirk).
4. **Variable end times** — grey on the right edge means the track stops before t=99.

---

## Step 03 — The REAL density & developmental stage (correcting Step 02)

**Scripts:** `03_true_density_stage.py`, `03_stage_plot.py`
→ **Table:** `03_true_density_stage.csv` · **Image:** `03_stage_plot.png`

![stage plot](03_stage_plot.png)

### The big correction
Steps 01–02 counted **annotated** `.geff` nodes (1–33/frame). Those are **sparse
labels, not real cells.** Each `.geff` stores the organisers' true estimate in
`attributes.geff.extra.estimated_number_of_nodes` (**estN**).

- estN is the **total estimated cells across the whole timelapse** (proved by
  physics: e.g. estN=31,117 can't fit in one 104³ µm crop — packing limit ≈2,150).
- **True cells per frame = estN / n_frames = 38 → 1,015.** *That's* the real density.
- Only a **median 3.6 %** of real cells are labeled (range 0.1 %–20 %).
  So the Step-02 heatmap shows **label density, not biology.**

### ⚠️ The annotation is INVERTED vs biology
| group | # | annotated /frame | TRUE /frame (median) | label fraction | biology |
|-------|---|------------------|----------------------|----------------|---------|
| `44b6` | 71  | 3.2 (looks sparse) | **397** | **0.8 %** | **LATE / dense** (segmentation) |
| `6bba` | 128 | 9.0 (looks dense) | **97**  | **9.8 %** | **EARLY / sparse** (gastrula) |

So "6bba = dense group, 44b6 = sparse group" from Step 02 is true for **labels** but
**exactly backwards for real development.** 44b6 is the biologically *densest/latest*
group yet the *most sparsely labeled*.

### Developmental stage (from true per-frame density, 5 log-bins ≈ E56 S0–S4)
| stage | true cells/frame | n | 44b6 | 6bba | Kimmel period |
|-------|------------------|---|------|------|---------------|
| S0 (earliest) | 38–73    | 47 | 1  | 46 | early gastrula |
| S1 | 74–144   | 38 | 4  | 34 | gastrula |
| S2 | 145–275  | 37 | 15 | 22 | late gastrula / bud |
| S3 | 276–526  | 46 | 28 | 18 | early segmentation |
| S4 (latest) | 527–1015 | 31 | 23 | 8  | segmentation / somites |

**Stage↔group confound:** 6bba skews early (S0–S1), 44b6 skews late (S3–S4).
An embryo-disjoint split therefore partly measures **stage shift**, not pure
fish-to-fish variation → prefer **stage-stratified / leave-one-STAGE-out** validation.

### Does this match the zebrafish standard? (the review)
- ✅ **Yes for the trajectory.** Sparse large cells (38/frame) → dense small packed
  nuclei (1015/frame) is exactly **gastrula → segmentation** (Kimmel 1995).
- ✅ Growth is **asynchronous/gradual**, not synchronous doubling → firmly **post-MBT**
  (rules out cleavage/early-blastula).
- ⚠️ **No exact hpf in the files.** `t` is unitless frame index (scale 1.0), crops are
  ~104³ µm sub-volumes of a ~600–700 µm embryo. Stage is read from **density**, not a
  timestamp. An exact hpf would need the Zebrahub frame interval + each crop's t-offset
  into its parent movie (ZSNS timelapses run t=0→514) — not present in the release.

---

## Open threads / next steps
- [ ] Confirm stage from **image cues** (nucleus size + packing), count-independent.
- [ ] Build a **leave-one-STAGE-out** validation split to break the group↔stage confound.
