# NeuroGolf 2026 — Data & Format

## Tasks (ARC-AGI-1, 400 training tasks)
- The 400 NeuroGolf tasks are the public **ARC-AGI-1 training set** (github `fchollet/ARC-AGI`, `data/training/`
  — same tasks, re-indexed `task001…task400`). Each task JSON has:
  - `train`: list of `{input, output}` grid pairs (the demonstrations).
  - `test`: held-out `{input, output}` pairs.
  - `arc-gen`: extra generator-produced pairs (from `ARC-GEN/tasks/task_<agi_id>.py`) — the generalization set.
- A grid is a 2-D list of integer colours `0..9` (0 = black), up to `30×30`.
- Local mirror in this workspace: `writeups/pntan17_9th_src/data/task*.json` (400) + `.../ARC-GEN/tasks/`
  (the per-task generators — treat as the RULE ORACLE, produce fresh holdout samples; never a lookup table).

## ONNX network format (official — `neurogolf_utils.py` is the authority)
- Input tensor `input`, output tensor `output`, both **one-hot `[1, 10, 30, 30]` float32**:
  cell colour `k` at `(r,c)` → channel `k` = 1.0; grids < 30 are TOP-LEFT anchored, the rest is zero-hot.
  Read-back: a cell's colour = the channel with value `> 0.0` (trailing all-zero rows/cols are trimmed).
- **Correctness** = exact `(output > 0.0)` one-hot equality on EVERY train + test + arc-gen example ≤ 30×30
  (not argmax). Generate 1000 fresh ARC-GEN samples (seed = task number) and require ZERO failures.
- opset 10, ir_version 10. Single input, single output.

## Validity rules (a network violating any of these scores nothing)
- Banned ops: `Loop`, `Scan`, `NonZero`, `Unique`, `Script`, `Function`, `Compress`, and any `*Sequence*` op.
- No subgraphs / nested graphs, no `model.functions`, no non-default opset domain.
- No initializer↔IO name collision, no duplicate `value_info`, no `kernel_time` in any tensor name.
- Every tensor must strict-shape-infer to concrete positive dims. File size ≤ 1.44 MiB per network.
- No lookup tables of public inputs/outputs, no per-example dispatch — the rule must generalize.

## Cost intuition (worked, from `patterns.md`)
- Terminal `Transpose(perm=[0,1,3,2])` (diagonal reflection) → cost 0 → score 25.
- One `Gather(axis=1)` palette recolor (length-10 index) → params 10 → score 22.697.
- Charged intermediate `[1,10,30,30]` float32 = 36000 bytes → score ≈ 14.5. AVOID; keep the big tensor at
  `output` (free) and reduce intermediates to scalars / short vectors / int8 / bool.
