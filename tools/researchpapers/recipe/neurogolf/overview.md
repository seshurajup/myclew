# NeuroGolf 2026 (IJCAI-ECAI Championship) — ARC-AGI Network Golf

Emit, for each of 400 ARC-AGI tasks, the SMALLEST ONNX graph that is functionally correct. This is a
program-synthesis / grid-reasoning comp, not a training comp: you are compiling each ARC transformation
rule into a tiny ONNX network.

## Score
Per task: `score = max(1.0, 25.0 - ln(max(1.0, cost)))`, where `cost = memory_bytes + params`.
- `params` = initializer elements + Constant-node value elements.
- `memory_bytes` = charged intermediate activation tensors. **The graph `input` and `output` tensors are
  FREE (uncharged).** Tensor bytes are the MAX over static shape-inference and the ORT runtime profiler.
- **MACs are FREE** — arithmetic is not charged. Spend compute to delete stored memory/params.
- A zero-cost graph (single terminal node, no initializers) scores the full 25.0.
Submission = a zip of `task001.onnx … task400.onnx`. Total score = sum over tasks (higher is better).

## The competition is over — this is a LATE-SUBMISSION rebuild
The deadline has passed. To score on the leaderboard you must JOIN the competition and download the official
`neurogolf_utils.py` scorer (the ground-truth authority for cost/validity), then submit the zip. Do NOT trust
any recorded score; always re-run the official scorer.

## The winning METHOD (distilled from the mined 1st/3rd/4th/5th/6th/9th-place writeups)
- **Agentic per-task workers.** One optimization loop per task, each starting from the best-known state
  (current best ONNX + builder + cost profile + score + target + promoted/rejected history + similar solved
  tasks + relevant idioms). A shared dashboard/MEMORY tracks progress across all 400.
- **Architecture REWRITES >> pruning.** Rewrites averaged +0.5 score/task; polishing an existing graph only
  +0.05. Agents are pushed to explore new representations, not tune the current one — with a concrete,
  headstrong target ("beat the baseline by ≥1.5, it's possible") to escape plateaus.
- **Recompute-not-store, end-in-Einsum.** Reduce charged intermediates to scalar / short-vec / int8 / bool;
  synthesize the full grid ONLY at the final node named `output` (which is free); put geometry in op
  ATTRIBUTES (pads/strides/dilations/kernel_shape/axes/equation/perm), not stored tensors. 9th place: 202/400
  tasks were single-node, 273 ended in `Einsum`.
- **Shared idiom catalogue + cross-task transfer.** A growing, per-score-band library of ONNX constructions
  (the mined `patterns.md`) is queried by a task's rule family to pick a target band + idiom; a construction
  that wins on one task is reused on similar tasks.

## How this repo drives it (roles + tools — nothing new to build)
- `leader` (live brain) orchestrates the 400 tasks + the shared MEMORY.md, sets per-task targets.
- `researcher` (live brain) does the ARC-solving + architecture-rewrite reasoning for a task.
- Deterministic python TOOLS (called via `fleet_dispatch`, no LLM inside them):
  - `arc-idioms` — query the `patterns.md` construction catalogue (idioms/families/cost-rules/exemplars by
    score band) for a task's rule family.
  - `onnx` (generic) / `arc-onnx-golf` (neurogolf wrapper) — emit a candidate ONNX, VERIFY functional
    correctness on train+test+arc-gen under the official one-hot semantics, and return the OFFICIAL cost
    (memory+params) + score. Trust ONLY its re-scored numbers.
  - `arc-worker-context` — assemble the rewrite-first per-task worker context + prompt.
  - If a needed python tool is missing, `researcher` calls `agent-author` to design + register it, then uses it.
