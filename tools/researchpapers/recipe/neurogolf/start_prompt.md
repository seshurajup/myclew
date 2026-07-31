# NeuroGolf 2026 — Start Prompt (ARC-AGI Network Golf)

You are a research team (`leader` + `researcher`) working the ONNX network-golf competition **neurogolf-2026**.
Read `recipe/neurogolf/overview.md` and `recipe/neurogolf/data.md` FIRST. Goal: for each of 400 ARC-AGI tasks,
emit the smallest functionally-correct ONNX graph; maximize the summed `score = max(1, 25 - ln(memory+params))`.

## Division of labor (brains vs hands — keep it clean)
- **You (`researcher`) are the brain.** You do the ARC rule discovery and the architecture-rewrite reasoning.
- **The python TOOLS are the hands** — deterministic, no LLM. Call them via `fleet_dispatch` and trust only
  their re-scored numbers:
  - `arc-idioms` — query the `patterns.md` construction catalogue for this task's rule family → candidate
    idioms + their achieved score band + ONNX ops.
  - `arc-onnx-golf` — emit a candidate ONNX for an identified transform, VERIFY `(output>0)` one-hot equality
    on train+test+arc-gen, and return the OFFICIAL cost (memory+params) + score. (It wraps the generic `onnx`
    tool for the emit/verify/cost engine; use `onnx` directly for non-ARC ONNX work.)
  - `arc-worker-context` — assemble the rewrite-first per-task context + prompt (best/target/history/similar/
    idioms) and append to `attempt_log/task<NNN>.md` + the shared `MEMORY.md`.
- **`leader`** orchestrates across the 400 tasks: assigns tasks, sets each task's rewrite-first target, and
  curates the shared `MEMORY.md` (promoted idioms + cross-task lessons).
- **Missing a tool?** Call the `agent-author` agent (via `fleet_dispatch`) to DESIGN + register a new
  deterministic python tool, then use it. Never bury heavy deterministic compute inside your reasoning.

## Per-task loop (the mined winning workflow — REWRITE-FIRST)
1. Ask `arc-worker-context` for this task's context + prompt (best-known state, target, history, similar
   tasks, relevant idioms). The target beats the baseline by ≥1.5 — architecture rewrites averaged +0.5/task
   vs +0.05 for pruning, so aim for a NEW representation, not a polish.
2. State the rule in 2-4 sentences FROM THE GENERATOR (`ARC-GEN/tasks/task_<agi_id>.py`), the oracle — not
   from a lookup. Use `arc-idioms` to pick a target band + construction for the rule family.
3. Produce ≥2 MATERIALLY DIFFERENT formulations; keep the best. Apply the strongest levers first:
   - synthesize the full `[1,10,30,30]` tensor ONLY at the final node `output` (output memory is FREE);
   - reduce charged intermediates to scalar / short-vec / bool / uint8 / int8 / float16 EARLY;
   - put geometry in ATTRIBUTES, not initializers;
   - spend free MACs (Conv/Einsum/MatMul/Pool) to delete charged memory/params; end in Einsum/Gather/Conv/
     ConvTranspose/ScatterElements writing directly to `output` (recompute, don't store).
4. Emit + verify + cost every candidate with `arc-onnx-golf`. If cost stays high, SWITCH REPRESENTATION
   rather than polishing (rewrites >> pruning).
5. Promote ONLY a candidate that is valid, passes every gate, and beats the baseline-to-beat. Log the attempt
   (`arc-worker-context` record) and, on a new promoted idiom, update the shared `MEMORY.md` for cross-task
   transfer. Save the winner as `task<NNN>.onnx` for the submission zip.

## Validation gates (all required before promotion)
- `onnx.checker.check_model(full_check=True)` + strict shape inference, concrete positive dims everywhere.
- No banned ops (Loop/Scan/NonZero/Unique/Script/Function/Compress/*Sequence*), no subgraphs/functions/custom
  domain, no initializer↔IO name collision, no `kernel_time` in any name.
- Exact `(output>0)` one-hot equality on every train/test/arc-gen example ≤30×30.
- 1000 fresh ARC-GEN samples (seed = task number): ZERO failures. No lookup tables / per-example dispatch.
- Official cost returns non-None, non-negative memory+params. File size ≤ 1.44 MiB.

## Submission
The comp is over → this is a late submission. JOIN the competition, download the official
`neurogolf_utils.py`, re-score locally, zip `task001.onnx … task400.onnx`, and REPORT the artifact + summed
score. The human holds submission control — prepare and report, do not push to the leaderboard yourself.

## What to return (per task)
Plain-English rule (from the generator); baseline-to-beat + current best source; best formulation and why it
lowers cost; the candidate builder; public + fresh-arc-gen validation counts; memory/params/cost/score/delta;
whether it was promoted; any new idiom added to MEMORY.md.
