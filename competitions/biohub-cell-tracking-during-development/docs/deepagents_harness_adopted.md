# Deep-Agent Harness: what we adopted from `deepagents` vs what we already had

Date: 2026-07-19. Source read: `langchain-ai/deepagents` actual code (`libs/deepagents/deepagents/middleware/` —
`summarization.py`, `_message_eviction.py`, `_overflow_clip.py`, `memory.py`, `filesystem.py`, `subagents.py`)
plus the `examples/ralph_mode` and `examples/better-harness` worked patterns. NOT the README.

**Bottom line (honest):** deepagents is a batteries-included agent harness, but its primitives map almost
1:1 onto what our runtime already has (Claude leader/researcher/trainer on tmux + `fleet_dispatch` over 284
deterministic Python agents + MEMORY.md + per-comp Postgres + ledger). The frontier patterns **converge on
what our fleet already does.** The ONE genuine delta we were missing is a formalized **context-management /
working-memory protocol** — offload large outputs to disk, keep the thread/inbox compact, drive from a
~200-line working memo. We did NOT adopt LangChain/LangGraph (wrong stack); the adoption is a tiny Python
helper + a generic protocol section in the leader/researcher brains.

## Gap-verification table (deepagents primitive → what WE already had → adopted?)

| deepagents primitive | Where in deepagents | What WE already had | Verdict |
|---|---|---|---|
| Planning / todo list | `middleware/` planning + write_todos | Leader task decomposition + TaskCreate/TaskUpdate + `orchestrate`/`plan-ingest` agents | **Already had** |
| Sub-agents w/ isolated context | `middleware/subagents.py`, `async_subagents.py` | Agent-tool spawn + leader/researcher/trainer tmux roles + `fleet_dispatch` workers (conclusion returns, tool output stays out of thread) | **Already had** |
| Tools / MCP | tool registration, MCP | `fleet_dispatch` (284 Python agents) + MCP servers | **Already had** |
| Human-in-the-loop | `middleware/permissions.py`, `_fs_interrupt.py` | Permission modes + human-gated Kaggle submit; leader escalate→human | **Already had** |
| Persistent memory | `middleware/memory.py` (store-backed) | MEMORY.md file-memory + per-comp Postgres + ledger + INSIGHTS.md | **Already had** |
| Virtual filesystem | `backends/filesystem.py`, `middleware/filesystem.py` | Real filesystem + scratchpad + `output/` conventions | **Already had** |
| Skills | `middleware/skills.py` | checkcode / researchpapers skills | **Already had** |
| **Context summarization + large-output offload** | `middleware/summarization.py` (evict → `/conversation_history/{thread}.md`), `_message_eviction.py` / `_overflow_clip.py` (oversized ToolMessage → stub at `/large_tool_results/{id}` + head/tail preview) | **Nothing.** Big score tables / logs / JSON went straight into the board thread and inboxes, poisoning context. | **ADOPTED (the delta)** |
| Eval-gated self-improvement (`examples/better-harness`) | outer agent edits allowed surfaces → test on `train`+`holdout` → keep only if combined pass-count improves | `lever-hunt` + `feasibility-gate` + LOEO/both-embryo CV discipline (propose lever → held-out verify → keep only if it genuinely wins). Built independently for biohub. | **Already had** |
| Autonomous fresh-context loop (`examples/ralph_mode`) | `while :; do cat PROMPT.md \| agent; done`; fresh context each iteration, filesystem+git as memory/worklog | `/loop` + long-running improve-cron; git-track + journal + MEMORY.md as durable worklog | **Already had (protocol now names it)** |

## The adopted delta, in detail

**1. Context-management protocol (generic, every comp)** — added to `identities/leader.md` and
`identities/researcher.md`:
- Large tool/worker outputs go to a disk artifact; the thread/inbox carries a SHORT summary + the path, not
  the dump.
- Compact working memory: ~200-line `docs/MEMORY.md` in context, full detail in `docs/MEMORY_ARCHIVE.md` on
  disk; promote reusable findings up, demote narrow ones down.
- Prefer an isolated sub-agent / fleet worker for heavy exploration (tool output stays out of the main thread).
- Fresh-or-summarized context each cycle; git/files are the durable worklog.

**2. `context-offload` fleet agent** — `fleet_agents/context_offload.py` (registered as `context-offload`;
test `test_fleet_agents/context_offload_test.py`, 22/22 green under `OMP_NUM_THREADS=1` — now covers the
two-tier `truncate_args` refinement below).
Deterministic, offline. This is the direct port of deepagents' `TOO_LARGE_TOOL_MSG` behavior:
- `offload(text, label)` → writes `output/run_artifacts/<comp>/<ts>_<label>.md` (RP_COMP-routed, same as the
  ledger) and returns `{path, bytes, lines, preview, stub}`. The `stub` = summary + path + head/tail preview
  (measured: a 235 KB / 5000-line input → a 1.3 KB stub → ~180× smaller in the thread).
- `read_slice(path, offset, limit)` → reads detail back on demand (deepagents' `read_file` offset/limit).
- Dispatch: `fleet_dispatch context-offload "<summary>" '{"text":...,"label":...}'`; read-back with
  `'{"mode":"read","path":...,"offset":...,"limit":...}'`.

## 2026-07-19 refinement pass — the 3 final code-grounded adoptions

A second read of the actual deepagents source (`summarization.py`, `subagents.py`, `better-harness/core.py`)
turned the protocol above into three concrete, additive, backward-compatible code changes. All local; no
LangChain/LangGraph. Full-fleet data-wise suite stays green (the two touched agents pass; the pre-existing
biohub-domain reds — div-temporal-feas / ledger / sub-journal / submit-verify — are in untouched modules).

**Adoption 1 — two-tier offload refinement** (`fleet_agents/context_offload.py`):
- (a) BREADCRUMB pointer confirmed: `offload()`'s `stub` is already a clean breadcrumb — summary + absolute
  path + line/byte counts + a read-back command + a head/tail preview (`context_offload.py:93-97`). Left as-is
  (backward-compatible).
- (b) NEW `truncate_args(text_or_dict, limit=TRUNCATE_LIMIT, *, path=None)` (`context_offload.py:126-162`):
  the deepagents `_should_truncate_args`/`_truncate_tool_call` port — a SEPARATE, LOWER threshold
  (`TRUNCATE_LIMIT=4_000` chars) than full offload (`OFFLOAD_LIMIT=20_000`). Clips one oversized arg / file-write
  / grep-style output IN PLACE (dicts & lists mutated in place; strings returned), keeping head+tail with a
  `...[N chars elided, full at <path> if offloaded]...` marker — so a single big output gets clipped at emit time
  BEFORE a full offload is ever needed. New `run()` verb `truncate` routes to it. `offload()` unchanged.
  MEASURED shrink (data-wise test): a 50,000-char arg → 4,055 chars (**91.9% smaller**); a 30,000-char dict
  value → 2,055 chars; small inputs pass through verbatim.

**Adoption 2 — subagent "final-answer-only" contract** (pure prompt, no code logic):
- One generic line added to `identities/leader.md` (CONTEXT MANAGEMENT section) and `identities/researcher.md`:
  *"a sub-agent's / fleet worker's CALLER sees ONLY its final message … put the COMPLETE self-contained answer
  in that final message and share ABSOLUTE PATHS for anything large."* Ported from
  `subagents.py:DEFAULT_SUBAGENT_PROMPT`.
- Same contract sentence added to `fleet_dispatch.py`'s module help/prompt text (`fleet_dispatch.py:11-15`).
  Dispatch logic untouched. Generic — helps every comp.

**Adoption 3 — blind-holdout keep-if-improves gate for PROMPT/HARNESS self-optimization ONLY**
(`fleet_agents/harness_opt_gate.py`, registered `harness-opt-gate`, handler #285; classified in coverage-audit
`Prompt-program` pack; test `harness_opt_gate_test.py`, 8/8 green):
- Pure deterministic function (no LLM, no subprocess pytest) porting `better-harness/core.py:run_experiment`'s
  acceptance rule: `gate(baseline_train, baseline_holdout, cand_train, cand_holdout)` ACCEPTS iff the COMBINED
  (train+holdout) pass-count STRICTLY improves. Optimizer sees only `train`; `holdout` is a BLIND accept/reject
  gate — a train-only gain that regresses the blind holdout is rejected. Inputs coerce from int / list / dict.
  Same-strata guard: a per-split total mismatch → `valid=False, accept=False`.
- DELIBERATELY scoped to the PROMPT-OPTIMIZATION side (prompt-optimize triad / skill-build / agent-author). Our
  ML gating (lever-hunt / feasibility-gate / per-embryo paired Wilcoxon LOEO) is MORE rigorous and remains the
  sole judge of model changes — NOT routed through this gate.

Fleet is now **285 handlers**. Side-fix: `context-offload` (registered earlier but never classified) added to
coverage-audit `Generic CORE` so `zero_unclassified` holds again.

## Triple-convergence (why this is the right pattern, not a fad)

The working-memory + offload pattern shows up independently in three frontier sources, which is the honest
justification for formalizing it:
1. **deepagents** — SummarizationMiddleware offloads evicted history to disk + summarizes; FilesystemMiddleware
   replaces oversized tool results with a stub + path + preview.
2. **Ralph mode** (Geoff Huntley, in deepagents `examples/ralph_mode`) — autonomous loop, fresh context each
   iteration, filesystem + git as the cross-iteration memory/worklog.
3. **9th-place neurogolf team** — ~200-line `MEMORY.md` working summary in context + `MEMORY_ARCHIVE.md` full
   detail on disk.

## What we deliberately did NOT do
- No LangChain / LangGraph dependency (wrong stack — our runtime is Claude agents + tmux + `fleet_dispatch`).
- No runtime rewrite. Additive + backward-compatible only: one new agent, two generic identity sections, this
  note. Existing 283 agents and all boards untouched.
