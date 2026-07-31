# How to Save Tokens with Claude Code

Practical, applied playbook for cutting Claude Code / LLM-agent token cost on this system.
Distilled from the token-optimization ecosystem (the "4 free repos" class of tools) plus what we've
actually wired into this machine. Ordered by impact.

## TL;DR — what's already applied here

- **1-hour prompt caching** — `ENABLE_PROMPT_CACHING_1H=1` in `~/.claude/settings.json` (`env`). Repeated
  context (MEMORY.md, tool history, system prompt) is billed at the cache-hit rate on any session that
  reuses it within the hour. Biggest zero-effort win.
- **`effortLevel: low`** — fewer reasoning tokens per turn. Already set. Raise to `medium`/`high` only for
  genuinely hard tasks, then drop back.
- **Permission allowlist** — `Bash(nvidia-smi*)`, `mcp__claude-in-chrome__tabs_context_mcp` in the project
  `.claude/settings.json`. Every permission prompt costs a full round-trip; allowlisting read-only commands
  removes those turns.
- **Token-saving plugin marketplaces registered** (install the ones you trust — they run hooks):
  `token-optimizer`, `context-mode`, `claude-code-toolkit`, `ponytail`, `headroom`.

## The levers, by impact

### 1. Context bloat is the real cost, not prompt length
Everything in the context window is re-sent every turn. Keep the always-injected files lean:
- **`MEMORY.md` / `CLAUDE.md` < 200 lines.** A 5,000-token memory index is a 5,000-token tax on *every*
  turn. Ours is ~1,800 tokens (89 lines) — healthy; keep it a one-line-per-fact index, push detail into
  the individual memory files that only load on recall.
- Delete stale memories and dead `[[links]]`; they're pure overhead.

### 2. Compress tool output before it reaches the model
Tool results (file dumps, JSON, logs, test output) are the largest single source of context pollution.
- **`headroom`** (installed isolated via `uv tool`, runs as a proxy/MCP) — compresses tool outputs, logs,
  files, and RAG chunks: ~15-20% fewer tokens for coding agents, 60-95% for JSON. Reversible, local-first.
- **`context-mode`** — sandboxes tool output (up to 98% reduction), persists session memory across turns.
- **`rtk` (Rust Token Killer)** — a CLI proxy that rewrites verbose dev-command output; 60-90% reduction on
  common commands. Single Rust binary, run in front of the agent.

### 3. Route work to the cheapest model that can do it
- Default to the smallest capable model; reserve the top tier for genuine architecture / multi-step reasoning.
- For **subagents doing high-volume mechanical work, set `model: haiku`** in the agent config — they run in
  their own context and return only a summary.

### 4. Offload wide reads to subagents
A subagent runs in its own context window and returns only its conclusion to the parent. Any
"investigate/search X across the codebase" task that would pull 30+ files into the main session should be a
subagent — the file dumps never enter your context, only the answer does. (This is exactly how the fleet's
`Explore`-style agents are used.)

### 5. Scope every request tightly
"Refactor the login function in `auth.ts`" beats "refactor the auth module." Smaller scope = less context
pulled, fewer tokens, more focused output. Vague asks make the model read broadly to disambiguate.

### 6. Compact deliberately, and mine what compaction loses
- Use `/compact` at natural task boundaries rather than letting auto-compaction fire mid-task.
- `token-optimizer` finds "ghost tokens" — bloated configs, unused skills, stale memory, model misrouting —
  the ~75% of waste that isn't just compressed command output.

## Fleet-side habits that keep our own token cost down

- **Work in Python, not chat.** Do multi-step analysis in a script and print one summary, instead of many
  tool round-trips. (Standing rule in this repo.)
- **Agents return findings, not file dumps.** Every fleet agent emits a one-line `log()` summary + a small
  structured `done()` payload — not raw data — so orchestration stays cheap.
- **Data-wise tests print PASS/FAIL counts, not full traces.** Keep verifier output terminal-cheap.

## Guardrail learned the hard way

Installing a token tool into a shared Python env can **clobber your torch/CUDA ABI**. `headroom-ai[all]`
pulled `torch 2.13+cu130`, which broke the 5090 (needs `+cu128`). **Always install agent tooling in an
isolated env** (`uv tool install` / `pipx`), never the `llm` env. If a pip run ever pulls torch, reinstall
`torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128` immediately.

## Sources
- KDnuggets — 7 Practical Ways to Reduce Claude Code Token Usage
- MindStudio — 5 Claude Code Skills That Cut Token Costs (benchmarked)
- Repos: `alexgreensh/token-optimizer`, `mksglu/context-mode`, `thoeltig/claude-code-toolkit`,
  `DietrichGebert/ponytail`, `headroomlabs-ai/headroom`, `rtk-ai/rtk`
- Video: Eric Tech — "4 Free Repos That Cut Claude Code Token Usage"
