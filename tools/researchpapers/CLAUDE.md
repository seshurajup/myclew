# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`researchpapers` is a local-first multi-agent runtime for machine learning research on Kaggle competitions. It orchestrates three AI agents (leader, researcher, trainer) to continuously iterate on experiments with minimal human intervention.

The system is NOT a generic chat interface—it's a complete operating system for ML experimentation with:
- tmux-based multi-agent runtime with fixed roles
- Forum-style discussion board for agent coordination
- Separate training queue service with its own board
- Opinionated experiment workflow (EDA → baseline → iteration)
- File-backed state (no database)

## Core Architecture

**Three-Agent Team:**
- **leader**: Orchestrates work, delegates tasks, reviews readiness, decides next steps
- **researcher**: Owns EDA, implementation, code/configs, experiment design, dry-runs
- **trainer**: Owns queue submission, result triage, summaries, trend reporting

**Four Layers:**
1. `researchpapers/runtime_cli.py` — tmux agent management, inbox delivery, supervisor loop
2. `.research-mvp-data/runtime/` — thread log, inboxes, agent state
3. `researchpapers/app.py` — runtime discussion board (FastAPI, port 7788)
4. `researchpapers/train_service/app.py` — job queue service (FastAPI, port 7799)

**Message Semantics:**
- Shared thread (`.research-mvp-data/runtime/thread.jsonl`): visibility, milestones, human-readable summaries
- Per-agent inbox (`.research-mvp-data/runtime/inbox/<agent>/`): executable work assignments

## Starting & Stopping the Runtime

```bash
# Start all services (runtime, app, train_service)
./start_research_mvp.sh start

# Stop all services
./start_research_mvp.sh stop

# Restart everything
./start_research_mvp.sh restart

# Check status
./start_research_mvp.sh status

# Optional: override host/port via env
APP_HOST=127.0.0.1 APP_PORT=7788 TRAIN_SERVICE_PORT=7799 ./start_research_mvp.sh start
```

**Web Boards** (once started):
- Runtime discussion board: `http://localhost:7788/runtime`
- Training queue board: `http://localhost:7799/`

**Tmux Interaction:**
```bash
# Attach to the tmux runtime session (agents run in tmux windows)
python -m researchpapers.runtime_cli --config researchpapers/runtime.toml attach

# Once attached:
# Ctrl-b n       next window
# Ctrl-b p       previous window
# Ctrl-b <0|1|2> switch to window (leader=0, researcher=1, trainer=2)
# Ctrl-b d       detach without stopping
```

## Runtime Commands (inside tmux or via CLI)

```bash
# View shared thread (latest 50 messages)
python -m researchpapers.runtime_cli --config researchpapers/runtime.toml thread tail -n 50

# List inbox messages for an agent
python -m researchpapers.runtime_cli --config researchpapers/runtime.toml inbox list <agent>

# Send a directed message to an agent
python -m researchpapers.runtime_cli --config researchpapers/runtime.toml delegate --from <sender> --to <recipient> "message text"

# View runtime status
python -m researchpapers.runtime_cli --config researchpapers/runtime.toml status
```

## Directory Structure & Conventions

**Workspace Layout:**
```
baseline/          — experiment runners, shell scripts, YAML configs
src/baseline/      — training code (entry point: train.py)
data/              — datasets and prepared data
eda/               — EDA scripts, notes, and figures
docs/              — experiment plans (design notes) and result summaries
output/            — checkpoints, metrics, logs, run artifacts
recipe/            — task recipes and competition context
.research-mvp-data/— runtime state (thread, inboxes, agent state)
```

**Versioned Iteration:**
- Each baseline version is numbered (v1, v2, v3, ...)
- Default naming: `baseline/experiments_v1/`, `baseline/run_experiments_v1.sh`, `output/baseline_v1/`
- Experiment design documented in `docs/baseline_vX_design.md`
- Results summarized in `docs/baseline_vX_results.md`

**One Formal Runner per Version:**
- Keep ONE canonical shell script per baseline version (e.g., `baseline/run_experiments_v1.sh`)
- The runner fans out to multiple configs rather than scattering separate entry scripts
- Runners must be queue-ready for `train_service`

## Training Code & Execution

**Training Launcher:**
```bash
python src/baseline/train.py --config baseline/experiments_v1/v1_2_hr_baseaug.yaml --fold 0
```

**Dry-Run (GPU-safe validation):**
```bash
python src/baseline/train.py --config baseline/experiments_v1/v1_2_hr_baseaug.yaml --fold 0 --dry-run
```

Dry-runs validate YAML schema, paths, and augmentation specs WITHOUT importing torch or using GPU. After a successful dry-run, the output shows a `--epochs 1 --max-iters 1 --single-gpu` command for GPU testing once the GPU is free.

**Key Training Observability:**
- Startup banner with clear experiment identification before training loop
- Per-run logs tee'd to `output/baseline_v<version>/<id>/train.log`
- MLflow integration for metrics tracking (controlled by `MLFLOW_*` env vars)

**PYTHONPATH Requirement:**
Some runtime agents require the correct PYTHONPATH:
```bash
export PYTHONPATH=tools/researchpapers:$PYTHONPATH
```

## Starting a Recipe Task

If the human asks to start a task under `recipe/<name>/`, the researcher MUST read first:
- `recipe/<name>/overview.md` — competition/task context
- `recipe/<name>/data.md` — data structure, evaluation metric, submission format
- `recipe/<name>/start_prompt.md` — initial research direction

**First phase MUST be EDA** before jumping to training. Put EDA scripts, notes, and charts in `eda/`.

Only move to baseline iteration after understanding:
- Data structure and shape
- Evaluation metric and scoring
- Submission format
- Major leakage risks
- Class imbalance or long-tail risks
- Obvious baseline directions

## Workflow Conventions

**Default Research Flow:**
1. Understand the task
2. Perform EDA (if new recipe task)
3. Design one baseline version with multiple experiments
4. Implement code and configs
5. Run minimal dry run (researcher's job)
6. Submit formal training through train_service (trainer's job)
7. Summarize results in `docs/`
8. Decide the next version (don't stop by default)

**Role Responsibilities (summarized):**

- **researcher**: Writes design notes in `docs/`. Code and configs belong in `src/baseline/` and `baseline/`. Always run dry-run before delegation.
- **trainer**: Writes result summaries in `docs/`. Submit queue-ready packages to `train_service`. Generate trend PNGs across baseline versions (e.g., `docs/baseline_v1_to_v2_top3_trend.png`).
- **leader**: Delegates via inbox, not just shared thread. Makes git commits for preserved code/script changes. Every 5 versions, review path against reference solutions.

**Key Rules:**
- Inbox messages are executable work assignments (not shared thread messages)
- If you need another agent to act, send a directed inbox message via `delegate`
- After delegating, add a brief shared-thread summary for visibility
- Keep updates concise and operational (file paths, next actions)
- Do NOT store experiment artifacts under `.research-mvp-data/`—that's control-plane state only
- Finish one baseline version → immediately choose next action (don't stop)

## Configuration & Runtime Setup

**Runtime Config:** `researchpapers/runtime.toml`
- Session name, agent names, tmux submit keys
- Env vars for all agents
- Timeouts and cooldowns

**Train Service Config:** Check `.research-mvp-data/` for active job submissions and queue state.

## Git Commits

After each preserved baseline version:
- `leader` makes a git commit with relevant code and script changes
- Commit message should summarize the version (e.g., "baseline_v1: control vs rich-aug A/B")

## Testing & Verification

**Dry-run approach:**
- GPU-safe: validates schema + paths + aug specs without importing torch
- Does NOT run the training loop
- Shows next command for `--epochs 1 --max-iters 1` GPU testing

**Integration verification:**
- Trainer and researcher coordinate via inbox delegation
- Trainer submits to train_service API at `http://localhost:7799/`
- Queue board shows submission status and logs

## Key Files to Know

- `README.md` — high-level project overview
- `AGENTS.md` — shared runtime contract for all three agents
- `researchpapers/identities/leader.md`, `researcher.md`, `trainer.md` — role-specific specs
- `researchpapers/ARCHITECTURE.md` — detailed runtime architecture
- `researchpapers/OPERATIONS.md` — operational procedures
- `researchpapers/runtime_cli.py` — main runtime supervisor and CLI
- `src/baseline/train.py` — training launcher with dry-run support
- `baseline/run_experiments_v*.sh` — versioned formal runner scripts

## Typical Development Tasks

**As a human coordinating the team:**
1. Send initial task to leader via the runtime board
2. Watch agents iterate via discussion board and queue board
3. Inject ideas/corrections directly into the discussion board when needed
4. Check logs via `./start_research_mvp.sh status`

**As researcher (inside tmux):**
1. Read task from inbox
2. Perform EDA if needed
3. Design baseline version, write design note in `docs/`
4. Implement code in `src/baseline/`, configs in `baseline/`
5. Run dry-run: `python src/baseline/train.py ... --dry-run`
6. Delegate to trainer with results

**As trainer (inside tmux):**
1. Read queue-ready package from inbox
2. Submit to train_service API
3. Wait for external results
4. Write result summary in `docs/`
5. Generate trend PNG
6. Report back to leader

## Dependencies

- Python 3.11+
- tmux (for multi-agent runtime)
- Core packages: FastAPI, Uvicorn, PyYAML, PyTorch, pandas, numpy, scikit-learn, matplotlib
- See `requirements.txt` and `pyproject.toml` for full list

## Common Debugging

**Runtime won't start:**
- Check logs: `tail -f logs/researchpapers_runtime.log`
- Ensure tmux is installed: `sudo apt install tmux`
- Verify PYTHONPATH if needed

**Agent not responding:**
- Check agent inbox: `python -m researchpapers.runtime_cli --config researchpapers/runtime.toml inbox list <agent>`
- Attach to tmux and check agent window for errors
- Check shared thread for blockers: `python -m researchpapers.runtime_cli --config researchpapers/runtime.toml thread tail -n 20`

**Train service not accessible:**
- Verify it's running: `./start_research_mvp.sh status`
- Check logs: `tail -f logs/researchpapers_train_service.log`
- Bypass proxy: `NO_PROXY=127.0.0.1 http_proxy= curl http://localhost:7799/`

**GPU memory issues:**
- Always run dry-run first (GPU-safe)
- Use dry-run output to test with `--epochs 1` once GPU frees
- Don't rely on trainer to fix GPU issues—that's researcher's concern
