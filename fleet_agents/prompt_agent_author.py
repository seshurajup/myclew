"""agent-author + agent-package — author the ADK agent bundle for autonomous-agent-prediction-beta and
package it into a valid submission.zip. Grounded in the demo + AgentForge champion structure:

  agent.yaml (single-orchestrator: name/model/instruction=!include/tools/skills/generate_content_config)
  prompts/system.md          — the champion 8-section prescriptive AutoML workflow (floor-first, gate,
                               trustworthy CV, safe FE, diverse zoo, AUC rank-blend, strategic submits, hedge)
  prompts/data_analyst.md    — EDA sub-agent instruction
  tools/data_analyst.yaml    — the sub-agent mounted as a callable tool
  skills/tabular-autopilot/  — the deterministic floor (from skill-build)

agent-package assembles the dir, validates the required manifest + allowed model ids, and zips at ROOT.
"""
from __future__ import annotations
import os
import shutil
from pathlib import Path
from .base import BaseAgent
from . import prompt_skill_build as SB

SYSTEM_MD = """# Autonomous ML Agent — operating manual

You solve tabular binary-classification tasks scored by ROC AUC in an offline, CPU-only sandbox. Budget:
~60 min wall clock, limited submit calls, limited token budget. ROC AUC is rank-based — do NOT tune a
threshold. The public score is only a SUBSET of the test set; do not overfit it.

## 1. Secure a valid floor FIRST
Immediately run the deterministic pipeline to guarantee a valid, competitive submission before anything else:
`run_skill_script(skill_name="tabular-autopilot", script_name="run_pipeline.py", args="--data-dir . --out submission.csv")`
Then ALWAYS gate before submitting:
`run_skill_script(skill_name="tabular-autopilot", script_name="check_submission.py", args="--sub submission.csv --sample sample_submission.csv")`
If CHECK_OK, `submit_predictions("submission.csv")` and record the returned submission id.

## 2. Understand the task
Delegate EDA to the `data_analyst` tool: shapes, target balance, missingness, high-cardinality columns,
train/test drift. Read target_col.txt if present.

## 3. Trustworthy validation
Prefer the pipeline's internal StratifiedKFold OOF AUC over the public leaderboard. A change ships only if it
improves OOF (and does not contradict the public subset).

## 4. Iterate deliberately, within budget
Call `get_status()` to watch time and remaining submit calls. Try improvements the skill supports (feature
handling, model mix). Prefer simple, fast, deterministic models (HistGBM/ExtraTrees/LogReg + LightGBM/XGBoost).
If time_minutes_remaining < 6 or tool_calls_remaining < 10, stop iterating and select finals.

## 5. Select complementary finals (hedge public/private noise)
`select_submission([...])` — choose TWO diverse finals: the best public score not contradicted by CV, and the
most robust OOF ensemble. The best test score among selections is your final.

## 6. Finish
Only after submitting AND selecting, end with a short text summary. Responding with text and no tool call ends
the session — never do it before you have submitted and selected.
"""

DATA_ANALYST_MD = """# Data Analyst

Profile the tabular task and report concisely: rows/cols, target column + class balance, numeric vs
categorical columns, high-cardinality categoricals, missingness per column, obvious leakage suspects, and
train/test distribution drift. Recommend a validation scheme (StratifiedKFold) and any feature handling.
Do not train models — analysis only.
"""

REVIEWER_MD = """# Reviewer (read-only critic)

You are a stronger, deterministic reviewer that the primary worker consults at uncertain checkpoints. You are
READ-ONLY: you cannot write files, run the pipeline, submit, or select. You only reason over the evidence the
worker gives you and return one structured verdict.

## Input protocol
The worker addresses you with EXACTLY ONE `REQUEST_TYPE:` tag (enum, one of):
  - CV_VS_LB_CONFLICT   — trustworthy OOF CV and the public subset disagree; which to trust
  - FE_DECISION         — should a feature-engineering / model change ship
  - LEAKAGE_SUSPICION   — a feature or split looks like leakage
  - SUBMIT_DECISION     — is the current submission safe to submit
  - FINAL_SELECTION     — which two finalists to select
followed by the evidence (OOF AUC, public score, drift, leakage signals, time/tool budget remaining).

## Output protocol
Return EXACTLY these three lines, nothing else:
NEXT_FLASH_ACTION: <one of USE_CURRENT | APPLY_FIX_AND_RUN | STOP_AND_SELECT>
MOST_IMPORTANT_PROBLEM: <the single most important problem, one sentence>
SMALLEST_REQUIRED_FIX: <the smallest concrete fix, one sentence — or "none" if USE_CURRENT>

Rules: prefer USE_CURRENT unless the evidence shows a real defect; never invent numbers not in the evidence;
trust trustworthy OOF CV over the public subset; if budget is nearly exhausted, prefer STOP_AND_SELECT. You
are read-only — recommend, never act.
"""

# ADK-allowed model ids (grounded); orchestrator uses flash, sub-agents use lite for token budget.
# reviewer uses a stronger deterministic (temperature 0) model consulted only at uncertain checkpoints.
MODELS = {"orchestrator": "gemini-3.5-flash", "analyst": "gemini-3-flash-preview",
          "reviewer": "gemini-3.5-pro"}


def _agent_yaml(skill_name, include_reviewer=True):
    # reviewer is APPENDED to the tools list (alongside data_analyst); when disabled the block is empty so
    # the emitted agent.yaml is byte-identical to the pre-reviewer output.
    reviewer_block = "  - agent_tool:\n      config_path: tools/reviewer.yaml\n" if include_reviewer else ""
    return f"""name: ml_agent
model: {MODELS['orchestrator']}
instruction: !include prompts/system.md
tools:
  - run_command
  - read_file
  - write_file
  - run_skill_script
  - load_skill_resource
  - submit_predictions
  - select_submission
  - get_status
  - agent_tool:
      config_path: tools/data_analyst.yaml
{reviewer_block}skills:
  - skills/{skill_name}
generate_content_config:
  temperature: 0.2
  max_output_tokens: 4096
  thinking_config:
    thinking_budget: 2048
    include_thoughts: false
"""


def _reviewer_yaml(reviewer_model=None):
    return f"""name: reviewer
description: Read-only stronger-model critic consulted at uncertain checkpoints; returns a structured NEXT_FLASH_ACTION verdict. No write/submit/select tools.
model: {reviewer_model or MODELS['reviewer']}
instruction: !include ../prompts/reviewer.md
tools:
  - read_file
generate_content_config:
  temperature: 0
  max_output_tokens: 1024
"""


def _data_analyst_yaml():
    return f"""name: data_analyst
description: Profiles a tabular dataset (shapes, target balance, missingness, drift) and recommends validation.
model: {MODELS['analyst']}
instruction: !include ../prompts/data_analyst.md
tools:
  - read_file
  - run_command
generate_content_config:
  temperature: 0.1
  max_output_tokens: 2048
"""


REQUIRED = ["agent.yaml", "prompts/system.md", "prompts/data_analyst.md", "tools/data_analyst.yaml",
            "skills/tabular-autopilot/SKILL.md", "skills/tabular-autopilot/scripts/run_pipeline.py",
            "skills/tabular-autopilot/scripts/check_submission.py"]


def author(out_dir, skill_name="tabular-autopilot", include_reviewer=True, reviewer_model=None):
    d = Path(out_dir); (d / "prompts").mkdir(parents=True, exist_ok=True); (d / "tools").mkdir(exist_ok=True)
    (d / "agent.yaml").write_text(_agent_yaml(skill_name, include_reviewer))
    (d / "prompts" / "system.md").write_text(SYSTEM_MD)
    (d / "prompts" / "data_analyst.md").write_text(DATA_ANALYST_MD)
    (d / "tools" / "data_analyst.yaml").write_text(_data_analyst_yaml())
    if include_reviewer:
        (d / "prompts" / "reviewer.md").write_text(REVIEWER_MD)
        (d / "tools" / "reviewer.yaml").write_text(_reviewer_yaml(reviewer_model))
    SB.build_skill(str(d), skill_name)
    return str(d)


def validate(out_dir):
    d = Path(out_dir); missing = [f for f in REQUIRED if not (d / f).exists()]
    return {"ok": not missing, "missing": missing}


def package(out_dir, zip_path=None):
    if not Path(out_dir).is_dir():
        raise FileNotFoundError(f"package: bundle dir does not exist: {out_dir}")
    zip_path = zip_path or (str(out_dir).rstrip("/") + ".zip")
    base = zip_path[:-4] if zip_path.endswith(".zip") else zip_path
    # zip CONTENTS at root (root_dir=out_dir) — not nested under the folder name
    shutil.make_archive(base, "zip", root_dir=out_dir)
    return base + ".zip"


class AgentAuthor(BaseAgent):
    name = "agent-author"
    thread = "R"
    kind = "config-gen"

    def run(self, q, worker):
        spec = self.spec(q)
        out = spec.get("out_dir") or "/tmp/aap_submission"
        include_reviewer = spec.get("include_reviewer", True)
        d = author(out, spec.get("skill_name", "tabular-autopilot"), include_reviewer, spec.get("reviewer_model"))
        v = validate(d)
        rv = "+reviewer" if include_reviewer else ""
        msg = f"agent-author: authored ADK bundle → {d} (agent.yaml+system.md+data_analyst{rv}+skill); valid={v['ok']} missing={v['missing']}"
        self.log(msg, kind="config-gen", recommendation="agent-config-eval on synthetic tasks, then agent-package")
        return self.done({"bundle_dir": d, **v}, msg)


class AgentPackage(BaseAgent):
    name = "agent-package"
    thread = "R"
    kind = "verdict"

    def run(self, q, worker):
        spec = self.spec(q)
        missing = [k for k in ("bundle_dir",) if k not in spec]
        if missing:
            return self.escalate(worker, "leader", f"agent-package needs spec keys {missing} — none provided")
        d = spec["bundle_dir"]
        v = validate(d)
        if not v["ok"]:
            return self.escalate(worker, "researcher", f"agent-package: bundle invalid, missing {v['missing']}")
        zp = package(d, spec.get("zip"))
        msg = f"agent-package: valid ADK bundle zipped at root → {zp}"
        self.log(msg, kind="verdict", recommendation="human-gated submit only (verify locally with agent-config-eval first)")
        return self.done({"zip": zp, **v}, msg)


_AUTHOR = AgentAuthor(); _PACK = AgentPackage()


def run(q, worker):
    return _AUTHOR.run(q, worker)


def run_package(q, worker):
    return _PACK.run(q, worker)
