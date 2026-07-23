"""base — the BaseAgent every fleet agent extends, so no agent misses the main features.

The fleet grew as loose handler functions; some had rich logging, some didn't; state/post/escalate/verify
were done ad-hoc. BaseAgent standardises the *contract* so every agent gets, for free:

  • spec(q)                      — reusable/spec-driven input parsing
  • load_state()/save_state()   — JSON state persistence under config/_auto/<name>.json
  • post(worker,to,msg)         — compact markdown to the board
  • log(summary,...)            — structured ledger entry
  • escalate(worker,to,msg)     — hand a decision to the leader/researcher
  • done()/escalated()          — the standard (status, data, to, message) return contract
  • verify()                    — the DATA-WISE self-test (runs test_fleet_agents/<name>_test.py)

Existing function-agents are wrapped as FunctionAgent(BaseAgent) so EVERY registered agent is a BaseAgent
with these features — nothing is missed — without a risky rewrite. New agents should subclass BaseAgent
directly and implement run() + verify().
"""
from __future__ import annotations
import json
import os
import subprocess
import sys
from pathlib import Path

COMP = Path(__file__).resolve().parent.parent
AUTO = COMP / "config" / "_auto"
TESTS = COMP / "test_fleet_agents"

# GPU-TRAINING HOLD — while this flag file exists, GPU-heavy trainers refuse to run and escalate instead.
# Used to enforce a human hold (e.g. the RTX 5090 power cap must be set to 400W before heavy training).
_GPU_HOLD_FLAG = AUTO / "gpu_train_hold.flag"


def gpu_train_held() -> bool:
    """True if a human has parked GPU training (thermal/power gate). Trainers must check this first."""
    return _GPU_HOLD_FLAG.exists()

# agents whose data-wise test is RED — they are QUARANTINED: they will not run their logic (which could be
# wrong) and instead escalate to be fixed. Populated by preflight(); persisted so the live fleet respects it.
_QFILE = AUTO / "quarantine.json"


def _load_quarantine() -> set:
    try:
        return set(json.loads(_QFILE.read_text()).get("red", []))
    except Exception:  # noqa: BLE001
        return set()


QUARANTINE: set = _load_quarantine()


class BaseAgent:
    name = "base"
    thread = "S"          # default board thread
    kind = "finding"      # default post kind

    # ---- input ----
    def spec(self, q) -> dict:
        return (q.get("spec") or {}) if isinstance(q, dict) else {}

    # ---- state ----
    def state_path(self) -> Path:
        return AUTO / f"{self.name.replace('-', '_')}.json"

    def load_state(self, default=None):
        p = self.state_path()
        if p.exists():
            try:
                return json.loads(p.read_text())
            except Exception:  # noqa: BLE001
                pass
        return {} if default is None else default

    def save_state(self, st):
        p = self.state_path(); p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(st, indent=2))

    # ---- outputs ----
    @staticmethod
    def _msg_sig(text: str) -> str:
        """Signature for dedup: normalise whitespace + drop trailing detail so same/similar messages collapse."""
        import re
        t = re.sub(r"\s+", " ", (text or "").strip().lower())
        return t[:160]                                    # header + opening = the identity of a repeated message

    def _is_duplicate(self, msg, window=3) -> bool:
        """True if the same/similar message is among the last `window` thread entries (don't re-spam the board)."""
        try:
            from researchpapers.fleet import post as _p
            thread = _p.runtime_dir() / "thread.jsonl"
            if not thread.exists():
                return False
            sig = self._msg_sig(msg)
            for ln in thread.read_text(errors="replace").splitlines()[-window:]:
                try:
                    if self._msg_sig(json.loads(ln).get("content", "")) == sig:
                        return True
                except Exception:  # noqa: BLE001
                    continue
        except Exception:  # noqa: BLE001
            return False
        return False

    def post(self, worker, to, msg, routine=False, kind=None, dedup=True):
        try:
            from researchpapers.fleet import post as _p
            # only dedup ROUTINE spam here; live/important messages pass through (post_thread applies the
            # same routine-gated rule + update_thread bypasses dedup, so status updates never break).
            if dedup and routine and self._is_duplicate(msg):
                return
            _p.post_thread(worker, to, msg, routine=routine, kind=kind or self.kind)
        except Exception:  # noqa: BLE001
            pass

    def log(self, summary, detail="", kind="finding", recommendation=""):
        try:
            from . import ledger
            ledger.log(self.name, summary=summary, detail=detail, kind=kind, recommendation=recommendation)
        except Exception:  # noqa: BLE001
            pass

    def record(self, change, cv=None, description=None, script="", train_set="golden12", lb=None,
               parent=None, stage=None, kept=None, observation=""):
        """Record a CV-RANKED experiment row in the journal (the sorted main table), not just a finding.
        Use whenever an agent produces a measured score/CV so training results show up ranked, not lost."""
        try:
            from . import ledger
            return ledger.record(change=change, script=script or self.name, cv=cv, lb=lb, train_set=train_set,
                                  parent=parent, stage=stage, kept=kept, observation=observation,
                                  description=description)
        except Exception:  # noqa: BLE001
            return {}

    def escalate(self, worker, to, msg):
        self.post(worker, to, msg, routine=False, kind="reason")
        return ("escalated", {}, to, msg)

    # ---- return contract ----
    def done(self, data, msg, to="all"):
        # FORCE composability: enrich every agent's output with the canonical flow keys (cv/config/nodes…)
        # so ANY agent chains to ANY agent in a workflow, regardless of how it names its own outputs.
        try:
            from . import flow
            if isinstance(data, dict):
                for ck, v in flow.canon(data).items():
                    data.setdefault(ck, v)
        except Exception:  # noqa: BLE001
            pass
        return ("done", data, to, msg)

    # ---- required ----
    def run(self, q, worker):
        raise NotImplementedError

    # kind → test-file stem where they differ (module name ≠ kind)
    _TEST_ALIAS = {"flow-gt-build": "flow_gt_builder", "conformal-predict": "conformal_prediction",
                   "dora-adapt": "dora_adapter",
                   "feasibility-map": "feasibility_gate",
                   "prompt-metric": "prompt_optloop", "prompt-dataset": "prompt_optloop"}

    def _test_stem(self) -> str:
        return self._TEST_ALIAS.get(self.name, self.name.replace("-", "_"))

    def verify(self) -> bool:
        """DATA-WISE self-test: run this agent's ground-truth verifier (test_fleet_agents/<name>_test.py)."""
        t = TESTS / f"{self._test_stem()}_test.py"
        if not t.exists():
            return False
        py = COMP / "research" / "cellmot_venv" / "bin" / "python"
        py = str(py) if py.exists() else sys.executable
        env = dict(os.environ); env["PYTHONPATH"] = str(COMP / "tools" / "researchpapers") + ":" + env.get("PYTHONPATH", "")
        r = subprocess.run([py, str(t)], capture_output=True, text=True, cwd=str(COMP), env=env, timeout=600)
        return r.returncode == 0

    def has_test(self) -> bool:
        return (TESTS / f"{self._test_stem()}_test.py").exists()

    def as_handler(self):
        return self.run


class FunctionAgent(BaseAgent):
    """Adapter: wraps an existing handler function as a BaseAgent so it gains the base contract/features
    and IS a BaseAgent (every registered agent extends BaseAgent, none missed)."""
    def __init__(self, name, fn, thread="S"):
        self.name = name; self.fn = fn; self.thread = thread

    def run(self, q, worker):
        # RESPECT the test: a quarantined agent (red data-wise test) does NOT run its (possibly wrong)
        # logic — it escalates to be fixed first. This makes tests governance, not decoration.
        if self.name in QUARANTINE:
            msg = (f"[{worker}] ⛔ AGENT `{self.name}` QUARANTINED — its data-wise test is RED. "
                   f"Not running its logic (could be wrong). Fix test_fleet_agents/{self._test_stem()}_test.py "
                   f"and re-run preflight before trusting this agent.")
            self.post(worker, "leader", msg, routine=False, kind="reason")
            return ("escalated", {"quarantined": self.name}, "leader", msg)
        return self.fn(q, worker)


def build_agents(handlers: dict) -> dict:
    """Wrap every handler as a BaseAgent instance → {kind: FunctionAgent}. Guarantees every agent extends
    BaseAgent and exposes the same features (spec/state/post/log/escalate/verify)."""
    return {kind: FunctionAgent(kind, fn) for kind, fn in handlers.items()}


def preflight(agent_names=None, post_summary=False, worker="preflight"):
    """Run the FAST data-wise verifiers (behavior-smoke covers all offline-testable agents) and QUARANTINE
    any agent whose test is RED. Returns {agent: passed}. Call at fleet startup so agents respect their tests.
    """
    py = COMP / "research" / "cellmot_venv" / "bin" / "python"
    py = str(py) if py.exists() else sys.executable
    env = dict(os.environ); env["PYTHONPATH"] = str(COMP / "tools" / "researchpapers") + ":" + env.get("PYTHONPATH", "")
    red = set()
    smoke = TESTS / "behavior_smoke_test.py"
    if smoke.exists():
        r = subprocess.run([py, str(smoke)], capture_output=True, text=True, cwd=str(COMP), env=env, timeout=600)
        for ln in r.stdout.splitlines():
            s = ln.strip()
            if s.startswith("❌"):
                name = s.split()[1] if len(s.split()) > 1 else ""
                if name:
                    red.add(name)
    QUARANTINE.clear(); QUARANTINE.update(red)
    _QFILE.parent.mkdir(parents=True, exist_ok=True)
    _QFILE.write_text(json.dumps({"red": sorted(red)}, indent=2))   # persist so the live fleet respects it
    if post_summary:
        try:
            from researchpapers.fleet import post as _p
            status = "⛔ RED agents quarantined: " + ", ".join(sorted(red)) if red else "✅ all agents green — none quarantined"
            _p.post_thread(worker, "all", f"[preflight] data-wise gate: {status}", routine=False, kind="verdict")
        except Exception:  # noqa: BLE001
            pass
    return {"red": sorted(red), "quarantined": len(red)}
