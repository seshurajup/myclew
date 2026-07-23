"""reasoning_exec_pack — the reasoning / agentic / meta EXECUTORS that are pure-Python runnable (ARC program
synthesis, code repair, judge attacks, test-time compute). Built real, verified offline:

  • program-search               — enumerate a small grid-transform DSL to find a program matching io examples (ARC).
  • program-synthesis-data-generator — sample DSL programs, apply to random grids → (input, program, output) triples.
  • program-golf-search          — program-search ranked by byte length (shortest correct program; code-golf).
  • fast-sim                     — vectorized BATCHED environment stepper (many envs stepped at once) for RL throughput.
  • code-repair-agent           — the localize→patch→VERIFY loop: apply a candidate patch, run F2P+P2P tests, accept/skip (Konwinski).
  • llm-judge-attacker          — craft inputs maximizing score DIVERGENCE across a judge panel (LLM-judge red-team).
  • ttc                         — test-time compute: augmentation-inverse-vote (AIRV) over a predictor (ARC lever).
"""
from __future__ import annotations
import itertools
import os
import subprocess
import sys
import time
import numpy as np
from .base import BaseAgent, COMP

# ---------------------------------------------------------------- grid-transform DSL
_OPS = {
    "id": lambda g: g, "fliplr": np.fliplr, "flipud": np.flipud, "transpose": lambda g: g.T,
    "rot90": lambda g: np.rot90(g), "add1": lambda g: g + 1, "mul2": lambda g: g * 2,
}


def _apply(prog, grid):
    g = np.asarray(grid)
    for op in prog:
        g = _OPS[op](g)
    return g


# ---------------------------------------------------------------- program-search
def program_search(examples, max_len=2, time_limit=None):
    """examples = [(input_grid, output_grid)]. Return the shortest op-sequence mapping every input→output.
    time_limit: wall-clock seconds to bound the enumeration (None = no bound). Guards empty examples → None."""
    if not examples:
        return None
    names = list(_OPS); t0 = time.time()
    for L in range(1, max(1, int(max_len)) + 1):
        for combo in itertools.product(names, repeat=L):
            if time_limit is not None and time.time() - t0 > time_limit:
                return None
            if all(np.array_equal(_apply(combo, i), np.asarray(o)) for i, o in examples):
                return list(combo)
    return None


def program_golf_search(examples, max_len=3, time_limit=None):
    """Shortest program (by serialized byte length) that solves all examples.
    time_limit: wall-clock seconds to bound the search (None = no bound). Guards empty examples → None."""
    if not examples:
        return None
    names = list(_OPS); best = None; t0 = time.time()
    for L in range(1, max(1, int(max_len)) + 1):
        for combo in itertools.product(names, repeat=L):
            if time_limit is not None and time.time() - t0 > time_limit:
                return best
            if all(np.array_equal(_apply(combo, i), np.asarray(o)) for i, o in examples):
                cand = list(combo)
                if best is None or len(",".join(cand)) < len(",".join(best)):
                    best = cand
        if best:
            break
    return best


# ---------------------------------------------------------------- program-synthesis-data-generator
def synthesize_data(n=50, prog_len=2, grid=4, seed=0):
    """Sample DSL programs, apply to random grids → (input, program, output) triples (self-verifying)."""
    n = max(0, int(n)); grid = max(1, int(grid)); prog_len = max(1, int(prog_len))
    rng = np.random.RandomState(int(seed)); names = list(_OPS); out = []
    for _ in range(n):
        prog = [names[rng.randint(len(names))] for _ in range(rng.randint(1, prog_len + 1))]
        g = rng.randint(0, 4, (grid, grid))
        out.append({"input": g.tolist(), "program": prog, "output": _apply(prog, g).tolist()})
    return out


# ---------------------------------------------------------------- fast-sim (batched env)
def batched_collect_step(positions, rewards_grid, actions):
    """Step B parallel 1-D collect envs at once. positions (B,), rewards_grid (B, N), actions (B,) in {0,1,2}.
    Returns (new_positions, per-env reward this step)."""
    pos = np.asarray(positions, int); A = np.asarray(actions, int); n = rewards_grid.shape[1]
    pos = np.clip(pos + (A == 1).astype(int) - (A == 2).astype(int), 0, n - 1)
    r = rewards_grid[np.arange(len(pos)), pos].copy()
    rewards_grid[np.arange(len(pos)), pos] = 0.0   # consume
    return pos, r


# ---------------------------------------------------------------- code-repair-agent (verify loop)
def verify_patch(candidate_code, test_code, timeout=15):
    """Run test_code against candidate_code in a subprocess; return whether the tests pass (the accept/skip gate)."""
    py = str(COMP / "research" / "cellmot_venv" / "bin" / "python")
    py = py if os.path.exists(py) else sys.executable
    prog = candidate_code + "\n" + test_code
    try:
        r = subprocess.run([py, "-c", prog], capture_output=True, text=True, timeout=timeout)
        return r.returncode == 0, (r.stderr.strip()[-200:] if r.returncode else "")
    except Exception as e:  # noqa: BLE001
        return False, str(e)


# ---------------------------------------------------------------- llm-judge-attacker
def craft_divergent_input(candidates, judges):
    """Pick the candidate input maximizing DISAGREEMENT (std) across a panel of judge score-functions.
    Guards empty candidates/judges → (None, 0.0, None)."""
    if not candidates or not judges:
        return None, 0.0, None
    best, best_div, scores_all = None, -1.0, None
    for c in candidates:
        scores = [float(j(c)) for j in judges]
        div = float(np.std(scores))
        if div > best_div:
            best, best_div, scores_all = c, div, scores
    return best, best_div, scores_all


# ---------------------------------------------------------------- ttc (test-time compute / AIRV)
def airv_vote(grid, predict_fn, augments):
    """Augmentation-inverse-vote: for each (fwd, inv) augmentation, predict on the augmented input, invert,
    and majority-vote the results. augments = [(fwd, inv)]. predict_fn(grid)->grid.
    Guards empty augments → the raw prediction on the identity grid."""
    if not augments:
        return np.asarray(predict_fn(np.asarray(grid)))
    votes = {}
    for fwd, inv in augments:
        pred = inv(predict_fn(fwd(np.asarray(grid))))
        key = np.asarray(pred).tobytes()
        votes.setdefault(key, [0, pred]); votes[key][0] += 1
    return max(votes.values(), key=lambda v: v[0])[1]


# ---------------------------------------------------------------- agents
class _B(BaseAgent):
    thread = "M"; kind = "finding"


class ProgramSearch(_B):
    name = "program-search"
    def run(self, q, worker):
        s = self.spec(q)
        missing = [k for k in ("examples",) if k not in s]
        if missing: return self.escalate(worker, "leader", f"program-search needs spec keys {missing} — none provided")
        ex = [(np.asarray(e[0]), np.asarray(e[1])) for e in s["examples"]]
        prog = program_search(ex, int(s.get("max_len", 2)), time_limit=s.get("time_limit"))
        msg = f"program-search: {'found program ' + str(prog) if prog else 'no program within depth'}"
        self.log(msg, kind="finding", recommendation="grow the DSL / depth for harder ARC tasks; ensemble w/ LLM")
        return self.done({"program": prog}, msg)


class ProgramGolf(_B):
    name = "program-golf-search"
    def run(self, q, worker):
        s = self.spec(q)
        missing = [k for k in ("examples",) if k not in s]
        if missing: return self.escalate(worker, "leader", f"program-golf-search needs spec keys {missing} — none provided")
        ex = [(np.asarray(e[0]), np.asarray(e[1])) for e in s["examples"]]
        prog = program_golf_search(ex, int(s.get("max_len", 3)), time_limit=s.get("time_limit"))
        msg = f"program-golf-search: shortest program {prog}"
        self.log(msg, kind="finding", recommendation="then byte-minimize via compression (code-compress-optimizer)")
        return self.done({"program": prog}, msg)


class SynthData(_B):
    name = "program-synthesis-data-generator"
    def run(self, q, worker):
        s = self.spec(q); data = synthesize_data(int(s.get("n", 50)), int(s.get("prog_len", 2)),
                                                 grid=int(s.get("grid", 4)), seed=int(s.get("seed", 0)))
        msg = f"program-synthesis-data-generator: generated {len(data)} self-verifying (input,program,output) triples"
        self.log(msg, kind="finding", recommendation="train a solver on these; filter by multi-sample consistency")
        return self.done({"n": len(data), "sample": data[:2]}, msg)


class FastSim(_B):
    name = "fast-sim"
    def run(self, q, worker):
        s = self.spec(q)
        missing = [k for k in ("positions", "rewards_grid", "actions") if k not in s]
        if missing: return self.escalate(worker, "leader", f"fast-sim needs spec keys {missing} — none provided")
        pos, r = batched_collect_step(s["positions"], np.asarray(s["rewards_grid"], float), s["actions"])
        msg = f"fast-sim: stepped {len(pos)} parallel envs (batched); total reward {float(r.sum()):.2f}"
        self.log(msg, kind="finding", recommendation="vectorize the comp's env for RL throughput (100x+)")
        return self.done({"positions": pos.tolist(), "reward": r.tolist()}, msg)


class CodeRepair(_B):
    name = "code-repair-agent"
    def run(self, q, worker):
        s = self.spec(q)
        missing = [k for k in ("candidate_code", "test_code") if k not in s]
        if missing: return self.escalate(worker, "leader", f"code-repair-agent needs spec keys {missing} — none provided")
        ok, err = verify_patch(s["candidate_code"], s["test_code"])
        msg = f"code-repair-agent: patch {'PASSES tests → submit' if ok else 'FAILS → skip'}" + (f" ({err[:60]})" if err else "")
        self.log(msg, kind="finding", recommendation="submit only if F2P+P2P pass; skip to avoid the wrong-patch penalty")
        return self.done({"tests_pass": ok, "error": err}, msg)


class JudgeAttacker(_B):
    name = "llm-judge-attacker"
    def run(self, q, worker):
        s = self.spec(q)
        missing = [k for k in ("candidates", "judge_scores") if k not in s]
        if missing: return self.escalate(worker, "leader", f"llm-judge-attacker needs spec keys {missing} — none provided")
        # judges provided as (name -> score value per candidate) precomputed table for offline testing
        cand = s["candidates"]; judge_scores = s["judge_scores"]  # list of lists: judge_scores[c] = [s_j...]
        best_i, best_div = -1, -1.0
        for i, sc in enumerate(judge_scores):
            div = float(np.std(sc))
            if div > best_div:
                best_i, best_div = i, div
        msg = f"llm-judge-attacker: candidate {best_i} maximizes judge divergence ({best_div:.3f})"
        self.log(msg, kind="finding", recommendation="verify against a LOCAL judge panel; defensive/eval framing only")
        return self.done({"chosen": best_i, "divergence": best_div}, msg)


class Ttc(_B):
    name = "ttc"
    def run(self, q, worker):
        s = self.spec(q)
        missing = [k for k in ("candidates",) if k not in s]
        if missing: return self.escalate(worker, "leader", f"ttc needs spec keys {missing} — none provided")
        # candidates = list of predicted grids (from test-time augmented inference); majority-vote (AIRV)
        cands = [np.asarray(c) for c in s["candidates"]]
        votes = {}
        for c in cands:
            k = c.tobytes(); votes.setdefault(k, [0, c]); votes[k][0] += 1
        best = max(votes.values(), key=lambda v: v[0])
        msg = f"ttc: AIRV majority-vote over {len(cands)} augmented predictions ({best[0]}/{len(cands)} agree)"
        self.log(msg, kind="finding", recommendation="pair with per-task test-time fine-tuning for the ARC lever")
        return self.done({"prediction": best[1].tolist(), "agreement": best[0]}, msg)


_PS = ProgramSearch(); _PG = ProgramGolf(); _SD = SynthData(); _FS = FastSim(); _CR = CodeRepair(); _JA = JudgeAttacker(); _TC = Ttc()


def run_ttc(q, worker): return _TC.run(q, worker)


def run_progsearch(q, worker): return _PS.run(q, worker)
def run_proggolf(q, worker): return _PG.run(q, worker)
def run_synthdata(q, worker): return _SD.run(q, worker)
def run_fastsim(q, worker): return _FS.run(q, worker)
def run_coderepair(q, worker): return _CR.run(q, worker)
def run_judgeattack(q, worker): return _JA.run(q, worker)
