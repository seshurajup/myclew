"""prompt-metric — the METRIC source for prompt optimization. Turns a NAMED metric (a JSON-board-safe string)
into the two callables the optimizers need: a score fn `score(pred, gold) -> [0,1]` and a `feedback(pred, gold)
-> str` (what's wrong — the reflection signal GEPA/APEX mutate on). This is the piece that lets dspy-prompt-
optimize be driven over the board: you can't send a Python metric function through JSON, but you CAN send the
NAME of a metric + its params, and this agent reconstructs the callable in-process.

Named metrics (extend freely): exact_match, norm_exact (case/space/punct-insensitive), contains, token_f1
(word-overlap F1), numeric (|Δ| ≤ tol), multiple_choice (A/B/C/D letter), regex_match (extract-then-compare),
json_field (compare a JSON field), keyword_coverage (fraction of required tokens present — LM-free demo metric).
For a competition's own metric, pass name='official' and this defers to the fleet's scorer/official-score.

Pure-python/stdlib; a BaseAgent with a data-wise test. Reused by dspy-prompt-optimize and any prompt loop.
"""
from __future__ import annotations
import json
import re
from .base import BaseAgent

_WORD = re.compile(r"[a-z0-9]+")


def _norm(s):
    return " ".join(_WORD.findall(str(s).lower()))


def _num(s):
    m = re.search(r"-?\d+\.?\d*", str(s))
    return float(m.group()) if m else None


def build_metric(name="norm_exact", spec=None):
    """Return (score_fn, feedback_fn). score_fn(pred, gold)->float in [0,1]; feedback_fn(pred, gold)->str.
    `name` is a JSON-safe string; `spec` carries params (tol, required, field, pattern, choices)."""
    s = spec or {}
    name = (name or "norm_exact").lower()

    def fb_default(pred, gold, ok):
        return "" if ok else f"expected ~ '{str(gold)[:60]}' but got '{str(pred)[:60]}'"

    if name in ("exact", "exact_match"):
        def score(p, g): return 1.0 if str(p).strip() == str(g).strip() else 0.0
        def feedback(p, g): return fb_default(p, g, score(p, g) == 1.0)

    elif name in ("norm_exact", "normalized"):
        def score(p, g): return 1.0 if _norm(p) == _norm(g) else 0.0
        def feedback(p, g): return fb_default(p, g, score(p, g) == 1.0)

    elif name == "contains":
        def score(p, g): return 1.0 if _norm(g) in _norm(p) else 0.0
        def feedback(p, g): return "" if score(p, g) else f"answer must contain '{str(g)[:60]}'"

    elif name in ("token_f1", "f1"):
        def score(p, g):
            pt, gt = set(_WORD.findall(_norm(p))), set(_WORD.findall(_norm(g)))
            if not pt or not gt:
                return 1.0 if pt == gt else 0.0
            inter = len(pt & gt); prec = inter / len(pt); rec = inter / len(gt)
            return 0.0 if inter == 0 else 2 * prec * rec / (prec + rec)
        def feedback(p, g):
            miss = set(_WORD.findall(_norm(g))) - set(_WORD.findall(_norm(p)))
            return "" if not miss else "missing: " + ", ".join(sorted(miss))

    elif name in ("numeric", "num_tol"):
        tol = float(s.get("tol", 1e-6))
        def score(p, g):
            a, b = _num(p), _num(g)
            return 1.0 if (a is not None and b is not None and abs(a - b) <= tol) else 0.0
        def feedback(p, g):
            a, b = _num(p), _num(g)
            if a is None: return "no number found in prediction"
            return "" if score(p, g) else f"got {a}, need {b} (±{tol})"

    elif name in ("multiple_choice", "mcq"):
        def _letter(x):
            m = re.search(r"\b([A-Da-d])\b", str(x))
            return m.group(1).upper() if m else None
        def score(p, g): return 1.0 if (_letter(p) and _letter(p) == _letter(g)) else 0.0
        def feedback(p, g): return "" if score(p, g) else f"choose {_letter(g)} (got {_letter(p)})"

    elif name in ("regex_match", "regex"):
        pat = re.compile(s.get("pattern", r"-?\d+\.?\d*"))
        def _ext(x):
            m = pat.search(str(x)); return m.group(m.lastindex or 0) if m else None
        def score(p, g): return 1.0 if (_ext(p) is not None and _norm(_ext(p)) == _norm(g)) else 0.0
        def feedback(p, g): return "" if score(p, g) else f"pattern-extract must equal '{str(g)[:40]}'"

    elif name == "json_field":
        field = s.get("field", "answer")
        def _f(x):
            try:
                return json.loads(x).get(field)
            except Exception:  # noqa: BLE001
                m = re.search(rf'"{field}"\s*:\s*"?([^",}}]+)', str(x))
                return m.group(1).strip() if m else None
        def score(p, g): return 1.0 if _f(p) is not None and _norm(_f(p)) == _norm(g) else 0.0
        def feedback(p, g): return "" if score(p, g) else f"JSON field '{field}' must equal '{str(g)[:40]}'"

    elif name in ("keyword_coverage", "keywords"):
        req = [str(t).lower() for t in (s.get("required") or [])]
        def score(p, g):
            need = [t for t in (g if isinstance(g, (list, tuple)) else req)]
            if not need: return 1.0
            have = set(_WORD.findall(_norm(p)))
            return sum(1 for t in need if t in have) / len(need)
        def feedback(p, g):
            need = [t for t in (g if isinstance(g, (list, tuple)) else req)]
            have = set(_WORD.findall(_norm(p)))
            miss = [t for t in need if t not in have]
            return "" if not miss else "missing: " + ", ".join(miss)

    else:
        raise ValueError(f"unknown metric '{name}'. Known: exact, norm_exact, contains, token_f1, numeric, "
                         f"multiple_choice, regex_match, json_field, keyword_coverage")
    return score, feedback


# DSPy adapter: wrap (score_fn) as a DSPy metric(example, pred, trace=None) using a configurable field.
def dspy_metric(name="norm_exact", spec=None, input_field="question", output_field="answer"):
    score, _ = build_metric(name, spec)
    def metric(example, pred, trace=None):
        gold = getattr(example, output_field, None)
        got = getattr(pred, output_field, None) if not isinstance(pred, str) else pred
        return float(score(got, gold))
    return metric


KNOWN = ["exact", "norm_exact", "contains", "token_f1", "numeric", "multiple_choice",
         "regex_match", "json_field", "keyword_coverage"]


class PromptMetric(BaseAgent):
    name = "prompt-metric"
    thread = "S"
    kind = "finding"

    def run(self, q, worker):
        spec = self.spec(q)
        name = spec.get("metric") or spec.get("name") or "norm_exact"
        try:
            score, feedback = build_metric(name, spec)
        except ValueError as e:
            return self.escalate(worker, "researcher", f"prompt-metric: {e}")
        # self-check on provided probes (pred, gold pairs) or a built-in sanity probe
        probes = spec.get("probes") or [("Paris", "paris"), ("London", "paris")]
        rows = [{"pred": p, "gold": g, "score": round(float(score(p, g)), 3), "feedback": feedback(p, g)}
                for p, g in probes]
        msg = (f"prompt-metric: '{name}' ready → score(pred,gold)∈[0,1] + feedback(pred,gold) for reflection. "
               f"probe: {rows[0]['pred']!r} vs {rows[0]['gold']!r} = {rows[0]['score']}. "
               f"Feed to dspy-prompt-optimize (metric='{name}') or run_dspy_optimizer via prompt_metric.dspy_metric.")
        self.log(msg, kind="finding", recommendation="pair with prompt-dataset (examples) to drive real optimization")
        return self.done({"metric": name, "probes": rows, "known": KNOWN,
                          "score_fn": "prompt_metric.build_metric", "dspy": "prompt_metric.dspy_metric"}, msg)


_AGENT = PromptMetric()


def run(q, worker):
    return _AGENT.run(q, worker)
