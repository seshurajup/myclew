"""prompt-dataset — the DATASET source for prompt optimization. Builds a trainset of {input, gold} examples
(the thing DSPy/GEPA measure a prompt against) from a JSON-board-safe spec, so dspy-prompt-optimize can be
driven without a Python caller hand-constructing dspy.Example objects. Sources, in priority order:

  spec['examples'] : inline list of {input, gold} (or [input, gold] pairs)         — most direct
  spec['file']     : path to .jsonl / .json / .csv with input_field/output_field   — bring your own data
  spec['synthetic']: a named generator (arithmetic, sentiment, multiple_choice)    — self-contained demo

Returns a JSON-safe list + a train/val split. `to_dspy()` converts to dspy.Example (lazy import; only when the
DSPy path is used). Pairs with prompt-metric (the score) — together they are the two inputs a prompt optimizer
needs. A BaseAgent with a data-wise test; stdlib only for the core (dspy import is optional + lazy).
"""
from __future__ import annotations
import csv
import json
import os
from .base import BaseAgent


def _coerce(ex, in_field, out_field):
    if isinstance(ex, dict):
        gi = ex.get(in_field, ex.get("input", ex.get("question", ex.get("text"))))
        go = ex.get(out_field, ex.get("gold", ex.get("answer", ex.get("label"))))
        return {"input": gi, "gold": go}
    if isinstance(ex, (list, tuple)) and len(ex) >= 2:
        return {"input": ex[0], "gold": ex[1]}
    return None


def _synthetic(name, n, seed):
    import random
    rng = random.Random(seed)
    out = []
    if name in ("arithmetic", "math"):
        for _ in range(n):
            a, b = rng.randint(1, 50), rng.randint(1, 50)
            out.append({"input": f"What is {a} + {b}?", "gold": str(a + b)})
    elif name in ("sentiment", "classify"):
        pos = ["great", "loved it", "excellent", "wonderful"]; neg = ["terrible", "hated it", "awful", "bad"]
        for _ in range(n):
            if rng.random() < 0.5:
                out.append({"input": f"Review: {rng.choice(pos)}.", "gold": "positive"})
            else:
                out.append({"input": f"Review: {rng.choice(neg)}.", "gold": "negative"})
    elif name in ("multiple_choice", "mcq"):
        for _ in range(n):
            ans = rng.choice("ABCD")
            out.append({"input": f"Q. Pick the marked option. A) x B) y C) z D) w  [correct={ans}]", "gold": ans})
    else:
        raise ValueError(f"unknown synthetic generator '{name}'. Known: arithmetic, sentiment, multiple_choice")
    return out


def build_trainset(spec):
    """Return {'examples': [{input, gold}], 'train': [...], 'val': [...], 'source': str}. JSON-safe."""
    s = spec or {}
    in_field = s.get("input_field", "input"); out_field = s.get("output_field", "gold")
    exs, source = [], None
    if s.get("examples"):
        exs = [e for e in (_coerce(e, in_field, out_field) for e in s["examples"]) if e and e["input"] is not None]
        source = "inline"
    elif s.get("file"):
        path = s["file"]; source = f"file:{os.path.basename(path)}"
        if not os.path.exists(path):
            raise FileNotFoundError(f"prompt-dataset file not found: {path}")
        if path.endswith(".jsonl"):
            rows = [json.loads(ln) for ln in open(path) if ln.strip()]
        elif path.endswith(".json"):
            rows = json.load(open(path)); rows = rows if isinstance(rows, list) else rows.get("examples", [])
        elif path.endswith(".csv"):
            rows = list(csv.DictReader(open(path)))
        else:
            raise ValueError("file must be .jsonl/.json/.csv")
        exs = [e for e in (_coerce(r, in_field, out_field) for r in rows) if e and e["input"] is not None]
    elif s.get("synthetic"):
        exs = _synthetic(s["synthetic"], int(s.get("n", 12)), int(s.get("seed", 0)))
        source = f"synthetic:{s['synthetic']}"
    else:
        raise ValueError("prompt-dataset needs one of: spec['examples'], spec['file'], spec['synthetic']")

    if not exs:
        raise ValueError("prompt-dataset produced 0 examples")
    frac = float(s.get("val_frac", 0.3))
    cut = max(1, int(round(len(exs) * (1 - frac)))) if len(exs) > 2 else len(exs)
    return {"examples": exs, "train": exs[:cut], "val": exs[cut:], "source": source, "n": len(exs)}


def to_dspy(examples, input_field="question", output_field="answer"):
    """Convert [{input, gold}] → [dspy.Example(...).with_inputs(input_field)]. Lazy dspy import."""
    import dspy
    return [dspy.Example(**{input_field: e["input"], output_field: e["gold"]}).with_inputs(input_field)
            for e in examples]


class PromptDataset(BaseAgent):
    name = "prompt-dataset"
    thread = "S"
    kind = "finding"

    def run(self, q, worker):
        spec = self.spec(q)
        try:
            ts = build_trainset(spec)
        except (ValueError, FileNotFoundError) as e:
            return self.escalate(worker, "researcher",
                                 f"prompt-dataset: {e} — pass spec['examples'] (inline), spec['file'] "
                                 f"(.jsonl/.json/.csv), or spec['synthetic'] (arithmetic|sentiment|multiple_choice).")
        sample = ts["examples"][0]
        msg = (f"prompt-dataset: built {ts['n']} examples from {ts['source']} "
               f"(train {len(ts['train'])} / val {len(ts['val'])}). e.g. input={str(sample['input'])[:50]!r} "
               f"gold={str(sample['gold'])[:30]!r}. Pair with prompt-metric → dspy-prompt-optimize.")
        self.log(msg, kind="finding", recommendation="feed train/val to dspy-prompt-optimize as spec['examples']")
        return self.done({"n": ts["n"], "train": ts["train"], "val": ts["val"], "source": ts["source"],
                          "examples": ts["examples"]}, msg)


_AGENT = PromptDataset()


def run(q, worker):
    return _AGENT.run(q, worker)
