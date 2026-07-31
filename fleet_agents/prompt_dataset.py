"""prompt-dataset — the DATASET source for prompt optimization. Builds a trainset of {input, gold} examples
(the thing DSPy/GEPA measure a prompt against) from a JSON-board-safe spec, so dspy-prompt-optimize can be
driven without a Python caller hand-constructing dspy.Example objects. Sources, in priority order:

  spec['examples'] : inline list of {input, gold} (or [input, gold] pairs)         — most direct
  spec['file']     : path to .jsonl / .json / .csv with input_field/output_field   — bring your own data
  spec['synthetic']: a named generator (arithmetic, sentiment, multiple_choice)    — self-contained demo
  spec['synthetic'] == 'tool_calls' : VERIFIED tool-call SFT data for the fleet's own two-tool contract,
                     built from the live 320-agent capability index — see build_tool_call_dataset()
  spec['synthetic'] == 'ledger'     : COMPETITION-GROUNDED (state -> next action) examples whose reward is
                     the real private-score gain — see build_ledger_tool_dataset()

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
    import random as _rnd
    _tr, _va = _split_by_prompt(exs, s.get("val_frac", 0.3), _rnd.Random(int(s.get("seed", 7))))
    return {"examples": exs, "train": _tr, "val": _va, "source": source, "n": len(exs)}


def to_dspy(examples, input_field="question", output_field="answer"):
    """Convert [{input, gold}] → [dspy.Example(...).with_inputs(input_field)]. Lazy dspy import."""
    import dspy
    return [dspy.Example(**{input_field: e["input"], output_field: e["gold"]}).with_inputs(input_field)
            for e in examples]


# ---------------------------------------------------------------- tool-call SFT data (the local-LLM pilot)
# Builds supervised examples that teach a small local model (Gemma-class, via Ollama/llama.cpp) to drive the
# fleet's two-tool contract from `agent_routing`: search_capabilities -> execute_capability.
#
# THE DESIGN DECISION, stated up front: we train the PROTOCOL, not the inventory.
#   Teaching a 4B model to name the right one of 320 agents from memory is both brittle (the roster changes
#   every time we add an agent) and the hardest part of the job. But `search_capabilities` ALREADY solves
#   retrieval — it is a scored index. So the skill actually worth training is:
#       (a) turn an ML request into a good search query,
#       (b) pick correctly FROM THE RETURNED SHORTLIST,
#       (c) fill `spec` with schema-valid types and echo `schema_digest`,
#       (d) recover from each structured error EXACTLY as the retry protocol prescribes.
#   All four survive a roster change; inventory memorisation does not.
#
# THE QUALITY MECHANISM, borrowed from the paper packs:
#   * every example is machine-verified through `execute_capability` before it enters the dataset — the
#     perfect-verifier property (afp unit 2). A "correct" label that the real validator rejects is a bug,
#     and here it cannot reach the training set;
#   * expected-FAILURE examples must return the exact error code they claim, not merely fail;
#   * hard negatives are the IDF matcher's RUNNER-UP — the plausible wrong pick (egm eq. 2's hardness idea);
#   * rejected candidates are kept and reported, never silently dropped (mse eq. 3's selection bias);
#   * a leakage gate measures request-vs-summary lexical overlap, because a templated generator can teach
#     "copy the rare word" instead of intent.
_TC_FRAMES = [
    "{v} {obj}", "I need to {v} {obj}", "can you {v} {obj}?", "please {v} {obj}",
    "next step: {v} {obj}", "help me {v} {obj}", "we should {v} {obj}",
    "the plan is to {v} {obj}", "goal — {v} {obj}", "task: {v} {obj}",
]
_TC_VERBS = ["run", "do", "set up", "handle", "take care of", "get going on", "kick off", "start"]
_TC_STOP = {"the", "a", "an", "of", "for", "to", "and", "or", "in", "on", "with", "by", "from",
            "this", "that", "its", "it", "is", "are", "be", "as", "at", "so", "we", "our", "per",
            "into", "than", "then", "not", "no", "any", "all", "one", "two", "you", "your", "can"}


def _tc_words(text):
    import re
    return [w for w in re.findall(r"[a-z0-9]+", (text or "").lower()) if w not in _TC_STOP]


# Domain synonyms, so a "hard" request can say the same thing in DIFFERENT words. Without this the
# generator produces requests that share ~77% of their tokens with the summary (measured), and a model
# can then solve every example by copying the rare word instead of understanding the intent — which the
# report's `mean_request_summary_overlap` gate exists to catch, and did.
_TC_SYN = {
    "quantization": ["8-bit conversion", "shrinking numeric precision"],
    "quantize": ["shrink numerically", "cut precision"], "quantized": ["reduced-precision"],
    "int8": ["8-bit"], "4bit": ["4-bit"], "low": ["reduced"], "bit": ["precision"],
    "cv": ["held-out evaluation"], "fold": ["evaluation split"], "folds": ["evaluation splits"],
    "split": ["partition"], "splits": ["partitions"], "validation": ["held-out checking"],
    "train": ["fit"], "training": ["fitting"], "trainer": ["fitting loop"],
    "detector": ["object finder"], "detection": ["finding objects"], "detect": ["find"],
    "tracker": ["trajectory linker"], "tracking": ["linking over time"],
    "blend": ["combine predictions"], "ensemble": ["committee of models"],
    "calibrate": ["fix probability scaling"], "calibration": ["probability scaling"],
    "notebook": ["submission script"], "submission": ["competition entry"],
    "kaggle": ["the competition platform"], "leaderboard": ["public standings"],
    "prompt": ["instruction text"], "agent": ["automated worker"], "agents": ["automated workers"],
    "pipeline": ["ordered chain of steps"], "score": ["metric value"], "metric": ["evaluation number"],
    "recall": ["fraction found"], "precision": ["fraction correct"],
    "gpu": ["accelerator"], "cuda": ["accelerator code"], "kernel": ["low-level routine"],
    "memory": ["footprint"], "latency": ["response time"], "throughput": ["items per second"],
    "attention": ["token mixing"], "transformer": ["attention network"], "layer": ["stage"],
    "embedding": ["vector representation"], "embeddings": ["vector representations"],
    "retrieval": ["search over documents"], "search": ["look up"], "index": ["lookup table"],
    "paper": ["publication"], "papers": ["publications"], "lesson": ["teaching unit"],
    "config": ["settings file"], "yml": ["settings file"], "spec": ["arguments"],
    "data": ["inputs"], "dataset": ["collection of inputs"], "features": ["input columns"],
    "model": ["network"], "models": ["networks"], "weights": ["parameters"],
    "loss": ["objective"], "optimizer": ["update rule"], "gradient": ["derivative signal"],
    "evidence": ["proof"], "gate": ["admission check"], "audit": ["review"],
    "report": ["write-up"], "graph": ["node-link structure"], "node": ["vertex"],
}


def _tc_paraphrase(cap, rng, hard=False):
    """A natural-sounding request for `cap`.

    hard=True is the anti-leakage path: drop the agent's own name tokens AND substitute domain synonyms,
    so the example states the same intent in different words. Measured effect on the corpus:
    request-vs-summary overlap falls from ~0.77 to well under half, and `low_overlap_frac` stops being 0.
    """
    summary = cap.get("summary") or cap["name"].replace("-", " ")
    words = _tc_words(summary)[:14]
    name_toks = set(_tc_words(cap["name"].replace("-", " ")))
    if hard:
        words = [w for w in words if w not in name_toks]
        out = []
        for w in words:
            alts = _TC_SYN.get(w)
            out.append(rng.choice(alts) if alts else w)
        words = [w for w in out if w]
        if len(words) > 7:                                    # keep it terse, drop mid-sentence filler
            words = words[:3] + words[-4:]
    obj = " ".join(words[:9]) or cap["name"].replace("-", " ")
    frame = rng.choice(_TC_FRAMES)
    return frame.format(v=rng.choice(_TC_VERBS), obj=obj).strip()


def _tc_valid_spec(cap, rng):
    """A schema-valid spec: start from the roster's own working example, then perturb VALUES (never types)
    so the model sees variety instead of one memorised constant."""
    schema, ex = cap.get("spec_schema") or {}, cap.get("spec_example") or {}
    out = {}
    for k, t in schema.items():
        base = ex.get(k)
        if t == "integer":
            out[k] = int(base) + rng.randint(0, 3) if isinstance(base, int) else rng.randint(1, 8)
        elif t == "number":
            out[k] = round(float(base) * rng.choice([0.5, 1.0, 2.0]), 4) if isinstance(
                base, (int, float)) else round(rng.random(), 3)
        elif t == "boolean":
            out[k] = bool(rng.getrandbits(1))
        elif t == "array":
            out[k] = list(base) if isinstance(base, (list, tuple)) else []
        else:
            out[k] = base if isinstance(base, str) and base else "auto"
    return out


def _tc_wrong_type(spec, schema, rng):
    """Break exactly ONE key's TYPE — the mistake `invalid_capability_arguments` exists to catch."""
    typed = [k for k, t in schema.items() if t in ("integer", "number", "boolean", "array")]
    if not typed:
        return None, None
    k = rng.choice(sorted(typed))
    bad = dict(spec)
    bad[k] = {"integer": "twelve", "number": "a lot", "boolean": "yes", "array": "one,two"}[schema[k]]
    return bad, k



def _split_by_prompt(examples, val_frac, rng):
    """Split train/val by PROMPT, never by example.

    Measured defect: these corpora repeat prompts — the same fleet state recurs, so a random example-level
    split put 21.3% of val inputs (60.6% of `next_agent`) into train as well. The model then scored on
    prompts it had memorised, and `next_agent` read 100% when the real decision skill was untested. Every
    example sharing an input string now lands on the SAME side.
    """
    groups = {}
    for e in examples:
        groups.setdefault(e["input"], []).append(e)
    keys = sorted(groups)
    rng.shuffle(keys)
    cut = max(1, int(len(keys) * (1 - float(val_frac))))
    train = [e for k in keys[:cut] for e in groups[k]]
    val = [e for k in keys[cut:] for e in groups[k]]
    return train, val


def build_tool_call_dataset(spec):
    """{'examples': [...], 'report': {...}} — verified SFT data for the two-tool protocol.

    spec keys: n (total target), seed, hard_frac (share of lexically-stripped requests), val_frac,
               kinds (subset of search|execute|recover_unknown|recover_args), shortlist (search limit).
    Every example carries `messages` (chat-format) plus the raw fields, so it feeds a LoRA SFT run directly.
    """
    import json as _json
    import json as _json_mod
    import random as _random
    from . import agent_routing as AR

    s = spec or {}
    rng = _random.Random(int(s.get("seed", 0)))
    n_target = int(s.get("n", 400))
    hard_frac = float(s.get("hard_frac", 0.35))
    shortlist = int(s.get("shortlist", 6))
    kinds = list(s.get("kinds") or ["search", "execute", "recover_unknown", "recover_args"])

    index = AR.capability_index()
    routable = [c for c in index.values() if (c.get("summary") or "").strip()]
    with_schema = [c for c in routable if c.get("spec_schema")]
    if not routable:
        raise ValueError("prompt-dataset tool_calls: the capability index is empty — is fleet_agents importable?")

    fmt = s.get("format", "compact")
    sys_msg = _sys_for(fmt)
    examples, rejected = [], []

    def _stub(q, worker):                                     # never runs a real agent while building data
        return ("done", {"stub": True}, None, "stub")

    def _emit(kind, cap, request, gold_obj, turns, verify):
        """Verify with the REAL validator, then keep or reject — nothing unverified enters the set."""
        vstatus, vdetail = verify()
        if not vstatus:
            rejected.append({"kind": kind, "capability": cap["name"], "why": vdetail})
            return
        msgs = [{"role": "system", "content": sys_msg}]
        for role, content in turns:
            msgs.append({"role": role, "content": content})
        rendered = _render_gold(gold_obj, fmt)
        msgs.append({"role": "assistant", "content": rendered})
        examples.append({"kind": kind, "capability": cap["name"], "input": request,
                         "gold": rendered, "gold_obj": gold_obj, "format": fmt,
                         "messages": msgs, "verified": vdetail, "hard": bool(hard)})

    # MEASURED saturation (gemma3n:e4b, 211 held-out): search 100%, recover_unknown 100%,
    # recover_args 94.6%, execute 70.2% exact. An equal split therefore spends >half the corpus on skills
    # the model already has. `kind_weights` shifts the mass onto `execute` — the one that is not solved —
    # while keeping a small anchor slice of the easy kinds so the output FORMAT does not drift.
    default_w = {"execute": 0.55, "recover_args": 0.25, "search": 0.12, "recover_unknown": 0.08}
    weights = dict(s.get("kind_weights") or default_w)
    tot_w = sum(weights.get(k, 0.0) for k in kinds) or 1.0
    for kind in kinds:
        per = max(1, int(n_target * weights.get(kind, 0.0) / tot_w))
        pool = with_schema if kind in ("execute", "recover_args") else routable
        if not pool:
            continue
        for i in range(per):
            cap = pool[(i * 7 + rng.randrange(len(pool))) % len(pool)]
            hard = rng.random() < hard_frac
            request = _tc_paraphrase(cap, rng, hard=hard)

            if kind == "search":
                # (a) query formulation: the gold is a search call whose query RETRIEVES the target
                request = MODE_DISCOVER + "\n" + request
                query = " ".join(_tc_words(cap.get("summary") or cap["name"])[:8]) or cap["name"]
                gold = {"tool": "search_capabilities", "query": query}
                def verify(cap=cap, query=query):
                    hits = [m["name"] for m in AR.search_capabilities(query, limit=shortlist)["matches"]]
                    return (cap["name"] in hits,
                            f"target at rank {hits.index(cap['name']) + 1}/{len(hits)}"
                            if cap["name"] in hits else "target not retrieved by its own gold query")
                _emit(kind, cap, request, gold, [("user", request)], verify)

            elif kind == "execute":
                # (b)+(c) pick from the shortlist, fill a schema-valid spec, echo the digest
                # a LONGER shortlist = more plausible distractors, which is what makes `execute` the
                # discriminative kind rather than a formatting exercise
                res = AR.search_capabilities(
                    " ".join(_tc_words(cap.get("summary") or cap["name"])[:8]),
                    limit=max(shortlist, int(s.get("exec_shortlist", 10))))
                shown = [{"name": m["name"], "summary": m["summary"][:110],
                          "spec_schema": m["spec_schema"], "schema_digest": m["schema_digest"]}
                         for m in res["matches"]]
                if cap["name"] not in [m["name"] for m in shown]:
                    rejected.append({"kind": kind, "capability": cap["name"],
                                     "why": "not in its own shortlist"})
                    continue
                # SHUFFLE — measured defect: the shortlist is built by searching the target's OWN summary,
                # so the gold landed at rank 1 in 94.8% of examples. "Always pick the first entry" then
                # scored 94.8% and a 2B model duly learned exactly that, reporting a meaningless 100%.
                # Randomising the order forces the model to READ the summaries. The `gold_rank1_frac` gate
                # below keeps this honest if the generator is ever changed again.
                rng.shuffle(shown)
                sp = _tc_valid_spec(cap, rng)
                gold = {"tool": "execute_capability", "name": cap["name"], "spec": sp,
                        "schema_digest": cap["schema_digest"]}
                def verify(cap=cap, sp=sp):
                    r = AR.execute_capability(cap["name"], sp, schema_digest_echo=cap["schema_digest"],
                                              dispatch=_stub)
                    return (bool(r.get("ok")), f"validator ok, {len(sp)} spec key(s)"
                            if r.get("ok") else f"validator said {r.get('error')}")
                _emit(kind, cap, request,
                      gold, [("user", request),
                             ("user", "Tool result:\n" + _json.dumps({"matches": shown}))], verify)

            elif kind == "recover_unknown":
                # (d) hallucinated name -> unknown_capability -> the protocol says SEARCH AGAIN
                fake = cap["name"].replace("-", "_") + "_v2"
                err = AR.execute_capability(fake, {}, dispatch=_stub)
                query = " ".join(_tc_words(cap.get("summary") or cap["name"])[:8]) or cap["name"]
                gold = {"tool": "search_capabilities", "query": query}
                def verify(err=err, query=query, cap=cap):
                    if err.get("error") != "unknown_capability":
                        return False, f"expected unknown_capability, got {err.get('error')}"
                    if err.get("retry", {}).get("action") != "search_capabilities":
                        return False, "retry protocol did not ask for a re-search"
                    hits = [m["name"] for m in AR.search_capabilities(query, limit=shortlist)["matches"]]
                    return (cap["name"] in hits, "error reproduced + recovery query retrieves the target")
                _emit(kind, cap, request, gold,
                      [("user", request),
                       ("assistant", _render_gold({"tool": "execute_capability", "name": fake,
                                                   "spec": {}}, fmt)),
                       ("user", "Tool result:\n" + _json.dumps(err, default=str))], verify)

            elif kind == "recover_args":
                # (d) wrong TYPE -> invalid_capability_arguments -> correct the args, do NOT re-search
                good = _tc_valid_spec(cap, rng)
                bad, broken_key = _tc_wrong_type(good, cap["spec_schema"], rng)
                if bad is None:
                    continue
                err = AR.execute_capability(cap["name"], bad, schema_digest_echo=cap["schema_digest"],
                                            dispatch=_stub)
                gold = {"tool": "execute_capability", "name": cap["name"], "spec": good,
                        "schema_digest": cap["schema_digest"]}
                def verify(err=err, cap=cap, good=good, bk=broken_key):
                    if err.get("error") != "invalid_capability_arguments":
                        return False, f"expected invalid_capability_arguments, got {err.get('error')}"
                    if err.get("retry", {}).get("action") != "correct_arguments":
                        return False, "retry protocol did not ask for corrected arguments"
                    if err.get("same_arguments_retryable") is not False:
                        return False, "the anti-loop flag was not set"
                    r = AR.execute_capability(cap["name"], good,
                                              schema_digest_echo=cap["schema_digest"], dispatch=_stub)
                    return bool(r.get("ok")), f"broke spec.{bk}, corrected version validates"
                _emit(kind, cap, request, gold,
                      [("user", request),
                       ("assistant", _render_gold({"tool": "execute_capability", "name": cap["name"],
                                                   "spec": bad,
                                                   "schema_digest": cap["schema_digest"]}, fmt)),
                       ("user", "Tool result:\n" + _json.dumps(err, default=str))], verify)

    # ---- quality gates, reported rather than assumed
    seen, dedup = set(), []
    for e in examples:
        key = (e["kind"], e["input"], e["gold"])
        if key in seen:
            continue
        seen.add(key); dedup.append(e)
    overlaps, ov_hard, ov_easy = [], [], []
    for e in dedup:
        cap = index.get(e["capability"], {})
        rw = set(_tc_words(e["input"])); sw = set(_tc_words(cap.get("summary", "")))
        o = len(rw & sw) / max(len(rw), 1)
        overlaps.append(o)
        (ov_hard if e.get("hard") else ov_easy).append(o)
    rank1 = ranked_n = 0
    for e in dedup:
        if e["kind"] not in ("execute", "recover_args"):
            continue
        try:
            shown = _json_mod.loads(
                e["messages"][2]["content"].split("Tool result:\n", 1)[1])["matches"]
            names = [m["name"] for m in shown]
            ranked_n += 1
            if names and names[0] == e["capability"]:
                rank1 += 1
        except (IndexError, KeyError, ValueError):
            pass
    hard_negs = 0
    for e in dedup:
        ranked = AR.normalize_request(e["input"], top=2)
        if len(ranked) > 1 and ranked[0]["agent"] != e["capability"]:
            hard_negs += 1
    n = len(dedup)
    rng.shuffle(dedup)
    cut = max(1, int(n * (1 - float(s.get("val_frac", 0.2)))))
    report = {
        "n": n, "rejected": len(rejected),
        "verified_frac": round(n / max(n + len(rejected), 1), 4),
        "by_kind": {k: sum(1 for e in dedup if e["kind"] == k) for k in kinds},
        "capabilities_covered": len({e["capability"] for e in dedup}),
        "index_size": len(index), "routable": len(routable), "with_schema": len(with_schema),
        "unroutable_no_summary": len(index) - len(routable),
        "mean_request_summary_overlap": round(sum(overlaps) / max(len(overlaps), 1), 4),
        "overlap_hard": round(sum(ov_hard) / max(len(ov_hard), 1), 4),
        "overlap_easy": round(sum(ov_easy) / max(len(ov_easy), 1), 4),
        "hard_frac_actual": round(len(ov_hard) / max(len(overlaps), 1), 4),
        "low_overlap_frac": round(sum(1 for o in overlaps if o < 0.34) / max(len(overlaps), 1), 4),
        "idf_top1_disagrees_frac": round(hard_negs / max(n, 1), 4),
        "gold_rank1_frac": round(rank1 / max(ranked_n, 1), 4),   # ~1/len(shortlist) if properly shuffled
        "shortlisted_examples": ranked_n,
        "reject_reasons": {},
    }
    for r in rejected:
        report["reject_reasons"][r["why"][:60]] = report["reject_reasons"].get(r["why"][:60], 0) + 1
    _tr, _va = _split_by_prompt(dedup, s.get("val_frac", 0.2), rng)
    return {"examples": dedup, "train": _tr, "val": _va, "report": report,
            "rejected": rejected, "source": "tool_calls(capability_index)"}


# ---- wire format for the TARGET the model learns to emit ------------------------------------------
# `compact` is the default: measured at 44% fewer tokens than JSON on real targets (48.4 -> 27.1) with
# 4 fragile syntax characters instead of 28. `json` remains available for comparison runs.
# Mode markers. `direct` (answer from memory) and `search` (discover first) are otherwise
# indistinguishable prompts with contradictory targets — see the note in build_history_tool_dataset.
MODE_RECALL = "MODE: recall — you know the catalog; answer with EXEC directly."
MODE_DISCOVER = "MODE: discover — you do not know the exact name; answer with SEARCH first."


def _render_gold(gold, fmt="compact"):
    from . import agent_routing as AR
    import json as _j
    return AR.to_compact(gold) if fmt == "compact" else _j.dumps(gold)


def _sys_for(fmt="compact"):
    from . import agent_routing as AR
    return AR.CAPABILITY_INSTRUCTIONS + "\n\n" + (
        AR.COMPACT_SPEC if fmt == "compact" else AR.TEXT_PROTOCOL)


def mine_failures(failures, spec=None):
    """The self-improvement step: a model's OWN mistakes -> new verified training examples.

    This is what makes the corpus a living artifact instead of a one-off build (standing rule). Two papers
    in `learning/paper_packs/` argue for exactly this and both were measured there:
      * afp unit 3 — an agent that carries its failed attempts outperforms independent retries at equal
        budget, so the failures are the highest-value data we have;
      * mse eq. 3 — estimating quality only from what you KEPT is biased upward, so failures must be
        mined rather than discarded.

    `failures`: [{"capability", "kind", "predicted"}] as emitted by llm_tool_train's evaluation — the
    example the model got wrong, and (for a wrong pick) WHICH capability it wrongly chose. That confusion
    is the signal: the new examples put the confused pair in the same shortlist, so the next round trains
    on the exact discrimination that failed rather than on more easy cases.

    Every generated example goes through the real validator, as always.
    """
    import json as _json
    import random as _random
    from . import agent_routing as AR

    s = spec or {}
    rng = _random.Random(int(s.get("seed", 0)))
    mult = int(s.get("mult", 4))                       # new examples per distinct failure
    index = AR.capability_index()
    fmt = s.get("format", "compact")
    sys_msg = _sys_for(fmt)

    def _stub(q, worker):
        return ("done", {"stub": True}, None, "stub")

    # collapse to distinct (kind, capability) with the set of things it was confused WITH
    buckets = {}
    for f in failures or []:
        cap = f.get("capability")
        if cap not in index:
            continue
        b = buckets.setdefault((f.get("kind", "execute"), cap), set())
        pred = f.get("predicted")
        if pred and pred != cap:
            b.add(pred)

    out, rejected, confusions = [], [], []
    for (kind, cap_name), confused in sorted(buckets.items()):
        cap = index[cap_name]
        for c in sorted(confused):
            confusions.append({"gold": cap_name, "predicted": c})
        for _ in range(mult):
            request = _tc_paraphrase(cap, rng, hard=rng.random() < 0.6)
            if kind in ("direct", "user_msg"):
                # A `direct` failure must mine MORE DIRECT examples, not search ones — the earlier version
                # fell through to the search branch and reinforced the wrong task, which is why the weakest
                # kind (weights-only recall, 60.9%) never improved from its own mistakes.
                #
                # `direct` has no shortlist, so a hard negative cannot be smuggled into one. What it does
                # have is NAME FAMILIES: 79 of 322 agents share a prefix (llm-* x8, audio-* x6,
                # tracker-* x5), and measurement showed no two agents have >50% summary overlap — so the
                # failures are DISCRIMINATION, not ambiguity. Mine the confused sibling alongside the
                # target so both surfaces are trained in the same batch.
                targets = [cap_name] + [c for c in sorted(confused) if c in index]
                fam = cap_name.split("-")[0]
                sibs = [k for k in index if k != cap_name and k.split("-")[0] == fam]
                targets += sorted(sibs)[:2]
                for tname in dict.fromkeys(targets):
                    tcap = index[tname]
                    req2 = ((MODE_RECALL + "\n") if kind == "direct" else "") + _tc_paraphrase(
                        tcap, rng, hard=rng.random() < 0.6)
                    sp2 = _tc_valid_spec(tcap, rng)
                    g2 = {"tool": "execute_capability", "name": tname, "spec": sp2,
                          "schema_digest": tcap["schema_digest"]}
                    if not AR.execute_capability(tname, sp2, schema_digest_echo=tcap["schema_digest"],
                                                 dispatch=_stub).get("ok"):
                        rejected.append({"capability": tname, "why": "validator rejected mined spec"})
                        continue
                    out.append({"kind": kind, "capability": tname, "input": req2,
                                "gold": _render_gold(g2, "compact"),
                                "messages": [{"role": "system", "content": sys_msg},
                                             {"role": "user", "content": req2},
                                             {"role": "assistant", "content": _render_gold(g2, "compact")}],
                                "hard": True, "mined": True, "confused_with": sorted(confused),
                                "verified": f"mined for {kind}; family `{fam}` siblings trained alongside"})
                continue

            if kind in ("execute", "recover_args") and cap.get("spec_schema"):
                res = AR.search_capabilities(
                    " ".join(_tc_words(cap.get("summary") or cap_name)[:8]), limit=10)
                shown = [{"name": m["name"], "summary": m["summary"][:110],
                          "spec_schema": m["spec_schema"], "schema_digest": m["schema_digest"]}
                         for m in res["matches"]]
                names = {m["name"] for m in shown}
                # FORCE the confused agents into the shortlist — train the discrimination that failed
                for c in sorted(confused):
                    if c in index and c not in names:
                        ci = index[c]
                        shown.append({"name": c, "summary": ci["summary"][:110],
                                      "spec_schema": ci["spec_schema"],
                                      "schema_digest": ci["schema_digest"]})
                        names.add(c)
                if cap_name not in names:
                    shown.append({"name": cap_name, "summary": cap["summary"][:110],
                                  "spec_schema": cap["spec_schema"],
                                  "schema_digest": cap["schema_digest"]})
                rng.shuffle(shown)
                sp = _tc_valid_spec(cap, rng)
                gold = {"tool": "execute_capability", "name": cap_name, "spec": sp,
                        "schema_digest": cap["schema_digest"]}
                r = AR.execute_capability(cap_name, sp, schema_digest_echo=cap["schema_digest"],
                                          dispatch=_stub)
                if not r.get("ok"):
                    rejected.append({"capability": cap_name, "why": str(r.get("error"))})
                    continue
                msgs = [{"role": "system", "content": sys_msg},
                        {"role": "user", "content": request},
                        {"role": "user", "content": "Tool result:\n" + _json.dumps({"matches": shown})},
                        {"role": "assistant", "content": _render_gold(gold, fmt)}]
            else:
                query = " ".join(_tc_words(cap.get("summary") or cap_name)[:8]) or cap_name
                hits = [m["name"] for m in AR.search_capabilities(query, limit=8)["matches"]]
                if cap_name not in hits:
                    rejected.append({"capability": cap_name, "why": "gold query does not retrieve target"})
                    continue
                gold = {"tool": "search_capabilities", "query": query}
                msgs = [{"role": "system", "content": sys_msg},
                        {"role": "user", "content": request},
                        {"role": "assistant", "content": _render_gold(gold, fmt)}]
            out.append({"kind": kind, "capability": cap_name, "input": request,
                        "gold": _render_gold(gold, fmt), "messages": msgs, "hard": True,
                        "mined": True, "confused_with": sorted(confused),
                        "verified": "validator ok (mined from a real failure)"})

    return {"examples": out, "rejected": rejected,
            "report": {"n": len(out), "rejected": len(rejected),
                       "distinct_failures": len(buckets),
                       "confusion_pairs": confusions[:40],
                       "note": ("mined from the model's own errors; confused agents are forced into the "
                                "shortlist so the next round trains the exact discrimination that failed")}}


# ---------------------------------------------------------------- competition-grounded tool-call data
# The synthetic generator above teaches the PROTOCOL. This one teaches JUDGEMENT, and it fixes the two
# weaknesses that generator's own quality report exposes:
#   * no lexical leakage — the model's input is a real COMPETITION STATE (current CV, what has been tried,
#     what was kept), not a paraphrase of the target agent's summary. There is no rare word to copy;
#   * a real reward — the label is the action that actually moved the PRIVATE score, read from a finished
#     competition's ledger, not "the validator accepted this".
#
# Source: any competition workspace with `docs/experiment_ledger.jsonl` (the fleet already writes it).
# First instance: birdclef-2026, where 11 experiments carry real (cv, public, private) outcomes and the
# CV->private estimator was validated at r=0.976 / LOO MAE 0.034 — so an action's value is measurable.
#
# Honest scale note: a finished competition yields TENS of examples, not thousands. That makes this a
# high-signal BENCHMARK and a fine-tuning seasoning, not a bulk corpus — and the report says so.
_LEDGER_ALIASES = {
    "audio-train": "audio-train", "kaggle-submit": "kaggle-submit",
    "adversarial-val": "adversarial", "adversarial": "adversarial",
    "xai": "xai", "xai_core": "xai", "cv-lb-calibrate": "cv-lb-calibrate",
    "blend-optimize": "blend-optimize", "calibrate": "calibrate",
    "nb-preflight": "nb-preflight", "submit-verify": "submit-verify",
    "lb-sync": "lb-sync", "metric-probe": "metric-probe", "det-sweep": "det-sweep",
    "tab-train": "tab-train", "distill": "distill", "pseudo-label": "pseudo-label",
}


def _ledger_actions(row, known):
    """Ledger `script` -> the real dispatchable agent kinds it names, in the order written.

    PRECISION MATTERS MORE THAN RECALL here, and an early version got this wrong: scanning `desc`/`change`
    too made EXP_03 (whose real lever was a Perch-only kernel) match the unrelated `baseline` agent from
    prose, attaching a genuine +0.20 reward to the WRONG action. A mislabelled example with a large reward
    is worse than no example. So only `script` — the field that records what actually ran — is read, and a
    row naming no dispatchable kind is SKIPPED rather than guessed at.
    """
    import re
    text = str(row.get("script", "")).lower()
    out = []
    for t in re.findall(r"[a-z][a-z0-9_-]{2,}", text):        # in written order, not alphabetical
        cand = _LEDGER_ALIASES.get(t, t if t in known else None)
        if cand and cand in known and cand not in out:
            out.append(cand)
    return out


def build_ledger_tool_dataset(spec):
    """{'examples': [...], 'report': {...}} — (competition state -> next action) examples with a REAL reward.

    spec keys: comp (workspace dir name, default 'birdclef-2026'), root (default /home/seshu/kaggle/2026),
               ledger (override path), val_frac, min_gain (only label actions whose private gain clears it).
    """
    import json as _json
    import os as _os
    from . import agent_routing as AR

    s = spec or {}
    root = s.get("root", "/home/seshu/kaggle/2026")
    comp = s.get("comp", "birdclef-2026")
    path = s.get("ledger") or _os.path.join(root, comp, "docs", "experiment_ledger.jsonl")
    if not _os.path.exists(path):
        raise FileNotFoundError(f"prompt-dataset ledger source: no ledger at {path}")

    rows = []
    for ln in open(path):
        ln = ln.strip()
        if not ln:
            continue
        try:
            rows.append(_json.loads(ln))
        except ValueError:
            continue
    rows.sort(key=lambda r: str(r.get("ts", "")))

    index = AR.capability_index()
    known = set(index)
    fmt = s.get("format", "compact")
    sys_msg = _sys_for(fmt)

    def _num(v):
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    examples, skipped = [], []
    best_private, tried = None, []
    for r in rows:
        acts = _ledger_actions(r, known)
        priv, cv = _num(r.get("private")), _num(r.get("cv"))
        gain = None if (priv is None or best_private is None) else round(priv - best_private, 5)

        if acts:
            # the STATE the agents faced BEFORE this experiment — no leakage of the answer
            state = {
                "competition": comp,
                "best_private_so_far": best_private,
                "best_cv_so_far": None,
                "experiments_done": len(tried),
                "already_tried": tried[-6:],
            }
            request = (f"Competition `{comp}`. {len(tried)} experiments run so far; "
                       f"best private score {best_private if best_private is not None else 'none yet'}. "
                       f"Already tried: {', '.join(tried[-6:]) or 'nothing'}. "
                       f"What is the next action?")
            cap = index[acts[0]]
            gold = {"tool": "search_capabilities",
                    "query": " ".join(_tc_words(cap.get("summary") or acts[0])[:8]) or acts[0]}
            hits = [m["name"] for m in AR.search_capabilities(gold["query"], limit=8)["matches"]]
            if acts[0] not in hits:
                skipped.append({"exp": r.get("exp"), "why": f"gold query does not retrieve {acts[0]}"})
            else:
                examples.append({
                    "kind": "ledger_next_action", "capability": acts[0],
                    "input": request, "gold": _render_gold(gold, fmt),
                    "messages": [{"role": "system", "content": sys_msg},
                                 {"role": "user", "content": request},
                                 {"role": "assistant", "content": _render_gold(gold, fmt)}],
                    "reward": gain, "exp": r.get("exp"), "cv": cv, "private": priv,
                    "state": state, "all_actions": acts,
                    "verified": f"gold retrieves {acts[0]} at rank {hits.index(acts[0]) + 1}",
                    "hard": True})
        else:
            skipped.append({"exp": r.get("exp"), "why": "ledger names no dispatchable agent kind"})

        if priv is not None:
            best_private = priv if best_private is None else max(best_private, priv)
        tried.append(str(r.get("change") or r.get("exp") or "?")[:44])

    rewarded = [e for e in examples if e["reward"] is not None]
    positive = [e for e in rewarded if e["reward"] > float(s.get("min_gain", 0.0))]
    cut = max(1, int(len(examples) * (1 - float(s.get("val_frac", 0.25)))))
    report = {
        "comp": comp, "ledger_rows": len(rows), "n": len(examples), "skipped": len(skipped),
        "with_reward": len(rewarded), "reward_positive": len(positive),
        "best_private_final": best_private,
        "mean_request_summary_overlap": None,       # by construction: the request is a STATE, not a summary
        "distinct_actions": len({e["capability"] for e in examples}),
        "skip_reasons": {},
        "note": ("state-grounded: the input never contains the target agent's own words, so there is no "
                 "lexical shortcut. Small by nature — one finished competition yields tens of examples."),
    }
    for sk in skipped:
        report["skip_reasons"][sk["why"][:60]] = report["skip_reasons"].get(sk["why"][:60], 0) + 1
    import random as _rnd2
    _tr, _va = _split_by_prompt(examples, s.get("val_frac", 0.25),
                                _rnd2.Random(int(s.get("seed", 7))))
    return {"examples": examples, "train": _tr, "val": _va,
            "report": report, "skipped": skipped, "source": f"ledger({comp})"}


# ---------------------------------------------------------------- next-agent-from-agent-output data
# The two generators above train tool SYNTAX and error recovery. Neither trains the actual job: choosing
# the NEXT agent from what PREVIOUS AGENTS RETURNED. Their examples only ever contain the model's own
# previous turn, so a model fitted on them can format a call but cannot run a chain.
#
# This one closes that gap with real data. `docs/experiment_decisions.jsonl` is the fleet's own posting
# log — 7.8k records of (agent, summary/finding, recommendation) from actual competition runs — and 2,480
# of those recommendations NAME the next agent, giving 104 distinct observed transitions. So the input is
# a genuine agent result string and the label is the agent the fleet actually went to next.
#
# Two example shapes:
#   `next_agent`  — H real prior agent outputs in the conversation -> the next agent (accumulated history)
#   `direct`      — a request with NO shortlist shown -> the exact execute_capability call, so the tool
#                   inventory must come from the WEIGHTS rather than the prompt (an explicit requirement;
#                   note it needs far more examples per agent than the shortlist route, and the 27 agents
#                   with empty summaries cannot be taught this way at all).
def _decisions(comp, root="/home/seshu/kaggle/2026", limit=None):
    import json as _json
    import os as _os
    path = _os.path.join(root, comp, "docs", "experiment_decisions.jsonl")
    if not _os.path.exists(path):
        raise FileNotFoundError(f"prompt-dataset: no decision log at {path}")
    out = []
    for ln in open(path, errors="replace"):
        ln = ln.strip()
        if not ln:
            continue
        try:
            out.append(_json.loads(ln))
        except ValueError:                                 # a truncated line is data loss, not a crash
            continue
    # Drop the PILOT's own rows, exactly as `local_pilot._decisions` does at serving time. The pilot writes
    # its picks into this same log, so leaving them in would train on inputs the pilot never sees -- the
    # train/serve drift that has bitten this pipeline repeatedly. Only 0.4% of windows today, but the share
    # grows with every pilot run, so the filter is preventive.
    from .local_pilot import SELF_KINDS as _SELF
    out = [r for r in out if str(r.get("agent") or "") not in _SELF]
    out.sort(key=lambda r: str(r.get("ts", "")))
    return out[-limit:] if limit else out


def _recommended_agent(rec_text, known):
    """The agent a recommendation names — longest match first, so `tracker-postproc` beats `tracker`."""
    rec = str(rec_text or "")
    hits = [a for a in known if len(a) > 4 and a in rec]
    return max(hits, key=len) if hits else None


def build_history_tool_dataset(spec):
    """{'examples': [...], 'report': {...}} — (accumulated agent output -> next agent) + memorisation.

    spec: comp, root, history (how many prior agent outputs to show, default 3), n (cap),
          kinds (subset of ['next_agent', 'direct']), direct_per_agent, seed, val_frac.
    """
    import json as _json
    import random as _random
    from . import agent_routing as AR

    s = spec or {}
    rng = _random.Random(int(s.get("seed", 0)))
    comp = s.get("comp", "biohub-cell-tracking-during-development")
    H = int(s.get("history", 3))
    kinds = list(s.get("kinds") or ["next_agent", "direct", "user_msg"])
    index = AR.capability_index()
    known = set(index)
    fmt = s.get("format", "compact")
    sys_msg = _sys_for(fmt)

    def _stub(q, worker):
        return ("done", {"stub": True}, None, "stub")

    rows = _decisions(comp, s.get("root", "/home/seshu/kaggle/2026"))
    examples, rejected = [], []
    transitions = {}

    if "next_agent" in kinds:
        for i, r in enumerate(rows):
            nxt = _recommended_agent(r.get("recommendation"), known)
            if not nxt or r.get("agent") not in known:
                continue
            # the accumulated history: this record plus up to H-1 real ones before it
            window = [w for w in rows[max(0, i - H + 1): i + 1] if w.get("agent") in known]
            if not window:
                continue
            hist_lines = []
            for w in window:
                out_text = str(w.get("finding") or w.get("summary") or "")[:340]
                hist_lines.append(f"agent `{w['agent']}` returned: {out_text}")
            request = (MODE_DISCOVER + "\nFleet run in progress. Recent agent output:\n"
                       + "\n".join(hist_lines) + "\n\nChoose the next agent to run.")
            cap = index[nxt]
            query = " ".join(_tc_words(cap.get("summary") or nxt)[:8]) or nxt
            hits = [m["name"] for m in AR.search_capabilities(query, limit=8)["matches"]]
            if nxt not in hits:
                rejected.append({"kind": "next_agent", "capability": nxt,
                                 "why": "gold query does not retrieve the recommended agent"})
                continue
            gold = {"tool": "search_capabilities", "query": query}
            transitions[(window[-1]["agent"], nxt)] = transitions.get((window[-1]["agent"], nxt), 0) + 1
            examples.append({
                "kind": "next_agent", "capability": nxt, "input": request,
                "gold": _render_gold(gold, fmt),
                "messages": [{"role": "system", "content": sys_msg},
                             {"role": "user", "content": request},
                             {"role": "assistant", "content": _render_gold(gold, fmt)}],
                "hard": True, "history_len": len(window),
                "from_agent": window[-1]["agent"],
                "verified": f"recommended `{nxt}` is dispatchable and retrievable (rank "
                            f"{hits.index(nxt) + 1})"})

    if "user_msg" in kinds:
        # The :7788 runboard lets a HUMAN address the fleet mid-run (`POST /api/runtime/messages`,
        # sender='human'). That message outranks whatever the agents are doing, so the model must learn to
        # follow it rather than continue the trajectory. There is no history to mine here — the channel is
        # wired but has carried zero human traffic in 8,097 thread rows — so these are SYNTHESISED, with
        # real agent output as the background the instruction must override.
        routable_u = [c for c in index.values() if (c.get("summary") or "").strip()]
        bg = [r for r in rows if r.get("agent") in known][-400:]
        per_u = int(s.get("user_msg_per_agent", 3))
        for cap in routable_u:
            for _ in range(per_u):
                instruction = _tc_paraphrase(cap, rng, hard=rng.random() < 0.5)
                noise = rng.sample(bg, min(2, len(bg))) if bg else []
                hist = "\n".join(
                    f"agent `{w['agent']}` returned: "
                    f"{str(w.get('finding') or w.get('summary') or '')[:220]}" for w in noise)
                request = (MODE_DISCOVER + f"\nUSER -> all: {instruction}\n\n"
                           + (f"Fleet run in progress. Recent agent output:\n{hist}\n\n" if hist else "")
                           + "The USER has sent the instruction above. Choose the next agent that carries "
                             "it out; the agent output is background only.")
                query = " ".join(_tc_words(cap.get("summary") or cap["name"])[:8]) or cap["name"]
                hits = [m["name"] for m in AR.search_capabilities(query, limit=8)["matches"]]
                if cap["name"] not in hits:
                    rejected.append({"kind": "user_msg", "capability": cap["name"],
                                     "why": "gold query does not retrieve target"})
                    continue
                gold = {"tool": "search_capabilities", "query": query}
                examples.append({
                    "kind": "user_msg", "capability": cap["name"], "input": request,
                    "gold": _render_gold(gold, fmt),
                    "messages": [{"role": "system", "content": sys_msg},
                                 {"role": "user", "content": request},
                                 {"role": "assistant", "content": _render_gold(gold, fmt)}],
                    "hard": True,
                    "verified": "user instruction overrides background agent output"})

    if "direct" in kinds:
        # MEMORISATION: no shortlist in the prompt at all. The inventory must be in the weights.
        per = int(s.get("direct_per_agent", 6))
        routable = [c for c in index.values() if (c.get("summary") or "").strip()]
        for cap in routable:
            for _ in range(per):
                request = MODE_RECALL + "\n" + _tc_paraphrase(cap, rng, hard=rng.random() < 0.5)
                sp = _tc_valid_spec(cap, rng)
                gold = {"tool": "execute_capability", "name": cap["name"], "spec": sp,
                        "schema_digest": cap["schema_digest"]}
                r = AR.execute_capability(cap["name"], sp, schema_digest_echo=cap["schema_digest"],
                                         dispatch=_stub)
                if not r.get("ok"):
                    rejected.append({"kind": "direct", "capability": cap["name"],
                                     "why": str(r.get("error"))})
                    continue
                examples.append({
                    "kind": "direct", "capability": cap["name"], "input": request,
                    "gold": _render_gold(gold, fmt),
                    "messages": [{"role": "system", "content": sys_msg},
                                 {"role": "user", "content": request},
                                 {"role": "assistant", "content": _render_gold(gold, fmt)}],
                    "hard": True, "verified": "validator ok, NO shortlist shown (weights-only recall)"})

    n_cap = int(s.get("n", 0))
    rng.shuffle(examples)
    if n_cap:
        examples = examples[:n_cap]
    cut = max(1, int(len(examples) * (1 - float(s.get("val_frac", 0.2)))))
    by_kind = {}
    for e in examples:
        by_kind[e["kind"]] = by_kind.get(e["kind"], 0) + 1
    per_agent = {}
    for e in examples:
        per_agent[e["capability"]] = per_agent.get(e["capability"], 0) + 1
    counts = sorted(per_agent.values())
    report = {
        "comp": comp, "decision_rows": len(rows), "n": len(examples), "rejected": len(rejected),
        "by_kind": by_kind, "history_len": H,
        "distinct_transitions": len(transitions),
        "top_transitions": [{"from": a, "to": b, "n": c}
                            for (a, b), c in sorted(transitions.items(), key=lambda kv: -kv[1])[:10]],
        "agents_covered": len(per_agent),
        "examples_per_agent_min": counts[0] if counts else 0,
        "examples_per_agent_median": counts[len(counts) // 2] if counts else 0,
        "unroutable_no_summary": len(index) - len([c for c in index.values()
                                                   if (c.get("summary") or "").strip()]),
        "note": ("next_agent inputs are REAL agent result strings (no paraphrase, no lexical shortcut); "
                 "`direct` shows no shortlist, so recall must come from the weights — which needs many "
                 "examples per agent and cannot cover agents whose summary is empty."),
    }
    _tr, _va = _split_by_prompt(examples, s.get("val_frac", 0.2), rng)
    return {"examples": examples, "train": _tr, "val": _va,
            "report": report, "rejected": rejected, "source": f"history({comp})"}


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
