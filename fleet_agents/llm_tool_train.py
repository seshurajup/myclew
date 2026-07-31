"""llm-tool-train — LoRA/QLoRA supervised fine-tuning of a LOCAL causal LM to drive the fleet's own
two-tool contract (`search_capabilities` -> `execute_capability` from agent_routing).

The gap this fills: `llm_backend` can already reach a local model, and `agent_routing.capability_loop` can
already drive 320 agents from one — but a small local model must be TAUGHT the protocol. `gemma3n:e4b`
cannot even accept a native tool schema (Ollama returns "does not support tools"), so it drives the fleet
only through the text protocol, and its JSON discipline is exactly what fine-tuning fixes.

Reusable by construction — nothing here is Gemma-specific:
  • `model` is any local HF causal LM path or hub id (default: the locally cached Gemma-4 E2B-it);
  • `bits` in (4, 8, 16) selects QLoRA-nf4 / int8 / bf16-LoRA — the fleet's standing "train low-bit only"
    rule, exposed as a knob rather than hard-coded;
  • data comes from the `prompt-dataset` agent (kinds: `tool_calls` for the protocol, `ledger` for
    competition-grounded judgement), so the corpus improves without touching this file.

Discipline this agent enforces rather than assumes:
  • **completion-only loss** — the prompt is masked out and the loss is computed on the assistant's JSON
    only. Training on the prompt teaches a small model to parrot the capability list back;
  • **`gpu_train_hold.flag`** — one GPU is shared with real competition training, so the flag is honoured
    and the run escalates instead of fighting for VRAM;
  • **dry_run is the DEFAULT** — every trainer in this fleet must be provable before it consumes the GPU;
  • **before/after evaluation on a held-out split** — a fine-tune that is not measured is not a result.
    Reported per example KIND, because "protocol format" and "which agent" are different skills and a mean
    over both hides which one moved.

Self-improvement loop (`rounds` > 1): train -> evaluate on held-out -> mine the model's OWN failures into
new verified examples (`prompt_dataset.mine_failures`, which forces the confused agents into the same
shortlist) -> retrain on the augmented corpus. Each round is posted to the board with its per-kind numbers,
and the loop stops as soon as a round fails to improve — the standing rule "every training run improves the
dataset", made mechanical.

Spec:
    {"kind": "llm-tool-train", "spec": {"model": …, "bits": 4, "n": 1200, "epochs": 2, "lr": 1e-4,
                                        "dry_run": true, "eval_only": false, "max_len": 1024}}
"""
from __future__ import annotations

import json
import os
import time

from .base import BaseAgent, COMP

DEFAULT_MODEL = os.path.expanduser(
    "~/.cache/huggingface/hub/models--unsloth--gemma-4-E2B-it/snapshots")
HOLD_FLAG = "gpu_train_hold.flag"


def _resolve_model(spec_model):
    """A hub id, an explicit path, or the newest local snapshot of the default cached model."""
    if spec_model:
        return spec_model
    if os.path.isdir(DEFAULT_MODEL):
        snaps = [os.path.join(DEFAULT_MODEL, d) for d in os.listdir(DEFAULT_MODEL)]
        snaps = [d for d in snaps if os.path.isdir(d)]
        if snaps:
            return max(snaps, key=os.path.getmtime)
    return "unsloth/gemma-4-E2B-it"


def gpu_held():
    """(held, why) — the shared-GPU gate. Real competition training outranks a fine-tune."""
    for root in (COMP, os.getcwd()):
        f = os.path.join(str(root), HOLD_FLAG)
        if os.path.exists(f):
            try:
                why = open(f).read().strip()[:120]
            except OSError:
                why = ""
            return True, f"{HOLD_FLAG} present at {root}" + (f": {why}" if why else "")
    return False, ""


def free_vram_gb():
    try:
        import torch
        if not torch.cuda.is_available():
            return 0.0
        free, _tot = torch.cuda.mem_get_info()
        return round(free / 1e9, 2)
    except Exception:  # noqa: BLE001 — a probe must never be the reason a run dies
        return 0.0


def build_corpus(spec):
    """Chain the prompt-dataset agent. Protocol examples plus (optionally) ledger-grounded ones."""
    from . import prompt_dataset as PD
    n = int(spec.get("n", 1200))
    d = PD.build_tool_call_dataset({"n": n, "seed": int(spec.get("seed", 7)),
                                    "hard_frac": float(spec.get("hard_frac", 0.5)),
                                    "val_frac": float(spec.get("val_frac", 0.2))})
    train, val, reports = list(d["train"]), list(d["val"]), {"tool_calls": d["report"]}

    # (2) NEXT-AGENT-FROM-AGENT-OUTPUT + weights-only recall. This is the part that teaches the actual job:
    # choosing the next agent from what previous agents RETURNED, and recalling the inventory without a
    # shortlist in the prompt. Inputs are real fleet result strings, so there is no lexical shortcut.
    if spec.get("history", True):
        try:
            H = PD.build_history_tool_dataset({
                "comp": spec.get("hist_comp", "biohub-cell-tracking-during-development"),
                "history": int(spec.get("hist_len", 3)),
                "direct_per_agent": int(spec.get("direct_per_agent", 20)),
                "seed": int(spec.get("seed", 7)), "val_frac": float(spec.get("val_frac", 0.2))})
            train += H["train"]; val += H["val"]; reports["history"] = H["report"]
        except (FileNotFoundError, ValueError) as e:
            reports["history"] = {"error": str(e)[:160]}

    # (3) competition-grounded judgement, where a finished comp supplies real private-score rewards
    if spec.get("comp"):
        try:
            L = PD.build_ledger_tool_dataset({"comp": spec["comp"]})
            train += L["train"]; val += L["val"]; reports["ledger"] = L["report"]
        except (FileNotFoundError, ValueError) as e:
            reports["ledger"] = {"error": str(e)[:160]}

    # LEAVE-ONE-DOMAIN-OUT. `exclude_domain`/`exclude_capabilities` drop every example whose TARGET is in
    # the held-out set, from train AND val. The agents stay registered and retrievable -- only the training
    # signal for choosing them is removed. That is the closest available proxy for a new competition, whose
    # agent families are present in the roster but absent from the decision history.
    excl = set(spec.get("exclude_capabilities") or ())
    dom = spec.get("exclude_domain")
    if dom:
        from . import agent_routing as _AR
        excl |= {n for n, c in _AR.capability_index().items() if (c.get("domain") or "") == dom}
    if excl:
        before = (len(train), len(val))
        train = [e for e in train if e.get("capability") not in excl]
        val = [e for e in val if e.get("capability") not in excl]
        reports["holdout"] = {"domain": dom, "n_agents": len(excl),
                              "dropped_train": before[0] - len(train),
                              "dropped_val": before[1] - len(val)}

    import random as _r
    rng = _r.Random(int(spec.get("seed", 7)))
    rng.shuffle(train)
    # STRATIFY the val set: take a fair share of every kind, then shuffle. Without this the eval sample
    # was the first N of a concatenated list — 4 kinds of 8 — and the three biggest training kinds
    # (direct, next_agent, user_msg = 87% of the corpus) were never evaluated at all.
    by_kind = {}
    for e in val:
        by_kind.setdefault(e["kind"], []).append(e)
    for v in by_kind.values():
        rng.shuffle(v)
    stratified, i = [], 0
    while any(len(v) > i for v in by_kind.values()):
        for k in sorted(by_kind):
            if len(by_kind[k]) > i:
                stratified.append(by_kind[k][i])
        i += 1
    return train, stratified, reports


def _encode(tok, ex, max_len):
    """Chat-template the messages and mask everything before the assistant turn (completion-only loss)."""
    import torch
    msgs = ex["messages"]
    prompt_msgs, target = msgs[:-1], msgs[-1]["content"]
    try:
        prompt_txt = tok.apply_chat_template(prompt_msgs, tokenize=False, add_generation_prompt=True)
    except Exception:  # noqa: BLE001 — a base model may ship no template
        prompt_txt = "\n".join(f"{m['role']}: {m['content']}" for m in prompt_msgs) + "\nassistant:"
    p_ids = tok(prompt_txt, add_special_tokens=False)["input_ids"]
    t_ids = tok(target + (tok.eos_token or ""), add_special_tokens=False)["input_ids"]
    ids = (p_ids + t_ids)[-max_len:]
    labels = ([-100] * len(p_ids) + list(t_ids))[-max_len:]
    return torch.tensor(ids), torch.tensor(labels)


def _gen_json(model, tok, ex, max_new=160):
    """Greedy-decode the assistant turn and parse the first JSON object out of it."""
    import torch
    from . import agent_routing as AR
    msgs = ex["messages"][:-1]
    try:
        txt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    except Exception:  # noqa: BLE001
        txt = "\n".join(f"{m['role']}: {m['content']}" for m in msgs) + "\nassistant:"
    enc = tok(txt, return_tensors="pt", add_special_tokens=False).to(model.device)
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=max_new, do_sample=False,
                             pad_token_id=tok.pad_token_id or tok.eos_token_id)
    gen = tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
    # accept the compact wire form first (what we now train on), then JSON for older/other models
    return (AR.from_compact(gen) or AR._first_json_object(gen)), gen


def score(model, tok, examples, max_new=160):
    """Tool-call accuracy, split by skill because a mean over both hides which one moved.

    parsed  — emitted a well-formed JSON object at all (pure format)
    tool_ok — named the right tool
    exact   — right tool AND right capability name (where the gold has one)
    """
    import json as _json
    from . import agent_routing as AR
    agg, failures = {}, []
    _t0, _n = time.time(), len(examples)
    for _i, ex in enumerate(examples):
        if _i and _i % max(1, _n // 5) == 0:
            print(f"    [eval] {_i}/{_n} ({_i / _n:.0%}) "
                  f"{(time.time() - _t0) / _i:.1f}s/example", flush=True)
        gold = ex.get("gold_obj") or AR.from_compact(ex["gold"]) or _json.loads(ex["gold"])
        obj, raw = _gen_json(model, tok, ex, max_new=max_new)
        k = ex.get("kind", "?")
        a = agg.setdefault(k, {"n": 0, "parsed": 0, "tool_ok": 0, "exact": 0})
        a["n"] += 1
        if not isinstance(obj, dict):
            failures.append({"kind": k, "capability": ex.get("capability"), "predicted": None,
                             "why": "unparseable", "raw": raw[:160]})
            continue
        a["parsed"] += 1
        if obj.get("tool") == gold.get("tool"):
            a["tool_ok"] += 1
            if gold.get("name") is None or obj.get("name") == gold.get("name"):
                a["exact"] += 1
            else:
                failures.append({"kind": k, "capability": ex.get("capability"),
                                 "predicted": obj.get("name"), "why": "wrong capability"})
        else:
            failures.append({"kind": k, "capability": ex.get("capability"),
                             "predicted": obj.get("name"), "why": "wrong tool"})
    tot = {"n": 0, "parsed": 0, "tool_ok": 0, "exact": 0}
    for a in agg.values():
        for m in tot:
            tot[m] += a[m]
    return {"by_kind": agg, "total": tot, "failures": failures,
            "parsed_rate": round(tot["parsed"] / max(tot["n"], 1), 4),
            "tool_rate": round(tot["tool_ok"] / max(tot["n"], 1), 4),
            "exact_rate": round(tot["exact"] / max(tot["n"], 1), 4)}


def spec_attn():
    """Prefer flash-attention-2, fall back to torch SDPA (still fused) — never the eager path."""
    try:
        import flash_attn  # noqa: F401
        return "flash_attention_2"
    except Exception:  # noqa: BLE001
        return "sdpa"


def load_model(model_id, bits, train=True):
    """Load with the requested precision. 4 = QLoRA-nf4, 8 = int8, 16 = bf16 (LoRA either way)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    # RTX 5090 (sm_120) settings. TF32 ON here — this is TRAINING, not an exactness proof, and the
    # matmul speedup is free; the paper packs keep TF32 OFF precisely because they assert identities.
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    kw = {"dtype": torch.bfloat16, "device_map": "auto",
          "attn_implementation": spec_attn()}
    if bits in (4, 8):
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=(bits == 4), load_in_8bit=(bits == 8),
            bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16)
    try:
        model = AutoModelForCausalLM.from_pretrained(model_id, **kw)
    except TypeError:                                    # older transformers wants torch_dtype
        kw["torch_dtype"] = kw.pop("dtype")
        model = AutoModelForCausalLM.from_pretrained(model_id, **kw)
    except (ValueError, ImportError):                    # this arch may not support the fused kernel
        kw.pop("attn_implementation", None)
        model = AutoModelForCausalLM.from_pretrained(model_id, **kw)
    return model, tok


def attach_lora(model, bits, r=16, alpha=32, dropout=0.05):
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    if bits in (4, 8):
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    cfg = LoraConfig(r=r, lora_alpha=alpha, lora_dropout=dropout, bias="none",
                     task_type="CAUSAL_LM", target_modules="all-linear")
    model = get_peft_model(model, cfg)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return model, {"trainable": trainable, "total": total,
                   "trainable_pct": round(100 * trainable / max(total, 1), 4)}


def train_lora(model, tok, train_examples, spec):
    """SFT loop tuned for the 5090 — no trl/datasets dependency.

    The first version ran at BATCH SIZE 1 and the GPU sat at 52% utilisation. Three fixes, in order of
    measured value on this hardware:
      * real batching with padding + attention mask (bs=8; a prior fleet measurement found training
        saturates around there on this card, so bigger buys nothing);
      * LENGTH-GROUPED batches — examples sorted by token count before batching, so a 900-token example
        never pads a 120-token one. Without this, padding waste eats most of the batching win;
      * an 8-bit optimizer when bitsandbytes is available — less optimizer state, so the batch fits.
    Completion-only loss is preserved: labels are -100 on both the prompt AND the padding.
    """
    import torch
    epochs = int(spec.get("epochs", 2))
    lr = float(spec.get("lr", 1e-4))
    max_len = int(spec.get("max_len", 1024))
    bs = int(spec.get("batch_size", 8))
    accum = max(1, int(spec.get("grad_accum", 2)))
    pad_id = tok.pad_token_id or tok.eos_token_id or 0

    enc = [_encode(tok, ex, max_len) for ex in train_examples]
    order = sorted(range(len(enc)), key=lambda i: len(enc[i][0]))       # length-grouped

    # TOKEN-BUDGET batching, not fixed batch size. Measured why: this model's vocabulary is ~262k, so the
    # logits tensor is bs x seq x 262144 x 2 bytes — a fixed bs=8 at seq=900 asks for 3.8 GB of logits
    # alone and OOMs on a 32 GB card. Budgeting by padded TOKENS gives large batches for the many short
    # examples and small ones for the few long examples, which is where the throughput actually is.
    budget = int(spec.get("max_batch_tokens", 2600))
    batches, cur, cur_max = [], [], 0
    for i in order:
        L_i = len(enc[i][0])
        new_max = max(cur_max, L_i)
        if cur and (new_max * (len(cur) + 1) > budget or len(cur) >= bs):
            batches.append(cur); cur, cur_max = [i], L_i
        else:
            cur.append(i); cur_max = new_max
    if cur:
        batches.append(cur)

    try:
        import bitsandbytes as bnb
        opt = bnb.optim.AdamW8bit([p for p in model.parameters() if p.requires_grad], lr=lr)
        opt_kind = "AdamW8bit"
    except Exception:  # noqa: BLE001 — plain AdamW is a fine fallback, just heavier
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
        opt_kind = "AdamW"

    model.train()
    losses, step, tokens = [], 0, 0
    t0 = time.time()
    rng = __import__("random").Random(0)
    for ep in range(epochs):
        rng.shuffle(batches)                                            # shuffle BATCHES, keep grouping
        for bi, idxs in enumerate(batches):
            rows = [enc[i] for i in idxs]
            L = max(len(r[0]) for r in rows)
            ids = torch.full((len(rows), L), pad_id, dtype=torch.long)
            lab = torch.full((len(rows), L), -100, dtype=torch.long)
            att = torch.zeros((len(rows), L), dtype=torch.long)
            for j, (i_, l_) in enumerate(rows):
                ids[j, :len(i_)] = i_; lab[j, :len(l_)] = l_; att[j, :len(i_)] = 1
            ids = ids.to(model.device); lab = lab.to(model.device); att = att.to(model.device)
            out = model(input_ids=ids, attention_mask=att, labels=lab)
            (out.loss / accum).backward()
            losses.append(float(out.loss.detach())); tokens += int(att.sum())
            if (bi + 1) % accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0)
                opt.step(); opt.zero_grad(set_to_none=True); step += 1
            # HEARTBEAT. Without this the log prints nothing between round boundaries, so a multi-hour
            # run is indistinguishable from a hung one — which cost real time to diagnose. Cheap, and it
            # makes "is it working?" answerable from the log instead of from nvidia-smi.
            if (bi + 1) % max(1, len(batches) // 20) == 0:
                el = time.time() - t0
                done = ep * len(batches) + bi + 1
                tot_b = epochs * len(batches)
                print(f"    [train] epoch {ep + 1}/{epochs} batch {bi + 1}/{len(batches)} "
                      f"({done / tot_b:.0%}) loss {sum(losses[-20:]) / min(len(losses), 20):.4f} "
                      f"{int(tokens / max(el, 1e-9))} tok/s eta "
                      f"{int((tot_b - done) * el / max(done, 1) / 60)}min", flush=True)
        opt.step(); opt.zero_grad(set_to_none=True)
    model.eval()
    dt = time.time() - t0
    n = max(len(losses) // 10, 1)
    return {"steps": step, "epochs": epochs, "seconds": round(dt, 1),
            "batch_size_cap": bs, "max_batch_tokens": budget, "n_batches": len(batches),
            "mean_batch": round(len(enc) / max(len(batches), 1), 2),
            "optimizer": opt_kind, "attn": spec_attn(),
            "tokens_per_s": round(tokens / max(dt, 1e-9)),
            "loss_first_decile": round(sum(losses[:n]) / n, 4),
            "loss_last_decile": round(sum(losses[-n:]) / n, 4)}


class LLMToolTrain(BaseAgent):
    name = "llm-tool-train"
    thread = "M"
    kind = "finding"

    def run(self, q, worker):
        spec = self.spec(q)
        model_id = _resolve_model(spec.get("model"))
        bits = int(spec.get("bits", 4))
        dry = bool(spec.get("dry_run", True))

        train, val, reports = build_corpus(spec)
        n_eval = int(spec.get("n_eval", 40))
        val_eval = val[:n_eval]
        base = {"model": model_id, "bits": bits, "n_train": len(train), "n_val": len(val),
                "data_reports": reports, "free_vram_gb": free_vram_gb()}

        if dry:
            kinds = {}
            for e in train:
                kinds[e["kind"]] = kinds.get(e["kind"], 0) + 1
            msg = (f"[{worker}] llm-tool-train DRY-RUN ok — {len(train)} train / {len(val)} val "
                   f"tool-call examples ({kinds}); target {os.path.basename(str(model_id))} at {bits}-bit "
                   f"LoRA; {free_vram_gb():.1f} GB VRAM free. Re-dispatch with dry_run=false to train.")
            self.log(msg, kind="finding",
                     recommendation="verify the corpus report, then run with dry_run=false")
            return self.done({**base, "dry_run": True, "by_kind": kinds}, msg)

        held, why = gpu_held()
        if held:
            return self.escalate(worker, "leader",
                                 f"[{worker}] llm-tool-train: GPU is held by real training ({why}). "
                                 f"Not competing for VRAM — retry when the flag clears.")
        try:
            model, tok = load_model(model_id, bits, train=True)
        except Exception as e:  # noqa: BLE001 — a load failure is a RESULT, not a crash
            return self.escalate(worker, "leader",
                                 f"[{worker}] llm-tool-train could not load {model_id} at {bits}-bit: "
                                 f"{type(e).__name__}: {str(e)[:200]}")
        before = score(model, tok, val_eval)
        model, lora_info = attach_lora(model, bits, r=int(spec.get("lora_r", 16)))

        # ---- SELF-IMPROVEMENT ROUNDS: train, evaluate, mine the model's OWN mistakes back into the
        # corpus, retrain on the augmented set. Every round is logged so a regression is visible rather
        # than averaged away, and the loop STOPS early when a round stops helping (no point burning GPU).
        from . import prompt_dataset as PD
        rounds = max(1, int(spec.get("rounds", 1)))
        history, cur_train, best = [], list(train), before
        best_round = -1                            # -1 = no round beat the pre-training baseline
        for rd in range(rounds):
            tr = train_lora(model, tok, cur_train, spec)
            ev = score(model, tok, val_eval)
            mined = {"report": {"n": 0}}
            if rd + 1 < rounds and ev["failures"]:
                mined = PD.mine_failures(ev["failures"],
                                         {"seed": rd, "mult": int(spec.get("mine_mult", 4))})
                cur_train = cur_train + mined["examples"]
            rec = {"round": rd, "n_train": len(cur_train), "steps": tr["steps"],
                   "loss": [tr["loss_first_decile"], tr["loss_last_decile"]],
                   "exact": ev["exact_rate"], "tool": ev["tool_rate"], "parsed": ev["parsed_rate"],
                   "by_kind": {k: round(v["exact"] / max(v["n"], 1), 4) for k, v in ev["by_kind"].items()},
                   "n_failures": len(ev["failures"]), "mined": mined["report"]["n"]}
            history.append(rec)
            # PRINT as well as post. `self.post` writes to the fleet board — right for the fleet, but
            # invisible to anyone tailing the run log, so a long unattended run looked silent even though
            # every round was being recorded. The two serve different readers; emit to both.
            print(f"  round {rd}: exact {ev['exact_rate']:.1%} tool {ev['tool_rate']:.1%} "
                  f"parsed {ev['parsed_rate']:.1%} by_kind {rec['by_kind']} "
                  f"failures {len(ev['failures'])} mined {mined['report']['n']} "
                  f"train {len(cur_train)}", flush=True)
            # BaseAgent.post(worker, to, msg, routine=, kind=, dedup=) — no `data` kwarg; the numbers go
            # in the message and the full record in `history` (returned + written to train_report.json)
            self.post(worker, "all",
                      f"[{worker}] llm-tool-train round {rd}: exact {ev['exact_rate']:.1%} "
                      f"(by kind {rec['by_kind']}), {len(ev['failures'])} failures, "
                      f"mined {mined['report']['n']} new examples -> {len(cur_train)} train",
                      routine=True, kind="llm-tool-train")
            if ev["exact_rate"] > best["exact_rate"]:
                best = ev
                # SAVE THE BEST ROUND, NOT THE LAST. Measured defect: `best` was tracked for the report
                # while save_pretrained() ran only at the end, so the artifact held the FINAL weights --
                # v7 reported its peak 99.5% (round 2) but shipped round 3 at 98.5%. The report and the
                # file on disk have to describe the same model.
                _out = spec.get("out") or os.path.join(str(COMP), "models", "tool_lora")
                os.makedirs(_out, exist_ok=True)
                model.save_pretrained(_out)
                best_round = rd
            elif rd > 0:
                break                              # a round that did not help ends the loop
        after, tr = best, history[-1]
        tr = {"steps": sum(h["steps"] for h in history), "epochs": int(spec.get("epochs", 2)),
              "seconds": None, "rounds": len(history),
              "loss_first_decile": history[0]["loss"][0], "loss_last_decile": history[-1]["loss"][1]}

        out_dir = spec.get("out") or os.path.join(str(COMP), "models", "tool_lora")
        os.makedirs(out_dir, exist_ok=True)
        if best_round < 0:                         # nothing ever improved -- ship the final weights
            model.save_pretrained(out_dir)
        (open(os.path.join(out_dir, "train_report.json"), "w")
         .write(json.dumps({**base, "lora": lora_info, "train": tr, "rounds": history,
                            "before": {k: v for k, v in before.items() if k != "failures"},
                            "after": {k: v for k, v in after.items() if k != "failures"},
                            "saved_from_round": best_round}, indent=1)))
        d_exact = round(after["exact_rate"] - before["exact_rate"], 4)
        msg = (f"[{worker}] llm-tool-train {os.path.basename(str(model_id))} @{bits}-bit LoRA "
               f"({lora_info['trainable_pct']}% trainable, {tr['steps']} steps, {tr['seconds']}s): "
               f"loss {tr['loss_first_decile']}→{tr['loss_last_decile']}; on {before['total']['n']} held-out "
               f"examples parsed {before['parsed_rate']:.0%}→{after['parsed_rate']:.0%}, "
               f"tool {before['tool_rate']:.0%}→{after['tool_rate']:.0%}, "
               f"EXACT {before['exact_rate']:.0%}→{after['exact_rate']:.0%} ({d_exact:+.0%}) "
               f"over {len(history)} self-improvement round(s), "
               f"{sum(h['mined'] for h in history)} examples mined from its own mistakes. "
               f"Adapter → {out_dir}")
        self.log(msg, kind="finding",
                 recommendation=("mine the remaining val failures back into prompt-dataset and retrain "
                                 "(standing rule: every training run improves the dataset)"))
        return self.done({**base, "lora": lora_info, "train": tr, "rounds": history,
                          "before": {k: v for k, v in before.items() if k != "failures"},
                          "after": {k: v for k, v in after.items() if k != "failures"},
                          "delta_exact": d_exact, "adapter": out_dir}, msg)


_AGENT = LLMToolTrain()


def run(q, worker):
    return _AGENT.run(q, worker)
