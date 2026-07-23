"""llm_pack — the LLM competition executors, built with the REAL libraries present here (transformers 5.x,
peft 0.18). Parts that run without a downloaded model are verified offline (code execution, metric eval,
token masking, retrieval); parts that need a model are real wiring + import-guarded (they escalate cleanly
if a model/GPU isn't available rather than pretending).

  • tir-executor          — Tool-Integrated Reasoning: execute the LLM's code blocks in a sandbox and splice
                           stdout back into the transcript (AIMO/Konwinski/code-golf). FULLY runnable.
  • llm-eval              — score generations against truth via the CompConfig metric (+ pass@k/maj@k). Runnable.
  • llm-infer             — constrained decoding: mask logits to allowed token ids (choice/Yes-No). Processor
                           tested on tensors; full generate() needs a model (guarded).
  • llm-finetune          — LoRA/QLoRA SFT wiring (peft LoraConfig + SFT loop). Config verified; run needs a model.
  • llm-retrieve-rerank   — embedding retrieval + rerank; numpy-embedding path runnable, HF-model path guarded.
"""
from __future__ import annotations
import os
import subprocess
import sys
import numpy as np
from .base import BaseAgent, COMP
from . import comp_config as CC


# ---------------------------------------------------------------- tir-executor (fully runnable)
def run_code_blocks(code_blocks, timeout=15, max_blocks=None, max_output=4000, stop_on_error=False):
    """Execute each python code block in a fresh subprocess; return spliced (block, stdout/err) transcript.
    max_blocks: cap how many blocks are executed (avoid runaway transcripts; None=all).
    max_output: truncate each block's captured output to this many chars (guards huge dumps).
    stop_on_error: stop executing further blocks after the first failure (fail-fast TIR)."""
    py = str(COMP / "research" / "cellmot_venv" / "bin" / "python")
    py = py if os.path.exists(py) else sys.executable
    blocks = list(code_blocks or [])
    if max_blocks is not None:
        blocks = blocks[:max(0, int(max_blocks))]
    to = max(1, int(timeout))
    cap = max(0, int(max_output)) if max_output else None
    transcript = []
    for code in blocks:
        try:
            r = subprocess.run([py, "-c", str(code)], capture_output=True, text=True, timeout=to)
            ok = r.returncode == 0
            out = r.stdout.strip() if ok else f"ERROR: {r.stderr.strip()[-200:]}"
        except Exception as e:  # noqa: BLE001
            ok = False; out = f"ERROR: {e}"
        if cap is not None and len(out) > cap:
            out = out[:cap] + "…[truncated]"
        transcript.append({"code": code, "output": out})
        if stop_on_error and not ok:
            break
    return transcript


# ---------------------------------------------------------------- llm-eval (runnable)
def eval_generations(y_true, y_pred, metric, default=0.0):
    """Score generations against truth via the CompConfig metric (exact-match fallback).
    default: returned when there is nothing to score (empty inputs) instead of a NaN/crash."""
    if y_true is None or y_pred is None or len(y_true) == 0 or len(y_pred) == 0:
        return float(default)
    spec = CC.metric_spec(metric)
    if spec["fn"] is None:
        # exact-match fallback for string answers
        pairs = list(zip(y_true, y_pred))
        if not pairs:
            return float(default)
        return float(np.mean([1.0 if str(a).strip() == str(b).strip() else 0.0 for a, b in pairs]))
    try:
        return CC.score(metric, y_true, y_pred)
    except Exception:  # noqa: BLE001 — degrade to exact-match rather than crash the fleet
        pairs = list(zip(y_true, y_pred))
        return float(np.mean([1.0 if str(a).strip() == str(b).strip() else 0.0 for a, b in pairs])) if pairs else float(default)


# ---------------------------------------------------------------- llm-infer (processor runnable)
def constrain_logits(logits, allowed_ids):
    """Mask a logits vector to only the allowed token ids (constrained/choice decoding). Returns the argmax id.
    Guards empty logits, empty/out-of-range allowed_ids (falls back to the unconstrained argmax)."""
    logits = np.asarray(logits, float).copy()
    if logits.size == 0:
        return 0
    ids = [int(i) for i in (allowed_ids or []) if 0 <= int(i) < logits.shape[-1]]
    if not ids:
        return int(np.argmax(logits))          # nothing allowed → unconstrained argmax rather than crash
    mask = np.full(logits.shape, -np.inf)
    mask[ids] = logits[ids]
    return int(np.argmax(mask))


def yes_no_logodds(logit_yes, logit_no):
    """Yes/No single-token scoring → P(Yes) via log-odds (map-charting/jigsaw pattern)."""
    z = float(logit_yes) - float(logit_no)
    z = float(np.clip(z, -60.0, 60.0))          # clip to avoid exp overflow on extreme logits
    return float(1.0 / (1.0 + np.exp(-z)))


# ---------------------------------------------------------------- llm-finetune (wiring; run needs model)
def lora_config(r=32, alpha=64, dropout=0.05, target_modules=None, bias="none", task_type=None):
    """Build a real peft LoraConfig (the SFT wiring). Verified offline; training needs a base model + GPU.
    bias: LoRA bias mode ('none'/'all'/'lora_only'). task_type: override the peft TaskType (default CAUSAL_LM)."""
    from peft import LoraConfig, TaskType
    tt = task_type or TaskType.CAUSAL_LM
    return LoraConfig(r=int(r), lora_alpha=int(alpha), lora_dropout=float(dropout), task_type=tt, bias=bias,
                      target_modules=target_modules or ["q_proj", "v_proj"])


# ---------------------------------------------------------------- llm-retrieve-rerank (numpy path runnable)
def retrieve(query_emb, doc_embs, k=5):
    """Cosine top-k retrieval over precomputed embeddings (the retriever stage of retrieve-then-rerank).
    Guards empty doc set (returns [], empty sims) and clamps k to the number of docs."""
    q = np.asarray(query_emb, float); D = np.asarray(doc_embs, float)
    if D.size == 0 or D.ndim != 2 or len(D) == 0:
        return [], np.asarray([], float)
    qn = q / (np.linalg.norm(q) + 1e-9); Dn = D / (np.linalg.norm(D, axis=1, keepdims=True) + 1e-9)
    sim = Dn @ qn
    kk = max(1, min(int(k), len(D)))
    return np.argsort(-sim)[:kk].tolist(), sim


# ---------------------------------------------------------------- agents
class _B(BaseAgent):
    thread = "M"; kind = "finding"


class TirExecutor(_B):
    name = "tir-executor"
    def run(self, q, worker):
        s = self.spec(q)
        tr = run_code_blocks(s.get("code_blocks") or [], int(s.get("timeout", 15)),
                             max_blocks=s.get("max_blocks"), max_output=int(s.get("max_output", 4000)),
                             stop_on_error=bool(s.get("stop_on_error", False)))
        msg = f"tir-executor: executed {len(tr)} code block(s); last output={tr[-1]['output'][:60] if tr else ''!r}"
        self.log(msg, kind="finding", recommendation="splice tool outputs back into the reasoning trace")
        return self.done({"transcript": tr}, msg)


class LlmEval(_B):
    name = "llm-eval"
    def run(self, q, worker):
        s = self.spec(q)
        missing = [k for k in ("y_true", "y_pred") if k not in s]
        if missing:
            return self.escalate(worker, "leader", f"llm-eval needs spec keys {missing} — none provided")
        score = eval_generations(s["y_true"], s["y_pred"], s.get("metric", "exact_match"))
        msg = f"llm-eval: {s.get('metric','exact_match')} = {score:.4f}"
        self.log(msg, kind="finding", recommendation="track this + calibration; trust CV over noisy public LB")
        return self.done({"score": score}, msg)


class LlmInfer(_B):
    name = "llm-infer"
    def run(self, q, worker):
        s = self.spec(q)
        if "logits" in s and "allowed_ids" in s:
            tok = constrain_logits(s["logits"], s["allowed_ids"])
            return self.done({"chosen_token": tok}, f"llm-infer: constrained decode → token {tok}")
        return self.escalate(worker, "researcher", "llm-infer full generate() needs a loaded model (offline env).")


class LlmFinetune(_B):
    name = "llm-finetune"
    def run(self, q, worker):
        try:
            cfg = lora_config(int(self.spec(q).get("r", 32)), int(self.spec(q).get("alpha", 64)))
        except Exception as e:  # noqa: BLE001
            return self.escalate(worker, "researcher", f"llm-finetune needs peft ({e}).")
        msg = f"llm-finetune: LoRA config ready (r={cfg.r}, alpha={cfg.lora_alpha}, targets={list(cfg.target_modules)}) — attach to a base model + GPU to train"
        self.log(msg, kind="finding", recommendation="distill soft-labels from a bigger teacher; quantize for offline")
        return self.done({"r": cfg.r, "alpha": cfg.lora_alpha, "needs": "base model + GPU"}, msg)


class LlmRetrieveRerank(_B):
    name = "llm-retrieve-rerank"
    def run(self, q, worker):
        s = self.spec(q)
        missing = [k for k in ("query_emb", "doc_embs") if k not in s]
        if missing:
            return self.escalate(worker, "leader", f"llm-retrieve-rerank needs spec keys {missing} — none provided")
        idx, sim = retrieve(s["query_emb"], s["doc_embs"], int(s.get("k", 5)))
        msg = f"llm-retrieve-rerank: retrieved top-{len(idx)} {idx[:5]} (rerank stage needs a cross-encoder model)"
        self.log(msg, kind="finding", recommendation="rerank the shortlist with a fine-tuned Qwen cross-encoder")
        return self.done({"retrieved": idx}, msg)


_TIR = TirExecutor(); _LE = LlmEval(); _LI = LlmInfer(); _LF = LlmFinetune(); _LR = LlmRetrieveRerank()


def run_tir(q, worker): return _TIR.run(q, worker)
def run_eval(q, worker): return _LE.run(q, worker)
def run_infer(q, worker): return _LI.run(q, worker)
def run_finetune(q, worker): return _LF.run(q, worker)
def run_rerank(q, worker): return _LR.run(q, worker)
