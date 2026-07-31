## Attribution

This solution is built on top of Parthenos's and Andreas' excellent public notebook, which I used as my starting point. The core infrastructure GPT-OSS 120B served via vLLM, the persistent Jupyter sandbox pool, the multi-attempt parallel inference loop, and the entropy-based scoring concept all originate from Parthenos's design. I credit that work as the essential foundation.

My contribution is a set of 7 targeted modifications that improved accuracy and robustness. This write-up documents both what I inherited and what I changed, along with the reasoning behind each decision.

---

## Overview

This solution is a pure inference optimization attempt. No training data, no fine-tuning. The task is to get a frontier-scale language model to correctly solve 50 IMO-level mathematical problems within a fixed time budget, with answers in the range [0, 99999].

The pipeline at a high level:

1. Serve GPT-OSS 120B locally via vLLM with FP8 KV-cache
2. For each problem, run 8 parallel reasoning attempts across 16 persistent Jupyter kernels
3. Each attempt streams a completion, executes Python tool calls (sympy, numpy, mpmath), and extracts the final answer from `\boxed{}`
4. An entropy-weighted majority vote ensemble selects the final answer
5. A dynamic time budget allocates more time to harder problems

Total inference time: ~4–5 hours for 50 problems on 1× H100 80GB.

---

## What I Changed (and Why)

### Improvement 1: Sharpened System Prompt

The most impactful single change. Parthenos's prompt was a solid 5-step protocol (UNDERSTAND / EXPLORE / PLAN / EXECUTE / VERIFY) which is a significant improvement on Andreas' notebook, but it had no explicit constraint on the answer range.

AIMO3 problems frequently require modular reduction. The raw answer might be a 20-digit number, and the problem asks for the last 5 digits. Without an explicit reminder, the model would often solve the problem correctly and then box the raw large integer, scoring zero.

**System Prompt:**

```
You are an elite mathematical problem solver competing at the International Mathematical Olympiad (IMO) level. Your sole objective is to produce the correct non-negative integer answer.

# Mandatory answer constraints:

The final answer MUST be a non-negative integer in [0, 99999]. If your raw answer exceeds 99999, the problem is asking for something modular (e.g., the last five digits, or the answer mod 10^5). Re-read the problem statement carefully before boxing your answer.

# Problem-Solving Protocol:

1. PARSE: Identify exactly what quantity is being asked for. Write it down explicitly.
2. EXPLORE: Consider at least two distinct solution strategies before committing. Note relevant theorems, constraints, and symmetries.
3. PLAN: Choose the most rigorous approach and outline key lemmas.
4. EXECUTE: Work step by step. Show all algebraic and logical steps clearly.
5. VERIFY: Before writing \boxed{}, independently re-derive or numerically confirm the answer. Check it satisfies every constraint. If verification fails, restart.

# Common pitfalls to avoid:

* Off-by-one errors in combinatorics and indexing
* Forgetting the modular reduction when the answer would otherwise exceed 99999
* Over-counting or under-counting in enumeration problems
* Sign errors and missed edge cases in algebraic manipulation
* Trusting an unverified intermediate result

# Verification checklist (complete before \boxed{}):

* Does the answer satisfy all constraints stated in the problem?
* Does it agree with small/simple special cases?
* Is it in [0, 99999]?
* Have you confirmed it via an independent route (code, substitution, or counting argument)?

# Output format:

Place your final numerical answer inside \boxed{}, e.g., \boxed{42}.
Never place an expression, fraction, or inequality inside \boxed{} — integers only.

Think step-by-step. Quality of reasoning is paramount.

```

**What I added:**

```
# Mandatory answer constraints:
The final answer MUST be a non-negative integer in [0, 99999].
If your raw answer exceeds 99999, the problem is asking for something modular
(e.g., the last five digits, or the answer mod 10^5). Re-read the problem
statement carefully before boxing your answer.
```

I also added a pre-boxing verification checklist consisting of 4 explicit questions the model must answer before writing `\boxed{}`:
- Does the answer satisfy all constraints in the problem?
- Does it agree with simple special cases?
- Is it in [0, 99999]?
- Have you confirmed it via an independent route?

And a common pitfalls list: off-by-one errors, forgotten modular reduction, over/under-counting, sign errors, trusting unverified intermediate results.

This was the highest-leverage change. It addressed a structural failure mode rather than a marginal improvement.

---

### Improvement 2: Condensed Library Hint

Parthenos's `preference_prompt` was a long multi-section description (~200 tokens) appended to the user turn for every problem. I condensed it to a single line:

**Preference prompt:**

```
Available libraries: math, numpy (numerical arrays/linear-algebra), sympy (exact symbolic computation, number theory, polynomial ops), mpmath (arbitrary-precision arithmetic, set to 64 decimal places), itertools, collections.

Use sympy for exact answers; numpy for large-scale numerics; combine both to derive symbolically and verify numerically.

```

This saves ~200 context tokens per attempt which is small in isolation, but it compounds across 8 attempts × 128 turns × 50 problems.

---

### Improvement 3: Tail-Windowed Entropy

Both my version and base notebook uses logprob entropy to measure how "confident" a completion was, for use in the ensemble. But there's an important difference in *what* entropy is measured over.

**Parthenos:** averages entropy over the full token stream.

**Mine:** averages entropy only over the last 256 tokens (`logprob_tail = 256`).

The reasoning: early in a completion, the model is exploring considering multiple approaches, writing "let me try...", reconsidering. This is high-entropy by design and doesn't tell you anything about how confident the model is in its *final* answer. The last 256 tokens are where the model commits: writing the final derivation, boxing the answer. Entropy here is a much cleaner signal.

---

### Improvement 4: Answer Extraction

Base `_scan_for_answer()` had two patterns:
1. `\boxed{N}` — the intended format
2. `"final answer is N"` — a prose fallback

The prose fallback sounds reasonable, but it fires constantly on intermediate working steps. Phrases like "the final answer is 42 for the base case, but let me now generalize..." appear throughout the model's reasoning. During streaming, this committed to wrong answers early.

I removed the prose fallback entirely, and restricted extraction to `\boxed{}` only, with two robustness improvements:
- **Comma handling**: `\boxed{42,000}` → 42000
- **Whitespace handling**: `\boxed{ 42 }` → 42
- **Last match wins**: returns the final `\boxed{}` occurrence in the window (most likely to be the definitive answer)

---

### Improvement 5: Ensemble Selection

This is where my approach differs most meaningfully from Parthenos's.

**Base `_select_answer()`:**
```python
weight = 1.0 / max(entropy, 1e-9)
answer_weights[answer] += weight  # accumulates 1/entropy per vote
score = total_weight               # pure entropy sum
```

This is **pure entropy weighting**. Vote count is tracked and displayed but never used in the score. A single attempt with very low entropy (high confidence) can override a 5-1 vote majority if the confident outlier has entropy near zero.

**My `_select_answer()`:**
```python
vote_share = votes / total_votes
confidence = 1.0 / max(median_entropy, 1e-9)
consensus = 1.0 if (votes == max_votes and confidence > 1.0) else 0.0
score = vote_share * 2.0 + confidence * 0.3 + consensus * 0.5
```

Vote share dominates (weight 2.0). Entropy is a tiebreaker (weight 0.3). The consensus bonus (0.5) nudges clear winners that are also confident.

The practical implication: Parthenos's method can be hijacked by a single very confident but wrong attempt. In the voting-ensemble of my solution notebook however, it will be much harder for a confident attempt to overturn the final answer selection. You need both majority votes and reasonable entropy to win. This matters specifically on hard problems where one attempt finds a clever shortcut (low entropy, confident) but gets the wrong answer, while the other 5 struggle through the full derivation (higher entropy) and arrive correctly.

A concrete example from my run — the 500×500 rectangles problem:

| Answer | Votes | Entropy | Parthenos score | My score |
|--------|-------|---------|-----------------|----------|
| 520    | 4     | 0.996   | 4.016           | 1.944    |
| 519    | 2     | 0.928   | 2.155           | 0.895    |
| 706    | 1     | 0.696   | 1.437           | 0.717    |

Both methods select 520 here. But consider a hypothetical where 706 had entropy 0.05: Parthenos's score would be 1/0.05 = 20.0, overruling the 4-vote majority. Mine would be 0.143 × 2.0 + 20.0 × 0.3 = 6.3 vs 520's 1.944 + 0.3 + 0.5 = 2.744. The confident outlier would still win under my formula in this extreme case, but the threshold for an outlier to override a 4-1 majority is much higher.

---

### Improvement 6: Median Entropy (Robustness)

Minor change: when computing per-answer confidence, I use median entropy across that answer's attempts rather than mean. This makes the score robust to a single outlier attempt that happened to produce very low entropy on a fluke.

---

### Improvement 7: Dynamic Uncatchable-Leader Early Stop

Parthenos's early stopping: halt when 4 attempts give the same answer (`early_stop = 4`).

My addition: also halt when the current leader's vote margin cannot be overturned even if every remaining attempt votes for second place.

```python
def _can_remaining_change_winner(self, valid_answers, attempts_completed):
    remaining = self.cfg.attempts - attempts_completed
    counts = Counter(valid_answers)
    top_two = counts.most_common(2)
    leader_count = top_two[0][1]
    second_count = top_two[1][1]
    return (second_count + remaining) > leader_count
```

If the leader has 4 votes and second place has 1 with 2 attempts remaining: `1 + 2 = 3 < 4` → stop early. This fires before the `early_stop = 4` threshold on many clear-consensus problems, freeing time budget for problems where it's genuinely needed.

---

## What Didn't Work / What I Didn't Change

- **Temperature and min_p**: left unchanged at 1.0 and 0.02. Tried lower temperatures briefly but diversity suffers.
- **More attempts**: going beyond 8 hits diminishing returns against the time budget.
- **Fine-tuning**: no training was performed. The base GPT-OSS 120B was strong enough that inference engineering yielded more gain per hour than fine-tuning in the available time.
- **Multiple system prompts**: tried running different system prompts across attempts (e.g., one "algebraic" and one "computational" persona). No measurable improvement, added complexity.

---

## What I Learned

The biggest insight from this competition is how much prompt engineering still matters even at 120B scale. The modular arithmetic reminder alone (a single paragraph addition to the system prompt) eliminated an entire class of otherwise-correct solutions that were scoring zero. The model knew how to solve the problems; it just needed to be reminded what format the answer needed to be in.

The second insight is about ensemble design. Entropy-based confidence scoring sounds sophisticated, but it's fragile when used as the primary signal. Vote count is a more robust aggregator across diverse reasoning paths where entropy is best used as a tiebreaker, not as the main criterion.

---

## Hardware & Runtime

| Item | Detail |
|------|--------|
| GPU | 1× H100 80GB (Kaggle competition hardware) |
| Model | GPT-OSS 120B, FP8 KV-cache, 65,536 context |
| Attempts per problem | 8 (parallel, 16 Jupyter kernels) |
| Avg time per problem | ~300–400 seconds |
| Total runtime (50 problems) | ~4–5 hours |
| Training | None |

---

## References

- Solution notebook: https://www.kaggle.com/code/varianceofx/let-me-improve-this-cooking?scriptVersionId=306081269
- Adapted notebook: https://huggingface.co/varianceofx/aimo3-2nd-place-solver-adapted
- Andreas' public base notebook: https://www.kaggle.com/code/andreasbis/aimo-3-gpt-oss-120b-with-tools
- Parthenos's public base notebook: https://www.kaggle.com/code/nihilisticneuralnet/44-50-let-me-over-cook
- vLLM: Kwon et al. (2023), PagedAttention, SOSP 2023
- SymPy: exact symbolic computation: https://sympy.org
- mpmath: arbitrary-precision arithmetic (64 d.p.): https://mpmath.org

---

*Thanks to Parthenos and Andreas' for sharing the base notebook openly, to the AIMO competition organisers, and to the Kaggle community for a great competition.*