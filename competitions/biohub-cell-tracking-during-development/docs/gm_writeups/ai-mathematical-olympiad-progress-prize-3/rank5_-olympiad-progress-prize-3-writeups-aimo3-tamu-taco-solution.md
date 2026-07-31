# TAMU-TACO 5th Place Solution for AIMO3

## Huggingface Space

Link: [TAMU-TACO Huggingface](https://huggingface.co/spaces/peiranli0930/AIMO3-TAMU-TACO)

## TL;DR

Our final submission did not train a new model. We used the pretrained `gpt-oss-120b` model and focused on making its inference behavior stable under the AIMO3 evaluation rules.

The main idea was to treat AIMO3 not only as a math reasoning benchmark, but as a sequential resource allocation problem. The system sees one hidden problem at a time, has a 5-hour GPU notebook budget, runs on one H100 80GB GPU, and cannot know whether future problems will be easy or hard. This means that every second spent on the current problem can help or hurt the rest of the run.

We found that private-score robustness was mainly controlled by the distribution of hard problems in the problem order, and by four coupled inference parameters:

```text
high_problem_timeout = 900
base_problem_timeout = 300
early_stop = 4
attempts = 8
```

To choose these values, we built `AutoResearchAgent`, a human-in-the-loop experiment engine that constructed an AIMO3-like local test set, stress-tested different hard-problem placements, ran repeated experiments, logged score and system metrics, and helped us select robust parameters. Human insight defined the search space and final decisions; the agent made the research loop measurable and repeatable.

![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F30948873%2F09bc96d70bcabd34e60ad0352dbefbf4%2Fs4.png?generation=1780602446926721&alt=media)

## 1. What Was Different About Our Approach

A straightforward way to think about this competition is: "How can we make the model solve each individual math problem better?"

That was not the framing that gave us the final gain. We instead asked:

> How should a strong pretrained reasoning model spend limited time and vRAM across an unknown 50-problem sequence?

This distinction mattered. If the system spends too much time on one extremely hard problem, it may lose the chance to answer several later problems that were actually solvable. If it spends too little time, it may cut off hard problems that the model was close to solving. If it runs too few parallel attempts, it leaves H100 memory underused. If it runs too many, it risks OOM or slower inference.

So the winning strategy was not a single prompt trick. It was a global inference policy that answered four questions for every problem:

- How much time must be protected for ordinary solvable problems?
- How much extra time can a near-solvable hard problem receive?
- How many independent reasoning paths should run in parallel?
- When is answer agreement strong enough to stop early?

The final notebook is the execution layer. The core contribution is the resource-allocation strategy and the experimental process used to lock it.

## 2. AIMO3 as a Sequential Resource Allocation Problem

The AIMO3 rules create several constraints that are easy to underestimate:

- Problems are hidden and served one by one.
- The next problem is not visible until the current problem is finished.
- The GPU notebook runtime is limited to 5 hours.
- Setup and vLLM startup consume part of that budget.
- The final submission runs on a single H100 80GB GPU.
- The answer must be an integer in `[0, 99999]`.
- Internet is disabled during the Kaggle rerun.

![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F30948873%2Fdbba5cf024c8ec671a73a2eaf2cd1db3%2Fs3.png?generation=1780602486661406&alt=media)

The most important consequence is that the system cannot plan with knowledge of the full test set. It has to decide how aggressively to spend compute before it knows what comes next.

This also changes how GPU memory should be viewed. If the current problem uses too few attempts, the unused vRAM cannot be transferred to a future problem, because that future problem has not been served yet. At the same time, using too much parallelism can cause memory pressure, lower throughput, or make a single problem consume too much of the global time budget.

In our view, AIMO3 was therefore a systems problem wrapped around a math reasoning benchmark.

## 3. Why Hard-Problem Order Was the Hidden Risk

The biggest source of private-test instability was not only whether a problem was hard. It was where hard problems appeared in the sequence.

A few hard problems can dominate runtime. If they appear early, an uncontrolled system can spend a large fraction of the 5-hour budget before seeing easier problems. If they appear in the middle, they can interrupt a stable pace. If they appear late, the system only handles them well if it has preserved enough time.

![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F30948873%2F20f7b0b7113be420d383313dff652816%2Fs5.png?generation=1780602506555970&alt=media)

This is why we did not tune for one random validation order. A parameter set can look good on a lucky shuffle and fail under an unfavorable private order. Our tuning objective was to find parameters that survived multiple hard-problem placements, not parameters that achieved the highest score in one run.

This was one of the central lessons of our solution: robustness came from modeling the distribution of hard problems across the run.

## 4. The Four Parameters That Controlled the Run

After early experiments and system analysis, we narrowed the critical search space to four parameters.

| Parameter | Final value | Role |
| --- | ---: | --- |
| `base_problem_timeout` | `300` seconds | Protects time for ordinary solvable problems. |
| `high_problem_timeout` | `900` seconds | Gives near-limit hard problems a chance without letting them drain the run. |
| `attempts` | `8` | Uses H100 vRAM through parallel reasoning paths. |
| `early_stop` | `4` | Stops once answer consensus is reliable enough. |

These parameters are coupled. They should not be understood as four independent knobs.

`base_problem_timeout` and `high_problem_timeout` define the time policy. The base timeout protects the run from spending all its time on early hard problems. The high timeout gives difficult but solvable problems enough room when the global budget allows it.

`attempts` and `early_stop` define the sampling policy. More attempts improve self-consistency and better use the GPU, but too many attempts increase memory and latency risk. A lower early-stop threshold saves time but may trust weak agreement. A higher threshold is more reliable but may wait too long.

The final values were selected to maximize stable score, not single-run peak score.

## 5. Building a Local Test Set `S`

The public and private test problems were hidden, and leaderboard feedback was too sparse for reliable tuning. We needed a local environment where we could repeatedly test inference policies.

We built a 100-problem local validation set `S`.

![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F30948873%2F48e733b6da5fdd91a6af27b138f083e1%2Fs7.png?generation=1780602515926402&alt=media)

The construction used the 10 official AIMO3 reference problems as difficulty anchors:

- Problems solved by GPT-5 high at pass@1 became Basic anchors.
- Problems solved only at pass@3 became Harder anchors.
- The problem not solved at pass@3 became the Hardest anchor.

Here, pass@1 means solving with one attempt, and pass@3 means allowing up to three independent attempts. We used this simple calibration to separate problems that were immediately within reach from problems that required more sampling or were beyond the model's reliable range.

We then screened candidate problems from public math sources including AIMO2, OpenMathReasoning, and Putnam-AXIOM. GPT-5 high judged difficulty and answer-format suitability, and human review was used to remove or replace unsuitable problems.

The final mix was:

```text
70 basic
20 harder
10 hardest
```

For tuning, we merged `harder + hardest` into a single Hard group:

```text
70 Basic + 30 Hard
```

This local test set was not used to train the model. It was used only to tune the inference policy.

## 6. Stress-Testing Hard-Problem Placement

For each candidate hyperparameter vector `H`, we tested four placements of the 30 Hard problems:

```text
hard_front50
hard_middle50
hard_back50
hard_random
```

Each placement was repeated 3 times, so one candidate `H` was evaluated through:

```text
4 placements x 3 repeats = 12 full evaluations per H
```

![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F30948873%2F9bb06f67ac5ece961abbf5bee43e2671%2Fs8.png?generation=1780602524303799&alt=media)

This protocol made the hidden private-test risk measurable. A configuration that was good only when hard problems were late or randomly spread out would not be selected. We wanted a parameter region that stayed strong when hard problems appeared in the front, middle, back, or random positions.

## 7. AutoResearchAgent

`AutoResearchAgent` was our experiment engine. It was not the whole idea, and it did not replace human judgment.

The human-designed part came first:

- Identify that the rules create a global resource-allocation problem.
- Identify hard-problem placement as the main robustness risk.
- Identify the four critical parameters.
- Define the local test set construction.
- Define the hard-placement stress tests.
- Define the two-phase search.

AutoResearchAgent then made this strategy executable:

- Build the local validation set `S`.
- Run candidate hyperparameter vectors across hard placements.
- Log per-problem correctness, elapsed time, valid answers, entropy, early stopping, timeout, OOM, GPU utilization, vRAM usage, CPU utilization, and RAM usage.
- Summarize each experiment suite.
- Use GPT-5 high to analyze summaries from a systems perspective.
- Propose the next candidate `H` within our predefined ranges.
- Leave final region selection to human expert review.

![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F30948873%2Fc43b7de56056913ed6ea5696e33dda69%2Fs9.png?generation=1780602532386371&alt=media)

The key point is that this was a human-in-the-loop research loop. The agent gave us scale, structure, and auditability. The core research judgment came from modeling the competition constraints correctly. The experiment phase used the same `gpt-oss-120b` / vLLM / Jupyter-tool inference stack as the final notebook, so the logged behavior reflected real runtime trade-offs rather than a lightweight proxy.

## 8. Search Design and Experiment Count

The initial values came from the competition clock and pilot experiments.

The full GPU notebook budget is 5 hours:

```text
5 hours = 18000 seconds
```

We reserved 600 seconds for setup and vLLM startup:

```text
18000 - 600 = 17400 seconds
```

With 50 problems, this gives:

```text
17400 / 50 = 348 seconds per problem
```

So the initial timeout values were:

```text
high_problem_timeout = 348
base_problem_timeout = 348
```

Early pilot tests used:

```text
early_stop = 3
attempts = 6
```

In 20 Basic problems from `S`, 16 produced at least 3 matching answers under this setup. GPU vRAM usage was roughly 67.8-71.6GB out of 80GB, showing that 6-way inference was already using the H100 effectively.

We then bounded the search:

```text
348 <= high_problem_timeout <= 1000
200 <= base_problem_timeout <= 348
3 <= early_stop <= 5
3 <= attempts <= 8
```

The lower bound of 200 seconds for `base_problem_timeout` came from the observation that the fastest correct AIMO3 example solution under the pilot setup took 212 seconds. The upper bound of 8 attempts came from memory. In our H100 80GB tests, 8-way inference with a 16k-32k total token budget per path used about 68.6-74.2GB of vRAM. Going beyond 8 attempts would increase OOM risk or require batching the same problem into multiple inference waves.

![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F30948873%2Ff818872b098105e2c024af10002806b6%2Fs11.png?generation=1780602540154555&alt=media)

The search had two phases.

### Phase 1: Time Allocation

We fixed:

```text
attempts = 6
early_stop = 3
```

Then we tuned:

```text
high_problem_timeout
base_problem_timeout
```

This phase had:

```text
46 candidate rounds
46 x 4 x 3 = 552 placement/repeat experiments
```

The stable high-score region was:

```text
high_problem_timeout = 894-906
base_problem_timeout = 288-306
```

After manual metric review, we locked the conservative values:

```text
high_problem_timeout = 900
base_problem_timeout = 300
```

### Phase 2: Parallel Sampling and Consensus

We fixed:

```text
high_problem_timeout = 900
base_problem_timeout = 300
```

Then we tuned:

```text
attempts
early_stop
```

This phase had:

```text
18 candidate rounds
up to 18 x 4 x 3 = 216 placement/repeat experiments
```

The best score-variance balance came from:

```text
attempts = 8
early_stop = 4
```

Overall, AutoResearchAgent covered 64 candidate rounds and up to 768 full placement/repeat experiments. Each full placement/repeat experiment ran the 100-problem local set `S` under one hard-problem ordering. The hyperparameter tuning environment used local research machines with 64 NVIDIA H100 80GB GPUs. This was separate from the final Kaggle inference environment, where the submitted notebook used one H100 80GB GPU.

## 9. Final Notebook Pipeline

The final Kaggle notebook implemented the locked policy with the pretrained `gpt-oss-120b` model.

![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F30948873%2Fdaacd00bd5f0cee196aac97694ea7f17%2Fs13.png?generation=1780602550546976&alt=media)

For each problem:

1. The notebook computes a dynamic time budget from the global remaining time and the number of problems left.
2. The budget is clipped between `base_problem_timeout=300` and `high_problem_timeout=900`.
3. Up to 8 independent reasoning attempts are launched through a local vLLM OpenAI-compatible API server.
4. Each attempt can call a pre-started Jupyter Python kernel for symbolic or numerical computation.
5. During streaming generation, an attempt can stop once it emits a valid boxed integer.
6. If the same valid integer appears 4 times, unfinished attempts are stopped early.
7. If no answer reaches the early-stop threshold, valid answers are combined with entropy-weighted voting.
8. The final output is forced to be an integer in `[0, 99999]`.
9. If no valid answer is found, the notebook returns `0` as a safe fallback.

Important runtime choices included:

```text
vLLM local API server
kv_cache_dtype = fp8_e4m3
gpu_memory_utilization = 0.96
context_tokens = 65536
batch_size = 256
--async-scheduling
--enable-prefix-caching
16 persistent Jupyter kernels for Python tools
```

The Jupyter kernels were pre-started and warmed with common math libraries such as `math`, `numpy`, `sympy`, `itertools`, `collections`, and `mpmath`. This avoided paying a cold-start cost when the model decided to use Python during a reasoning trace.

## 10. Answer Aggregation

We used two layers of answer aggregation.

### Common of N

If 4 attempts produced the same valid integer answer, we treated that as reliable consensus and stopped the rest of the attempts for that problem.

This was important because easy and medium problems often reached consensus before all 8 attempts finished. Early stopping saved time for later hard problems.

### Entropy-Weighted Voting

If no answer reached 4 votes, we used entropy-weighted voting among valid parsed answers.

For each generated token, vLLM returned top logprobs. We converted the top-5 logprobs into probabilities:

```text
p_i = exp(logprob_i)
```

Then we computed token entropy:

```text
H_token = -sum_i p_i * log2(p_i)
```

The mean entropy of an attempt was the average token entropy over generated tokens with available logprobs. Lower entropy means the model distribution was more concentrated, so that attempt received more weight:

```text
vote_weight = 1 / mean_entropy
```

This gave us a fallback when the model did not reach strict answer consensus.

## 11. Experiments and Directions We Did Not Use

Early in the competition, we explored the usual questions:

- Which open-source reasoning model should be the backbone?
- Is extra fine-tuning or RL training necessary?
- How can we make the output format reliably contain `\boxed{}`?
- How should tool-integrated reasoning be implemented?
- How many parallel samples can the H100 support safely?
- How should we stop attempts without losing reliability?

The final decision was to not fine-tune `gpt-oss-120b`. The pretrained model already had strong mathematical reasoning ability. The larger gain came from configuring inference so that the model's ability was used reliably under the exact competition constraints.

We also avoided optimizing only for public leaderboard feedback or a single local shuffle. The hidden private order was too important. Once we understood that hard-problem placement could change the whole run, the research effort moved toward local stress tests and robust parameter selection.

## 12. External Data and External Models

The final Kaggle notebook did not train on external data and did not call external services during inference.

Outside the final notebook, we used public math datasets only to build the local validation set `S` for inference-parameter tuning. Candidate sources included:

- AIMO2
- OpenMathReasoning
- Putnam-AXIOM

GPT-5 high was used outside the Kaggle submission for difficulty filtering, test-set construction support, and research-summary analysis. It was not part of the final inference notebook.

## 13. Hardware

Final Kaggle submission environment:

```text
Platform: Kaggle Notebook
GPU: 1 x NVIDIA H100 80GB
Internet: disabled
Model: pretrained gpt-oss-120b
Training: none
```

AutoResearchAgent tuning environment:

```text
Platform: local research machines
GPU: 64 x NVIDIA H100 80GB
OS: Ubuntu 24.04
CPU: AMD EPYC 9005 Series
RAM: 4TB 6000MT/s ECC DDR5 RDIMM
```

The distinction is important: the final winning notebook was a single-H100 Kaggle submission. The larger research compute was used for experiment automation and parameter selection.

## 14. What Set Us Apart

We believe there were five main differentiators.

First, we modeled the rules, not only the math problems. The sequential serving format, unknown order, 5-hour runtime, and single H100 memory limit directly determined the optimal inference policy.

Second, we identified hard-problem placement as the key robustness risk. This pushed us to test front, middle, back, and random hard-problem distributions instead of relying on one shuffle.

Third, we found the four parameters that actually controlled the run: `base_problem_timeout`, `high_problem_timeout`, `attempts`, and `early_stop`.

Fourth, we built AutoResearchAgent to make the research loop measurable. It logged score, variance, timeout, OOM, entropy, valid-answer behavior, and resource usage for every candidate setting.

Fifth, we matched the final notebook to the H100 system. Parallel attempts used GPU memory aggressively, persistent Jupyter kernels used CPU/RAM for tools, early stopping protected time, and dynamic timeout allocation protected the full 50-problem run.

## 15. Main Takeaway

With a strong pretrained reasoning model, model training is not the only path to a winning solution. In AIMO3, correct problem modeling was decisive.

`gpt-oss-120b` had the mathematical ability. The challenge was to expose that ability under a strict, sequential, time-limited evaluation. Our solution was to convert the competition rules into a measurable inference-time resource-allocation problem, stress-test the hidden hard-problem order risk, and lock a robust policy before final submission.

The broader lesson is that for reasoning benchmarks with strict runtime constraints, inference policy can be as important as model choice.