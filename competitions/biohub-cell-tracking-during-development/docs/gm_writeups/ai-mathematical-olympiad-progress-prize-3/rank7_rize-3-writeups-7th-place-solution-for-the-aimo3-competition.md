# 7th  Place Solution for the AIMO3 Competition

The AI Mathematical Olympiad - Progress Prize 3 is a challenging competition that encourages participants to develop open-source AI systems capable of solving high-level mathematical reasoning problems. For me, this competition was not only a leaderboard challenge, but also a valuable opportunity to study how modern large language models reason, fail, recover, and improve under different inference strategies.

My solution is built upon a strong public baseline notebook, and my main goal was to test and reflect on related approaches, conduct several experimental attempts, and gradually identify the strategy that I considered the most suitable. In particular, I focused on prompt design, response analysis, answer extraction, and inference-time aggregation, hoping to better understand how small changes in reasoning guidance and sampling behavior can influence the final performance. I hope that sharing my observations and research can provide some value to the community.

# 1.Overview
My solution is based on the strong public baseline notebook shared by Andreas Bisiadis and Parthenos.

This solution is mainly an inference-time optimization attempt. No additional training or fine-tuning is used. The goal is to make a large open-weight language model solve AIMO3 mathematical problems as reliably as possible under the fixed competition time budget, with final answers in the range [0, 99999].
The pipeline at a high level:
- Use GPT-OSS 120B as the main reasoning model
- Use a concise IMO-style prompt with strict boxed-answer format
- For each problem, run up to 8 independent reasoning attempts
- Allow the model to use Python tools for calculation and verification
- Extract valid integer answers from \boxed{}
- Aggregate answers with simple inverse-entropy weighting
- Use frequency-based early stopping to save inference time
- Control the per-problem budget under the global notebook time limit

The final version favors simplicity and stability. I tested more complex prompt and entropy variants, but the submitted version kept the components that were easier to interpret and more stable in my submissions.

# 2. Details of the Submission
## 2.1 Main Focus
My submission mainly explores two practical questions:
- How does prompt complexity affect mathematical reasoning performance, and is a highly structured prompt necessarily better than a concise task-focused prompt?
- How should multiple generated answers be aggregated under the competition time limit, especially when comparing simple entropy-weighted aggregation with more complex entropy-based strategies?

### 2.1.1 Prompt Complexity vs. Stability
One practical question I explored was how prompt design can help the model avoid common mistakes in mathematical reasoning. In olympiad-style problems, a wrong answer is often not caused by a complete lack of relevant knowledge, but by smaller failures during the reasoning process: choosing a solution path too early, making arithmetic or algebraic mistakes, overlooking constraints, ignoring boundary cases, or failing to verify the final result.

Inspired by Parthenos's five-step protocol — UNDERSTAND, EXPLORE, PLAN, EXECUTE, and VERIFY — I tested a more complex prompting strategy in some experimental versions.  These prompts encouraged the model to follow a more deliberate problem-solving process: first understand the problem, then explore possible strategies, make a plan, execute the solution, and finally verify the result. I also added reminders to check arithmetic, test special cases, consider constraints, and use Python for calculations or verification when appropriate. The goal was not to teach the model new mathematics, but to make its reasoning process less careless and more self-checking.

In practice, longer prompts introduced extra instructions and did not always give a stable advantage. They also consumed more context and sometimes made the reasoning process less direct. Therefore, in the final version, I chose a simpler prompt that focused on the most important constraints: the model should act as a strong olympiad-level solver, the answer must be a non-negative integer in the required range, and the final answer must be placed inside \boxed{}.

```python
system_prompt = (
    'You are a world-class International Mathematical Olympiad (IMO) competitor. '
    'The final answer must be a non-negative integer between 0 and 99999. '
    'You must place the final integer answer inside \\boxed{}. '
)

tool_prompt = (
    'Use this tool to execute Python code. '
    'The environment is a stateful Jupyter notebook. '
    'You must use print() to output results.'
)

preference_prompt = (
    'You have access to math, numpy and sympy to solve the problem.'
)
```

This choice reflects a balance between guidance and simplicity. The prompt provides the essential task framing and output constraints, while Python tool usage, multiple independent attempts, answer extraction, and entropy-weighted aggregation handle much of the practical reliability. In this way, prompt design helped reduce common errors mainly by clarifying the task, enforcing the answer format, and encouraging a careful mathematical-solving mindset, without making the final system overly complicated.

I should note that this was not a rigorous ablation study of prompt engineering. Rather, it was a practical exploration during the competition, and the final prompt was selected because it was simple, stable, and consistent with the rest of the inference pipeline.

| Prompt Strategy | Main Design | Entropy Strategy Setting | Number of Submissions | Public LB Score Range | Observation |
|---|---|---|---:|---|---|
| Final submitted prompt | Short and direct prompt focusing on IMO-level role, valid answer range, and boxed-answer format | Same entropy strategy: simple inverse-entropy weighting | 6 | 36–40<br>(Scored 40 twice) | More stable in practice and better aligned with the rest of the inference pipeline |
| More structured error-aware prompt | Explicitly guides the model through understanding, exploration, planning, execution, and verification | Same entropy strategy: simple inverse-entropy weighting | 6 | 35–39 | Helpful in principle, but did not show a stable advantage in my submissions |

Based on these submission results, I finally chose the shorter prompt for the final submission. Although the more structured prompt was designed to reduce common reasoning errors, such as premature conclusions, missed constraints, and insufficient verification, it did not provide a consistently better leaderboard result in my tests. One possible reason is that a longer prompt introduces more instructions for the model to follow, which may consume context, make the reasoning process less direct, and increase sensitivity to sampling randomness. In contrast, the shorter prompt kept only the most important constraints and worked more smoothly with the rest of the system, including Python tool use, multiple independent attempts, answer extraction, and entropy-weighted aggregation.

### 2.1.2 Entropy-Based Answer Aggregation
In addition to the simple inverse-entropy weighting strategy, I also tested a more complex entropy metric inspired by Parthenos. Instead of using only the mean entropy of generated tokens, this variant tried to model several aspects of reasoning confidence: the average uncertainty level, the confidence of later tokens closer to the final answer, the variance of entropy across the reasoning process, the proportion of sustained high-entropy tokens, and the existence of long low-entropy streaks.

Although the complex entropy metric was more expressive, I finally selected the simpler inverse-entropy weighting strategy for the final submission. The main reason was robustness. The complex metric introduced several hand-designed components and hyperparameters, such as the position decay factor, high-entropy threshold, low-entropy threshold, and the weights of each component. With only a limited number of leaderboard submissions and a small number of attempts per problem, these extra choices were difficult to tune reliably.

| Aggregation Strategy | Same Prompt Setting | Main Idea | Number of Submissions | Public LB Score Range | Observation |
|---|---|---|---:|---|---|
| Simple inverse-entropy weighting | Yes | Use mean token entropy as an uncertainty signal, then weight each valid answer by 1 / entropy | 6 | 36–40<br>(Scored 40 twice) | Simple, stable, and easy to interpret |
| More complex entropy metric | Yes | Combine mean entropy, position-weighted entropy, entropy variance, high-entropy penalty, and low-entropy streak reward | 3 | 36–39 | More expressive, but did not show a stable advantage |

In addition, token-level entropy is only an approximate confidence signal. A more complicated transformation of this signal may amplify noise rather than provide better calibration. In my tests, the complex strategy achieved scores in the range of 36–39, while the simpler strategy achieved 36–40. Therefore, I did not observe a sufficiently stable advantage from the complex variant. I chose the simple inverse-entropy method because it was easier to interpret, less dependent on tuning, and more consistent with the practical constraints of the competition.

## 2.2 What Was Tried and Limitations
During development, I also explored several practical variants that were not selected for the final submission:
- Longer Per-Problem Budget: Giving each problem more time could help difficult cases, but it risked consuming too much time early and hurting later problems. I therefore kept a dynamic budget strategy.
- More Aggressive Early Stopping: Stopping earlier after repeated answers can save time, but it may also lock in a repeated wrong answer. I kept early stopping conservative and used it mainly for efficiency.
- More Extensive Parameter Tuning: I considered broader tuning of temperature, min_p, and timeout settings, but leaderboard feedback and compute resources were limited. The final settings were chosen for practical stability rather than theoretical optimality.
Overall, these attempts suggested that more computation or more aggressive control rules do not always improve performance. Under the competition constraints, I preferred a simpler and more stable configuration.

Because of the limited number of submissions and other practical constraints, my prompt exploration was not exhaustive. I still believe that better prompt designs could potentially bring meaningful improvements. However, among the variants I actually tested, the alternative prompts did not show a clear performance gain. Therefore, my final choice was based on empirical testing and the actual scores observed in practice.

# 3. Hardware & Runtime

| Item | Detail |
|---|---|
| GPU | 1× H100 80GB |
| Model | GPT-OSS 120B, FP8 KV-cache, up to 81,920 context tokens |
| Attempts per problem | 8 parallel attempts |
| Jupyter kernels | 16 persistent Jupyter kernels |
| Avg time per problem | ~300–400 seconds |
| Total runtime (50 problems) | ~4–5 hours |
| Training | None |

# 4. Conclusion and Acknowledgements
I would like to sincerely thank Kaggle and the host team for organizing this challenging and inspiring competition. I am also deeply grateful to the outstanding contributors in the community, especially Andreas Bisiadis, parthenos, and many others who shared strong notebooks, thoughtful discussions, and detailed write-ups. Their work provided valuable references for my own testing, reflection, and improvement. I have deep respect for their contributions, and I hope that sharing my observations and research can provide some value to the community.

## Reference
- Andreas Bisiadis[Link to Kaggle Code](https://www.kaggle.com/code/andreasbis/aimo-3-gpt-oss-120b-with-tools)
- parthenos[Link to Kaggle Code](https://www.kaggle.com/code/nihilisticneuralnet/44-50-let-me-over-cook)
- parthenos[Link to Kaggle Code](https://www.kaggle.com/code/nihilisticneuralnet/43-50-aimo-3-gpt-oss-120b-weighted-entropy)
- vLLM: Kwon et al. (2023), PagedAttention, SOSP 2023
- SymPy: exact symbolic computation: [https://www.sympy.org/en/index.html](https://www.sympy.org/en/index.html)

## Competition Link
- Competition overview page: [Link](https://www.kaggle.com/competitions/ai-mathematical-olympiad-progress-prize-3/overview)
- Competition data page: [Link](https://www.kaggle.com/competitions/ai-mathematical-olympiad-progress-prize-3/data)

*Thanks to Parthenos and Andreas Bisiadis for openly sharing the base notebook, to the AIMO competition organizers for hosting the competition, and to the Kaggle community for the valuable discussions and contributions.*