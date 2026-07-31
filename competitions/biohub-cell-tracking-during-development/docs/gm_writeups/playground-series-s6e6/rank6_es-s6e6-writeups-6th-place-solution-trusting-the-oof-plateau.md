# 6th Place Solution: Trusting The OOF Plateau

Congrats to @optimistix for the win, and to the rest of the top 10: @xx1263, @nybbler, @cindyxue1122, @milanfx, @pchoi85, @jerry34, @vaibhavnakrani, and @liornis.

This was only my second competition, and compared with the last one there were a lot more overfitted public LB scores here. The core of my solution was built after the first week, with some final tweaks in the last days, but I honestly did not expect it to place this well. In the end, trusting local OOF mattered more than chasing the public LB.

I finished 6th with a 92-model OOF probability stack:

| Public | Private | OOF |
|---:|---:|---:|
| 0.97130 | 0.97054 | 0.970718 |

The solution was a wide stack of public artifacts and locally trained/tuned models that reached a good plateau early. Most of the rest of the competition was spent trying new ideas and deciding not to use them.

## Final Stack

The final stack used 92 models, a mix of public artifacts and models I trained/tuned locally with 5-fold validation.

Family mix:

| Family | Count |
|---|---:|
| RealMLP / RealMLP blends | 24 |
| XGBoost | 19 |
| CatBoost | 12 |
| LightGBM | 5 |
| public XGB artifacts | 5 |
| public RealMLP artifacts | 4 |
| public stacked probability artifacts | 4 |
| FM-style models | 3 |
| FT-Transformer | 2 |
| public CatBoost artifacts | 2 |
| public TabM artifacts | 2 |
| TabM | 2 |
| TabPFN-style artifacts | 2 |
| ExtraTrees / GSSpec / HGB / MLP / prototype / TabR | 1 each |

The strongest single member in the pool was already one of the public OOF stackers, with 0.970350 OOF, while the final full stack reached 0.970718. So the useful lift from stacking was real, but small. That was the main feel of the competition for me: lots of reasonable models, very tiny margins.

This was not my highest local OOF run. Later I had chains that scored a bit higher locally, but they were narrower and more patchy.

## Public Work

Public notebooks and shared artifacts were very useful. I used them mostly as probability surfaces or recipes to port locally, not as blind final answers.

The most important public contributors for my stack were:

- @kaiseitakahashi for the public OOF stacker that was the strongest single member in my pool
- @cdeotte for the GPU logistic stacker plus RealMLP/CatBoost/XGBoost artifacts and recipes
- @debatreyabiswas for the SAGA meta-stacker surface
- @yekenot, @donmarch14, @kirill0212, @syedsalmanashraf, @nawfeelrahman1124444, @pilkwang, @imrancoder786 and others for extra model/artifact ideas that went through the same local OOF checks

## Workflow

I used Codex heavily for this competition. Not in the sense of "ask AI for the winning solution", but as a coding/research loop that kept moving while I reviewed results.

Most of the work looked like this: find a public notebook or a local idea, port it into the repo, run proper OOF, register the prediction files, test whether it helped the stack, then either keep it or log why it was dead. That made it possible to try many more ideas than I would have by hand, while still forcing everything through the same local OOF filter.

For me the final decision came down to trusting the stable OOF stack instead of chasing the highest public-looking variant.