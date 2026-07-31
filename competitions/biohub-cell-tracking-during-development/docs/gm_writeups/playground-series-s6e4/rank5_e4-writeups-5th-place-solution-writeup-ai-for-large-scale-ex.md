# 5th Place - AI for large scale experimentation - s6e4

This was my second time ever competing in Kaggle Playground Series so I am happy to finnish in the top 5. Learning from s6e3 (13th place), This time I was aiming for diversity and efficiency in experimentation. The goal was to have experiments running for 24/7, with a steady flow of new models. Therefore I focused on structuring my codebase and optimising my workflow to maximize the use of AI (codex, claude code). Before the competition, I created a simple tool to use AI to analyse my experiments. This tool allowed me to push tasks to a queue and have codex check on them ones they were done; Either to fix failed training or write a summary of the results and propose a follow-up experiment. This allowed me to have a steady flow of new and diverse models.

However, this also created new challenges. One of the hardest parts was filtering the models and deciding which ones to include in the final ensemble. This was especially difficult because the balanced accuracy metric was noisy, and local CV did not always match the public leaderboard.

I also learned a lot from the differences between OpenAI and Anthropic models. In my experience, GPT models often struggled to generate genuinely new ideas. They tended to suggest similar approaches repeatedly, and because the metric was noisy, they often failed to recognize when a line of experimentation was no longer promising. Claude was much more creative, but its lower rate limits made it harder to use for continuous experimentation.

Most of the neural network models below were produced by running Codex in an iterative loop. I gave Codex the goal of evaluating new models by comparing their errors against those of the current best ensemble, with the objective of maximizing complementary errors while maintaining a reasonably strong CV score.

Keeping track of incremental improvements became difficult, but I think this workflow was genuinely useful. It likely helped move the solution from around the top 15 to the top 5. That said, the majority of the progress still came from manual iteration, model selection, and judgment. Codex was especially helpful for fixing issues and automatically restarting training runs, which made the experimentation process much smoother.

## Final Ensemble Summary

My final submission was a stacked ensemble built from 119 base models. Each base model produced class probabilities for the three target classes, and these probabilities were used as features for a multiclass LightGBM stacker. The selected ensemble had a cross-validation score of approximately `0.98178` and achieved my best private leaderboard score, `0.98133`.

The strongest individual models were mostly boosted-tree models, especially XGBoost, LightGBM, and CatBoost variants. The best-performing feature sets combined target-encoded categorical variables, features derived from numeric values after rounding or digit extraction, distribution-matching signals from the original data, and features based on the rule discovered from the original dataset.

Beyond the top individual models, my goal was to aim for diversity. The final ensemble included many different model families and feature views: boosted trees, neural tabular models, grouped CNN-style models, graph-based models, random forests, AutoML models, and several smaller specialist models. The goal was not only to add high-CV models, but also to include models that made different mistakes.

The main feature-engineering themes were:

- Target encoding variants, where categorical values or pairs of values were replaced with smoothed target statistics computed in a leakage-safe way.
- Features extracted from rounded numeric values, decimal patterns, and digit-level representations of the tabular inputs.
- Features based on the discovered original-data rule, including rule scores, class probabilities derived from those scores, confidence measures, and rule-by-feature interactions.
- Pair-interaction features that captured relationships between two original variables at a time.
- Features marking rows close to important rule thresholds, such as values near soil, rainfall, temperature, or wind cutoffs.
- Features from a denoising autoencoder trained to learn compact representations of the tabular inputs.
- A decomposed classification strategy in one final base model: first separating the middle class from the two outer classes, then separating the two outer classes from each other.

The original-rule-based features came from the exact rule discovered in the original dataset. The rule used threshold conditions on soil moisture, rainfall, temperature, and wind speed, together with crop growth stage and mulch usage. I used the binary rule flags, derived rule scores, class probabilities computed from those scores, confidence measures, and interactions with stage, mulch, and boundary indicators as features for several base models.

### Model Families In Final Ensemble

| Model family | Count | CV score range |
|---|---:|---:|
| XGBoost | 25 | `0.9179-0.9804` |
| CatBoost | 19 | `0.9665-0.9796` |
| Custom tabular neural network variants | 16 | `0.9661-0.9754` |
| 1D CNNs over grouped tabular features | 14 | `0.9638-0.9780` |
| RealMLP tabular neural network | 11 | `0.9683-0.9798` |
| TabM tabular neural network | 9 | `0.9683-0.9798` |
| LightGBM | 6 | `0.9618-0.9803` |
| TabTransformer tabular neural network | 4 | `0.9724-0.9762` |
| Graph neural network | 4 | `0.9706-0.9718` |
| Random forest / GPU random forest | 3 | `0.9563-0.9686` |
| Histogram gradient boosting | 2 | `0.9786-0.9796` |
| AutoML | 2 | `0.9669-0.9776` |
| FT-Transformer tabular neural network | 2 | `0.9674-0.9776` |
| Public/external ensemble | 1 | `0.9798` |
| k-nearest-neighbor model | 1 | `0.8320` |

The 1D CNN models arranged engineered tabular features into a meaningful sequence, grouping related features together, then used convolutional layers to learn local patterns across nearby feature groups. Some variants used raw standardized scalar inputs, while the stronger hybrid variants converted numeric and categorical features into learned tokens before applying convolutional layers.

The custom tabular neural networks included several related feed-forward architectures. They used learned embeddings for categorical features, normalized hidden layers, GELU or SELU activations, dropout, label smoothing, and class weighting. Some variants added residual connections, input gates, explicit pair-interaction features, random Fourier features, variable-selection layers, or DeepFM-style components that combine linear terms, learned pairwise interactions, and dense neural layers. These models were mainly included for diversity, since they used the same engineered feature families differently from boosted trees.

### How The Ensemble Was Produced

The ensemble was produced by collecting out-of-fold and test-set probability predictions from all available base models. Each model was first scored by balanced accuracy on out-of-fold predictions. Highly redundant models were then pruned by comparing their predicted labels against already-selected models: if a candidate model agreed too closely with a stronger selected model, it was removed. The final submitted version used an agreement threshold of about `0.9964`, leaving 119 base models.

After pruning, a LightGBM multiclass stacker was trained on the base-model probabilities. I used seed averaging and cross-validation to reduce variance (5 different seeds).

I ended up using this threshold-sweep stacking approach because the public leaderboard and local CV were both extremely noisy. It was difficult to know whether adding a new base model truly improved the ensemble, even when averaging multiple seeds. I tried several ways to evaluate whether new base models helped, but both CV and leaderboard scores varied enough to make single-run comparisons unreliable.

The pruning-threshold sweep gave me a more systematic way to compare nearby ensemble variants. One of the sweep runs produced one of the highest CV scores across my experiments, even though its public leaderboard score was only `0.98079`. I decided to trust the stronger CV signal, and that choice paid off: this became my best private leaderboard submission with a private score of `0.98133`.

Thanks for all the public contributions including:
- [Original Data Exact Formula](https://www.kaggle.com/code/cdeotte/original-data-exact-formula)
- [PSS6E4 xgb CV:0.979805](https://www.kaggle.com/code/yunsuxiaozi/pss6e4-xgb-cv-0-979805)
- [PSS6E4 lgb baselineCV:0.97943](https://www.kaggle.com/code/yunsuxiaozi/pss6e4-lgb-baselinecv-0-97943)
- [PS6E4 - Tab Transformer - Claude Vibe Coding](https://www.kaggle.com/code/include4eto/ps6e4-tab-transformer-claude-vibe-coding)
- [PG S6E4 - RealMLP - [CV 0.97802 LB 0.97685]](https://www.kaggle.com/code/mahoganybuttstrings/pg-s6e4-realmlp-cv-0-97802-lb-0-97685)
- [Predicting Irrigation Need](https://www.kaggle.com/competitions/playground-series-s6e4/discussion/694877)
- Finally, big thanks to @cdeotte . I learned a lot from your public posts from s6e3. P.S. I hope to get hired by nvidia someday :)