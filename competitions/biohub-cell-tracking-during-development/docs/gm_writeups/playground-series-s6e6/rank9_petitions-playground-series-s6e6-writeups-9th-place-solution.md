# 9th place solution

The final submission is a L2 regularized logistic regression stack sitting on top of 63 diverse base models (public LB 0.97121, CV 0.970355). All of the value came from having a wide, decorrelated pool of models underneath it.

## How it evolved

I started with GBDTs. LightGBM, XGBoost and CatBoost all landed around 0.964 out of the box and clustered really tightly. Feature engineering moved things a bit: colours (u-g, g-r and friends), magnitude summaries, redshift ratios, and target encoded categoricals. The biggest surprise was RealMLP (the "better by default" tabular net recipe), which quietly became my strongest single model at about 0.9693 and ended up a backbone of the stack.

After that it turned into a stacking game. Every time I added a genuinely different base model, the stack nudged up a little, and the 63 base version was the best of them.

The interesting and frustrating part is that everything caps around 0.970 for single models and 0.9704 for stacks. I confirmed that wall maybe ten different ways. The leftover errors are basically GALAXY vs STAR confusion right at redshift near zero, and with only ugriz photometry plus redshift there just is not enough signal to split them. Every strong model also ends up correlated above 0.70 because they all lean on the same redshift and target encoding signal, so "strong and decorrelated" is very hard here.

## What did not work

A few things I tried that did not work, in case it saves someone time:

- **Cleaning up "mislabeled" rows (confident learning / cleanlab).** The label noise is real and it sits right in the GALAXY vs STAR zone, but pruning or down-weighting those rows did nothing or slightly hurt. The data is synthetic, so train and test might share the same noise, and cleaning the training set just pulls you away from what you are scored on ? maybe
- **New feature representations to break the model correlation (flux magnitudes, luptitudes, RBF and Nystroem features).** These genuinely decorrelated the models, error correlation dropped to about 0.48 to 0.54, which was exciting, but each one was too weak on its own (around 0.94 BA) to help the stack. Strong and decorrelated was very tuff.
- **Just pushing the models harder.** Bigger RealMLP, more epochs, Optuna tuning, and a 100 member bag were all basically flat.
- **Pseudo-labeling the test set and generative pretraining.** Washed every time.
- **Recovering the original SDSS labels from the sky coordinates.** The generator reuses a chunk of the real coordinate values, so I hoped I could match rows back to the real catalog, but the coordinate to label matching did not hold up.

## On the agent

This whole run was driven by an autonomous but supervised agent system I have been building inside Claude Code. It runs propose, build and validate loops and pauses for me at the gates.

Calling it autonomous is a bit generous though. It needs a lot of steering, and it has a real habit of deciding it has hit the ceiling and quietly winding down. So a good chunk of my job was talking it back into the fight, pointing it at the next idea, and approving or rejecting proposals. Once it gets going again it is good, but it will not push through a plateau on its own. Still a fun way to compete.

**Additional notes**:

- ran 144 experiments in all, and the final submission is experiment number 91.
- Every finding and every proposal gets logged.
- hold a one small change per experiment policy (most of the time), so we can see what is actually moving the needle.

Thanks to everyone who shared notebooks and discussion, learned a lot. 

Edit: Thank you @cdeotte for the awesome starter notebook. Helped a lot.