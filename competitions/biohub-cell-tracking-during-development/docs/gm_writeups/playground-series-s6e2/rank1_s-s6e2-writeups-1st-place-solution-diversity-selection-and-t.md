# 1st Place Solution — Diversity, Selection, and Trusting the CV–LB Relation

![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F11017380%2Ff25a388a818dd5c954511a92550fcbea%2FS6E2Solution%20Diagram.drawio%20(1).png?generation=1772326924535398&alt=media)

Wow — I’m genuinely so happy to be writing 1st place solution!!

I had been competing in this series for well over a year, so finally finishing **1st place** means a lot to me.

I’m a little busy at the moment, but I still wanted to get the main ideas written down while everything was fresh, so for now I used an LLM to help organize the solution into a readable write-up. I’ll probably come back and revise / expand some parts later.

Huge thanks to @cdeotte, @tilii, @mahoganybuttstrings, @optimistix— and definitely **not limited to them**. I learned so much from the discussions, comments, shared ideas, and from this community as a whole throughout the competition.

## Final Result

**Final Submission:**
- CV: **0.9557801**
- Public LB: **0.95396**
- Private LB: **0.95535**

**Best Submission I obtained:**
- CV: **0.9557901**
- Public LB: **0.955394**
- Private LB: **0.95536**

**Best CV I obtained during the competition:**
- CV: **0.955865**
- Public LB: **0.955393**
- Private LB: **0.95534**

However, I did **not** choose that highest-CV submission as my final submission, because I suspected **split overfitting**.

This point ended up being one of the most important lessons of the competition, and it is also why I think the better title for this write-up is not *“Trust your CV”*, but rather:

> **Trust the CV–LB relation.**

## 1. Overall Strategy

In this competition, I did not aim to build one overwhelmingly dominant model.

Instead, my overall strategy was:

> Create many slightly different models → select effective combinations → combine them with a simple linear model.

More concretely:

1. Generate multiple feature representations.  
2. Train many different model types.  
3. Collect approximately 150 OOF predictions.  
4. Use Optuna to search for good subsets.  
5. Combine selected OOFs with Ridge regression.  

The central idea was **diversity**.

I was not looking for one “magic” model. Rather, I wanted many reasonably strong models that made **slightly different mistakes**, and then combine them in a simple and robust way.

## 2. Feature Engineering

Rather than searching for one optimal feature set, I created multiple representations of the same base features. Each transformation acted as a component that could later be combined flexibly.

### 2.1 Binning

I discretized numerical features using:

- Quantile-based binning (`qcut`)
- Equal-width binning (`cut`)
- Simple rounding (for example, Age/5, BP/10)

Tree-based models sometimes benefit from grouped values. Different binning schemes produce slightly different split structures, which helps increase diversity.

### 2.2 Digit Features

For numerical variables, I extracted:

- Integer digit positions (units, tens, hundreds, etc.)
- The first few decimal digits when applicable

This may look unusual, but digit-based features sometimes capture hidden structure in tabular competitions.

### 2.3 Treating All Features as Categorical

I also converted all base features into string format and treated them as categorical variables.

### 2.4 Frequency Encoding

For each feature, I added its frequency (how often each value appears).

Rare values can carry useful signal, and frequency encoding often complements target encoding well in boosting models.

### 2.5 Genetic Programming Features

Using `gplearn`, I generated nonlinear interaction features automatically.

The goal was not necessarily to create individually dominant features, but rather to introduce alternative representations that improve ensemble diversity.

### 2.6 Extracting Signals from the Original Dataset

Because this competition dataset was generated from an original dataset, I extracted additional statistics from the original data, such as:

- Target mean
- Smoothed target mean
- Weight of Evidence (WoE)
- Entropy

This can be viewed as a kind of external target encoding, and in synthetic Playground competitions this type of information is often useful.

### 2.7 Denoising Variational Autoencoder (DVAE)

I trained a Denoising Variational Autoencoder on the base features.

A DVAE adds noise to the inputs and learns to reconstruct the original data. The encoder then provides compressed latent representations.

The main purpose here was not necessarily to improve single-model CV, but to create additional nonlinear representations that increase OOF diversity.

## 3. Model Training

Using the feature sets above, I trained a wide range of models:

- XGBoost
- LightGBM
- CatBoost
- RealMLP
- Regularized Greedy Forest (RGF)
- TabICL
- AutoGluon

All OOF predictions were generated using **5-fold StratifiedKFold with shuffle=True and random_state=42**.

Performance was evaluated using **overall OOF AUC**.

## 4. Representative Single-Model Performance

Below are representative OOF AUC scores for models that were selected at least five times during the Optuna search.

All scores were computed using 5-fold StratifiedKFold and evaluated using overall OOF AUC.

| Feature Set | Model | CV |
| --- | --- | --- |
| BASE+BIN+DIGIT+ALL_CATS | AutoGluon | 0.955747 |
| BASE+BIN+DIGIT+ALL_CATS | RealMLP | 0.955739 |
| ORIG+TE+EMB* | RealMLP | 0.955726 |
| BASE+GP_FEAT+ALL_CATS | RealMLP | 0.955720 |
| BASE+BIN+DIGIT+ALL_CATS | CatBoost | 0.955686 |
| BASE+TE | XGBoost | 0.955663 |
| BASE+BIN+DIGIT+ALL_CATS+FREQ | LGBM | 0.955652 |
| BASE+ALL_CATS | XGBoost | 0.955619 |
| ORIG+ALL_CATS | XGBoost | 0.955599 |
| BASE+ORIG+ALL_CATS | XGBoost | 0.955597 |
| BASE | XGBoost | 0.955575 |
| DVAE+ALL_CATS | XGBoost | 0.955426 |
| BASE+BIN+DIGIT+ALL_CATS | RGF | 0.954980 |
| BASE (subsample 100k × 5) | TabICL | 0.954971 |

\* ORIG+TE+EMB: all features target-encoded and combined with internal embeddings in RealMLP.

![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F11017380%2F8c16758acdde67ab8c2e947dc575748d%2Forigfeat.drawio.png?generation=1772326776510376&alt=media)

The differences between individual models were small.

However, some models with slightly lower standalone CV — such as **RGF** and **TabICL** — were still selected frequently during ensemble search.

This was a good reminder that **ensemble contribution is not determined solely by individual CV performance**.

## 5. OOF Selection with Optuna and Ridge Ensemble

In total, I generated approximately **150 OOF predictions**.

Simply averaging all of them hurt performance, so I used **Optuna** to search for effective subsets:

- 2500 trials
- Each trial selects a subset of OOF predictions
- Objective: maximize overall OOF AUC

Only about one-tenth of the OOFs were consistently selected.

For the final combination, I used **Ridge regression**.

Ridge worked best because:

- It is simple and stable
- It handles correlated predictions well
- It reduces overfitting compared with more flexible meta-models

In my experiments, nonlinear meta-models tended to overfit, while Ridge produced more consistent improvements.

## 6. Training on Full Data

For GBDT and RGF models:

- I retrained on the full dataset after generating OOF
- I used **20 different random seeds**
- I averaged predictions across seeds
- I set `n_estimators` to **1.25 ×** the average best iteration obtained during CV

In my experiments, this full-data retraining strategy worked better than simply averaging fold models.

## 7. Trust the CV–LB Relation, Not Just the Best CV

This was probably the most important practical lesson from this competition.

At one point, I obtained a submission with **CV = 0.955865**, which was my highest CV. However, I did **not** trust it enough to make it my final submission.

Why?

Because I started to suspect **split overfitting**.

I ran many hill-climbing experiments and also compared multiple actual submissions to study the relation between CV and leaderboard performance. In particular, I submitted ensembles stopped at different stages and compared their **CV–LB behavior**.

What I observed was roughly this:

- Up to around **CV 0.95578**, the CV–LB relation looked reasonably consistent
- But once the CV started going beyond **0.95578+**, the relation clearly became worse
- Improvements in CV no longer translated into similarly reliable LB improvements
- In some cases, higher CV looked more like exploitation of fold-specific behavior than genuine generalization

Because of this, I decided to select my final submission from the range around:

> **CV 0.95578 to 0.95580**

rather than blindly choosing the numerically best CV.

In hindsight, I think this was the correct decision.

## 8. What Did Not Work

The following methods did not improve CV in my experiments:

- Pseudo labeling (both soft and hard)
- Knowledge distillation
- Very deep GBDT models
- High-order interaction expansion
- Autoencoders other than DVAE
- Nonlinear stacking
- Averaging too many OOFs without selection
- Public leaderboard climbing

This competition reinforced a simple lesson:

> More complexity does not necessarily produce better generalization.

## 9. Key Takeaways

For beginners and intermediate competitors, I think the main lessons are:

- You do not always need one dominant “magic” model
- Small differences in feature representation can create useful diversity
- Models with slightly lower CV can still be valuable if they add complementary signals
- Selection can be just as important as generation
- Simple ensembles are often more robust than flexible ones
- A strong CV number alone is not enough — what matters is whether the **CV–LB relation remains trustworthy**
- Careful validation design is essential when tuning, stacking, and searching many combinations

The most important parts of my solution were:

- stable cross-validation
- controlled diversity
- aggressive but disciplined selection
- simple and robust ensembling
- cautious final submission choice based on CV–LB consistency

## 10. Cross-Validation Strategy and Leakage Considerations

Finally, I want to discuss CV design a bit more, because it was central to this solution.

### 10.1 Matching CV Splits Between OOF and Ensemble

When generating OOF predictions and training an ensemble model, the CV splits should be aligned.

If the fold splits used to:

- generate OOF predictions for base models, and
- train the ensemble model

are different, leakage can occur.

Why?

Because OOF predictions are only truly out-of-fold **with respect to the split used to generate them**. If the ensemble is evaluated using a different split, then the meta-model may end up training on samples whose base predictions were indirectly influenced by those same samples.

That can produce overly optimistic CV.

### 10.2 Early Stopping and Subtle Leakage

Even when the fold splits are the same, there is still a subtle issue if early stopping is used.

For example:

- A base model uses fold A as validation
- Best iteration is chosen based on fold A
- The OOF prediction for fold A is produced using that selected iteration

Strictly speaking, fold A has now influenced model selection.

So while this is standard practice and usually acceptable in Kaggle, it is not perfectly leakage-free from a theoretical perspective.

This effect is often small, but it is worth keeping in mind, especially when stacking many models and squeezing out tiny CV gains.

### 10.3 Nested K-Fold as the Strict Solution

The theoretically correct approach is **Nested K-Fold**:

- Outer folds: generate truly out-of-fold predictions
- Inner folds: tune hyperparameters / determine early stopping

Procedure:

1. Split data into outer folds
2. For each outer fold:
   - run inner CV on the training portion
   - determine hyperparameters / best iteration
   - retrain on the full outer-training data
   - predict on the outer-validation fold
3. Combine outer-fold predictions into OOF

This ensures that no sample is used, directly or indirectly, to tune the model that predicts it.

However, the computational cost becomes extremely large, especially when managing around **150 OOFs**.

### 10.4 Practical Trade-Off in Kaggle

In practice, especially in Kaggle:

- fully nested CV is often too expensive
- early stopping within standard K-Fold is widely accepted
- leakage risk is controlled by limiting model flexibility and avoiding over-searching

Realistic safeguards include:

- using fixed CV splits
- avoiding overly flexible meta-models
- limiting ensemble size
- checking CV–LB consistency carefully
- avoiding blind public LB chasing

In many Playground competitions:

- train and test distributions are very similar
- training data is larger than test data

Under those conditions, a well-designed validation scheme is often more reliable than public LB movement alone — but only if the **CV–LB relation remains healthy**.

### 10.5 Practical Takeaway

Strictly speaking, Nested K-Fold is the only fully leakage-free solution when:

- hyperparameter tuning
- early stopping
- model selection
- ensemble selection

are all involved.

But in real competitions, the practical goal is:

> **Minimize leakage risk and overfitting while preserving a stable CV–LB relation.**

In my case, I prioritized:

- fixed StratifiedKFold splits
- controlled ensemble size
- simple Ridge stacking
- monitoring CV–LB behavior through actual submissions
- not blindly trusting the highest CV

That balance between theory and practicality was, I believe, one of the keys to finishing in **1st place**.

## Closing

This competition was a great reminder that ensembling is not just about adding more models.

It is about:

- generating meaningful diversity
- selecting carefully
- keeping validation honest
- and knowing when **not** to trust a tiny CV gain

In the end, my best raw CV was **not** my best decision.

The winning decision came from trusting the **CV–LB relation**, not the leaderboard, and not the single highest CV in isolation.