# March Machine Learning Mania 2026: 5nd Place Solution

Final Score: 0.1174602 (MSE/Brier Score) | Rank: 5rd / 3,485 teams

## Context

Business context: https://www.kaggle.com/competitions/march-machine-learning-mania-2026/overview

Data context: https://www.kaggle.com/competitions/march-machine-learning-mania-2026/data

## Philosophy

-   Using the Elo rating system as the core method, it purely relies on historical match data for Data drive prediction, eliminating the need for manual feature engineering.

-   The Male's and Female's championships are modeled independently, using different hyperparameters (K-factor, weighting strategy).

-   Post-process the champion to strive for a lower Brier Score

## Approach

### Data Preparation

| File                               | Purpose                          |
| :--------------------------------- | :------------------------------- |
| `MRegularSeasonCompactResults.csv` | Male's Regular Season Results    |
| `MNCAATourneyCompactResults.csv`   | Male's NCAA Tournament Results   |
| `MTeams.csv`                       | Male's Team Information          |
| `WRegularSeasonCompactResults.csv` | Female's Regular Season Results  |
| `WNCAATourneyCompactResults.csv`   | Female's NCAA Tournament Results |
| `WTeams.csv`                       | Female's Team Information        |
| `SampleSubmissionStage2.csv`       | Submission Template              |

#### Competition type marking and weight allocation

Male's and Female's events are respectively labeled as regular season (tourney=0) and tournament (tourney=1), and assigned different weights:

| Gender | Regular season weight | Tournament Weight |
| :----- | :-------------------- | :---------------- |
| Male   | 1.00                  | 0.75              |
| Female | 0.95                  | 1.00              |

-   Male: Regular season weight > Tournament weight, placing more trust in regular season performance with large sample size 

-   Female: Tournament weight ≥ Regular season weight, reflecting that tournament performance has stronger indicative power for the strength of Female's teams 

After merging the regular season and the tournament, sort by `(Season, DayNum)`  time. 

## Elo Rating System

### Core algorithm (`calculate_elo`)

Update the scores for each game sequentially in chronological order:

1.  **Initial Rating**: Initial rating for all teams initial_rating = 1200
2.  **Win Rate Estimation**: Using Logistic Elo Formula

$$
P(\text{win}_A) = \frac{1}{1 + 10^{(R_B - R_A) / \text{width}}}
$$

​	**where** $\text{width} = 1200$

3. **core Update**：

$$
R_{\text{winner}}' = R_{\text{winner}} + w \cdot K \cdot \text{MoV} \cdot (1 - P(\text{win}_{\text{winner}}))
$$

$$
R_{\text{loser}}' = R_{\text{loser}} + w \cdot K \cdot \text{MoV} \cdot (0 - P(\text{win}_{\text{loser}}))
$$

- $K$ = K-factor（Control the magnitude of the impact of each game on the rating）
- $w$ = Competition weight (different for regular season/tournament)
- $\text{MoV}$ = Margin of Victory Multiplier (in this scenario alpha=None , i.e.,  $\text{MoV}=1$, point differential adjustment not enabled) 

4. Score Lower Limit: Default Unlimited (`lowerlim = -inf`)
5. Loss function : Record Brier Score components for each game  $(1 - P(\text{win}_{\text{winner}}))^2$

###  Hyperparameter configuration

| parameter                           | Male | Female |
| :---------------------------------- | :--- | :----- |
| initial_rating                      | 1200 | 1200   |
| K-factor                            | 125  | 190    |
| width                               | 1200 | 1200   |
| alpha (Score difference adjustment) | None | None   |

The K-factor for Female is significantly higher than that for Male (190 vs 125), making the Female's team scores more sensitive to single-game results and reflecting the characteristic of a faster-changing competitive landscape in Female's events.

### Season Statistics Summary(`create_elo_data`)

Calculate each team's Elo statistical indicator per season using only regular season data:

| Indicator     | Instructions                                         |
| :------------ | :--------------------------------------------------- |
| Rating_Mean   | Season Elo Mean                                      |
| Rating_Median | Season Elo Median                                    |
| Rating_Std    | Season Elo Standard Deviation (Stability Indicator)  |
| Rating_Min    | Season Low Elo                                       |
| Rating_Max    | Season High Elo                                      |
| Rating_Last   | Elo after the last regular-season game of the season |
| Rating_Trend  | Elo linear regression slope (intra-season trend)     |

**Model Evaluation**: Calculate the average Brier Score only on tournament matches.

## Prediction

### **Submit prediction generation**

Using each team's  `Rating_Last`(final Elo rating at the end of the season) from the 2026 season to generate win probability predictions:

1. Parse the ID column in  `SampleSubmissionStage2.csv` , extract `Season`、`T1_TeamID`、`T2_TeamID`
2. Merge the Elo data of the 2026 season for both genders, and construct `TeamID → Rating_Last`  lookup dictionary 
3. Apply the Elo Logistic formula:

$$
P(T1\ \text{wins}) = \frac{1}{1 + 10^{(R_{T2} - R_{T1}) / 1200}}
$$

## Post-Processing

Post-processing modifications are made to the predicted champion teams UCLA and Michigan. This is a high-risk, high-reward strategy that significantly reduces the Brier Score when the champion prediction is correct, but incurs the maximum penalty in all relevant matches when the prediction is incorrect.

## Pipeline Summary

```
Original game data (Male/Female × Regular Season/Tournament)
    ↓
Mark the competition type (tourney) & Allocate weights (weight)
    ↓
Merge & Sort by (Season, DayNum) time
    ↓
Elo update per game (K=125 for male / K=190 for female, calculated independently)
    ↓
Extract Rating_Last from the 2026 season → Logistic formula generates T1 win rate
    ↓
Post-processing: Champion Post-processing
    ↓
Output submission.csv
```

## What Could Be Improved

-   **No Feature Engineering** : Only Elo ratings are used, without utilizing box score data (efficiency rating, rebound rate, assist-to-turnover ratio, etc.) ​

-   **No seed information** : The tournament seed difference was not used as a feature, although historically the seed difference has been highly correlated with knockout results 

-   **No external data fusion** : No integration of effective signals such as betting odds, injury information, home and away status, etc. 

-   **Statistical aggregation redundancy**: Seven seasons' statistical indicators were calculated, but only  `Rating_Last`

## Sources

- https://www.kaggle.com/code/lennarthaupts/calculate-elo-ratings/notebook
- https://wncviz.com/NCAABrackets/KaggleBrackets.html
- https://sportsbook.fanduel.com/navigation/ncaaw?tab=national-champion
- https://sportsbook.draftkings.com/leagues/basketball/ncaab?category=team-futures&subcategory=champion