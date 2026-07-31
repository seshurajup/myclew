# 2nd Place Solution - Recursive NN + GBDT Ensemble Model

Run TimeCongratulations to @lgreig and @hardyxu52 for your top 3 finishes. I especially enjoyed competing in the top ranks with @hardyxu52 for the second time.

### Recursive Prediction

After reaching a plateau at a score of 18, I realized that I couldn’t go any further and decided to rebuild the entire pipeline to make it recursive. Thanks to this approach, I had access to 13 past lag values and was able to use them as features.

### Target Transformation

The target variable follows a Tweedie distribution. I transformed it to make the problem more suitable for L2 loss by applying a square root operation.

$$\text{sales}_t = \sqrt{\text{sales}_t}$$

### Features as Estimators

My hypothesis was that there was a proportional relationship between the variables total_orders, sell_price, and sales.

$$\text{sell_price} = \text{sell_price_main} \times (1 - \max(\text{type_}_{0..6}\text{_discount}))$$
$$\hat{\text{sales}}\_{t,n}^{price} = \text{sales}\_{t-n} \times \frac{\text{sell_price}\_{t-n}}{\text{sell_price}\_{t}}$$

$$\hat{\text{sales}}\_{t,n}^{order} = \text{sales}\_{t-n} \times \frac{\text{total_orders}\_{t}}{\text{total_orders}\_{t-n}}$$

At the end I used average of last 14 price based sale estimation as a base value. Subtracted that from our target and tried to explain remaining variance using regression models.

$$\hat{sales}_t = \text{mean}(\hat{\text{sales}}\_{t,1..14}^{price})$$
$$\text{sales}_t = \text{sales}_t - \hat{\text{sales}}_t$$

### Feature Engineering

If I remember correctly, I have 117 features. These mostly consist of lags: 1, 2, 3, 4, 5, 6, 7, 14, 21, and 28.

### Imputing

I imputed numerical values with 0 and categorical values with "unknown."

### Model

I used RealMLP and LightGBM until the last month. Then, I added XGBoost, CatBoost, and TabM models. The benchmark is derived from the [public notebook](https://www.kaggle.com/code/greysky/rohlik-recursive-prediction?scriptVersionId=222710520):

| Model    | Public Score | Private Score | Run Time | GPU      |
|----------|-------------|--------------|---------|----------|
| CatBoost | 19.046      | 18.809       | 13m42s  | Enabled  |
| XGBoost  | 18.382      | 17.814       | 14m52s  | Enabled  |
| LightGBM | 18.254      | 17.836       | 19m39s  | Disabled |
| TabM     | 18.725      | 18.502       | 1h10m   | Enabled  |
| RealMLP  | 17.936      | 17.794       | 45m8s   | Enabled  |
| Ensemble | 17.643      | 17.385       | 1h52m   | Enabled  |

### Validation Strategy

I used the last two weeks as the validation set. Due to time and GPU constraints, I mostly conducted my experiments using only LightGBM.

### Salt Noise Strategy

Since I used model predictions as features, the predictive power of the variables was lower in the test set. In the training set, I randomly set 5% of these variables to null to reduce their importance during training. This approach increased the score by a small amount, but I stopped using it later because it was overengineering.

### Inference

My main notebook took 1 hour and 50 minutes to train and perform inference, achieving scores of 17.58/17.35 on the public and private leaderboards.

### Kudos

A big thanks to David Holzmüller and his team for this great repository:
[PyTabKit: Tabular ML models and benchmarking (NeurIPS 2024)](https://github.com/dholzmueller/pytabkit)