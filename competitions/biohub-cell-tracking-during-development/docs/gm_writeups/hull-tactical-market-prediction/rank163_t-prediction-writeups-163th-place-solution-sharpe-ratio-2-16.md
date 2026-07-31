# 163th Place Solution: Sharpe ratio 2.16

Thanks to Hull Tactical and Kaggle for this competition. It was our first real step into quantitative finance.

During the competition, we learned several technical details that are easy to miss. One important point was how to use the API’s batch processing correctly. If this is handled incorrectly, positions based on excess return quantiles can become invalid. Another key point is that the test set is part of the training data, so a fair evaluation should exclude the last 180 days of training data. We also found that using backfill (bfill) to impute missing values causes data leakage and can reduce the score by about 0.6. Because the early historical data contains many missing values, we chose to drop those records from training instead. For position sizing, we used tanh to map excess returns to positions, which worked better than sigmoid in this setting. 

After fixing these data handling issues, our score reached about 2.19. More importantly, the Sharpe ratio proved highly robust, holding steady at 2.19 on the training data and 2.16 six months later. However, the model still showed a clear correlation with the overall S&P 500 trend: it performed better when the index was rising, and the ranking tended to fall when the market declined.

When retraining the model with the corrected data, we did not use TimeSeriesSplit for cross-validation. In hindsight, that was a limitation, and proper time-series validation would be a useful next step.

```

def convert_ret_to_signal(x: float, bound: float = 0.006, scale: float = 3.0) -> np.ndarray:
    """
    Convert an array of returns to discrete trading signals 0, 1, 2
    """
    # Step 1: 截断到 (-bound, bound)
    x_clipped = np.clip(x, -bound, bound)
    
    # Step 2: 归一化到 [-1, 1]
    x_norm = x_clipped / bound  # 现在 x_norm ∈ [-1, 1]
    
    # Step 3: 放大到 [-scale, scale] 以激活 tanh 的非线性
    x_scaled = x_norm * scale   # ∈ [-scale, scale]
    
    # Step 4: tanh 映射到 (-1, 1)，再平移到 (0, 2)
    y = np.tanh(x_scaled) + 1.0  # ∈ (0, 2)

    return y
```