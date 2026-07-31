# [Private LB 162nd] solution

Firstly, huge thanks to Kaggle and Jane Street for organizing such an interesting competition. Also, a very sincere thanks to the Kaggle community; the discussions here helped me a lot to understand mistakes and I learned a lot.

#### **1. Overview**

The solution is a hybrid model, combining an adaptive neural network with a static GBDT ensemble to address the non-stationary nature of the financial time-series data.

#### **2. Final Architecture: Weighted Ensemble**

The final prediction is a linear blend of two models:

*   **Model A (70% weight):** An online learning TabM neural network (my contribution).
*   **Model B (30% weight):** A static LightGBM ensemble (my teammate's contribution).

```python
# Final prediction logic
y_final = (0.7 * y_tabm_pred) + (0.3 * y_lgb_pred)
y_final = np.clip(y_final, -5, 5)
```

---

#### **3. Model A: Online Learning TabM (PyTorch)**

This model was designed for continuous adaptation to new market data.

*   **Architecture:**
    *   TabM-based MLP (3 hidden layers, 512 units, ReLU activation).
    *   Data I/O and processing were handled with the Polars library for performance.
*   **Online Learning Mechanism:**
    *   The model was continuously fine-tuned on a rolling window of the most recent 3-4 days of data.
    *   A retraining cycle was triggered daily after new data became available.
    *   Fine-tuning was performed for 10 epochs using an AdamW optimizer with a learning rate of `1e-5`.
    *   A smaller learning rate and mixing old data helped avoid catastrophic forgetting.

---

#### **4. Model B: Static LightGBM Ensemble (Teammate Contribution)**

This model, provided by my teammate, served as a stable baseline.

*   **Architecture:** An ensemble of multiple LightGBM models, with predictions averaged.
*   **Features:** Utilized the 79 base features plus 9 lagged `responder` values from the previous day.
*   **Purpose:** Trained once on a large historical dataset to capture long-term, stable patterns in the data.

---

#### **5. Validation Strategy**

*   During training, I was not able to find a good LB/CV correlation. I kept the last 100 days for validation, then trained on the whole dataset again for submission.
*   A local simulation of the Kaggle API was used to ensure robustness and avoid timeouts, based on this [reference notebook](https://www.kaggle.com/code/chumajin/janestreet-updated-simulator-for-time-series-api).
    *   This allowed for end-to-end local testing of the entire inference and retraining pipeline. It was critical for debugging and validating the online learning strategy before submission.

*This is my first write-up, please correct me if I wrote something wrong.*