# 1st Place Solution

I would like to express my gratitude to the organizers for hosting such an exciting competition.
Now, all I can do is hope that the submission passes the final tests.

As I mentioned in another discussion, my model is trained only on the data provided in the “NFL Big Data Bowl 2026 – Prediction” competition.
Achieving strong performance with such a limited dataset required powerful data augmentation, well-designed loss functions, and an effective model architecture.

Train and inference notebook are here.  
https://www.kaggle.com/code/chack3/nfl2026-1st-place-inference  
https://www.kaggle.com/code/chack3/nfl2026-1st-place-train  

---

# Solution Overview

## 1. Feature Engineering

### 1.1 Dynamic Features

* Use the **20 frames** immediately before the pass.
* Extract the following **10-dimensional** features for each frame:

  * `x, y`
  * `sin(o), cos(o)`
  * `sin(dir) * s, cos(dir) * s`
  * `x - ball_land_x, y - ball_land_y`
  * `x - receiver_x, y - receiver_y`

### 1.2 Static Features

* Total **12 dimensions**:

  * One-hot encoding of `player_role` (4 dims)
  * Number of prediction frames (1)
  * The number of frames before the pass used as input (1)
  * Passer’s final-frame coordinates (2)
  * Ball landing coordinates (2)
  * Player’s final-frame coordinates (2)

### 1.3 Target

* Predict the **displacement (Δx, Δy)** from the final input frame.
* The final trajectory is obtained by adding the predicted displacement to the final input coordinates.

---

## 2. Model Architecture
### 2.1 Overall Structure

* A **play-level model** handling up to 22 players.
* Player dimension is frequently reshaped/transposed into the batch axis.

![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F1630583%2Fad0c358c11eb57e4e48bfe8aae31c35e%2Fmodel.png?generation=1764969335701434&alt=media)
### 2.2 Dynamic Feature Encoder

* **Input:** `(10, player, 20 frames)`
* **Output:** `(640, player, 1 frame)`
* Depthwise Conv1d (kernel size = 3) × 7

  * Each feature → 64 dims → concatenated into 640 dims
* No padding (to emphasize the last frame)
* 20 frames → 5 frames → only the **final frame** is used

### 2.3 Static Feature Encoder

* **Input:** `(10, player)`
* **Output:** `(64, player)`
* Conv1d(kernel=1) → BatchNorm → SiLU

### 2.4 Merge Layer

* Concatenate dynamic features (640) + static features (64)
* Project to **256-dimensional** player features

### 2.5 Inter-player Interaction Layer

* **Input:** `(256, player)`
* **Output:** `(256, player)`
* Transformer Encoder × 3 layers
* Player masks for variable player counts
* FFN uses **SwiGLU**

### 2.6 Decoder

* **Input:** `(256, player)`
* **Output:** `(2, player, 48 frames)` + 4 auxiliary outputs
* 256 → 1536 → reshape → `(32, 48 frames)`
* Apply multiple Conv1d (kernel size = 3)
* Final projection: 32 → 2 (xy) + 4 (auxiliary terms)

---

## 3. Training Setup

### 3.1 Cross Validation

* Group **5-fold CV** using `game_id`
* Perform 5-fold CV **three times** with different splits

### 3.2 Optimization

* **RAdam** optimizer
* **EMA** with decay = 0.9995
* 210 epochs, evaluated every 6 epochs

---

## 4. Loss Functions

### 4.1 Main Loss: GaussianNLLLoss

* Predicts both mean and variance
* Automatically downweights samples with large variance → robust learning
* Performed better than SmoothL1
* Frame-wise weighting provided no meaningful improvements
* Variance is constrained to positive values using:
  **softplus(var) + 1e-3**

### 4.2 Auxiliary Loss

* Computed from predicted xy:

  * Velocity (1st difference)
  * Acceleration (2nd difference)
* Also uses GaussianNLLLoss
* Helps stabilize and accelerate learning

---

## 5. Data Augmentation

### 5.1 Rotation Augmentation

* 50% chance to randomly rotate the entire play around the mean player position
* Rotation angle sampled uniformly from **0–360°**
* Initially used a Gaussian distribution (σ = 5°),
  but increasing variance improved performance → switched to uniform
* Likely beneficial because the target depends on **rotation-invariant relative positions**

### 5.2 Predicting From Earlier Frames

* Important for long-pass predictions
* Instead of starting prediction at pass-attempt frame,
  use up to **20 frames earlier**
* The model thus predicts:

  * (earlier frames) + (original prediction frames)
* Improves long trajectories significantly
* Add a static feature indicating “how many frames earlier”

  * (always 0 during test)

The upper diagram shows an example where the data consists of 30 input frames and 9 output frames.
Blue indicates the portion of the input that is fed into the model, red indicates the data the model is asked to predict, and gray indicates data that is not used.

When we want the model to start predicting from 5 frames before the pass, the configuration becomes as shown in the lower diagram.
With this adjustment, the output becomes a 14-frame prediction.
This procedure is applied to all players within a play using the same set of parameters.

<p align="center">
  <img src="https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F1630583%2F9a7df2828bba3ad2ebe6bb0fa65f857b%2Fframe_aug.png?generation=1764890503442923&alt=media"
       width="1280">
</p>

### 5.3 Vertical Flip

* Flip the play along the X-axis

---

## 6. Data Processing

### 6.1 Outlier Removal

Remove plays with abnormal length or missing passer:

```
too_long_ids  = ["2023091100_3167", "2023122100_1450"]
no_passer_ids = ["2023091001_3216", "2023112606_4180", "2023121009_3594"]
```

### 6.2 Other Processing

* If `play_direction == left`, rotate the play by **180 degrees**

---

## 7. Ensemble

* Use an ensemble of **100+ models**
* All models share the same architecture
* Diversity arises from:

  * Different feature configurations
  * Different CV splits
* Ensemble method: **simple average**

---

## 8. Ablation Study
Changes in scores across epochs for each of the 5 folds.

Looking at the results side by side, it’s interesting to see that rotation augmentation and early-frame prediction help reduce overfitting, the auxiliary loss stabilizes training, and GaussianNLLLoss contributes to improving accuracy.

![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F1630583%2Fe2a712f5d6e31dd8ab0d05c703463542%2Fabu.png?generation=1764846156985608&alt=media)

---