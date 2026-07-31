# 5th Place Solution

First of all, I would like to thank the competition hosts and Kaggle for organizing this interesting challenge. Below is my solution writeup.

***

# Solution Overview
## Keypoints
- predict the difference (delta x, delta y) from the previous frame
- encode which future frame the embedding is intended to predict using RoPE.
- huberloss with muon optimizer
- 2step training
  - 1st step: train using 2023 NFL competition data (all players) and this competition data (only player_to_predict = true players).
  - 2nd step: train using 2023 NFL competition data (all players) and this competition data (all players, pseudo labeling).
- EMA decay=0.995

## Model Architecture
I spent most of my time improving the model architecture.

![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F4705999%2Fe96560d257e87f2e5814bef7f2f5ee37%2F.png?generation=1767943433183304&alt=media)