# 5th place solution

Thanks to the host and kaggle for hosting such an interesting competition.
I would also like to thank all of the participants and teammates( @takashisomeya @nomorevotch @fuumin621 ) for a great time.

Our solution consists of two stages: NN and GBDT. We will show you how in detail.

## ■stage1 NN part overview
- tracking data and images as input(player-player distance < 2 and player-ground)  
- inference of sequential frames at once  
- CNN + LSTM  

## Input to NN
### [1]tracking data
Use the following tracking data.
- distance
- distance_1(player1)
- distance_2(player2)
- speed_1
- speed_2
- acceleration_1
- acceleration_2
- same_team(bool)
- different_team(bool)
- G_flag(bool)

If player is G, fill distance and XXXX_2 values with -1.
same_team and different_team are flags for whether the players are belong to the same/different team.
G_flag means the player-ground pair flag.

### [2]Images + Bbox
- Concat the following three in the channel direction
    - video frames of +-1 frame cropped around the helmet. 
    - helmet bbox mask
- Image size
    - player-player pair   ：crop size = max(average bbox width, average bbox height) * 3
    - player-ground pair ：crop size = max(bbox width, bbox height) * 3
    - Resize the cropped image to 128x128.

We used sequential frames containing at least one frame with a distance < 2.
(At this time the data may contain frames of distance > 2.)
- [1]：B x N x 10  
- [2]：B x N x 3 x 128 x 128  
(B:batch_size, N:Sequential frames (e,g. 16,32,48,64))  

![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F3584397%2F4e835e68beeb3243da667319ac771c14%2Fcnn_input.jpg?generation=1677942599176145&alt=media)

Sequential frames (N) are cut out with different strides during training and inference.   
training: No duplicate frames  (stride == N)  
inference: Duplicate frames(stride < N, Duplicate frame results are averaged.)  

## Augmentations during training
Use the following augmentations.
- HorizontalFlip
- RandomBrightnessContrast
- OneOf
    - MotionBlur
    - Blur
    - GaussianBlur  
- Ramdom frame dropout (40-60% for images and 20-60% for tracking data)

## NN Model
The overall NN model architecture is as follows  
- Endzone/sideline images go through a shared CNN backbone.  
- The CNN backbone uses the TSM module.  
　https://www.kaggle.com/competitions/nfl-impact-detection/discussion/209403  
- Concatenate features extracted by CNN with tracking features  
- BiLSTM layers + FC layer infer sequential frames at once  

![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F3584397%2Fcc3cb2c68d98704faeb57d48d11ecea4%2Fcnn_model.jpg?generation=1677942632702738&alt=media)

## ■stage2 GBDT part overview

The key feature in this model is the logit from stage1.
The goal is to further improve the score by combining logit with tracking data and other data to create a binary classification model.

## Data

- distance <= 2
- swap player1 and player2 features then concatenate them vertically to the original data.
- average swap and original  features for final prediction

## Features
### Raw value
- x_position, y_position, speed, distance, orientation, acceleration, direction, sa, jersey_number of each player
- distance between players
- frame number
- nn_pred

### Helmet
https://www.kaggle.com/code/ahmedelfazouan/nfl-player-contact-detection-helmet-track-ftrs

### Simple computational features

The following are calculated for x_position, y_position, speed, distance, orientation, acceleration, direction, sa
- Absolute difference between players, multiplied by
- Difference from the average of all players in each frame

### Aggregate features
For distance, nn_pred, sa, distance, speed
- Aggregate features for (game_play, position), (game_play, player), (game_play, team), (game_play, step)
- Aggregate features for each (game_play, player_1, player_2)
- shift, diff(-3~3) for each (game_play, player_1, player_2).

## model
- lgbm
- xgboost

## ■Ensemble 
### stage1 (NN part)
Created models on different backbones and different sequence lengths as follows
* backbone
  * resnet18,34,50
  * resnext50
  * efficientnet b0,b1
* sequence length
  * 16,32,48,64
### stage2 (GBDT part)
Two models were created with the same features
* LightGBM
* XGBoost

### Forward Selection
Created models for (almost) all combinations of the above, and use Forward Selection 
* Forward Selection was based on the excellent kernel by chris here.
       https://www.kaggle.com/code/cdeotte/forward-selection-oof-ensemble-0-942-private/notebook
* It is a simple method. so we expected to avoid overfit.
* The following models were finally selected by Forward Selection

| sequence length | backbone | gbdt | cv |
| --- | --- | --- | --- |
| 64 | resnext50 | xgb | 0.7918 |
| 64 | resnext50 | lgb | 0.7906 |
| 64 | effib0 | lgb | 0.79 |
| 32 | resnext50 | lgb | 0.7935 |
| 32 | effib0 | lgb | 0.7881 |
| 16 | resnext50 | xgb | 0.7906 |
* Final submit is CV:0.8016 ,LB : 0.7902, PB : 0.7913

## Threshold
We simply blend predictions of selected models (x5fold), and determined by a single threshold.
* We used two threshold. 
   * predictions themselves
   * percentile of the predictions
* We also tried voting ensemble , but decided not to use it because the LB score was better with a single threshold.

## Other tips
In the inference notebook, the following were introduced to avoid OOM and timeout.
  * using lru_cache for read image at high speed
  * PyTurboJPEG loads images faster than OpenCV
  * Polars helps reducing submission time.

## Acknowledgments
zzy's excellent kernel is very helpful in our pipeline.  
https://www.kaggle.com/code/zzy990106/nfl-2-5d-cnn-baseline-inference