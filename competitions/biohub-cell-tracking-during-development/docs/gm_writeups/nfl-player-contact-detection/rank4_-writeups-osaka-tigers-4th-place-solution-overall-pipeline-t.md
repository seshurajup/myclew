# 4th place solution Overall pipeline & tabular part - Osaka Tigers

We really appreciated the hosts and the kaggle team for organizing the competition. Moreover, we would also like to thank all the participants who joined. We could enjoy this competition and write up our solutions. 

I would like to thank team members, @bamps53, @nyanpn and @kmat2019, who have the top talent to analyze the task. I could discuss and enjoy the competition. 

# Overview
Simple solution outline is attached pic.
[![pipeline.png](https://i.postimg.cc/pLQ1qbCW/pipeline.png)](https://postimg.cc/VJ6Rkh2p)

In the 1st stage we predict the contact by multiple CNN. In the 2nd stage CNN prediction(s), tracking and helmet data are aggregated and created features to input GDBT.  Lastly we compute 5 models averaged value and optimize threshold for both player-player and player-ground contact.
 

# 1st stage CNN
## k mat model
Details are written in https://www.kaggle.com/competitions/nfl-player-contact-detection/discussion/391719.
We can obtain both Endzone and Sideline prediction values. 

## camaro model
will come up soon

# 2nd stage aggregation & binary classification models

We excluded player-player pairs with distance > 3, and the remaining ~880K rows were used to train 2nd stage models. During inference time, we assigned 0 to pair with distance > 3 and predicted only the remaining data.

## Created features
Because our CNN predictions are so strong, more than 90% of the top 30 important features were CNN-related features. Below are part of the features we have created.
### Tracking 
 - distance between two players
 - distance/x_position/y_position from step0
 - distance from around player (full/same team/different team )
 - distance between team center
 - distance to second nearest player
 - current step / max step
 - lag / lead of acc, speed, sa etc
 - max/min/mean of x, y, speed, acc, sa, distance group by (play, step), (play, step, team) and (play, player1, player2) x/y positon diff from step=0
 - ”interceptor” features
  - find playerC who meet the following conditions and add distance(A, C) and ∠BAC to the features of playerA-playerB (to detect that C intercepts between A-B)
     - ∠BAC < 30deg
     - distance(A, C) < distance(A, B) and distance(B, C) < distance(A, B)
### Helmet
 - bbox aspect ratio
 - bbox overlap
 - lag / lead of bbox coordinates
 - bbox center x,y std/shift/diff
 - distance of bbox centers
### CNN prediction and  meta-features
 - oof predictions of 1st stage CNNs
 - max/min/std of predictions group by (play, step) and (play, player1, player2)
 - 5/11/21 rolling features
  - to complement CNN predictions on frames without helmets
 - lag / diff
 - around players’ player-ground prediction value
#### Combinations
 - registration errors from helmet-tracking coordinate transform (similar to 6th place solution, and previous NFL’s 1st place solution by K_mat)

### Models
We trained four GBDT models with different combinations of 1st stage CNNs. We also added one NN model ("camaro2" in the figure above) and calculated the simple average of these 5 models. Predictions were binarized with separate thresholds optimized for player-player and player-ground respectively.

 - LightGBM
   - K_mat A + Camaro1 Public 0.795/Private 0.792
   - K_mat B + Camaro 1
   - K_mat B
 - xgboost
   - K_mat B + Camaro 1
 - Camaro 2
### tips
- rolling features for CNN prediction values are most important in our models.
- judging from permutation feature importance, ‘minimum distance between players in the game_play’,  ‘distance between away team mean and home team mean’ and ‘player-player distance’ are important tracking features to increase score.
- We did not use early-stopping to train the GBDTs because the optimal number of rounds for MCC is always longer than AUC.

### not wroked for models
- Catboost
- Residual fit
- Meta Features by non CNN (e.g. logistic regression prediction values/ k-means clustering feature)
- Separate player-player and player-ground model
- 1DCNN
- External NFL data
- Focal loss

# not worked for overall
- Adding previous competition pseudo labeling data
- Removing noisy label
- all29 assignment and its prediction
- 2.5D or 3D CNN, but should have dug more..
- Aggregate near frame information