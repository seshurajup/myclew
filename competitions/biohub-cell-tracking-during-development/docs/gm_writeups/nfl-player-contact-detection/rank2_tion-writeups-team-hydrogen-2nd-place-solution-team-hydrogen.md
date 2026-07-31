# 2nd place solution - Team Hydrogen

Thank you for another great NFL challenge! As the previous NFL competitions it was well prepared and had quick feedback cycles anytime that the community had questions. We would like to highlight @robikscube, one of the hosts, who even supplied a strong tabular baseline to get started. 

## Validation
The test data is rather small compared to the large training set and only consists of 61 plays. Thus, local validation becomes even more important than usual. To evaluate our models, we used Stratified Group KFold cross validation on the `game_key` and public LB usually followed any local CV improvements with only a small random range of a few points and with blends being a bit more stable than single models (5 folds or a handful of fullfits). Our best local CV was 0.807 for the blend including 2nd stage and about 0.802 for a single model including 2nd stage. 

## Models and architecture
The core ideas and central building blocks of our models are based on our concepts of the previous DFL competition (https://www.kaggle.com/competitions/dfl-bundesliga-data-shootout/discussion/359932) utilizing 2D/3D CNNs capturing temporal aspects of videos. This architecture has already served us well in multiple video sports projects and competitions and also turned out to be highly competitive here.

In this competition we found longer time steps to work better and we got our best single model results using a time step of 24 frames, two times in both directions. We crop the region of interest for each potential contact based on helmet box information. In most models, we resize the crop, so that all boxes have about the same size. We concatenate endzone and sideline views horizontally to enable early fusion. Additionally, we encode tracking data directly into the CNN models. This has the main advantage that we can mostly rely on a single stage solution, and are less prone to overfitting on a 2-stage approach with out-of-fold CNN predictions. We step-wise encoded tracking features based on their importance in tabular models.

The main architecture of our approach looks like the following:

![model-architecture](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F2675447%2F81c89d45e01bd4e3c444b8d64ddcd091%2Farch2.png?generation=1677767191925623&alt=media)

We will now explain in detail each of the channels. We have slight variations of these channels across models in our ensemble, but the core concept is the same. Please note that the order of channels is always the same, and the order is only changed for visual clarity in above architecture visualization, also only showing three channels, while our models use mostly five. Frames 552, 600 and 648 show the first channel in the foreground, while frame 576 shows the second frame, and 624 the fifth frame.

**First channel**
The first channel depicts the region of interest of the potential contact only using the grayscale image. For each view, we take the center of the two (or one) boxes and then crop a total rectangle of width 128 and height 256. We then put both views next to each other resulting in a 256x256 input size. For most of our models we try to keep the aspect ratio based on box information and crop more information downwards than upwards to better capture the full body of players.

**Second channel**
Here we put a mask of the boxes to allow the model to clearly learn which players it should try to predict the contact for. We mask the two boxes with a value of 255. If there is only one box, or if there is a ground contact, we only mask this one box. We additionally mask all other boxes in this crop with 128.

**Third channel**
The most important feature is the distance between two players. The CNN model itself can only learn the distance between players to some degree. So in this channel we directly decode the distance as derived from tracking information. Conveniently, there is a nice cutoff at around 2 yards where basically no contacts are present any longer. So we just multiply the distance by 128, giving us values between 0 and 255 that we encode in this channel.

**Fourth channel**
A very important feature was whether both players are from the same team. So here we just encode 255 if both are from the same team, and 128 otherwise.

**Fifth channel**
Finally, we saw that distance traveled of players from the last time point is helpful in tabular models. So similar to distance between players, we encode this feature separately for both players, or one in case of ground attack.

For all tracking feature channels, we stick to uint8 encoding which means we lose some precision for the features, but it helps with overfitting to it and can be seen as a binning between 256 bins similar to what GBM models do. The great benefit of encoding these features is that the CNNs can learn all the spatial and temporal information of such tracking features directly.

As the 2D backbone, we used `tf_efficientnetv2_s.in21k_ft_in1k` and `tf_efficientnetv2_b3` architecture and pre-trained weights from the timm library. We train all our models for 4 epochs and cosine schedule decay and AdamW optimizer. Checkpoints are always on last epoch.

## Augmentations
Specifically mixup proved to be very useful in preventing quick overfitting. While it may appear counterintuitive to work well with the encoded feature channels, it likely acted as a good regularization. 
During training, we randomly shifted the image frame within a range of +-3 frames to the closest matching frame calculated from the current step. Furthermore, we used a small shift of +-1 for a subset of the model as test time augmentation in the ensemble. 

## Tracking and helmet interpolation
For the random frame shift augmentation, it was helpful to interpolate the tracking information from 10 Hz to 60 Hz. We tried a few different methods, but simple linear interpolation proved to be sufficient and is robust. We also added missing helmet box information using linear interpolation. While this definitely added some noise and false positives, overall it seemed to have helped catching a few more contacts in very crowded situations. We also use this interpolation for inference in our submissions.

## Ensemble & Inference
Our final ensemble consists of 6 models, and 3 seeds for each of them. All final models were retrained on the full data. We tried to add some diversity by different crop strategies and step sizes.

|Backbone| Description | Step size | CV |
| --- | --- | --- | --- |
|tf_efficientnetv2_s.in21k_ft_in1k|No scaling of the crops|24|0.7899|
|tf_efficientnetv2_b3|Slightly zoomed-in crops|24|0.7953|
|tf_efficientnetv2_b3|Inverted feature channel encoding|24|0.7987|
|tf_efficientnetv2_b3|No interpolation for boxes of other players|24|0.7989|
|tf_efficientnetv2_b3|Inverted feature channel encoding|12 (4 times)|0.7988|
|tf_efficientnetv2_s.in21k_ft_in1k|Smaller step size|6|0.7890|  

&nbsp;

We made full use of the recently added kernel with 2 T4 GPUs by parallelizing the pipeline and spawning two threads (1 CPU core for each to preprocess) each covering one half of the plays. All model predictions were averaged and subsequently fed to a stage 2 LGBM model. The final blend has a CV score of around 0.805 before the second stage.

### Stage 2
We use a LGBM model with only a few carefully selected features including stage 1 ensemble probabilities, `nfl_player_id_1` to `nfl_player_id_2` distance and their lags. Other notable features are "step_pct", encoding the current step based on the play length and normalized X and Y positions on the field. Basically, using the average position of the two players and normalizing to one quarter of the field to prevent overfitting to single plays. 

In the early stages of the competition, our 2nd stage model gave a great boost in score, specifically after adding the extra tracking features, while in the end the stage 1 predictions were almost on-par, showcasing how the stage 1 CNNs already efficiently learn from the encoded tracking feature channels.

Finally, we blend the LGB predictions with the smoothed raw predictions (window of 3) from the ensemble in a 50:50 ratio.

Our final solution has a CV score of 0.807, a public LB of 0.796, and a private LB of 0.796, exhibiting strong consistency and generalizability.

Huge shoutout to my teammates @philippsinger and @ybabakhin!