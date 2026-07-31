# 3rd place solution, single stage approach

I'd like to thank the I'd like to thank organisers for a very interesting challenge (especially @robikscube for providing very useful answers and helping teams). It was interesting to participate.

## Overview

The approach is single-stage, trained end-to-end with a single model executed per player and step interval (instead of per pairs or players) and predicting for all input steps range the ground contact for the current player and contact with 7 nearest players. The model has a video encoder part to process input video frames and a transformer decoder to combine tracking features and video activations.

![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F743064%2Fa93573968664b0e0af1534e3819e9d49%2FKaggle%20model_1.png?generation=1677986296204472&alt=media)

### Video encoders

The video encoders used a number of input video frames around requested steps and produced activations at corresponding steps at downsampled resolution, usually for 16 steps with corresponding 96 frames using every second frame for input.

I used a few different models for video encoders:

- 2d imagenet pretrained models + 3d Conv layer (credits to the Team Hydrogen solution of one of previous competitions). 3 input frames around the current step are converted to grayscale and used as an input to 2d model, with the results combined using 3d conv. Usually larger models performed better for me, with the best performing model based on the convnext large backbone. Other Convnext based models or DPN92 also worked ok.
- 2d imagenet pretrained models + TSM, with the color inputs for every 2nd or 3rd frame and TSM like activation exchange between frames before every convolution. Worked better with smaller models like convnext pico or resnet 34 (would probably work better with larger models if the TSM converted model were pretrained on video tasks).
- 3D/Video models like CLIP-X (X-CLIP-B/16 was the second best performing model) or the Video Swin Transformer (performed okeish but not included in the final submission).

Video frames were cropped to 224x224 resolution with the current player's helmet placed at the center/top part of the frame and scaled so the average size of helmets in surrounding frames would be scaled to 34 pixels.
I applied augmentations to randomly shift, scale, rotate images, shift HUV, added blur and noise.

For video model activations (at the 32x downsampled 7x7 resolution) I added the positional encoding and learnable separate sideline / endzone markers.
Optionally the video activations may be encoded using transformers per frame in a similar way as done in DETR but I found it has little to no impact on the result.

### Transformer player features / video activations decoder

The idea is to use attention mechanisms to combine the players features with other surrounding players information and to query the relevant parts of the images.

For particular player and step, I selected the current player features for surrounding -7..+8 steps and for every step I selected up to 7 nearest players within 2.4 yards, so in total 16 steps * (7+1) players inputs.

For every player/step input I used the following features, added together using per feature linear transformation to match the transformer features dim:
- position encoding for the helmet pos on the sideline and endzone video, if within 128 pixels from the crop.
- is it visible on sideline and endzone frames
- pos encoding for the step number
- is player the current selected player
- is player from the same team as the current player or not
- player position (not xy but the role from the tracking metadata)
- speed over +- 2 frames
- signed acceleration over +- 2 frames
- distance to the current player, both values and one hot encoding over +- 2 frames
- relative orientation, of the player relative to player-player0 and of player0 relative to player, encoded as sin and cos over +- 2 frames
- for visible helmets, I also added the activations from the video at the helmet position directly to player features. The idea was - it's most likely relevant and may help to avoid using the attention heads for the same task, but I found no difference in the final result.

Player/step features are used as inputs/targets for a few iterations of transformer layers:
- For all step/player input, I applied the transformer decoder layer with the query over video activations from the same step. 
- For all step/player inputs I applied the transformer encoder with the self attention over all  players/steps:

```
        # video shape is HW*2 x T*B x C
        # player_features shape is P, T, B, C
        # where P - players, T - time_steps, B - batch, C - features, HW - video activations dims
        x = player_features
        for step in range(self.num_decoder_layers):
            x = x.reshape(P, T*B, C)  # reshape to move time steps to batch to use attention only over the current step
            x = self.video_decoders[step](x, video)

            x = x.reshape(P*T, B, C) # attention over all players/steps
            x = self.player_decoders[step](x)
```

I tested with the number of iterations between 2 and 8 and the results were comparable, so I used 2 iterations for most of models.

## Data pre-processing

Mostly to smooth the predicted helmets trajectory, smoothed the prediction to find and remove outliers and interpolated/extrapolated.
During the early test the impact on the performance was not very large, so not conclusive.

![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F743064%2Feb690ac7195ad8d03205681175d3c979%2Fplayers_trajectory_pp.png?generation=1677916456068367&alt=media)

## Training

For training I selected all players and steps with helmet detected on at least one video (so model would have the tracking features for a few steps before or after the player was visible for the first/last time). I have not excluded any samples using other rules.

I used the AdamW optimiser with quite a small batch size of 1 to 4 and CosineAnnealingWarmRestarts scheduler with the epoch size of 1024-2048 samples, trained for 68 epochs. It takes about 6-10 hours to train a single model on 3090 GPU.
I evaluated model every time the scheduler reaches the min rate at epochs 14, 36 and 68.

I used the BCE loss with slight label smoothing of 0.001..0.999 (it was a guess, I have not tuned hyperparameters much).

I added aux outputs to the video models to predict if the current player has contact with other players or ground and heatmap of other player helmets with contacts, but the impact on the score was not very large.

## Prediction

The prediction is very straightforward, for model with the input interval of 11 or 16 steps I run it with the smaller offset of 5 steps to predict over the overlapped intervals for every player.

predictions = defaultdict(list)  # key is (game, step, player1, player2)

Every prediction between the current and another player, it's added to the list at the dictionary key (gameplay, step, min(player0, player), max(player0, player))
and all predictions are averages. Usually predictions for the pair of players at a certain step would include predictions with each player as the current one and a few step intervals when the current step is closer to the beginning, middle and end of the intervals.

When ensembles multiple models, their predictions are added to the same predictions dictionary, with better models added 2-3 times to increase their weight.
In total, I used 7 models for the best submission.

## Individual models performance

| Video model type, backbone | Notes                                      | Private LB score |
|----------------------------|--------------------------------------------|------------------|
| Convnext large, 2D + 3D conv| 16 steps/96 frames, skip 1 frame.         | 0.7915           |
| Convnext base, 2D + 3D conv| 16 steps/96 frames, skip 1 frame.          | 0.786            |
| DPN92, 2D + 3D conv        | 16 steps/96 frames, skip 1 frame.          | 0.784            |
| X-CLIP-B/16                | 11 steps/64 frames, skip 1 frame.          | 0.791            |
| X-CLIP-B/32                | 11 steps/64 frames, skip 1 frame.          | 0.784            |
| Convnext pico, TSM         | 63 steps/384 frames, skip 2 frames.        | 0.788            |
| Convnext pico, 2D + 3D conv| 64 steps/384 frames, skip 2 frames.        | Local CV slightly worse than TSM |
| 2 best models ensemble |  Convnext large and X-CLIP-B/16,   | 0.7925 |
| 6 models ensemble |  Without DPN92, re-trained on full data with original helmets  | 0.7932 |
| 6 models ensemble |  Without DPN92, re-trained on full data with fixed helmets      | 0.7934 |
| 7 models ensemble |  Convnext large  added with weight 3 and X-CLIP-B/16 with weight 2. Models trained on different folds.  | 0.7956 |

## What did not work

- Training Video Encoder model using aux losses before training transformer decoders. Video Encoder overfits.
- Adding much more tracking features to player transformer inputs. When added the history over larger number of steps for each player input, the transformer encoder overfits.
- Larger models with TSM
- Fix players/helmets assignment in the provided baseline helmets prediction. On some folds the impact was negligible, on some the score has improved by ~ 0.005 even without re-training models. On the private LB the score was similar with and without helmets fixed. One submitted model was using the original data pre-processing, another using more complex pipeline with helmets re-assigned.

## Local CV challenges

To check for possible issues with models generalisation, I decided to split to folds using the sorted by game play list of games, with the first 25% of games assigned to fold 0 validation fold and so on.

I found to have not only the difference between folds in score, but models/ideas performing well on one fold may work much worse on another.
For example, I found on the fold 2, the models with the very large receptive field over time/steps (384 steps, over 6 seconds,  convnext pico based models in the submission) performed by about 0.008 better than the best larger models, while the score fo such models was by the similar 0.007 worse on the fold 3.

All this made the local validation much more challenging and harder to trust. Taking into account the private dataset is even smaller than every fold, I expected to see a significant shakeup.

## Player helmets re-assignment

Since it was not part of the best submission, added as a separate post: https://www.kaggle.com/competitions/nfl-player-contact-detection/discussion/392392

Instead of the data pre-processing described above, I used the estimated tracking -> video transformation to interpolate/extrapolate missing helmets information. The best result was when I discarded the first or the last predicted helmet position and extrapolated by 8 steps maintaining the difference with the position predicted from tracking and tracking->view transformation.

The submission source is available at https://www.kaggle.com/dmytropoplavskiy/nfl-sub-place3