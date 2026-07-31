# 2nd place solution: SED + CNN with 7 models ensemble

Congratulations to all the winners! Thanks to Kaggle and Cornell Lab of Ornithology for hosting this interesting competition.

This is my first solo gold medal and I am glad to have this result.

This competition shared a lot of similarity to the past BirdClef competitions(2020/2021/2022). Thus I spent a lot of time gathering the solution shared by the top teams in the past competitions. Special thanks to all of you for sharing such important information!

Let me briefly introduce my solution. I will update the solution for more details in a couple of days.

# Most important (7 models ensemble!)
Please see the notebook below.
[openvino is all you need!!](https://www.kaggle.com/code/honglihang/openvino-is-all-you-need)

# Training data

Here is my training data.

- 2023/2022/2021/2020 competition data
- [2020 additional competition data](https://www.kaggle.com/competitions/birdclef-2023/discussion/398318)
- additional training data from xeno-canto, including 2023 comp species in both foreground and background(records with 2023 comp species only in background which is less than 60 seconds are included). 

I intended to collect more records from ebird site, but I realized that ebird data is not public and cannot be used. I [asked the host](https://www.kaggle.com/competitions/birdclef-2023/discussion/393023) and confirmed that. Thanks [@tomdenton](https://www.kaggle.com/tomdenton) answering my questions.

Thus, my training pipeline does not contain records from ebird site.

# Model Architecture

First, I used SED architecture. The same as yours.

backbones are:

- tf_efficientnetv2_s_in21k
- seresnext26t_32x4d
- tf_efficientnet_b3_ns

All of them are trained on 10sec clip.

Second, I used CNN proposed by [2nd place of 2021 competition](https://www.kaggle.com/competitions/birdclef-2021/discussion/243463)

backbones are:

- tf_efficientnetv2_s_in21k
- resnet34d
- tf_efficientnet_b3_ns
- tf_efficientnet_b0_ns

All except b0 are trained on 15sec clip. b0 is trained on 20sec clip.

# Pseudo Labeling and Hand Labeling

I have used SED model to generate pseudo label and extracted the potential nocall using quantile threshold. Then, I hand labeled the potential nocall by hearing the record. I hand labeled about 1800 records but did not see improvement. Maybe the pseudo label contains more FP rather than FN. I did not have time to further investigate the prediction.

# Model Training

augmentations:

- GaussianNoise
- PinkNoise
- Gain
- NoiseInjection
- Background Noise(nocall in 2020, 2021 comp + rainforest + environment sound + nocall in freefield1010, warblrb, birdvox)
- PitchShift
- TimeShift
- FrequencyMasking
- TimeMasking
- OR Mixup on waveforms
- Mixup on spectrograms.
- [With a probability of 0.5 lowered the upper frequencies](https://www.kaggle.com/competitions/birdsong-recognition/discussion/183269)
- self mixup for records with 2023 species only in background.(60sec waveform -> split to 6 \* 10sec -> np.sum(audios,axis=0) to get a 10sec clip)

I have used weights (computed by primary_label and secondary_labels) for Dataloader in order to cope with unbalanced dataset.

# Training stages

For training I have used 2 stage training:

1. Pretrain on all data(834 species).
2. Finetune on 2023 species(264 species).

In both stages, I first train model with CrossEntropyLoss, and then train on with BCEWithLogitsLoss(reduction='sum'). Model converges faster with CrossEntropyLoss than BCEWithLogitsLoss, but BCEWithLogitsLoss gives better score.

To give more diversity, models are trained on different windows and different mixup rate, and some of them only trained on CrossEntropyLoss. And also 3 of the models are fintuned on 30s clip.

# CV strategy

- For each validation sample - slice the first 60 seconds to pieces -> predict each piece -> max(sample_predictions, dim=pieces).

CV does not show correlation with LB, but it seems that the right ways to improve the LB are those which do not significantly decrease CV. So I monitored the CV when tuning the pipeline.

# Inference

For SED model, feed model 10 sec chunk BUT apply head only on centered 5 sec reduced CNN image and use max(framewise, dim=time).

Also, tta(2s) is used for SED model.

Important: convert pytorch model to openvino model significantly reduce inference time(about 40%). (eca_nfnet_l0 backbone ONXX cannot be converted to openvino because the stdconv layer in timm use train mode of F.batch_norm in forward method). That is the magic of ensembling 7 models.

# Ensemble

I spent quite a lot of time understanding the metrics. The ensemble is as followed:

1. (weighted average, 0.84 on LB, 0.76 on private) Apply weighted average on raw logit. This ensemble does not make sense for me because the output logit of models differ and should not be simply added, otherwise the result is biased. But considering that the absolute value of logit may also contribute to the score and it does give the best LB, so I choose it for final submission to have a gamble.
2. (rank average, 0.83 on LB, 0.75 on private) Convert the logit to rank and apply weighted average on rank. I think this is the reasonable way to ensemble, considering that I don't have a reliable CV. <strong>Be careful that in this comp submission is padded with 5 rows of 1, thus the ranking should start from 0 to prevent the largest ranking to be 1 after convert rankings to percentile form.(For example, 0, 0.333, 0.666 is good while 0.333, 0.666, 1.0 is bad. otherwise the score will decrease due to the potential false positive of those ranked 1.0) </strong>

Basically all of my single model's performance reaches about 0.81-0.82 on LB, so the weight of each model is similar to give diversity although higher public lb requires large weight of sed v2s and sed b3_ns.

Looking at the private LB, weighted average seems to be better. emmm......Why?

# What did not work

- [contrast](https://www.kaggle.com/competitions/birdsong-recognition/discussion/183269)
- EMA
- BirdNet
- adjusting logit according to previous and next 5s clip
- q transform

Inference Notebook: https://www.kaggle.com/code/honglihang/2nd-place-solution-inference-kernel
github: https://github.com/LIHANG-HONG/birdclef2023-2nd-place-solution