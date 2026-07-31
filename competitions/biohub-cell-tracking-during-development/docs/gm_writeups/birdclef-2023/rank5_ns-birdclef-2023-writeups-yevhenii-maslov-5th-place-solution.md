# 5th place solution

First of all, thanks to the Cornell Lab of Ornithology and the Kaggle Team for hosting this competition. It was a great opportunity to learn something new.

In this post, I want to present a summary of my solution.

### Datasets

* 2023/2022/2021 competition data
* Additional Xeno-Canto data containing 2023 comp. species in foreground and background
* ESC50 noise
* No-call noise from 2021 competition data

### Models

I used SED architecture (same as in @tattaka 4th place [solution](https://www.kaggle.com/competitions/birdclef-2021/discussion/243293)) with following backbones:

* tf_efficientnet_b1_ns
* tf_efficientnet_b2_ns
* tf_efficientnet_b3_ns
* tf_efficientnetv2_s_in21k

### Training

I trained all models in two steps:
* Pretrain with 2022/2021 data. I used only white noise (p=0.5) for this step
* Finetune on 2023 data with the following augmentations:
	* For waveform - Mixup (p=1) and OneOf([White noise, pink noise, brown noise, noise injection, esc50 noise, no-call noise]) (p=0.5)
	* For spectrogram - Two time masks (p=0.5 each) and one freq mask (p=0.5)

Training details:
* All models were trained on 5-sec clips
* 4-fold stratified CV split
* I used both primary and secondary labels
* BCEWithLogitsLoss with weight for each sample based on the rating
* AdamW - 5e-4 lr, 1e-3 weight decay for most of the models
* CosineLRScheduler with default parameters
* 40 epochs - the best score was almost always in the last epoch, so in addition to 4 folds, I also trained the 5th model using all available data. This full-fit version was consistently better than one-fold by 0.002-0.003 public and private LB.
* Some models were finetuned on soft/hard pseudolabels. 

My best model was trained with a combination of actual labels for competition data and hard pseudo labels for XC data. It has 0.72714/0.81836 private/public LB and 10 minutes submission time. 

### Inference

The probabilities from 15 models (folds) were averaged. 6 of these models have the same first three layers of the backbone, so the number of models is not very fair)

I've also used ONNX runtime and ThreadPoolExecutor - thanks to the @leonshangguan for sharing this [notebook](https://www.kaggle.com/code/leonshangguan/faster-eb0-sed-model-inference).

Inference kernel:  https://www.kaggle.com/evgeniimaslov2/birdclef-5th-place-code
Github: https://github.com/yevmaslov/birdclef-2023-5th-place-solution