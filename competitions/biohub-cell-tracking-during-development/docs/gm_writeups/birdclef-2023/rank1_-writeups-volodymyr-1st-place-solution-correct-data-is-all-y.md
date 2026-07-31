# 1st place solution: Correct Data is All You Need

Hi, Kagglers!

Let's start our journey in the tricky world of Audio Bird Data and Modelling, but before this, a few very important words:

*I would like to thank the Armed Forces of Ukraine, Security Service of Ukraine, Defence Intelligence of Ukraine, and the State Emergency Service of Ukraine for providing safety and security to participate in this great competition, complete this work, and help science, technology, and business not to stop but to move forward.*

# If You Only Knew the Power of A100s GPUs

I have managed to run 294 experiments: half of them with 5 folds and half of them with full data training. So, all in all, many hypotheses were checked and, of course, most of them were rejected :) So let's take a look.

# Data, Data is Everywhere 

## Let's Start from 2023 Training Data

If you take a look at `train_metadata["primary_label"].value_counts()`, you may notice some strange maximum magic number: 
```
barswa     500
wlwwar     500
thrnig1    500
eaywag1    500
comsan     500
          ... 
lotcor1      1
whctur2      1
whhsaw1      1
afpkin1      1
crefra2      1
Name: primary_label, Length: 264, dtype: int64
```
Why do we have a maximum of 500 representatives of some species? I do not know the 100% answer, but I have a strong hypothesis - a bug in [XC API](https://github.com/ntivirikin/xeno-canto-py). I cannot remember the exact place in the code, but the overall problem lies in the data loading pipeline. Here's how it works:

1. Download meta files - json files.
2. Iterate over all urls in the meta file(s) and download them.

BUT if you have more than 500 files for one species - on the first stage, you will have several json files (maximum number of files in one json metafile = 500), and here we have the problem! On the second stage, the API takes into account only one json for each species and ignores the next ones, so you will have a maximum of 500 files per species.

NOTE: I am not sure whether it is fixed in the latest version of the API, but I have used a commit from the previous year, and it was there.

From this bug, we can clearly understand that using the fixed API can hugely enrich our training dataset.

## Other more boring stuff

- 2023/2022/2021/2020 competition data
- [2020 additional competition data](https://www.kaggle.com/competitions/birdclef-2023/discussion/398318)
- [Zenodo](https://www.kaggle.com/competitions/birdclef-2023/discussion/394358#2179605)
- Xeno-Canto 

## Data Preparation

### Training Data

In order to make validation more robust:

- Split samples of species with only one representative into 2 splits. This is done in order to have at least one CV split with each species in train AND val splits.
- Remove some duplicates manually.
- Remove duplicates by the next rule: Two samples have same: duration, author, primary_label.

### Additional training data

From the 2023/2022/2021/2020 competition data plus Xeno-Canto data, I have selected only files with this year's primary labels and added them to the final stage of training.

### Pretrained Dataset 

When I was using only 2023 training data in the final stage of training, pretraining on 2022/2021/2020 competition data boosted the score a lot. But after adding additional training data, pretraining stopped working on the leaderboard (though it still increased local validation). In the last week, I decided to return to pretraining experiments. This granted me one position up in the public leaderboard and two positions up in the private leaderboard - so, Kagglers, don't forget to revisit even rejected hypotheses :) 

Why and when did it work? Compared to previous pretraining experiments, I have:
- Filtered out 2023 train data duplicates not only by id but also by 'author + primary_label', as was suggested [here](https://www.kaggle.com/competitions/birdclef-2023/discussion/395843)
- Taken species that are present in 2023/2022/2021/2020 competition data + 2020 additional competition data and only if there are more than 10 representatives of the species. Overall, 822 species. 
- Added additional files for selected species from Xeno Canto.

###  Zenodo

I have selected nocall regions and used them as background augmentation.

### Data Experiments That Did Not Work 

- Massive pretraining on all Xeno Canto data.
- [Background noise from 2021 2nd place](https://www.kaggle.com/datasets/christofhenkel/birdclef2021-background-noise) as Background augmentation 
- [ESC50](https://www.kaggle.com/datasets/mmoreaux/environmental-sound-classification-50) as Background augmentation 
- Selecting only High Quality samples (>=32kHz) from additional data 
- Maybe some other ideas out of 200+ experiments that I have just forgotten

# Validation: Be soft like cmAP, Do not be hard like F1

Finally! We do not have to select a threshold on completely different training data compared to soundscape data, come up with super [sophisticated schemes](https://www.kaggle.com/competitions/birdclef-2021/discussion/243463) or fall 19 places (as I did in 2021)

I have used pretty much the same validation scheme as in previous years' competitions:
- Stratified CV on 5 Folds
- Take max prob from each 5 second clip over time across ALL sample

IMPORTANT: For Padded cmAP it is pretty important to take mean across folds, NOT to do Out Of Fold !!! 

Of course, absolute numbers of CV and LB are different:
- Best Public LB:   0.84444 (4 fours :) )
- Best Private LB: 0.76392
- Best CV: 0.9083368282233681  

But the rank correlation was pretty good. CV improvement in 0.0x (and more) resulted in improvement on LB. I have nearly all CV results for my experiments, so I hope I will have time to publish a paper with a detailed ablation study and a CV-LB correlation study.

# Training

I have taken a look at @philippsinger [presentation](https://www.youtube.com/watch?v=NCGkBseUSdM) and understood how strongly I was overfitting all the time.

Due to time and device constraints, I have chosen the following scheme:
1. Validate the hypothesis on CV and submit the first 2-3 folds.
2. For ensembling retrain on full train data, so you have one model for each setup

Training Details:
- 50 Epochs
- Adam
- CosineAnnealing from 1e-4 (or 1e-3) to 1e-6
- Focal loss 
- 64 BS
- 5 second chunk 
- SUPER IMPORTANT: Class sampling weights
```
sample_weights = (
    all_primary_labels.value_counts() / 
    all_primary_labels.value_counts().sum()
)  ** (-0.5)
```
- Same setups for pretrain and finetune

Stages:
1. Pretrain - refer to `Pretrained Dataset `
2. Tune  only on scored species 

# Model

Because of computational constraints, we couldn't use the golden rule of Deep Learning: Stack More Layers!

So I have dived a bit in inference optimization techniques:
- ONNX - this worked pretty well for me. It improved the inference time slightly and allowed me to reduce the number of custom dependencies in the inference notebook.
- Quantization -  I spent more than a week experimenting with it, but unfortunately, I had no success :( 
-  openvino -  I didn't use or try this, I just read about it the [2nd place description](https://www.kaggle.com/competitions/birdclef-2023/discussion/412707) and burnt my chair 

Overall, my final submission is an ensemble of 3 Sound Event Detection (SED) models with the following backbones:
- eca_nfnet_l0 (2 stages training; Start LR 1e-3)
- convnext_small_fb_in22k_ft_in1k_384 (2 stages training; Start LR 1e-4)
- convnextv2_tiny_fcmae_ft_in22k_in1k_384 (1 stage training; Start LR 1e-4)

It was pretty important to tweak the starting learning rate for different architectures!!!

# Augmentations

I was pretty picky about augmentation selection, so my final models used next ones:
- Mixup : Simply OR Mixup with Prob = 0.5
- BackgroundNoise with Zenodo nocall
- RandomFiltering - a custom augmentation: in simple terms, it's a simplified random Equalizer
- Spec Aug: 
   - Freq: 
      - Max length: 10
      - Max lines: 3
      - Probability: 0.3
   - Time:
      - Max length: 20
      - Max lines: 3
      - Probability: 0.3

# Small inference tricks

- Using temperature mean: `pred = (pred**2).mean(axis=0) ** 0.5`
- Using Attention SED probs * 0.75 + Max Timewise probs * 0.25

All these gave marginal improvements but it is was a matter of first 3 places :) 

# Other stuff that created a carbon footprint but did not improve my LB score

This section will be far from complete but let's add something that I have in mind now:
- [2021 2nd place model](https://www.kaggle.com/competitions/birdclef-2023/discussion/412707). I have tried (like I did in 2022) but unfortunately it did not work for me 
-  Pretrain on whole Xeno Canto
- Train on larger chunks. The same result occurred if I inferred on smaller chunks or on same length chunks
- Colored Noise augmentations 
- CQT or [LEAF](https://github.com/denfed/leaf-audio-pytorch)
- Specific finetuning: smaller LR, smaller number of epochs, freeze backbone, different LRs for backbone and head
- Loss on Attention SED probs + Loss on Max Timewise probs
- Deep Supervision 
- Different `alpha` for MixUp
- Transformer architectures. For example [ECAPA TDNN](https://speechbrain.readthedocs.io/en/latest/API/speechbrain.lobes.models.ECAPA_TDNN.html) 

# Closing words

I hope you have not fallen asleep while reading. Finally, I want to thank the entire Kaggle community, congratulate all participants and winners.
Special thanks to Cornell Lab of Ornithology, LifeCLEF, Google Research, Xeno-canto, @stefankahl, @tomdenton, @holgerklinck. All of you were super active in discussions, shared datasets and interesting materials, answered all questions, and of course, prepared such a cool competition!"

# Resources 
**Inference Kernel** : https://www.kaggle.com/code/vladimirsydor/bird-clef-2023-inference-v1/notebook
**GitHub** : https://github.com/VSydorskyy/BirdCLEF_2023_1st_place
**Paper** : TBD