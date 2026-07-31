# 3rd solution

I am the one posting but this is team work with @christofhenkel and @theoviel : team NVBird! 

Thanks everyone for the interesting competition, and congratulations to the winners ! We are very happy with the 3rd place, even though we missed the win by a very tiny margin.

## Overview
Our pipeline is summarized below. A key ingredient of our solution is to use unlabeled soundscapes for pseudo labeling and model distillation. A number of models were trained with the training data, then used to predict labels on 5 second clips from unlabelled soundscapes. These were added to the original training data to train a new set of models used for the final submission. 

<img src="https://i.ibb.co/QJQrdFf/Bird-CLEF-pipe.png" alt="Bird-CLEF-pipe" border="0">
## Data

Overall, we relied on knowledge acquired during previous competitions, and added some extra samples to fight class imbalance.

We use this year’s competition data plus the additional data from Xeno Canto shared in the forum, plus records from previous year competitions for the same species as this year.  When a record name appeared in several competitions we picked the most recent one (their content is not always identical form one competition to the next one)
We capped the number of records per species to 500, keeping the most recent ones. Indeed, adding all extra data leads to severe class imbalance that was detrimental to model accuracy.
Low frequency classes are upsampled so that there are at least 10 samples for each class in the training folds. 

To train models, we use the following preprocessing and augmentations:

We didn’t use all the training data. For each record we use a random crop of 5 seconds clip among the first 6 seconds, or the last 6 seconds of a record. If record length is smaller than 5 seconds, random padding so that the middle of the signal is between 2 and 3 seconds in the 5 sec resulting clip. For most of the models we used time shifting with a one second window as the only augmentation besides mixup. An exception are some models which are inspired by Birdclef23 2nd place SED models (https://github.com/LIHANG-HONG/birdclef2023-2nd-place-solution/blob/main/configs/sed_v2s.py)and use the same augmentations as used there
We use an additive mixup: primary labels are the max of primary labels of the two audios to be mixed. Secondary labels are the concatenation of secondary labels.
We mostly use image models that take log mel spectrograms as input. For these we compute mel spectrograms with parameters chosen to have an image size of 224x224 or 288x288 depending on the image model we use. Input waveforms are normalized to have a std of 1. 
## Models
### First Level models

The cpu-only requirement was quite constraining for submissions, but this does not apply for pseudo-label generation, so we could use more backbones for first level models. When ensembling several models larger than the ones used at second level we perform what is known as model distillation. This is a rather powerful technique in general.

Models used include:
- Efficientvit_b0.224.in1k on 224x224 log mel spectrograms
- Efficientvit_b1.r288_in1k on 288x288 log mel spectrograms
- A variety of CNNs (efficientnets, mobilenets, tinynets, mnasnets, mixnets) and Efficientvits ([b0, b1](https://arxiv.org/pdf/2205.14756), [m3](https://arxiv.org/pdf/2305.07027) trained on 224x224 log mel spectrograms.
- SED model with tf_efficientnetv2_s_in21k on 128x313  log mel spectrograms?
- We also fine-tuned aves-large and [aves-base](https://github.com/earthspecies/aves?tab=readme-ov-file#birdaves), which is a recent  waveform based model.

Depending on the pipeline one or more of these models were used to predict pseudo labels on unlabelled soundscapes. We trained most models on full data with 5 different seeds.

### Second Level models
We used efficientvit-b0 primarily and mnasnet-100 on 224x224 log mel spectrograms. Efficientvit-b0 showed great performances while still being very fast to infer. 5 folds take 40 minutes to submit using ONNX. We tried several models with similar throughput to effvit-b0, and decided to also use an mnasnet-100 for diversity. Since computing log mel spectrograms is slow on CPU we decided to use the same mel spectrogram hyperparameters for all models in the inference notebook, to only have to do log mel spectrogram transformation once and use the same input for our ensemble models.

For training second level models we added the unlabelled soundscapes with the predicted pseudo labels to the training data. This looks simple but it took several attempts to find the correct way to do it. What worked fine was to use rather large batch sizes (128), 
We used the two following strategies;
- Add extra soundscapes to each batch. Each soundscape is split into 5 second clips, which means we added 48 x 4 = 192 clips with pseudo labels to the 128 samples with actual labels in each batch.
- Add 128 samples with pseudo labels, taken from random soundscapes this time. 

### Final ensemble

At the end we had an ensemble of 14 model weights via 3 pipelines, one pipeline per team member:

#### Pipeline 1 - 5 seeds
- Level 0: efficientvit_b0 
- Level 1: efficientvit_b0

#### Pipeline 2 - 2 seeds + 2
- Level 0: efficientvit_b1, mobilenetv2, efficientnet_b0, efficientnetv2_b0, efficientvit_b0, efficientvit_m3, aves-base, aves-large
- Level 1: 2x mnasnet-100

- Level 0: Efficientvit_b0, mixnet_s, mnasnet_100, tinynet_b, efficientvit_b0, efficientvit_b1, mobilenetv3
- Level 1: 2x efficientvit_b0, 

#### Pipeline 3 - 5 seeds
- Level 0  efficientvit_b1_288, 5x efficientnet_v2_s sed,  aves-base, aves-large 
- Level 1 5x efficientvit_b0

That ensemble scored 0.689970 private, and 0.742124 public.

## Training

It was a bit tricky to tweak parameters without a validation set. We have two training pipelines with different parameters, and it is rather unclear what actually mattered. 

We used BCEWithLogitsLoss, without any label smoothing. Labels were defined by primary labels. Secondary labels were used to mask loss: loss for secondary labels is multiplied by 0. Reason is that we don’t know if a secondary label occurs at the start and at the end of the record, but they could. Given the uncertainty, we mask the loss for these. Masking secondary label loss improves LB by about 0.01.

When using pseudo labels on unlabelled soundscapes we set their secondary labels to be empty. 

We used AdamW or Ranger optimizers. The number of epochs for the second level was way higher than for first level ones. For instance in one pipeline first level models are trained with 30 epochs while second level models are trained with 88 epochs.
## Post Processing

We used several post processing, to incorporate soundscape-level information.

The first one worked well on public LB with a 0.02 boost initially. Experiments after competition end show that its effect is much smaller (0.004 on our best unselected submission), and even detrimental on our best selected submission (- 0.002). The idea is that if a bird appears anywhere in the soundscape then its probability to appear in any 5 second slice is increased.
For a given soundscape, once we have a 48x182 logits array or predictions P, we compute the maximum P_max of P over the time dimension. We then replace P with:
P + (P_max + P.mean() - P_max.mean()) * 0.8.
The 0.8 weight can be tuned further.

The second postprocessing had a smaller impact on LB the first time it was tried but it had a larger impact of +0.01 on our best selected sub. The idea is to smooth predictions for each 5 second clip by blending it a bit with the previous two and the following two clips. We used a convolution for smoothing predictions with the kernel [0.1, 0.2, 0.4, 0.2, 0.1].

## What did not work

A lot. Main issue was that we could not find a reliable local validation scheme. We mainly looked at train data cross-validation like most of the participants, but also put some effort in looking at past BirdCLEF train / test differences individually for each year. We thought that this might help to understand what augmentations might be bridging the train (= xeno-canto) / test (= PAM soundscape recordings) gap. However, the results were quite inconclusive. Probably because soundscapes are quite different each year.  Whatever we tried lost correlation with the public LB once scores were high enough.

One thing that seemed to work great and didn’t in the end was the post processing based on max probability per soundscape.

Another one was the Aves model which improved public LB by about 0.01 each time it was used, but appeared to be detrimental on private LB.

We tried a number of augmentations, including those documented in previous competitions or in academic papers, but none seemed to really help.

ONNX did not bring much speedup (maybe 10%) and Openvino was much faster (almost 2x speedup compared to pytorch). However we noticed a drop of about 0.01 when using Openvino compared to ONNX, and decided not to use it. Maybe the speedup from openvino would have been compensated by the larger number of models we could use in the final ensemble?

Our final model selection did not work that well in the end. We had individual models submitted with private scores above 0.70, but we did not include them in our ensemble given their relatively low public scores.

Thanks for reading !

Edit, link to code (alphabetical order):
- [CPMP's part](https://github.com/jfpuget/birdclef-2024)
- [Dieter's part](https://github.com/ChristofHenkel/kaggle-birdclef24-3rd-place-solution-dieter)
- [Theo's part](https://github.com/TheoViel/kaggle_birdclef2024)
- Best selected submission: https://www.kaggle.com/cpmpml/birdclef-2024-inf-ens-08