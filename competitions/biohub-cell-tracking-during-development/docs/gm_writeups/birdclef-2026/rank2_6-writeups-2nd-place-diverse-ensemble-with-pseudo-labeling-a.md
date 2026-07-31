# TL;DR: 

Public Perch+Improved distilled SED+Own CNN pipeline+Insecta specialist

# Acknowledgements

Thanks to the organizers and Kaggle for hosting such an interesting yearly competition, I really wanted to participate last year already but didn't have the availability.

Thanks to last year's top 5 solutions which were a big help in setting up my pipeline, especially [2nd place](https://www.kaggle.com/competitions/birdclef-2025/writeups/volodymyr-vialactea-2nd-place-journey-down-the-rab) whose [XC-pretrained backbones](https://www.kaggle.com/datasets/vladimirsydor/bird-clef-2025-pretrained-models) are a big part of my solution, and [1st place](https://www.kaggle.com/competitions/birdclef-2025/writeups/nikita-babych-1st-place-solution-multi-iterative-n) for the general roadmap and insecta specialist backbone and idea.

From this year's competition, I used some iterations of the public Perch notebooks, most notably [this one](https://www.kaggle.com/code/hideyukizushi/bird26-reproduce-perch-protossm-resssm-inf-train) and [this one](https://www.kaggle.com/code/mtoshidesu/test-0-948) for my final submission. I also used the public ["Distilled SED" notebook](https://www.kaggle.com/code/tuckerarrants/bc2026-distilled-sed) pipeline although I ended up making some modifications.

The public discussions e.g. ["What is your best single model LB score ?"](https://www.kaggle.com/competitions/birdclef-2026/discussion/683791) and ["How much did pseudo-labeling help you?"](https://www.kaggle.com/competitions/birdclef-2026/discussion/698504#3457225) also contained useful information.

In all, these resources were very useful to me since I'm a relative beginner.

# General strategy

When I started the competition there were already strong Perch-based notebooks, and I quickly realised that ensembling with my CNN models (even the early 0.8+ ones) worked very well. So my main goal was to develop my pipeline while preserving ensembling potential, if possible between my own models as well.

# Datasets

I used almost exclusively the competition data, except for the insecta model where I used @nikitababich's [extra XC data from last year](https://www.kaggle.com/datasets/nikitababich/birdclef2025-1st-place-extra-data). 

# Data split and validation

For my own pipeline I used a fixed held-out set consisting of focal clips, labeled soundscapes, and unlabeled soundscapes. Some soundscape-only species (most notably all the insecta sonotypes) were train-only, which made validation blind to those species, but I thought it would be better to feed them to the model. 

I tried many validation strategies along the way but ultimately they were all unreliable and I used LB as the main signal. I tried to engineer some LB proxy metrics with a subset of species, and to use pseudo-labeled data with high confidence. The only metric that stayed correlated to LB throughout was the AUC score on labeled soundscapes, but correlation was very low (~0.2).

# Mel specs

I experimented with various mel specs along the way. In general, lower hop_length and higher n_mels improved the models at the cost of higher inference time (no surprises there). It also looks like using lower hop_length, then resizing, works better than reaching the target size through hop_length directly.

However, for the XC-pretrained backbones, using the original pre-trained mel specs always worked better, so I used those for most submissions: 128 mels, 512 hop_length, 2048 FFT, 20 Fmin, 16000 Fmax.

# Window length

I experimented with 5s, 20s, and a mix of 20s focal/5s soundscapes. In the end, the 20s framework similar to @nikitababich's 2025 solution consistently underperformed (maybe I implemented it wrong). I used 5s windows for most of my ensemble models, except one with the 20s/5s mix, and the insecta model which uses 20s (didn't test is but it seemed more principled).

# Backbones

My final submission consists of Efficientnet_v2_s, Efficientnet_b0 and NFNet backbones (+Perch). Early on I experimented with many different backbones, and NFNet consistently performed best. Some others looked promising, but I couldn't make them improve through the pseudo-labeling phases (most notably ViT never got past 0.9). Towards the end, Efficientnet versions started to beat NFNet.

I also trained some Convnext models that I used for pseudo-labeling. They were the most diverse CNN compared to Efficientnet/NFNet, but scored lower on LB and took too long to infer, to I didn't use them in the ensemble. Interestingly PB seems to indicate those models were stronger than I thought.

I trained PaSST models for some pseudo-labeling steps, they had good validation metrics but I couldn't fit them in the 90 minutes inference to check LB.

Based on last year's top solutions I initially planned to use a SED head + GeM pooling system. However, GeM actually hurt my scores everytime I A/B tested it, so I defaulted to average pooling. 

# Loss functions

In my early experiments I tried a lot of loss functions, and BCE always performed better.

However when I reached the later pseudo-labeling rounds, I started experimenting again, originally with the goal of getting more model diversity: soft AUC loss had low correlation with BCE models, but lower LB scores.

In the end, my best models were with a combination of AUC loss and a small component of BCE loss (I settled with 0.25).

# Augmentations

Waveform mixup, frequency and time masking were beneficial and used throughout, other augmentations made LB worse. I didn't manage to make background noise augmentation work even though it seems principled.

# Perch distillation

I tried Perch distillation in the beginning and got around 0.02 improvement across backbones, but also noticed it significantly increased correlation between my models. So I decided to not use it at all to preserve ensembling potential (unclear if that was a good idea or not).

# Self-labeling rounds

Similar to last year's most top solutions, I used iterative pseudo-labeling rounds to gradually improve my models.
Since I was still learning, I made some new design choices along the way, so the score increases can't be attributed solely to the pseudo-labeling steps. 
I will try to describe both processes simultaneously since that is how I worked, so this section will be a bit long-winded.

I started by training a variety of backbones on the competition data while establishing my pipeline. Inspired by 2025 5th place, I then did one round of **focal pseudo-labels** with my best models (v2_s, NFNet, RegNetY, ViT) and raw Perch. The motivation was to detect any secondary labels in the focal clips. At this stages I tried using a pretrained NFnet from @nikitababich's 2025 solution and also reduced hop_length to 512, which boosted my LB score to 0.917 (0.910 PB), otherwise the scores from that stage are about the same as I was getting with Perch KD before.

**Round 1** of soundscapes pseudo-labels was generated from Efficientnet_b4, NFNet, Convnext and PaSST models, plus Perch predictions via a gating mechanism, since their calibrations were too different.
During this round I was using 20s focal clips mixed with 5s soundscape chunks. Pseudo-labeled chunks were added through mixup. My best models were using the backbones from the previous round and cosine warm restarts. My best model was with NFnet, it scored 0.919 LB but 0.926 PB, I kept it in my final ensemble.

**Round 2** of pseudo-labels was generated from two NFnet, Efficientnet_b4, PaSST and Convnext models. Here I started having difficulties improving my models. After reading [this discussion](https://www.kaggle.com/competitions/birdclef-2026/discussion/698504) and especially @aliozanmemetoglu's comments, I stopped mixing the soundscapes into the labeled data and started substituting them instead (with 50% probability, upped to 60% for my last models). I also settled with 5s windows exclusively for the later experiments. The best model was NFnet with 0.922 LB (0.930 LB), but Efficientnet and Convnext got equivalent scores. I also experimented with lower hop_length and stride which consistently improved LB at the cost of inference time.

**Round 3** pseudo-labels were generated with NFNet, Efficientnet and Convnext models, and I also included labels from the public Perch and distilled SED notebooks. I started using the XC-pretrained backbones from last year's 2nd place and their mel specs. Thanks to those, my models started breaking 0.930 LB, but the PB data shows that some of my earlier models were actually as strong. The best model from this round was an Efficientnet_v2_s that scored 0.934LB/0.933 PB, but I used a slightly weaker version of it in my ensemble.

**Round 4** pseudo-labels were generated with Efficientnet and NFnet models, as well as the public Perch and distilled SED labels. I started training models with soft AUC loss, originally with the goal to add model diversity, but I eventually converged with a soft AUC + 0.25 BCE setting that gave me the best LB scores, with pretty high correlation with the BCE models. I also tried to train 20s models again but they scored significantly lower (again, maybe faulty implementation). The best model was an Efficientnet_v2_s with 0.938 LB/0.935 PB, it is my 3rd ensemble member.

**Round 5** was a last minute copy-paste of round 4's best models with new pseudo-labels, I didn't have the slots to check their LB but adding one of them as 4th ensemble member improved LB and PB slightly.

I used pseudo-powers for each rounds of 1.55, 1.0, 1.2, 1.2, 1.2 for BCE models, and 1.0 for soft AUC + BCE models (decided from the resulting pseudo-labels distribution).

Recap of my best scores from each stage. All scores are single checkpoint, no TTA/post-processing. Some checkpoints use SWA averaging (when training was stable enough).

Stage | Best LB | PB |
| --- | --- | --- |
Competition data only | 0.881 | 0.869 |
With Perch KD (later discarded) | 0.896 | 0.891 |
Focal pseudo-labels round | 0.917 | 0.91 |
Soundscape pseudo-labels round1 | 0.919 | 0.926 |
Soundscape pseudo-labels round2 | 0.922 | 0.929 |
Soundscape pseudo-labels round3 | 0.934 | 0.933 |
Soundscape pseudo-labels round4 | 0.938 | 0.935 |
Soundscape pseudo-labels round5 | 0.933 | 0.932 |

# Insecta model

I followed the recipe from @nikitababich, training with amphibia and insecta samples from competition and extra XC data (upweighting the competition part), and started from his checkpoint from last year.
The model didn't actually help with amphibia, but adding it for insecta only raised LB by 0.002, and by 0.004 with 5x upweighting (50% of the ensemble).
I tried training this model on pseudo-labels with various settings (self-labeled or using the ensemble labels) but all led to worse LB.

# "Public Perch" ensemble member

I took a Perch embeddings-based model from [this notebook](https://www.kaggle.com/code/mtoshidesu/test-0-948), which scored a reproductible 0.936 LB/0.930 PB (it was itself a copy of [this one](https://www.kaggle.com/code/youssefmo942009/lb-0-948)). All my attempts to improve on this branch failed to I ended up just trusting the hivemind's successive experiments. I expected some LB overfitting, but it was still diverse from the rest of my models so worth adding.

# "Distilled SED" ensemble member

@tuckerarrants's Distilled SED pipeline was working really well on public notebooks and with my models as well (~0.4 correlation with Perch and with my CNNs). I tried improving its score but due to low compute and submissions remaining I ended up with a somewhat different setup: I removed distillation and instead used the XC-pretrained v2_s backbone, with softAUCloss+0.25BCE loss (the recipe with the best scores in my pipeline). This version scored 0.929 LB/0.938 PB, and still ensembled well with the rest.

# Ensembling decisions

I spent a lot of time trying to devise ensembling strategies using my held-out unlabeled soundscapes set.
In the end the judge was LB score, but some metrics had decent correlation with ensemble performance, that I used for some of my model choices: individual LB, signal-to-noise ratio, and lift. Model diversity, measured with Spearman correlation, was not necessarily a good predictor, but that might be an artifact of how I ran my evaluations. For the insecta specialist model, all metrics were useless at predicting the best weight and checkpoints (most likely due to the anti-correlation with the other models).

Inter-model Spearman correlation on oof data:

| |**Perch** | **SED** | **r1 NFnet** | **r3 Effnet** | **r4 Effnet** | **r5 NFnet** | **Insecta specialist** |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **Perch** | 1.000 | 0.410 | 0.428 | 0.538 | 0.522 | 0.494 | 0.415 |
| **SED** |  | 1.000 | 0.437 | 0.456 | 0.420 | 0.390 | 0.344 |
| **r1 NFnet** |  |  | 1.000 | 0.638 | 0.529 | 0.481 | 0.382 |
| **r3 Effnet** |  |  |  | 1.000 | 0.792 | 0.682 | 0.444 |
| **r4 Effnet** |  |  |  |  | 1.000 | 0.706 | 0.457 |
| **r5 NFnet** |  |  |  |  |  | 1.000 | 0.404 |
| **Insecta specialist** |  |  |  |  |  |  | 1.000 |

Using Perch and SED as pseudo-labeling teachers raised their correlation from ~0.4 to ~0.5. Insecta specialist correlations are computed on insecta only.  The figures are slightly wrong for the SED branch because I didn't generate oof for my last iteration. 

# Final ensemble

| Model | LB | PB | Note | Solo inference time |
| --- | --- | --- | --- |
| Perch | 0.936 | 0.930 |  | 23min |
| Distilled SED-based v2_s | 0.929 | 0.938 | AUC+BCE loss | 25min |
| 1st round NFNet | 0.919 | 0.926 | BCE loss, 20s clips with 5s chunks | 17min |
| 3rd round v2_s | 0.933 | 0.928 | BCE loss | 6min |
| 4th round v2_s | 0.938 | 0.935 | AUC+BCE loss | 6min |
| 5th round NFNet | 0.933 | 0.932 | AUC+BCE loss | 9min |
| Insecta specialist b0 | NA | NA | 50% of insecta ensemble | 8min |

With some shared I/O and initialisation, the total inference time is around 87min.

The weights were mostly equal, I slightly reduced the weights from the r3-r4-r5 round models since they are heavily correlated.

The individual model scores moved a lot between LB and PB, but they mostly canceled out and the overall score was stable.
My final submission scored 0.959 LB - 0.960 PB, and most of my ensembles scored similar on LB and PB, despite individual models scoring very differently. The ensembles that degraded on PB were the ones where I experimented with overweighting certain models, so I credit the equal-weighted ensemble and the relative lack of LB overfitting strategies with my final result (with of course a touch of luck).

# Post processing

I used the "sonotype mirroring" and "temporal continuity" steps from the public notebooks, they gave a consistent small boost. All other techniques failed or were neutral, most notably any form of smoothing. I still kept the gaussian smoothing in the public SED pipeline (I didn't have the subs to test it).

# AI usage

In the recent competitions I saw that a lot of top-scoring solutions used AI in some way, and the Birdclef discussions when I joined were all about @tom99763's experiments, so I decided to stop being a dinosaur and got a Claude subscription.

I used it through CLI to generate code, analyse results, and summarize discussions. In the end it was a great accelerator and I don't think I would have scored in this bracket without it. However as a result my code will be quite bloated and there are some unintended choices (e.g. my 20s/5s mix setting wasn't originally intended).

# Compute usage

I used mostly remote compute, with a bit of Kaggle notebooks, for this competition. The total cost is around 200$ + 2 months of Claude sub (around 40€).

# To be added: inference notebook, Github repo, fixes...

I will come back and edit/answer questions over the next few days.

Edit 1: [Inference notebook](https://www.kaggle.com/code/tennogh/2nd-place-0-960-submission) added with model weights

Edit 2: [GitHub repo](https://github.com/FlorentinGe/Birdclef2026-2ndplace) (will do a few minor fixes e.g. hard paths)