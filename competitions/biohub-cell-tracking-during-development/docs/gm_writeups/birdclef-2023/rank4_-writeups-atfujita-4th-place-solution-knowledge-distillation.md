# 4th Place Solution: Knowledge Distillation Is All You Need

First of all, thanks to Kaggle and Cornell Lab of Ornithology for hosting this interesting competition. And I would like to emphasize the thanks to the bird sound recordists who provided their data through xeno-canto.

### Solution Summary
- Knowledge Distillation is all you need.
- Adding no-call data, xeno-canto data, and background audios(Zenodo) is effective.

### Datasets
Additional datasets can be found [here](https://www.kaggle.com/datasets/atsunorifujita/birdclef-2023-additional).
- Bird CLEF 2023
- Bird CLEF 2021, 2022 (for pretraining)
- ff1010bird_nocall (5,755 files for learning no call)
- xeno-canto files not included in training dataset CC-BY-NC-SA (896 files) and CC-BY-NC-ND (5,212 files).
- Zenodo dataset (background noise)
- esc50 (rain, frog)
- aicrowd2020_noise_30sec (background noise for pretraining)

### Models
- My models are based on [the BirdCLEF 2021 2nd place solution](https://www.kaggle.com/competitions/birdclef-2021/discussion/243463) using [timm](https://github.com/huggingface/pytorch-image-models).
- My solution consists of a total of 4 models, all with eca_nfnet_l0 as the backbone. Each set is slightly different.

1. MelSpectrogram (sample_rate: 32000, mel_bins: 128, fmin: 20, fmax: 16000, window_size: 2048, hop_size: 512, top_db=80.0, NormalizeMelSpec) following [this solution](https://www.kaggle.com/competitions/birdclef-2021/discussion/243293).
    - Public: 0.8312, Private: 0.74424
2. Balanced sampling with the above settings.
    - Public: 0.83106, Private: 0.74406
3. [PCEN](https://github.com/daemon/pytorch-pcen) (sample_rate: 32000, mel_bins: 128, fmin: 20, fmax: 16000, window_size: 2048, hop_size: 512).
    - Public: 0.83005, Private: 0.74134
4. MelSpectrogram (sample_rate: 32000, mel_bins: 64, fmin: 50, fmax: 14000, window_size: 1024, hop_size: 320).
    - Public: 0.83014, Private: 0.74201

### Training
Knowledge distillation of pre-computed predictions in [Kaggle Models](https://www.kaggle.com/models/google/bird-vocalization-classifier/frameworks/TensorFlow2/variations/bird-vocalization-classifier/versions/1) (bird-vocalization-classifier) was the hallmark of my solution. This model cannot complete inference in less than 2h, but it is very powerful. cmAP_5 on my validation dataset was 0.9479. So I tried to extract useful information from this model.

**The Kaggle models pre-computed predictions were created [here](https://www.kaggle.com/code/atsunorifujita/extract-from-kaggle-models/notebook)**.

According to this [paper](https://arxiv.org/abs/2106.05237), they argued that efficient distillation required 1. consistent input, 2. aggressive mixup, and 3. a large number of epochs. So I did 2 and 3 because it was challenging to integrate the Kaggle model into my training pipeline. It certainly looked effective.

I chose only approaches that improve both CV and LB. In this competition, I just couldn't believe in one or the other.

- Using 5 StratifiedKFold. Only 1 fold did not cover all classes, so the remaining 4 were used for training.
- The evaluation metric is padded_cmap1. The best results were the same with padded_cmap5 in most cases.
- Use primary_label only
- no-call is represented by all 0
- loss: 0.1 * BCEWithLogitsLoss (primary_label) + 0.9 * KLDivLoss (from Kaggle model)
  - Softmax temperature=20 was best (tried 5, 10, 20, 30).
- I used randomly sampled 20-sec clips for training. Audio that is less than 20 sec is repeated.
- epoch = 400 (Most models converge at 100-300)
- early stopping(pretraining=10, training=20)
- Optimizer: AdamW (lr: pretraining=5e-4, training=2.5e-4, wd: 1e-6)
- CosineLRScheduler(t_initial=10, warmup_t=1, cycle_limit=40, cycle_decay=1.0, lr_min=1e-7, t_in_epochs=True,)
- mixup p = 1.0  (It was better than p=0.5)<br/>

 
I also encountered a situation where the training time was significantly longer when pretrained with past competition data. Thanks to everyone who suggested solutions.

#### Augmentation
- OneOf ([Gain, GainTransition])
- OneOf ([AddGaussianNoise, AddGaussianSNR]
- AddShortNoises esc50 (rain, frog)
- AddBackgroundNoise from [Zenodo](https://www.kaggle.com/competitions/birdclef-2023/discussion/394358#2179605). The 60 minutes with the fewest bird calls were extracted from each dataset and divided into 30 sec (training only).
- AddBackgroundNoise from aicrowd2020_noise_30sec and ff1010bird_nocall (pretraining only).
- LowPassFilter
- PitchShift

#### Hardware
- 1 * RTX 3090
- Pretraining time: 30-40h per model. 
- Training time: 9-12h per model.

### Inference
Reducing the inference time is the part I was having trouble with. Thanks to @leonshangguan for sharing his [effective approach](https://www.kaggle.com/code/leonshangguan/faster-eb0-sed-model-inference), I was able to use 4 models.

- Ensemble with simple averaging.
- 4 models using PyTorch JIT. total 110min.
- **[Inference notebook](https://www.kaggle.com/code/atsunorifujita/4th-place-solution-inference-kernel)**

### What didn’t work
- focal loss.
- Split a low-sample class.
- Backbone other than eca_nfnet_l0 and eca_nfnet_l1.
- Optimizers (adan, lion, ranger21, shampoo). I tried to create a custom normalize-free model but failed.
- [CMO](https://github.com/naver-ai/cmo) (mixup worked better when using distillation).
- [CQT](https://github.com/KinWaiCheuk/nnAudio/blob/master/Installation/nnAudio/features/cqt.py) (slow and degraded).
- Change to first stride(1, 1). CV was good but inference takes a long time and not a single model is completed in less than 2h.
- Pretraining Zenodo data (story before trying distillation).
- Distillation training from scratch (not use imagenet weights). It didn't converge in the amount of time I could tolerate.
- MelSpectrogram, PCEN, and CQT integrated into input channels in all combinations. There was no synergistic effect.

### Ablation study
| Name | Public LB | Private LB |
| --- | --- |--- |
| BaseModel | 0.80603 | 0.70782 |
| BaseModel + Knowledge Distillation | 0.82073 | 0.72752 |
| BaseModel + Knowledge Distillation + Adding xeno-canto | 0.82905 | 0.74038 |
| BaseModel + Knowledge Distillation + Adding xeno-canto + Pretraining | 0.8312 | 0.74424 |
| BaseModel + Knowledge Distillation + Adding xeno-canto + Pretraining + Ensemble (4 models) | 0.84019 | 0.75688 |

### Acknowledgments
My solution builds on contributions from participants in this and past competitions, birding enthusiasts, and the work of the machine learning community. thank you very much 🙏 

**Training Code**: https://github.com/AtsunoriFujita/BirdCLEF-2023-Identify-bird-calls-in-soundscapes