# 5th solution

First of all, thanks to the organizers for hosting this competition.
And congrats to all winners.

### Data
2024 data

### Model
| model                          | backbone       | weights     | public   | private  |
|--------------------------------|----------------|-------------|----------|----------|
| raw signale                    | hgnetb0        | 3/5 folds   | 0.720354 | 0.666671 |
| spectrum                       | efficientnetb0 | 1/5 folds   | 0.708198    | 0.672360    |
| raw+spectrum                   | hgnetb0+effb0  | 1/5 folds   | 0.698408 | 0.660503 |
| ensemble                       | ...            | 0.5+0.4+0.1 | 0.743960 | 0.687173 |

+ 1. raw signal model
```
# x bsx160000

#bsx80000x2
x = x.view(bs, -1, 2)           
#bsx2x80000
x = torch.transpose(x, 2, 1)    
#bsx2x1000x80
x = x.view(bs, 2, -1, 80)         
feature = self.backbone(x)   
```
+ 2. spectrum model
```
torchaudio.transforms.MelSpectrogram(
                32000,
                n_mels=512,
                f_min=0,
                f_max=16000,
                n_fft=2048*2,
                hop_length=512,
                normalized=True,
            ),

torchaudio.transforms.AmplitudeToDB(top_db=80)
```
+ 3. mix model

```commandline
raw_f  = self.raw_model(x)
spec_f = self.spec_model(x)
feature = torch.cat([raw_f,spec_f],dim=1)
x=self.fc(feature)

## spectrum is resized to 256x256 for fast speed, 
## other params for mel spectrum is the same with spectrm model.
```
### Preprocess
+ 1. wave/max(wave) if max(wave)>1
### Augmentation
+ XY cut out for spectrum model
### Train
+ 1. Random sample 5 seconds data for trainning, 
first 5 seconds data for validation.
+ 2. Mixup p=1.
+ 3. BCEWithLogitsLoss

### inference
+ openvino

### extra
I make a post to discussion the method feeding raw signal to vision model. Here is the link [good luck](https://www.kaggle.com/competitions/birdclef-2024/discussion/511763)

inference [notebook](https://www.kaggle.com/code/cooolz/5nd-solution)