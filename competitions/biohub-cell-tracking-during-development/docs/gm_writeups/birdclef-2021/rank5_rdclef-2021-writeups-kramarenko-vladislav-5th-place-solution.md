# 5th place solution

Thanks to the organizers for an interesting contest and @cpmpml for the motivation.

**My decision is based on my public code posted here:**
https://www.kaggle.com/c/birdsong-recognition/discussion/183269
https://github.com/vlomme/Birdcall-Identification-competition

**The basic solution easily gets silver. What I changed:**
- Switched to SED +1%.
- Lowered threshold +1% . 
- Left only birds from region of record +0%
- Changed ensemble averaging +1%

## Pre-processing:
Used log-melspectrograms
n_fft = 1536, sr = 21952, hop_length = 245, n_mels = 224, len_chack 448, image_size = 224 * 448, 5 seconds

## Models:
Ensemble of 14 models (sed_resnet50, sed_resnest50, sed _efficientnet-b0)

## Augmentations:
- For contrast, I raised the image to a power of 0.5 to 3. at 0.5, the background noise is closer to the birds, and at 3, on the contrary, the quiet sounds become even quieter.
- Slightly accelerated / slowed down recording
- Add a different sound without birds(rain, noise, conversations, etc.)
- Added white, pink, and band noise. Increasing the noise level increases recall, but reduces precision.
- With a probability of 0.5 lowered the upper frequencies. In the real world, the upper frequencies fade faster with distance

## Train:
- Used BCEWithLogitsLoss. For the main birds, the label was 1. For birds in the background 0.3.
- Used loss:
```
train_los1 = nn.BCEWithLogitsLoss()(prediction['clipwise_output'], true)
train_los2 = nn.BCEWithLogitsLoss()(prediction["segmentwise_output_max"], true)
train_loss = (train_los1 + train_los2)/2
```
- I didn't look at metrics on training records, but only on validation files (train_soundscapes)
- 20-40 epochs (24 hours on gtx1060)

## Postprocessing
- If there was a bird in the segment, I increased the probability of finding it in the entire file.
- Model ensemble averaging
```
proba1 = proba.prod(axis = 0) ** (1.0/len(proba))
proba = proba**2
proba = proba.mean(axis=0)
proba = proba**(1/2)
proba = (proba + proba1)/2
```

Since the training took a long time and other approaches didn't work, I switched to other competitions and have hardly taught any new models in the last month. It's a shame that a little was not enough, I will try to do better