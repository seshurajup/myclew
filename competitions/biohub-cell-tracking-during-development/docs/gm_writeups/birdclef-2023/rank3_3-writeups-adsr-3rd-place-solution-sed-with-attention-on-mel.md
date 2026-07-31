# 3rd place solution: SED with attention on Mel frequency bands

First of all, thanks to the organizers Sohier Dane, Stefan Kahl, Tom Denton, Holger Klinck and all involved institutions (Kaggle, Chemnitz University of Technology, Google Research, K. Lisa Yang Center for Conservation Bioacoustics at the Cornell Lab of Ornithology, LifeCLEF, NATURAL STATE, OekoFor GbR and Xeno-canto).

This year’s competition was a welcome change compared to previous challenges. The cMAP evaluation metric eliminated the need for threshold tuning, while the inference time limit encouraged a focus on efficient models with a good balance between accuracy and speed.

In this post I want to briefly introduce some aspects of my solution. A more detailed description will be provided later as an update or in the upcoming working note.

### Quick summary
- Modified SED architecture with attention on frequency bands
- Addressing domain shift with reverb augmentation 
- Using freezed TorchScipt models and precalculated inputs to speed up inference
- Addressing fluctuating inference time by setting a timer in inference notebook

### Datasets
- 2021/2023 competition data
- [2020 extended xeno-canto data](https://www.kaggle.com/c/birdsong-recognition/discussion/159970)
- [2023 extended xeno-canto data (all files with 2023 species as primary label)] (https://www.kaggle.com/datasets/mariotsaberlin/xeno-canto-extended-metadata-for-birdclef2023)
- BirdCLEF 2019 soundscapes (2021 species only & nocall/noise)
- [DCASE 2018 Bird Audio Detection Task (nocall/noise)](https://dcase.community/challenge2018/task-bird-audio-detection)
- [Some nocall/noise files from datasets of previous competitions/solutions] (https://www.kaggle.com/datasets/theoviel/bird-backgrounds)

### Data preparation
- Convert files to 32 kHz (if necessary)
- Convert extended (downloaded) xeno-canto files to FLAC
- Add duration information for each file to dataset metadata 
- Add first 10 seconds interval of all xeno-canto files to training set
- Split training set into 8 folds (but mostly only 3 folds were used)

### Model input
- Log Mel spectrogram of 5 second audio chunks (n_fft= 2048, hop_length=512, n_mels=128, fmin=40, fmax=15000, power=2.0, top_db=100)
- Normalized to 0…255
- Converted to 3 channel RGB image 

### Model backbone/encoder architectures (from [timm](https://github.com/huggingface/pytorch-image-models))
- tf_efficientnet_b0_ns
- tf_efficientnetv2_s_in21k

I also tried resnet50, resnet152, tf_efficientnet_b2_ns, tf_efficientnet_b3_ns, tf_efficientnet_b4_ns, efficientformer_l3, tf_efficientnetv2_m_in21k, densenet121 and eca_nfnet_l0 but none of those were included in inference ensemble because in my case tradeoff between performance and inference time was not as good as for EffNetB0 or EffNetV2s.

All models used pretrained ImageNet weights and served as feature extractor combined with a custom classification head. As classifier I used a modified SED head with attention on frequency bands instead of time frames. The intuition behind this is, that species in soundscapes often occupy different frequency bands. In original SED architecture, feature maps representing frequency bands are aggregated via mean pooling and attention is applied on features representing time frames. If attention is instead applied on frequency bands it can help to distinguish species vocalizing at the same time but with different pitch. The modification can be achieved simply by rotating the Mel spectrogram by 90 degrees before feeding it to the original SED network.

### Data augmentation (esp. to deal with weak/noisy labels and domain shift between train/test set)
- Select 5s audio chunk at random position within file:
 - Without any weighting
 - Weighted by signal energy (RMS)
 - Weighted by primary class probability (using info from pseudo labeling)
- Add hard/soft pseudo labels of up to 8 bird species ranked by probability in selected chunk
- Random cyclic shift
- Filter with random transfer function
- Mixup in time domain via adding chunks of same species, random species and nocall/noise
- Random gain of signal amplitude of chunks before mix
- Random gain of mix
- Pitch shift and time stretch (local & global in time and frequency domain)
- Gaussian/pink/brown noise
- Short noise bursts 
- Reverb (see below)
- Different interpolation filters for spectrogram resizing  
- Color jitter (brightness, contrast, saturation, hue)

In soundscapes, birds are often recorded from far away, resulting in weaker sounds with more reverb and attenuated high frequencies (compared to most Xeno-canto files where sounds are usually much cleaner because the microphone is targeted directly at the bird). To account for this difference between training and test data (domain shift), I added reverb to the training files using impulse responses, recorded from the Valhalla Vintage Verb audio effect plugin. During training, I randomly selected impulse responses and convolved them with the audio signal with a 20% chance, using a dry/wet mix control ranging from 0.2 (almost dry signal) to 1.0 (only reverb).

I didn’t use pretraining followed by finetuning, instead I trained on all 2021 & 2023 species + nocall (659 classes). Background species were included with target value 0.3. For inference, predictions were filtered to the 2023 species (264 classes).

### Speed up inference and deal with submission time limit

Due to variations in hardware and CPUs used to run inference notebooks, the number of models that could be ensembled varied. To prevent submission timeouts, I set a timer in the notebook to ensure completion within the 2-hour limit. If the timer reached approximately 118 minutes, inferencing was stopped and results were collected for models and file parts predicted up to that point. Results for unfinished models/file parts were masked before averaging predictions. Using this method, I couldn’t determine the exact number of models that could be ensembled. In early submissions, I could only ensemble 3 models without risking timeouts. Later, I prioritized inference speed over model diversity by using models with the same input (no variation in FFT size, number of Mel bands etc.). Now I could precalculate and save Mel spectrogram images to RAM for all test files in advance and reuse those for all models. I also converted models to TorchScript. With these optimizations, I could ensemble at least 7 models, depending on architecture (e.g. 4x EfficientNetB0 + 3x EfficientNetV2s) without setting a timer.

My best single model used an EfficientNetV2s and scored 0.83386 on public leaderboard (0.74104 on private LB). The best single model with highest score on private leaderboard used a ResNet50 backbone (0.7482 private LB / 0.83288 public LB). My best ensemble on private LB (0.76365) was a mix of 8 models (5x EfficientNetB0 + 3x EfficientNetV2s) with simple mean averaging of single model predictions.

### Some things I tried but gave up on because I couldn’t get them to work well enough
- [Model soup] (https://arxiv.org/abs/2203.05482)
- MultiLabelSoftMarginLoss (instead of BCEWithLogitsLoss)
- Knowledge Distillation
- Finetuning using only 2023 species data
- Converting models to ONNX or openvino format (speed up was only achieved for small batch sizes)
- Any postprocessing (e.g. amplify probabilities of detected species in neighboring windows or entire file)

### Citations
- [Kong, Qiuqiang, Yin Cao, Turab Iqbal, Yuxuan Wang, Wenwu Wang, and Mark D. Plumbley. "PANNs: Large-Scale Pretrained Audio Neural Networks for Audio Pattern Recognition." arXiv preprint arXiv:1912.10211 (2019).] (https://arxiv.org/abs/1912.10211)
- [Code for PANNs paper] (https://github.com/qiuqiangkong/audioset_tagging_cnn/)
- [S. Adavanne, H. Fayek & V. Tourbabin, "Sound Event Classification and Detection with Weakly Labeled Data", Proceedings of the Detection and Classification of Acoustic Scenes and Events 2019 Workshop (DCASE2019), pages 15–19, New York University, NY, USA, Oct. 2019] (https://dcase.community/documents/workshop2019/proceedings/DCASE2019Workshop_Adavanne_45.pdf)
- [Introduction to Sound Event Detection by Hidehisa Arai] (https://www.kaggle.com/code/hidehisaarai1213/introduction-to-sound-event-detection/notebook)
- [Lasseck M (2019) Bird Species Identification in Soundscapes. In: CEUR Workshop Proceedings.] (http://ceur-ws.org/Vol-2380/paper_86.pdf)
- [https://xeno-canto.org/] (https://xeno-canto.org/)
- [timm (PyTorch Image Models)] (https://github.com/huggingface/pytorch-image-models)
- [Audiomentations] (https://github.com/iver56/audiomentations)

### Inference Notbook
[https://www.kaggle.com/mariotsaberlin/bc23-3rd-place-solution-refactored] (https://www.kaggle.com/mariotsaberlin/bc23-3rd-place-solution-refactored)

### Working Note
[Lasseck M (2023) Bird Species Recognition using Convolutional Neural Networks with Attention on Frequency Bands. In: CEUR Workshop Proceedings.] (https://ceur-ws.org/Vol-3497/paper-175.pdf)