# 4th place solution: Team Cerberus

Thank you to Kaggle, the hosts, and all the competitors. Participating in this exciting competition has been an amazing experience. Here's a look at our 4th-place solution. This achievement was truly a team effort, with equal contributions from @ajobseeker and @tamotamo. I'm grateful to have had the chance to work with them in this competition.

Update (2024-06-23):
Added the inference notebook and training code.
Inference Notebook: https://www.kaggle.com/code/yokuyama/bc24-4th-place/notebook
Train code(melspec models): https://github.com/yoku001/BirdCLEF2024-4th-place-solution-melspec
Train code(raw signam models): https://github.com/tamotamo17/BirdCLEF2024-4th-place-solution-raw-signal

## TL;DR
* Ensemble of Melspec Models and Signal Models
* TTA
* OpenVINO
* Post Processing

## Scores

| Model | Public Score | Private Score | Public Score (+TTA) | Private Score (+TTA) |
| ------------- | ------------- | ------------- | ------------- | ------------- |
| Melspec Model B (inception-next-nano) | 0.668 | 0.623 |  |  |
| Melspec Model A  (rexnet_150)  | 0.676 | 0.641 | 0.690 | 0.649 |
| Melspec Model A (seresnext26ts)| 0.682 | 0.645 | 0.693 | 0.651 |
| Raw signal Model C (tf_efficientnet_b0_ns) | 0.673 | 0.620 | 0.691 | 0.636 |
|   Weighted Mean  |0.717|0.667| 0.731 | 0.676 |
|Weighted Mean + Geometric Mean  ||| 0.732 | 0.677 |
|Weighted Mean + Geometric Mean  + Smoothing ||| 0.741 | 0.685 |
|Weighted Mean + Geometric Mean  + Smoothing + Cut-off (Final Sub) ||| 0.7469 | 0.6877 |
|Weighted Mean + Geometric Mean  + Smoothing + Cut-off + Max With Neighbors||| 0.749 | 0.689 |

* Weighted Mean
    * `0.15*Model B + 0.25*Model A (rexnet_150) + 0.3*Model A (seresnext26ts) + 0.3*Model C`
* Geometric Mean
    * `(0.15*Model B + 0.25*Model A (rexnet_150) + 0.3*Model A (seresnext26ts) + 0.3*Model C) + 0.3*(Model A (rexnet_150) * Model C)**(0.5)`
    * By adding the geometric mean of the MelSpec Model(Model A) and the Raw-signal Model(Model C) to the ensemble, we achieved a slight improvement in our score. Choosing this ensemble at the end allowed us to fortunately remain in the prize-winning positions.
    * We saw a big risk of overfitting, so we decided not to spend any more time adjusting the ensemble weights.

## Model A: 2021-2nd Melspec CNNs
We heavily referenced the code from the [2023 2nd place solution](https://github.com/LIHANG-HONG/birdclef2023-2nd-place-solution) to build our training and inference pipeline for this model. Big thanks to @honglihang for sharing such valuable information. Their codebase was incredibly strong, and with just a few modifications, we were able to create a single model with an LB score of 0.68.

* Dataset
    - BC2024
    - Some models pretrained on 2021, 2022, and 2023's datasets.
    - Random 15-20 seconds from audio at training, first 5 seconds at validation.

* Preprocessing
    - n_mels=128, n_fft=2048, f_min=0, f_max=16000, hop_length=627, top_db=80. 

* Data Augmentation
    - AddBackgroundNoise ([datasets](https://www.kaggle.com/datasets/honglihang/background-noise))
    - Gain
    - Noise Injection
    - Gaussian Noise
    - Pink Noise
    - Mixup
* Model
    - `seresnext26ts`
    - `rexnet_150`
        - 2021-2023 pretrained
* Loss Function
    - BCELoss
    - Class sampling weights proposed by [1st place of 2023 competition](https://www.kaggle.com/competitions/birdclef-2023/discussion/412808).

## Model B: Simple Melspec CNNs

* Dataset
    - BC2024 + xeno-canto-additional-cleaned(see `Validation Strategy` section)
    - Random 5 seconds from audio for training, first 5 seconds for validation.

* Data Augmentation
    - AddBackgroundNoise ([datasets](https://www.kaggle.com/datasets/honglihang/background-noise))
    - Gain
    - Noise Injection
    - Gaussian Noise
    - Pink Noise
    - Mixup
    - [Sumup](https://www.kaggle.com/c/birdclef-2023/discussion/412922)

* Model
    - `inception-next-nano` with attention head
        * InceptionNeXt with the same scaling as ConvNeXt-nano

        ```
        from timm.models.inception_next import _create_inception_next
        from timm.models.inception_next import InceptionDWConv2d
        from timm.models._registry import register_model

        @register_model
        def inception_next_nano(pretrained=False, **kwargs):
            print("inception_next_nano")
            model_args = dict(
                depths=(2, 2, 8, 2), dims=(80, 160, 320, 640),
                token_mixers=InceptionDWConv2d,
            )
            return _create_inception_next('inception_next_nano', pretrained=False, **dict(model_args, **kwargs))
        ```

### Model C: Raw signal CNN
This model is inspired by the HMS [2nd place solution](https://www.kaggle.com/competitions/hms-harmful-brain-activity-classification/discussion/492254). A big thanks to @cooolz! They provided a detailed explanation of [them solution](https://www.kaggle.com/competitions/birdclef-2024/discussion/511535).

* Dataset
    - BC2024
    - Removed duplicate data by referring to  [[this link]](https://www.kaggle.com/code/robbynevels/bc24-duplicate-audio-files/).
    - Added several samples from the minority class using data from xeno-canto.
    - Applied stratified 5-fold cross-validation grouped by author.
    - Classes with fewer than 15 samples were upsampled to 15 samples during training.

* Preprocessing
    - Used the first 5 seconds of each audio sample.
    - Downsampled the audio to half the original rate (from 32000 Hz to 16000 Hz).
    - Reshaped the downsampled audio data from a size of 80000 to 625x128.

* Data Augmentation
    -   Annotated 50 background segments from unlabeled data and added them as background noise.
    -   Gain
    -   Noise Injection
    -   Gaussian Noise
    -   Pink Noise
    -   Random Volume
    -   Mixup
    -   Cutmix
* Model
    - `tf_efficientnet_b0_ns` with SED head
* Loss Function
    - focal loss

## Validation Strategy
Using the training data for validation didn't give us reliable results, so we switched to a synthetic data approach.

1. We sampled files for 40 out of 182 classes from the xeno-canto-additional dataset and cropped the segments where the birds were vocalizing to create a clean dataset.
1. We sampled audio files containing only background noise (without bird calls) from the unlabeled soundscape dataset.
1. We combined the clean dataset and background noise to create a test-like dataset with time-series labels.
1. Using this synthetic dataset, we calculated the ROC AUC score. 

Although this validation method did not perfectly correlate with the LB results, it provided more reasonable outcomes compared to using the first 5-second crop method. For more details, please refer to [this notebook](https://www.kaggle.com/code/yokuyama/quant-valid-synthetic-data/notebook).

## TTA
To enhance the accuracy of our time series predictions, we employed techniques similar to sub-pixel super-resolution. Instead of predicting just the 5-second frames during inference, we also predicted frames shifted by 2.5 seconds. We then combined these results as a TTA. This method helped in refining the overall predictions.

![img](https://raw.githubusercontent.com/yoku001/kaggle-static-resouces/main/img/birdclef2024/zu1.drawio.png)

## OpenVINO + INT8 Post Training Quantization
To speed up our model's inference time, we used OpenVINO. Additionally, we implemented [post-training quantization](https://docs.openvino.ai/2024/openvino-workflow/model-optimization-guide/quantizing-models-post-training/basic-quantization-flow.html) to convert our model to INT8.

For the quantization calibration dataset, we used our model's **training dataset**, applying augmentations like background noise addition and gain changes. We believed that these augmentations would help create a quantized model better suited to handle a wider range of test data scenarios.

The results were impressive: our inference speed improved dramatically, with the quantized model running **30-40%** faster.

When performing quantization, the selection of layers to be quantized was crucial. We observed that excluding the head layers from quantization tended to improve the model's accuracy.

```
names = ['/head/Gemm/WithoutBiases', '/global_pool/Pow', '/global_pool/GlobalAveragePool', '/global_pool/Pow_1', '/global_pool/Clip']
quantized_model = nncf.quantize(
    model, calibration_dataset, subset_size=600,
    ignored_scope=nncf.IgnoredScope(names=names),
)
```

We began working on quantization just three days before the submission deadline, leaving us insufficient time to thoroughly verify the combination of ensemble and quantization. So, we used a quantized model in only one of our two final submissions. (trade-off: we reduced the number of TTA runs for this submission.)

We ended up finding that enabling quantization did not significantly impact the public/private scores.

## Post Processing
By using several tricks, we were able to improve our scores by approximately 0.01 on both the private and public leaderboards.

* Smoothing
    * [Similar to the 6th place team](https://www.kaggle.com/competitions/birdclef-2024/discussion/511527), we improved our scores by taking the moving average of adjacent segments.
* Cut-off
    * Birds that appear once in 4 minutes of audio are more likely to reappear compared to other audio. Recognizing that the probability could be low due to noise, overlapping calls with other birds, and inference slices cut by bin boundaries, we halved the value if the model had a confidence of 0.10 or less in all 48 bins of the 4-minute audio. In other words, we halved the probability if there were no birdsong (0.10 or less) in all 48 sections, and left the number unchanged if there was birdsong at least once in any of the 48 sections.
* Max With Neighbors
    * Select the maximum value including the previous and next two rows. For a 30-second test sample, if the inference values of a label are [0.1, 0.3, 0.5, 0.2, 0.4, 0.1], modify them to [0.5, 0.5, 0.5, 0.5, 0.4]. We conducted many tests with different datasets and found a 25% probability of score decrease, so it was not included in the final submission. 

## What didn't work
* Pseudo labeling using unlabeled data.
* Data cleansing and hand-labeling for training data.
* Using novel loss functions. BCE and Focal Loss performed almost the best.
* Manifold Mixup, D-Mixup
* PCEN
* CWT, CQT, VQT
* Trainable frontends: Leaf, trainable filterbank, trainable stft, Conv1D
* Reparameterized model
* Mobilenet V4
* BirdNET embeddings