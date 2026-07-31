# 5th Place Solution — DINOv3 Semantic Segmentation

I would like to thank Kaggle and Recod.ai/LUC for hosting this interesting competition. I finished 5th on the private leaderboard.

**Final scores:**

* Public LB: **0.422**
* Private LB: **0.376**
* [**inference code**](https://www.kaggle.com/code/wowfattie/recodfinal)
* **training code attached at the end**

## Overview

My solution focused on simplicity and robustness. I treated the task primarily as a semantic segmentation problem.

For training mask generation, when an image contained multiple copy-move instances, I used the union of all manipulated regions as the training target. In other words, the model was trained as a binary manipulated-region segmenter rather than as an instance segmentation model.

At inference time, I also did not separate multiple copy-move instances. This was a deliberate simplification.

Two factors contributed most significantly to the final score:

1. Using external datasets to improve generalization.
2. Using the strongest available pretrained DINOv3 backbone.

## Validation

I mainly validated my models on the 48 supplemental images provided by the host, because these multi-panel images were the closest match to the actual test set.

## Backbone and Segmentation Head

I used pretrained DINOv2/DINOv3 backbones with a very simple segmentation head.

The model takes the final patch tokens from the ViT backbone, reshapes them into a 2D feature grid, and applies a single 1×1 Conv2d head to produce one logit map.

During training, I downsampled the ground-truth mask to match the patch-grid resolution:

* 14×14 for DINOv2
* 16×16 for DINOv3

During inference, after predicting the low-resolution mask, I simply resized it back to the original image size using bilinear interpolation.

## Augmentations

The most useful augmentations were simple image-level augmentations:

```python
A.RandomRotate90(p=1.0)
A.HorizontalFlip(p=0.5)
A.GaussianBlur(p=0.15)
A.ToGray(...)
A.HueSaturationValue(...)
A.RandomBrightnessContrast(p=0.9)
A.ImageCompression(p=0.25)
```

I used fairly aggressive color augmentation because the manipulation signal is not always color-specific. The model needed to learn structural and local consistency cues rather than memorize color distributions.

## Important Training Choices

* BF16 autocast
* Full fine-tuning
* FlashAttention-2
* Input image size: 1024 × 1024
* Vanilla BCE loss

## Important Inference Choices

* 8-bit inference to fit within the 16GB memory limit of a T4 GPU
* SDPA attention
* Input image size: 1024 × 1024
* Four-way `rotate90` test-time augmentation

## Post-processing

The post-processing pipeline was simple:

1. Create a high-confidence binary mask using a fixed threshold.
2. Compute the area of this high-confidence mask by summing its active pixels.
3. If the high-confidence area is smaller than `min_area`, classify the sample as authentic and stop.
4. Create the final binary mask using the configured threshold `mask_thr`.
5. If the final mask contains no active pixels, classify the sample as authentic and stop.

## Early Results

I started with a DINOv2 model. Its validation score on the supplemental images was only **0.092**, although the validation score on a split of the provided training set was much higher.

After inspecting the training data, I found that although the dataset was not small, it contained only a limited variety of image types. As a result, the fine-tuned model did not generalize well to the supplemental set, which was visually quite different from the training set.

## External Datasets

To improve generalization, I searched for external datasets that could increase the diversity of the training data. I eventually added the following datasets:

* [CASIA](https://www.kaggle.com/datasets/divg07/casia-20-image-tampering-detection-dataset)
* [GRIP](https://www.grip.unina.it/download/prog/CMFD/)
* [FAU](https://www.cs1.tf.fau.de/research/multimedia-security/code/image-manipulation-dataset/#collapse_1)

Validation results on the supplemental images were:

| Training data                                 | Validation score |
| --------------------------------------------- | ---------------: |
| Competition training set only                 |            0.092 |
| Competition training set + CASIA              |            0.185 |
| Competition training set + GRIP               |            0.095 |
| Competition training set + FAU                |            0.129 |
| Competition training set + CASIA + GRIP + FAU |            0.240 |

Although these external datasets are not biomedical, their diversity helped the model learn more general copy-move detection cues.

## Better Backbones

Because copy-move detection is a difficult task, stronger pretrained backbones made a large difference.

Upgrading the backbone from **DINOv2 Giant** to **DINOv3 Huge+** improved the validation score from the earlier range to around **0.35**. Since these two models are relatively close in size, I believe the improvement mainly came from the stronger pretraining of DINOv3.

I then used the original **DINOv3 7B** model, which further improved the validation score to **0.51**. With additional hyperparameter tuning, the best validation score exceeded **0.56**.

Overall, the final solution remained simple: a strong pretrained ViT backbone, a minimal segmentation head, diverse training data, and lightweight post-processing.