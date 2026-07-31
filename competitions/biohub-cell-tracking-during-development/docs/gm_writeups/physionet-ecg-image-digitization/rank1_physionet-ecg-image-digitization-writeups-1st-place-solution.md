# 1st place solution

We would like to thank PhysioNet and Kaggle for organizing this ECG signal prediction competition. Special thanks to @hengck23 for sharing the image rectification pipeline. This pipeline includes Stage 0 (image rotation and homography) and Stage 1 (rectifying homography-transformed images). We have integrated both stages into our workflow and developed a strategy to predict all ECG leads directly from these rectified images.

## Overview

- Prediction based on rectified images
  - Transforming various image styles into a standardized format minimizes inconsistencies from different perspectives. This helps the model ignore spatial distortions and focus on essential heart patterns, resulting in more accurate and reliable ECG predictions.

- Remapping from high resolution image
  - Input images are generated using two different approaches to maximize feature preservation:
  - 1. Homography-based: Rectifying the image at a high resolution before remapping.
  - 2. Direct Remapping: Skips the homography conversion and performs remapping directly from the source.

- Resampling in the Fourier domain
  - Instead of linear interpolation, resampling is performed in the Fourier domain using scipy.signal.resample. This method is better suited for ECG signals.

## Data Preparation
To validate the model, the first ten training samples were used as a validation set.

## Pipeline
### 1. Preprocessing
Two methods were used to generate input images for the prediction model. The first method transforms the rotated image into a 3200x2400 resolution via a homography matrix, followed by grid point remapping. The second method rescales the grid points using the homography matrix and performs remapping directly on the rotated image. In both cases, the lower region containing the ECG signals is cropped for use while the top portion containing personal information is discarded. 

Subsequently, images are converted to grayscale and concatenated with coordinate-based features as model inputs, providing the network with both visual intensity and spatial context to ensure more accurate ECG signal prediction.

![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F3137289%2F61565f7a7b6f09bc3a6c20a028492afb%2Fcrop.png?generation=1769153210601324&alt=media)

### 2. Prediction Strategy 1
Features from the last and second-to-last blocks of the backbone are used as inputs for numerical prediction. Since the ECG images consist of four vertically stacked data groups, these features are partitioned by height into four segments. These segments are then expanded to four times the input image width.

![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F3137289%2Fdc76d19ee753e8560886fe0f94bd42dc%2FStrategy_1_.png?generation=1769153248239656&alt=media)

### 3. Prediction Strategy 2
Features from the last block of the backbone are used as inputs for numerical prediction. Given that the ECG images contain four vertically stacked data groups, the features are partitioned by height into four segments. These segments are then expanded to four times the input image width.

![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F3137289%2F079c86a643208a53d900dc71bed5ba84%2FStrategy_2_.png?generation=1769153261663258&alt=media)

### 4. Prediction Strategy 3
Features from the last block of the backbone are used as the input for numerical prediction. Unlike Strategy 1 and Strategy 2, instead of partitioning features by height, the height dimension is first merged using max pooling. The channels are then split into four segments and expanded to four times the input image width.

![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F3137289%2F98f5b42ba6777ef0ae7e219ef0e2aba1%2FStrategy_3_.png?generation=1769153273732175&alt=media)

### 5. Post-Processing
Based on the model output, the middle 10,000, 15,000, or 20,000 outputs are selected and subsequently resampled. Following this, three types of lead blending are performed based on ECG characteristics and Einthoven’s Law to refine the signals
 - Mix the first quarter of the long II with the II
 
 $$II = (II\times1.35+long II)\div2.35$$

 - Apply 'II = I + III' if np.abs(II-I-III).mean()<0.01

 $$I = (I+(II-III)\times0.6)\div1.6$$
 
 $$II = (II+(I+III)\times0.3)\div1.3$$
 
 $$III = (III+(II-I)\times0.6)\div1.6$$

 - Apply 'aVR + aVL + aVF = 0' if np.abs(aVR+aVL+aVF).mean()<0.01

 $$aVR = (aVR+(-aVL-aVF)\times0.5)\div1.5$$
 
 $$aVL = (aVL+(-aVR-aVF)\times0.5)\div1.5$$
 
 $$aVF = (aVF+(-aVR-aVL)\times0.5)\div1.5$$
 
### 6. TTA
A total of three Test-Time Augmentation (TTA) strategies were implemented. These involve gamma adjustments (1.0, 0.9, and 1.1) for brightness control, alongside cropping and resizing of the Stage 1 inputs.
 - Cropping: The Stage 1 inputs were resized to 1560x1248, followed by a center crop of 1440x1152.
 - Resizing: The input resolution for Stage 1 was resized to 1600x1280.

### 6. Training Details
SmoothL1Loss was used as the loss function, with AdamW as the optimizer and a batch size of 1. The learning rate was set to 1e-4 and managed by a cosine learning rate schedule.

#### Augmentation
Synthetic blank ECG images were generated to prevent the model from predicting anomalous values when encountering occlusions or noise. Random variations were also applied to the ground truth data to increase diversity. Furthermore, Moiré patterns, flares, and overlaid non-rectified images were introduced to increase training difficulty and enhance model robustness.

![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F3137289%2Fc9a0cee2bcd48c7889c77e0cf89eb94e%2FAugmentation.png?generation=1769155658015257&alt=media)

## Results
A total of ten models were ensembled. For validation, 30 images were selected by taking three segment (0006, 0009, and 0012) from ten image id.

The implementation of TTA improved overall performance by 0.07, while the Einthoven’s Law-based correction provided an additional 0.06 boost.

| Model | Validation | LB (Private) | Backbone | Input Size | Output Size | Predication Strategy | Ensemble Weight |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | 26.79 |  | convnextv2_tiny | 5056x1280 | 4x20000 | Strategy 1 | 20 |
| 2 | 26.21 |  | convnextv2_base | 2528x1280 | 4x10000 | Strategy 1 | 32 |
| 3 | 26.50 |  | tf_efficientnetv2_m | 5056x1280 | 4x20000 | Strategy 2 | 30 |
| 4 | 26.03 |  | hgnetv2_b6 | 5056x1280 | 4x20000 | Strategy 2 | 11 |
| 5 | 26.46 |  | mambaout_base | 5056x1280 | 4x20000 | Strategy 2 | 8 |
| 6 | 25.65 |  | caformer_m36 | 2528x1280 | 4x10000 | Strategy 2 | 9 |
| 7 | 26.77 | 22.98 (22.75) | convnextv2_tiny | 5056x1280 | 4x15000 | Strategy 3 | 31 |
| 8 | 26.56 |  | inception_next_small | 5056x1280 | 4x15000 | Strategy 3 | 23 |
| 9 | 26.76 |  | inception_next_base | 5056x1280 | 4x20000 | Strategy 3 | 27 |
| 10 | 26.78 |  | inception_next_base | 5056x1280 | 4x20000 | Strategy 1 | 25 |

### 1. Crossing signal
To evaluate model robustness, ECG-Image-Kit was used to select cases where leads overlap or cross each other. Results demonstrate that even in scenarios where Lead V3 and Lead II partially intersect, the model remains capable of generating accurate predictions.

![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F3137289%2Fdc0e4bfe13abb9b5a57376b14b85797a%2Fresult.png?generation=1769153309536138&alt=media)

### 2. Resampling
SNR differences were compared between post-processing using scipy.signal.resample and linear interpolation. (Only test one image at local)

||  Scipy.signal.resample   | Linear interpolation  |
|  :----:  |  :----:  | :----:  |
|SNR| 23.834  | 22.176 |

### 3. Remapping
SNR differences were compared between remapping from high-resolution and low-resolution images. (Only test one image at local)

||  Remapping from high resolution   | Remapping from low resolution  |
|  :----:  |  :----:  | :----:  |
|SNR| 23.834  | 21.267 |

### 4. Rectification
Two primary enhancements were implemented for the original rectification method to improve output quality
1. Grid point positions are calculated using a weighted average from the feature map to preserve floating-point precision.
2. Surface fitting is utilized to identify and remove outliers, with the resulting gaps filled through interpolation or the regression function itself. This approach ensures smooth image edges and significantly enhances the overall structural integrity of the rectified images.

![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F3137289%2F0ab95dbc80abfa356d2989bb3986c466%2Fr1.png?generation=1769158218152490&alt=media)
![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F3137289%2F1b3c2accc5b039265498c97b45ef7203%2Fr2.png?generation=1769158229129404&alt=media)

## Other
Image 3 was generated by scanning Image 1, but we found offset between grid and signal in Image 3 is different from that in Image 1.

![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F3137289%2Fad3cbbee180c0fcb7dea28f2582fd72b%2Fdiff.png?generation=1769155710423238&alt=media)