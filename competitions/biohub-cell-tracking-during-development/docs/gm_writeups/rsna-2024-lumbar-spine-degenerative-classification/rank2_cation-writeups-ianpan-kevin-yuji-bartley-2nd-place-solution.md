# 2nd place solution

First, I would like to express my gratitude to Kaggle and RSNA for hosting this excellent competition.
I'm also grateful to my teammates who worked hard till the end.
Our solution is a simple blend of our individual predictions and small post-processing. My teammates will likely share their solutions in the replies to this post. I'll describe my solution and post-processing below.

inference code: https://www.kaggle.com/code/yujiariyasu/rsna-lumbar-spine-2nd-place-solution

# Summary
My solution is an ensemble of small models. I worked separately on axial and sagittal. Additionally, I created separate models for each target.
Basically, all models predict 3 targets: ['normal_mild', 'moderate', 'severe']. I used models that treat data from different levels and left/right as the same, without considering these distinctions. In the end, I used the team's ensemble oof to remove noisy labal data and retrain the classification model.

# Axial
First, I classify which slices to use for predicting each level. I used @hengck23 code for this - thank you always for your significant contributions!
Next, I estimate the regions within each image to use for severity prediction. I trained YOLOX using the provided data.

<img src="https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F3460291%2Fcdd5586eea2b4020457c4ad8b69b07a4%2F2024-10-18%2013.31.18.png?generation=1729225900120808&alt=media" width="800">

Finally, I trained classification models using ConvNeXt Small. For spinal predictions, I directly use the regions estimated by YOLOX. For non-spinal predictions, I use only the left or right half of the image, allowing me to treat left and right labels equally.

# Sagittal
First, I classify slices suitable for predicting spinal and subarticular targets. I used 2.5D images and a simple Timm model.
Next, I estimate regions for each level within the images. I trained YOLOX using data shared by my teammate @brendanartley - thank you for your excellent contribution!

<img src="https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F3460291%2F4e96087f8f0251caf9d7553da5d55c9f%2F2024-10-18%2013.35.14.png?generation=1729226157298856&alt=media" width="400">

Using boxes, level each level horizontally and then crop.

<img src="https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F3460291%2F81d7c0015e4a5bf975cd54d1e100c6ef%2F2024-10-18%2013.39.21.png?generation=1729226474453903&alt=media" width="400">

Finally, I perform classification using a MIL model that accepts 5 images. The backbone is ConvNeXt Small. For spinal and subarticular, I use 5 slices centered on those predicted in the 1st stage. For foraminal, I use 5 slices centered between the spinal and subarticular slices. Some models use T1/T2 in separate channels, while others use only one.

# noise reduction
My teammate discovered label noise in train dataset, so we removed samples with high loss. Using our ensemble oof (CV: 0.3687), we excluded samples where the difference between the label and the predicted value was 0.8 or greater. Due to imbalanced data, we needed to apply coefficients to the moderate and severe categories. This magic improved our score by 1% on both public and private leaderboards. I came up with this idea just two days before the deadline, so I didn't have time to try various methods or coefficients. There are likely better approaches.

This is a brief overview of my solution. There are many more intricate details that I couldn't include here.

training code: https://github.com/yujiariyasu/rsna_2024_lumbar_spine_degenerative_classification

# Ensemble and Post-processing
I simply weighted-averaged the predictions of each member, then applied post-processing only to spinal predictions.
For each study, I multiplied the highest predicted spinal-severe value among the 5 levels by 1.25.

Again, thanks to my teammates!