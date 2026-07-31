# 1st Place Solution for the Vesuvius Challenge - Surface Detection Competition

Thanks to the organizers for this fun and important competition, and congratulations to all the winners!

## Context

- __[https://www.kaggle.com/competitions/vesuvius-challenge-surface-detection/overview](https://www.kaggle.com/competitions/vesuvius-challenge-surface-detection/overview)__
- __[https://www.kaggle.com/competitions/vesuvius-challenge-surface-detection/data](https://www.kaggle.com/competitions/vesuvius-challenge-surface-detection/data)__

## Overview of the approach
Our solution consisted of an nnU-Net ensemble plus post-processing.

## Details of the submission

### nnU-Net
First, we would like to appreciate @jirkaborovec 's work! His [notebook](https://www.kaggle.com/code/jirkaborovec/surface-nnunet-training-inference-with-2xt4) served as the starting point of our training pipeline.

#### Single Model Strategy

- We used all available data for training and built our baseline nnU-Net model (Model 1) with the following settings:

```
patch size: 128
batch size: 2
epochs: 4000
```

- We then fine-tuned Model 1 with larger patch sizes of 192 and 256, training for 250 epochs each.

- Due to the extension of the competition, we trained additional models from scratch (for 4000 epochs) with patch sizes of 160, 192, and 224 to increase model diversity. Among these, the 192-patch model was included in our final ensemble.

- As for the single model performance, , and two of our best single models are (due to submission quota, we only did basic post processing on single model submission for A/B tests, so full post processing may produce better results):

| Setting | Public LB | Private LB |
|----------|-----------|------------|
| fine-tuned 192-patch model at 250 epochs    | 0.577     | 0.614     |
| from-scratch 192-patch model at 4000 epochs    | 0.587     | 0.613     |

#### Ensemble Strategy

We prepared two sets of 4-model ensembles as our final submissions:

- **Set 1:** Baseline 128-patch Model 1 (weight: 0.12) + fine-tuned 192-patch model (0.28) + fine-tuned 256-patch model at 100 epochs (0.18) + fine-tuned 256-patch model at 250 epochs (0.42).
- **Set 2:** Baseline 128-patch Model 1 (weight: 0.12) + fine-tuned 192-patch model (0.28) + fine-tuned 256-patch model at 250 epochs (0.18) + from-scratch 192-patch model at 4000 epochs (0.42).

For each ensemble, we fused the post-softmax probabilities using the assigned model weights and applied different thresholds and post processing methods (will be discussed in the following section).

| Ensemble | Public LB | Private LB | Threshold |
|----------|-----------|------------|-----------|
| Set 1    | 0.613     | 0.620      | 0.20      |
| Set 2    | 0.606     | 0.627      | 0.26      |

### Post-processing
As discussed in many of the forum posts, minimizing holes and cavities in the predicted masks was essential for getting a good score.
To that end, we applied five post-processing steps:

&#48;. We removed mask components that have less than 20K voxels.

Then for each sheet (where a sheet is a connected-component in a predicted mask), we did the following:

&#49;. Applied scipy.ndimage.binary_closing(), with a spherical footprint of radius 3. This closes holes and cavities of radius 3 or less.

&#50;. Patching: To repair larger holes, we represented each sheet as a height map, and filled any gaps in the height map by linear interpolation across the gap. We did this separately along the x and y directions, and averaged the two results, weighted by distance to the nearest gap-edge. The projection axis for creating the height map was chosen to be the one that gave the largest projected sheet area. This method is capable of significant repairs: 
   
![](https://github.com/Paul-G2/VesuviusSurfaceDetection/raw/main/PatchHoles03.jpg)

&#51;. We created a simple function to plug any small (1-voxel) remaining holes. For each voxel in a given sheet, we look at it's 2 x 2 x 2 neighborhood
and add whatever voxels are needed to make that neighborhood 6-connected watertight. For example:

![](https://github.com/Paul-G2/VesuviusSurfaceDetection/raw/main/Cubes.jpg)

(This method was inspired by the scikit-image euler_number() function, which uses such neighborhoods to compute the Euler number.) We don't have a proof that our function plugs all 1-voxel holes, but it seems to work well in practice. The algorithm is implemented as a lookup table, with 256 entries for the 256 possible neigborhoods. To create the lookup table, we used a very high tech scotch-tape-and-paper model to visualize all the cases:

![](https://github.com/Paul-G2/VesuviusSurfaceDetection/raw/main/PaperCubes1.jpg)

(Ordinary dilation will also plug holes, but it tends to add too many voxels, which can degrade the dice score.)

&#52;. Finally,  we called scipy.ndimage.binary_fill_holes() on the whole mask, which fills arbitrary-size cavities.

Our best version of this post-processing pipeline also included a check of the number of holes before and after patching, because the patching can actually introduce small holes, for example when the sheet is very curved. In such cases we discard the patch. 

The following table shows the cumulative effect of each post-processing step on our best nnU-Net ensemble:

| Step | Public Score | Private Score |
|-------|--------------|---------------|
| No post-proc | .572 | .596 |
| Remove small components | .586 | .614 |
| Plug small holes | .598 | .622 |
| Patch large holes | .601 | .625 |
| Binary closing | .606 | .627 |
| Fill_holes | .606 | .627 |

## What didn't work
Unfortunately, we did not find an effective solution to the problem of touching sheets. We just relied on nnu-Net to minimize their occurrence.

## What we misled by public LB

- logits fusion performs better on private LB, but we trusted (overfitted) on probability fusion
- larger threshold (Like 0.35-0.4 is the best searched value on our model on training set) performs better on private LB

## How Vibe Coding helps

Initially, our ensemble pipeline got resources issues (GPU memory, hard disk space, ...) even with 2 models ensemble, ChatGPT helps us redesign the pipeline, and enables 4 models ensemble.

## Sources

- __[https://www.kaggle.com/code/jirkaborovec/surface-nnunet-training-inference-with-2xt4](https://www.kaggle.com/code/jirkaborovec/surface-nnunet-training-inference-with-2xt4)__
- [0.627 solution notebook](https://www.kaggle.com/code/tonylica/nnunet-4-model-7-5-2-1-final-submit-so-long?scriptVersionId=300373304)