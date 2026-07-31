# Solution: Public LB 8th place | Private LB 3rd place (updated 2026-04-28)

Please read the attached pdfs below

*New update after the competition ended*

# Segmentation of image into panels
1. Segment Anything 3 model is prompted with domain-specific text cues to detect individual panels within a composite scientific figure. Multiple prompts are tried in sequence:
   ["outlined scientific images", "microscopic images", "bordered rectangles", ...]
2. Accept the first prompt that returns any segmentation masks.

# Pre-processing of panels
1. Apply CLAHE, which corrects for brightness/exposure manipulations
2. Detect text and arrows which are overlaid on top of the images, and blur them out.

# SIFT + Matching
1. Extract keypoints from each panel using SIFT, which is scale and rotation invariant.
2. Candidate pairs are matched using FLANN approximate nearest-neighbor matcher using the SIFT keypoints.
3. Filter out low-probability candidates by using Lowe's ratio test
4. Use RANSAC (via homography) to filter out bad matches and enforce geometric consistency between panels

# Tightening the regions
1. instead of matching the entire panel, only segment around the tight bounding box of the RANSAC inlier matched keypoints - this handles cases where only a *partial* region of one image is duplicated, rather than the full crop.
2. Remove candidates where there is high spatial overlap between the images (SAM3 detection error of the same region), or the two are vastly different areas (geometric inconsistency)

# Rerun segmentation
1. Re-run SAM3, but now having prompts targeting image context rather than panel structure - seeking biological objects rather than borders. The prompts are:
   ["biological cells", "scientific blobs", "item of interest", ...]
2. If SAM3 finds nothing, then inject deterministic crops of the overall image. This ensures every figure has at least some regions to compare. For example: split image into halves, thirds, quadrants, etc.
3. Re-run previous steps with relaxed parameters:
   CLAHE -> text/arrow suppression -> SIFT -> FLANN -> Lowe's ratio test -> RANSAC -> tighten bounding boxes -> remove rejections

# Interesting findings

No model training was used. Things that did not work:

- training a ViT UNET / segformer / DinoV3 / mask2former to predict duplicates. For that matter, extending training on other copy-forge datasets like Casia, DefactoCopyMove, figshare_wb, kaggle-dsbowl, etc.
- approaches in academic literature did not work for me: BCMNet, Beit Base, Busternet, LBRT
- initially I trained my own custom DETR model to identify where panels were (using noisy training data by saying anything non-white background was part of a panel). However, SAM3 model from Facebook was slightly better. I even tried some unique ideas like Comic Panel Detection NN and DocLayout YOLO DoyLayNet.
- using cellpose to segment cells, then performing matching based on that (too slow)
- other keypoint methods (ROMA V2) are promising but too slow
- sherloq (https://github.com/GuidoBartoli/sherloq) which was only able to match duplicate text
- photoholmes (https://github.com/photoholmes/photoholmes) wasn't good at copy-forge detection
- Forensically (https://29a.ch/photo-forensics/#forensic-magnifier) - i recreated the algorithm in Python, but the hyperparameters are too fiddly to work with. Would probably be valuable as an uncorrelated approach to find more duplicates.
- imagetwin, imachek, proofig, figcheck - these all cost money and essentially the competition is to create free open-source versions of these

I also handlabeled an extra dataset of a few hundred duplicate pairs from PubPeer/RetractionWatch. However, I did not have enough time to use it in my validation set beyond trying out different SAM3 prompts on it.

The key insight to this competition was to look at your model performance on the supplemental test data. Empirically you can see that training a NN simply on (synthetic) individual images does not translate well to the supplemental data which was "figure"-based (multiple panels). Therefore a completely different approach was required