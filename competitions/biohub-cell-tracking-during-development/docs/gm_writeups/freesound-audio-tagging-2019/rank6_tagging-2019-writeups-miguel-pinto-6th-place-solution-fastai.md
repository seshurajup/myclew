# 6th place solution fastai

Here are the write-up and code for my solution!

**Blog post:** https://link.medium.com/Kv5kyHjcIX
**Code:** [https://github.com/mnpinto/audiotagging2019](https://github.com/mnpinto/audiotagging2019)

**Summary:**
* Models: xresnets
* Image size: 256x256
* Mixup sampling from a uniform distribution
* Horizontal and Vertical Flip as new labels (total 320 labels)
* Compute loss only for samples with F2 score (with a threshold of 0.2) less than 1.
* Noisy data: ~3500 "good noisy samples" used the same way as curated data
* TTA: Slice clips each 128px in the time axis (no overlap), generate predictions for each slice and compute the `max` for each class.
* Final submissions: 1) Average of 2 models: public LB 0.742, private LB 0.74620; 2) Average of 6 models: public LB 0.742, private LB 0.75421.

**Additional observations:**
* I found better results when using random crops of 128x128 and rescale to 256x256, comparing to random crops of 128x256 and rescale to 256x256, I was expecting the opposite.
* I wonder why max_zoom=1.5 works; I would not expect so.

**Acknowledgements:**
@daisukelab thanks for the code to generate the mel spectrograms! Thanks to everyone that contributed in the discussions or with kernels. And finally, thanks to the organizers for this great competition!