# 2nd place solution

Thanks to Kaggle, the hosts, and our fellow competitors for this very interesting competition. In the following, we want to give a rough overview of our 2nd place solution. We joined this competition in the last 3 weeks and worked hard to fill the knowledge gap to last and previous years’ participants. 

As always, this has been an incredible team effort and has been equal contribution by @christofhenkel, @ilu000 and @philippsinger - I was just the lucky one winning the roll to post the solution :)

**TLDR**

Our solution is an ensemble of several CNNs, which take a mel spectrogram representation of a 30 sec wav-crop as input. We used mixup and added background noise as an augmentation method to improve generalization of our models. For inference, we predict on 5 sec snippets and refine the result by a binary bird/nobird classifier and postprocessing to account for metadata.

**Validation**

I am sure most participants are aware that a robust validation setup is quite difficult in this competition given the fact that test contains different species, and specifically also two additional sites for which we have no validation labels at all. We still tried to come up with a somewhat robust validation setup. 

All our models are only fit on short clips and we always evaluate on train soundscapes. That means that the idea of having multiple folds is redundant here and we basically have only one full validation set containing all soundscape files. 
One thing we quite soon noticed was that if you evaluate on full soundscapes, the validation F1 score is significantly higher than on LB. Our final validation on the full soundscapes was close to 0.84. We figured that this has mostly to do with the presence of 3 full songs in validation that do not contain any calls at all. We also saw on the sample submission score that at least public LB contains more birds than the full soundscape dataset would suggest. So as a first step, we mostly focused on evaluating all but these three songs for our validation score, let’s call it CV-3 (~0.81).

To make it even more robust, we decided to introduce bootstrapping with the following steps:

- Remove 3 songs without calls
- For k times (e.g., 10) sample 80% of the remaining songs - this should emulate the full test dataset (public+private)
- Apply any kind of threshold selection technique, post processing, etc. on this data as we have to do the same when submitting (as we have a combination of public / private there and don’t know what is what).
- For j times (e.g., 50) sample 65% of the remaining songs - this should emulate the private test dataset.
- Calculate the score on each of these j samples.
- Report average, median, min, max, std scores across all k times j (e.g., 500) subsets

This is how such an evaluation then looks like:

![](https://i.imgur.com/4yOisgS.png)

**Code Pipeline and data setup**

We used github for code storage and versioning and neptune.ai for logging and sharing our experiments. To reduce CPU bottleneck we could have preprocessed mel spectrograms to disk, but in order to be flexible with respect to trying different hyperparameters we instead performed mel spec transformation on GPU using [torchaudio](https://pytorch.org/audio/stable/index.html). We also did mixup augmentation on the GPU and used mixed precision training to further speed up runtime. For all models we used pytorch with CNN backbones from [timm](https://github.com/rwightman/pytorch-image-models/). 

**Binary classifier**

We trained a binary classifier to predict bird / no bird in order to try various ideas with respect to pre- and postprocessing. In the end, we only use it for one postprocessing step. For this, we used 3 datasets containing binary labels of 10sec recordings (freefield1010, warblrb10k, BirdVox-DCASE-20k) available [online](http://dcase.community/challenge2018/task-bird-audio-detection). The model is very similar to SED model used in several past solutions.

Backbones: seresnext26t_32x4d, tf_efficientnet_b0_ns

**Bird classifier**

Our models were pretty similar and were all trained on 30 sec random crops of the train_short data. 30 seconds was beneficial as we do not know where the labels are (weak labels). To account for the 5sec snippet format of test data, we reshaped the 30sec crops into 6x 5sec parts before feeding through the backbone. After the backbone we reshaped the data again to re-arrange to the 30sec representation by concatenating the respective time segments and then used simple pooling of time and frequency dimension before forwarding through a simple one layer head which gave us the 398 bird classes. We naively used the union of primary and secondary label as target. For inference then we directly fed 5sec snippets to the model.

![](https://i.imgur.com/M81KcGr.png)

We used the following backbones: resnet34, tf_efficientnetv2_s_in21k, tf_efficientnetv2_m_in21k, eca_nfnet_l0

We trained with BCE loss using Adam optimizer and cosine annealing schedule. We saw improvements using the following tricks:

- Use the rating for weighting the recordings contribution to the loss. The assumption is that recordings with a lower rating have worse quality with respect to audio and label and should contribute less to model training. In detail we weight each sample by rating/max(ratings).
- Label smoothing. We used label smoothing to account for noisy annotations and absence of birds in “unlucky” 30sec crops.
- Clever augmentation. Similar to past solutions we used no-bird background noise and mixup as main augmentation methods. For background noise we used a mix of no-call parts of this years validation set and past years data. We also not only used mixup between recordings but also within a recording by mixing between the 5 sec parts. In mixup we also weight the labels and sample weights accordingly.

**Ensembling**

The ensembling of our models was straightforward since all output the same shapes. We took a simple mean of the predictions after a step of post-processing which is explained in the next paragraph. At the end we used 9 models which differ mostly on hyperparameters and backbones and fitted each model with 6 different seeds. Our final kernel ran in approximately 1h, so there was still quite some room in the kernel.

**Post processing**

The first step for post processing involved choosing an appropriate threshold for making hard predictions for which birds are present in a 5 second segment in soundscapes. As we all know, given the f-score metric, this is one of the most crucial steps of the solution. Even though optimizing a hard threshold on validation and applying it on LB worked quite well, we understood that there are some issues with that approach.

First, we quickly realized that test and regular validation had different proportions of nocalls and calls which was also apparent from the different sample submission scores (only nocalls). This means that in general you wanted to predict more birds on LB meaning lowering thresholds to a certain degree could be helpful for improving public LB. We also accounted for this imbalance in our validation setup by removing the three nocall songs (see above, CV-3).

Second, choosing hard thresholds can be problematic when you introduce new blends to your solution. Each new model has certain shifts in probabilities for all and certain birds, so the global thresholds can shift quite a bit. Now it became hard for us to properly judge if new models work well in the blend on validation and LB based on the merit of the models, or only based on some arbitrary probability / threshold shifts that emerged from it. And it was unclear what is a result of random fluctuation, or model properties.

To that end, we decided to move to a percentile based thresholding approach. In detail, this meant that we set a certain percentile of predictions we want to do on a validation or test set, and calculated the according threshold that way. We did this by flattening all predictions, and then calculating the threshold. On CV-3 this looked for example like that:

``threshold = np.percentile(y_preds.flatten(), 0.9987)``

The more birds a set contains, the lower the percentile can be if predictions are decently ranked. The good thing now with this approach was that we could keep the percentile stable, and just exchange models, blends and other post processing and if the quality in our ranking of predictions improved, also the score improved given this fixed percentile, because we always predict the same amount of records.

After we had this setup, we played a bit with changing the percentile on LB to check how test differs in that sense. We found the optimum on public LB to be at around 0.9980 meaning that quite a few more birds are present. In our final sub we chose 0.9981 and made another gamble with 0.9973. The better sub was clearly 0.9981, and actually even a bit higher could have neted us a potential first place (closer to best percentile on validation).

In theory the gamble was legit, because private LB even had more birds as imminent from sample submission. But at the same time it seems that the ranking of predictions was worse, so that lower percentiles introduce too many FPs, meaning that more conservative setting was better. By and large, our choice based on a combination of validation and LB was a very robust one in the end, and we believe that this percentile based approach was way more stable and robust than individual threshold optimization.

Additionally, we employed several smaller post processing steps to improve the predictions including attempts like: (1) increasing the probability of birds in songs based on their average prediction probability, (2) smoothing neighboring predictions, or (3) adjusting predictions by the predictions from our binary models. We also removed some unlikely predictions based on distance in space and time given the metadata very similar to how 4th place did.

**What did not work**

In the end quite a few things we tried ended up in the blend fostering the diversity in it. But naturally, there are also many different things that did not work. One thing to note is TTA which we could not make work. We had quite some time left in the kernel runtime, so this was a natural area to explore, but TTA with mel spectrograms is not as straightforward as with usual CV data. Furthermore, we tried to explore pseudo tagging in different versions, but also could not improve our blends with it.

Thanks for reading. Questions are very welcome. 
Christof, Pascal & Philipp