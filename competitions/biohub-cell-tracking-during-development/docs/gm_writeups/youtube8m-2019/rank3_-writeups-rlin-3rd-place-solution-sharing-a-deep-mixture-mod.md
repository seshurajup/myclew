# 3rd place solution sharing: A Deep Mixture Model with Online Distillation

At first, I would like to thanks google research for providing another interesting video understanding challenge. This competition really provides fun to many of my weekends in the last 4 months.  

Overall, my solution follows the widely-used system design: candidate generation and ranking. ![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F132211%2F38572e71d7518a50136e28e4fca386b5%2Fsystem.PNG?generation=1571208082982881&amp;alt=media)

A quick offline analysis suggests that the top20 topics(among 1000 topics) cover over 97% of the positive labels. The segment level classifier is directly finetuned from video-level classifer(with the same structure).

I found larger model can generally perform better in the video dataset but will quickly overfit the smaller segment dataset. In this competition, I tried another approach to increase model capacity by training multiple models. Our final model is a 2-layer mixture model with online distillation.  Each of the MixNeXtVLAD model is a mixture of 3 NeXtVLAD model. So in total, we trained 12 NeXtVLAD models in parallel using 4 Nvidia 1080 TI GPUs.  The online distillation part can effectively prevent the whole model to overfit the smaller dataset. 

![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F132211%2F6d516e4eb35e009469a87a8176afca78%2Fmix_mix.PNG?generation=1571208477789758&amp;alt=media)
 

![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F132211%2F0c96b3aff80ffdaa1bbbb8a14a41fe5a%2Fmix.PNG?generation=1571209130114152&amp;alt=media)

More details about the model will be included in the research paper and shared in this post once I finish the writing : )

If you are interested in the performance of models I have tried, following are the results:
![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F132211%2Fa082666359eb8ed9f0314a71f0d87a21%2Fresults.PNG?generation=1571705893788412&amp;alt=media)

 Iuse all the available data for training, including the validation set, because the performance on local validation dataset is highly aligned with public LB.