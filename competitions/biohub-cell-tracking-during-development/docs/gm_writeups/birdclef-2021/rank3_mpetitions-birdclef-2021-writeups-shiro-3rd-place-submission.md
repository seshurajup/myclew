# 3rd place submission

First of all, thank you to the Host, the Kaggle Team for having organised such a good competition. I am very happy to have participated in this one.

In the following, I want to explain roughtly what was the approach I used.

**Summary**:

The final solution was an ensembling of 18 checkpoints trained on different CNNS and fold. During training each models are trained on a clip of 20 seconds. A post processing is also applied durgin inference based on the prediction at the "clip level" (20 seconds) and the "segment level" (5 seconds)

**Explanation:**

The first observation we can make is that :
-	For training, we have weak label. So we can use a supervised learning method based on stochastic gradient descent to minimise the loss function.
-	For testing, we are looking to predicting every 5 seconds. Therefore, there is a gap, between the label given to the model during training (sample level) and during inference (5-second segment-level) .

To overcome this issue, I implement  an architecture which allows getting prediction for a clip (20 seconds or more depending of your hardware) and also make prediction every 5 seconds in the clip. 

*Why training on a clip of 20 seconds and not 5 seconds ?* 
The hypothesis I have made is this one : *training on small clip will introduce noise to our training. *
Indeed, we don’t know where and which birds are singing in the 5-second clip , so it is possible to select a clip with nocall for instance. Increasing the length of the clip, allow to reduce this noise.
Then, these clips are divided into segments of 5 secondes. For each of these segment, I create a mel-spectrogram based on the torchlibrosa package. Therefore, for a clip of 20 seconds, 4 spectrograms will be generated. We can use these spectrogram to feed a Deep Learning model.

**Model :**

The architecture of the model is composed of :
-	A backbone based on SED model (CNN) which can use different architectures such as seresnet50, EfficientNetB2, EfficientNetB3, etc with the model coming from this github : https://github.com/rwightman/pytorch-image-models
the backbone has two outputs : the first one is for classifying each segment (one output for every 5 seconds), the second one is for classifying each timestep of the segment. I use the second output  to decide if a bird is present in the 5 seconds or not for inference on test (I took the max probability across the time-step axis for each class). I used the first one for feeding the attention block of my "sequential model" used during training.
-	An attention blocks in order to compute the final output (clip-level prediction) which will be needed for the training. The Attention block is applied on the first output of the backbone (the dimension should be (BS, 4, num_class) )
During the training, we are trying to minimise the output of the attention block with a cross entropy loss. We are trying here to predict the birds present in the 20 seconds clip. I use label smoothing of 0.05 as the label was still noisy. That could help to reduce overfitting. The metric used to select best checkpoint is the PR auc, as F1 score is based on precision and recall, I thought it would be a good idea to use PR AUC for each class then average it in order to have a robust estimation of the model.
During inference, I am using the outputs of each timesteps of the segments. The dimension should be (BS, 4, timesteps, num_classes). I take the max probability across the timesteps dimension and got a vector of (BS, 4, num_classes). These vector corresponds to the prediction for each 5-second segment inside the 20 seconds clip. Then a post-processing is applied, using the final output (prediction at 20-second clip-level).
 
<a href="https://ibb.co/hFfvZ6n"><img src="https://i.ibb.co/nMCSnGV/Processus.jpg" alt="Processus" border="0"></a>

Different CNNs have been trained such as SeResNet50,  EfficientNetB2, EfficientNetB3, EfficientNetB4, EfficientNetB5, EfficientNetB6, EfficientNetB7. My initial goal was to train each of these model with a 5-folds strategy but because of hardware issues(too long to train), I did not train them on all folds. After that, I just ensemble these models by CNN type. 

**Post processing Inference :**

To do that I create two ensemblings : one based on the final output (20 sec clip) and the second one based on the max of the prediction of the timesteps segments. Then, to decide if a bird appeared, I need to look first if the bird appeared in the ensembling's prediction of the max timesteps segments (based on a threshold t1) and if the bird also appeared in the ensembling's final prediction (based on a threshold t2)
 

<a href="https://ibb.co/mF6B9Mx"><img src="https://i.ibb.co/JF5dvMh/ensembling.jpg" alt="ensembling" border="0"></a>

Data augmentation used : mixup (signal and spectrogram worked)

Hardware : GTX 1080TI + Colab