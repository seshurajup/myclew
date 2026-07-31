# Summary
My solution consists of three EfficientNet-B7 models (I used the Noisy Student pre-trained weights). I did not use external data, except for pre-trained weights. One model runs on frame sequences (a 3D convolution has been added to each EfficientNet-B7 block). The other two models work frame-by-frame and differ in the size of the face crop and augmentations during training. To tackle overfitting problem, I used mixup technique on aligned real-fake pairs. In addition, I used the following augmentations: AutoAugment, Random Erasing, Random Crops, Random Flips, and various video compression parameters. Video compression augmentation was done on-the-fly. To do this, short cropped tracks (50 frames each) were saved in PNG format, and at each training iteration they were loaded and reencoded with random parameters using ffmpeg. Due to the mixup, model predictions were “uncertain”, so at the inference stage, model confidence was strengthened by a simple transformation. The final prediction was obtained by averaging the predictions of models with weights proportional to confidence. The total training and preprocessing time is approximately 5 days on DGX-1.
# Model and code
You can use my [code](https://github.com/NTech-Lab/deepfake-detection-challenge) to reproduce the result.
# Key ingredients
## Mixup on aligned real-fake pairs
One of the main difficulties of this competition is a severe overfitting. Initially, all models overfitted in 2-3 epochs (the validation loss started to increase). The idea, which helped a lot with the overfitting, is to train the model on a mix of real and fake faces: for each fake face, we take the corresponding real face from the original video (with the same box coordinates and the same frame number) an do a linear combination of them. In terms of tensor it’s  
```python
input_tensor = (1.0 - target) * real_input_tensor + target * fake_input_tensor
```
where target is drawn from a Beta distribution with parameters alpha=beta=0.5. With these parameters, there is a very high probability of picking values close to 0 or 1 (pure real or pure fake face). You can see the examples below:  
![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F409174%2Fb2c49a9ba741584da43e34306ea65505%2Fmixup_example.jpg?generation=1592040431827143&amp;alt=media)
Due to the fact that real and fake samples are aligned, the background remains almost unchanged on interpolated samples, which reduces overfitting and makes the model pay more attention to the face.
## Video compression augmentation
In the paper \[1\] it was pointed out that augmentations close to degradations seen in real-life video distributions were applied to the test data. Specifically, these augmentations were (1) reduce the FPS of the video to 15; (2) reduce the resolution of the video to 1/4 of its original size; and (3) reduce the overall encoding quality. In order to make the model resistant to various parameters of video compression, I added augmentations with random parameters of video encoding to training. It would be infeasible to apply such augmentations to the original videos on-the-fly during training, so instead of the original videos, cropped (1.5x areas around the face) short (50 frames) clips were used. Each clip was saved as separate frames in png format. An example of a clip is given below:  
![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F409174%2F621e97ea700ccd2e6930ca18835936f0%2Fclip_example.jpg?generation=1592040594526237&amp;alt=media)
 For on-the-fly augmentation, ffmpeg-python was used. At each iteration, the following parameters were randomly sampled (see \[2\]):
- FPS (15 to 30)
- scale (0.25 to 1.0)
- CRF (17 to 40)
- random tuning option
## Model architecture
As a result of the experiments, I found out that the EfficientNet models work better than others (I checked ResNet, ResNeXt, SE-ResNeXt). The best model was EfficientNet-B7 with Noisy Student pre-trained weights \[3\]. The size of the input image is 224x192 (most of the faces in the training dataset are smaller). The final ensemble consists of three models, two of which are frame-by-frame, and the third works on sequence.
### Frame-by-frame models
Frame-by-frame models work quite well. They differ in the size of the area around the face and augmentations during training. Below are examples of input images for each of the models:  
![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F409174%2F6b85a72e94d2d33b9f92393ff1ab702a%2Ffirst_and_second_model_inputs.jpg?generation=1592040647615387&amp;alt=media)
### Sequence-based model
Probably, time dependencies can be useful for detecting fakes. Therefore, I added a 3d convolution to each block of the EfficientNet model. This model worked slightly better than similar frame-by-frame model. The length of the input sequence is 7 frames. The step between frames is 1/15 of a second. An example of an input sequence is given below:  
![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F409174%2Fb92e6241ef295c6f719fd4987144c58f%2Fthird_model_input.jpg?generation=1592040677053674&amp;alt=media)
## Image augmentations
To improve model generalization, I used the following augmentations: AutoAugment \[4\], Random Erasing, Random Crops, Random Horizontal Flips. Since I used mixup, it was important to augment real-fake pairs the same way (see example). For a sequence-based model, it was important to augment frames that belong to the same clip in the same way.  
![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F409174%2F1a8e42e972ea9a755cd29ef49217ffb2%2Faugmented_mixup.jpg?generation=1592040717314961&amp;alt=media)
## Inference post-processing
Due to mixup, the predictions of the models were uncertain, which was not optimal for the logloss. To increase confidence, I applied the following transformation:  
![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F409174%2F747b96390d72602fec15438d627e92ed%2Fpred_transform.jpg?generation=1592040745015626&amp;alt=media)
Due to computational limitations, predictions are made on a subsample of frames. Half of the frames were horizontally flipped. The prediction for the video is obtained by averaging all the predictions with weights proportional to the confidence (the closer the prediction to 0.5, the lower its weight). Such averaging works like attention, because the model gives predictions close to 0.5 on poor quality frames (profile faces, blur, etc.). 
## References
\[1\] Brian Dolhansky, Russ Howes, Ben Pflaum, Nicole Baram, Cristian Canton Ferrer, “The Deepfake Detection Challenge (DFDC) Preview Dataset”  
\[2\] [https://trac.ffmpeg.org/wiki/Encode/H.264](https://trac.ffmpeg.org/wiki/Encode/H.264)  
\[3\] Qizhe Xie, Minh-Thang Luong, Eduard Hovy, Quoc V. Le, “Self-training with Noisy Student improves ImageNet classification”  
\[4\] Ekin D. Cubuk, Barret Zoph, Dandelion Mane, Vijay Vasudevan, Quoc V. Le, “AutoAugment: Learning Augmentation Policies from Data”