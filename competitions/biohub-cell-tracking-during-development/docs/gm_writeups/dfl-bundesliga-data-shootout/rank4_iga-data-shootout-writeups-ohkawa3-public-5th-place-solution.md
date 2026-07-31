# Public 5th place solution

Thank you for organizing very exciting competition.
And also the following codes helped me a lot, thank you very much! @its7171
https://www.kaggle.com/code/its7171/dfl-benchmark-training  
https://www.kaggle.com/code/its7171/dfl-benchmark-inference

# Overview
+ Image grayscaling, resizing (1024x576), center crop (960x512)
+ Difference from previous frame and apply absolute
  + Due to the small amount of training data, I thought it would be useful to explicitly extract video features.
+ 3 class classification (binary cross entropy) by efficientnet_b1 with input of 11 channels in total (±5 frames before and after)
+ 5-models ensemble

# Augumentation
+ Fliplr and random crop from 1024x576 to 960x512
+ Affine transformation with parameters varying linearly with time
  + I wanted to synthesize camera zoom, pan, etc. and use them for model training.
  + Assuming that the camera parameters vary linearly within a small range, I created the following function

  ``` python
  def hoge(src_imgs):
    # len(src_imgs)==11+1 frames

    # sampling parametors
    angle = random.uniform(-0.1, 0.1)
    translate_x = random.uniform(-1, 1)
    translate_y = random.uniform(-1, 1)
    scale = random.uniform(-0.0005, 0.0005)

    dst_imgs = torch.zeros_like(src_imgs)
    for i,t in enumerate(-6,6):
        dst_imgs[i] = torchvision.transforms.functional.affine(src_imgs[i],
                                                angle=float(t*angle),
                                                translate=(t*translate_y, t*translate_x),
                                                scale=1.0 + t*scale,
                                                interpolation = torchvision.transforms.InterpolationMode.BILINEAR)
    return dst_imgs
  ```

  The left column in the figure below shows 11 frames of video before augment, 
  and the center and right columns are examples of the augment applied.
 
![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F1630583%2F33870c3bcd56543625c2009f2e27af43%2Faugument100000.gif?generation=1665905790619453&alt=media)

![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F1630583%2Fb292f8832e026f08f0eb4d655baef581%2Faugument175000.gif?generation=1665906158387381&alt=media)

![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F1630583%2F3b9a6d71b4729efab357078adb0471ff%2Faugument075000.gif?generation=1665906759837057&alt=media)

# Label design
For the labels, I used a Gaussian distribution (sigma=20 frames) as shown below. 
These soft label designs are same for the three events.
Binary cross-entropy was used for the loss function.

![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F1630583%2F38f26f2c7f32dff2713f658fdfdd189a%2Flabel_design.png?generation=1665906218833070&alt=media)

# Ensemble.
Ensemble of 5 efficientnet_b1 models with different CVs.

## Additional annotations
It was the last four days and I was frazzled.
As a last ditch effort, I tried to annotate all the unannotated time.
After annotating 5 videos (1606b0e6_0, 407c5a9e_1, 4ffd5986_0, 9a97dae4_1, ecf251d4_0) I ran out of steam and started training the model.

# Speeding up inference
I spent quite a bit of time trying to parallelize cv2.VideoCapture with Dataloader, but it did not work.
I also tried decoders such as decord, dali, pyav, howevwer these did not work either.
Finally, I used cv2.VideoCapture for video encoding,
and grayscaling and resizing of the image were applied on CPU (np.array),  and then difference, absolute , creating input-batch  (rolling & copy) were done on the GPU (nn.sensor).

This speed up allowed us to infer on 6 models (efficientnet_b1 with 11x960x512 tensor) within the time limit.
Due to make margins, we submitted an ensemble of five models, finally.

# Scores
|  submit         |  LB  |
| ----            | ---- |
|  ①： my baseline    ( 512x288 )    |  0.625  |
|  ②： ① + larger iamge ( 1024x576 )   |  0.717  |
|  ③： ② + affine augument   |  0.748  |
|  ④： ③ + 5-model ensemble  |  0.768  |
|  ⑤： ④ +  extra annotation  |  0.792  |