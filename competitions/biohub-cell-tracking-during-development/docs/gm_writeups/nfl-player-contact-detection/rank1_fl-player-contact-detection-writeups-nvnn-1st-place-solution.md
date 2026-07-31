# 1st place solution

Thanks to NFL and Kaggle for hosting this interesting competition.
My approach comprises three main components
- A weak xgb model to remove easy negative samples
- A CNN to classify contact
- A xgb model to post-process the output.

Since my xgb preprocessing was not really good compare to other teams (CV ~ 0.72),I will only elaborate on my CNN and post-processing method in this write-up.
**1. 3D CNN for Video Classification**
***1.1 Input generator***

I separate the modeling and training of player-player (PP) and player-ground (PG) contacts.
The PP model is trained using input from three sources, namely endzone video, sideline video, and tracking data. On the other hand, the PG model is trained using input from only two sources, namely endzone video and sideline video. Notably, including tracking data does not result in improved performance for the PG model.

***1.1.1 Input generator for PP model***

The endzone and sideline videos are processed similarly. Firstly, I extract 18 images from neighboring frames, namely {frame[-44], -37, -30, -24, -18, -13, -8, -4, -2, 0, 2, 4, 8, 13, 18, 24, 30, frame[37]}. The frame[-44] represents 44 frames prior to the current sample's estimated frame. This sampling technique enables the model to observe more frames close to the estimated frame. 

Next, I mask the players' heads in contact with a black or white circle to guide the model's attention to the relevant players. Rather than using an additional channel, I mark the players' heads directly into the image. I made this decision to maintain the input's 3-channel format, which maximizes the utilization of the pretrained weight file. Finally, I crop each image around the players' contact area using a crop size of 10 times the mean helmet box size within the specified frame range.

To enable the tracking data to be stacked with images from the endzone and sideline, I simulate the tracking data as images. To accomplish this, I use the OpenCV cv2.circle function to plot each player's position in a specific step on a black background. I assign two different colors to represent the two teams, and players in contact are depicted with bigger and brighter circles (radius is 5, and pixel value is 255), while background players are depicted with smaller and darker circles (radius is 3, and pixel value is 127). By integrating this information into the input, the model can learn the interaction of all players from a bird's eye view. The input to the PP model is displayed in the GIF below.

![](https://i.ibb.co/sKJ5zHP/output.gif)

***1.1.2 Input generator for PG model***
The endzone and sideline videos are processed similarly to the PP model, with the exception that the PG model uses a longer input sequence of 23 neighboring frames, ranging from [-54, -48, -42, -36, -30, -24, -18, -13, -8, -4, -2, 0, 2, 4, 8, 13, 18, 24, 30, 36, 42, 48, 54]. 

The PG model does not include simulated tracking images as they do not improve the PG CV score. 

Unlike the PP model, I can use a longer sequence of images in the PG model because the tracking images are not included. In the PP model, the maximum sequence length that can fit into my GPU is 18 images.

***1.2 Model***
Given that the input appears to resemble an action classification task rather than a standard 3D classification, I opted to use an action recognition model to address this problem. After reviewing the [mmaction2 repository](https://github.com/open-mmlab/mmaction2), it became clear that the CSN series delivered the most impressive results in the Kinetics-400 dataset. As a result, I selected resnet50-irCSN and fine-tuned it for this particular task.

***1.3 Training***

During training, I apply the following augmentations to the endzone and sideline images and randomly swap them. As for the tracking images, I only use horizontal and vertical flips as augmentations.
```
base_aug = [
        A.RandomResizedCrop(always_apply=False, p=1.0, height=cfg.img_size, width=cfg.img_size, scale=(0.7, 1.2), ratio=(0.75, 1.3), interpolation=1),
        A.OneOf([
            A.RandomGamma(gamma_limit=(30, 150), p=1),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.3, p=1),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, p=1),
            A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=1),
            A.CLAHE(clip_limit=5.0, tile_grid_size=(5, 5), p=1),
        ], p=0.6),
        A.HorizontalFlip(p=0.5), 
        A.ShiftScaleRotate(shift_limit=0.0, scale_limit=0.1, rotate_limit=15,
                                        interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_CONSTANT, p=0.8),
        A.Cutout(max_h_size=int(50), max_w_size=int(50), num_holes=2, p=0.5),
    ]

cfg.train_transform = A.ReplayCompose(base_aug)
```

I used a linear scheduler for the learning rate and trained the model for one epoch. In the final submission, I trained the model using all available data with 4 seeds.

**2. XGB Postprocessing**
I employed a simple xgb model to combine the predictions of pre-xgb and cnn. Through experimentation, I discovered that the optimal feature for post-processing in PP and PG models slightly differs.

***2.1 PP postprocessing***
First, I calculated an ensemble probability from the CNN and preprocessing xgb model as follows: prob = 0.2pre_xgb_prob + 0.8cnn_prob. 
Then, I used the probability from the 20 neighboring steps as features for the xgb model, i.e., {prob(-10), prob(-9), ..., prob(0), prob(1), ..., prob(9)}, where prob(-10) represents the probability of the same pair of players in the prior 10 steps.
This postprocessing method improved my PP CV score by approximately 0.005.

***2.2 PG postprocessing***
 I calculated an ensemble probability from the CNN and preprocessing xgb model as follows: prob = 0.15pre_xgb_prob + 0.85cnn_prob. 
The feature to xgb model are 
- The ensemble probability from the 30 neighboring steps {prob(-15), prob(-14), ..., prob(0), prob(1), ..., prob(14)},  
- The pre_xgb_prob and cnn_prob from the 20 neighboring steps.
This postprocessing method improved my PG CV score by approximately 0.04.

P/S. Thanks chatGPT for making my explanation better!!