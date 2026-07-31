# 2nd Place Solution

Thanks to organizers for this interesting challenge and congrats everyone who enjoyed it! Since this competition can be approached in a variety of ways, I'm looking forward to see everyone's solution.

# Overview of my solution
My pipeline consists of:
- **Optical Flow to get global(camera) motion.**
- **Ball Detector to find a ball on the field.**
- **Cost Minimization of Ball Trajectory to refine the ball locations over multiple frames.**
- **2-stage Action Recognition to classify the events.**
- **Ensemble and post-processing.**

Once I found that the attention or cropping near the ball is effective to improve the accuracy of action classifier, I focused on refining the performance of the detector and classifier. 

<img src="https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F2938236%2Fa7acfaad0313497aa834f8cc4f6a1987%2F00_solution_summary.png?generation=1665797501214798&alt=media" alt="pipeline" width="650"/>

---

# 1. Optical Flow
It's difficult to detect the ball when:
- it is occluded behind players.
- its background is the stand.
- balls not on play are placed near sideline.

So my idea is to use multiple frames to detect the ball on play which is usually moving by the optical flow.
At first, I implemented the optical flow with high resolution image since balls are very small, but it's slow and not suitable for this competition. So I adopted the low resolution OpticalFlow to predict only the global (camera) motion. Flow itself cannot tell which one is the ball on play, but the difference between warped image and original image is very informative.
<img src="https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F2938236%2F0d77a90b9a5503f2635e8f3acf28aa07%2F01_motivation_opticalflow.png?generation=1665797595915623&alt=media" alt="opticalflow_motivation" width="650"/>
I used RAFT to predict optical flow. It is trained with the provided video by self-supervised manner.
<img src="https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F2938236%2F5251a987e82a7fe227028e05d037cbe9%2F02_implement_opticalflow.png?generation=1665797635231529&alt=media" alt="opticalflow_implementation" width="650"/>

---

# 2. Ball Detection
To make the ball detector, I repeated training and annotation several times. 
(Annotation -> Train Detector -> Predict -> Annotation -> Train Detector -> Predict -> …)
I selected CenterNet because I do not like annotation. Annotating point is much easier than rectangle. 

Since the target(ball) is very small. Inputs to the model is high resolution images.
To achieve both inference speed and accuracy, I used the initial part of the encoder. 
<img src="https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F2938236%2F940076494a1baa149a903c2f85a0a705%2F03_detection.png?generation=1665797710229958&alt=media" alt="balldetector" width="650"/>

Inputs to the detector consist of 9 channels:
- normal RGB image of the current frame
- difference between the current frame and previous frame reconstructed by the optical flow.
- difference between the current frame and next frame reconstructed by the optical flow.

The latter two drastically improved the accuracy of detector especially in the crowded scene.
<img src="https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F2938236%2F2a3a23bd77aa4ffd9b5de7d5d5043601%2F04_detection_example.png?generation=1665797754239777&alt=media" alt="detection_example" width="650"/>

---

# 3. Cost Minimization of Ball Trajectory
Ball location can be estimated more accurately by using multi-frame information. I applied cost minimization on selecting the ball path. 
If we know the trajectory of the ball, we can estimate where it will go next. In the following case, people can select the next point correctly even if the ball is hard to seen.
<img src="https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F2938236%2Fc78dbfa829f5fbfebef0b1eaff79dea5%2F05_motivation_costmin.png?generation=1665797922011099&alt=media" alt="ball_path" width="450"/>

As I want to consider both detector's score and distance based probability, I defined 
  Cost = Function (Confidence of Detector, Distance between Nodes)
and minimized the total cost of the path by Dijkstra Algorithm .

It can create a more natural and accurate trajectory than just connecting points with the highest scores.
<img src="https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F2938236%2F1b6f0c5060a0c92c8cc22b9bd544a2c0%2F06_costminpath.png?generation=1665798032110142&alt=media" alt="ball_path_image" width="650"/>
(If this competition was for the ball detection, I would get first place!)

Videos can be seen here. Left one is the max confidence path and right one is the min cost path.
https://twitter.com/i/status/1582747059165495297

---

# 4. Event Classifier (Action Recognition)
Now we know where the ball is. Next step is to predict event. I used ball trajectory and images cropped near ball over 1 sec to predict action(4 class) and regression of residuals.
Since the labeled intervals are very short, it's difficult to make the long sequential training data. I extracted only 1 sec frames randomly from labeled intervals as an input to the model.

The model consists of :
1) 2D CNN to extract features from cropped images
2) 1D CNN to extract features from ball path
3) 1D CNN to predict event from upper two sets of features 
<img src="https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F2938236%2Fdb149dcd08edcd2d364c9d251cda80db%2F07_eventdet_1ststage.png?generation=1665798262050233&alt=media" alt="event_detect_1st" width="650"/>

---

# 5. Event Classifier [2nd Stage]
The first classifier cannot predict the event with long-term information.
At second stage, long sequential inputs(4 sec) can be adopted because the inputs of the model is only feature vectors which is much more memory-efficient than the images.
<img src="https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F2938236%2Fe6ade8933d3f927f5365f78873d5e71e%2F08_eventdet_2ndstage.png?generation=1665798290526211&alt=media" alt="second_stage" width="650"/>

---

# 6. PostProcessing
Simply, combination of gaussian weight and peak detection.
- Peak detection by max pooling
- Gaussian filter on the “challenge” score because the peak of “challenge” is ambiguous compared to the other two classes.
<img src="https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F2938236%2F8dc5676759cd43223099db8b52ff56ad%2F09_postporc.png?generation=1665798310710848&alt=media" alt="post_process" width="650"/>

---

# Final Thought
Since the number of samples of the events is not so large and the most of the area in a image is useless, I thought the model would hard to learn and would overfit easily. So I soon decided to use the ball trajectory.
That's why I really amazed that many kagglers suceeded the simple end-to-end approach. I've learned a lot and still need to learn a lot from this competition. Thanks again to kaggle community!