# 19th Place Solution

First off, a huge thank you to the hosts for organizing such an insightful competition and to the Kaggle team for providing a great experience. Congratulations to all the winners!

This was my first time seriously participating in a Kaggle competition, and I learned a tremendous amount about Data Science throughout the process. My approach relies heavily on papers regarding pedestrian trajectory prediction and previous competition writeups. I hope this summary helps other entry-level Kagglers as much as the community helped me.

## **Summary**

My solution consists of an ST-Transformer for encoding and a TXP-CNN for decoding. The core challenge of this competition was handling the uncertain and multimodal nature of future trajectories while minimizing RMSE. To address this, I focused on modeling uncertainty directly in the loss function.

## 1. **Confidence Head & Loss Function**

I believe this was the most critical component of my solution.  I assumed the future trajectory distribution follows a Gaussian distribution and attached a "Confidence Head" to the model to predict the variance (σ^2) alongside the coordinates. The model was trained using Gaussian Log Negative Likelihood (NLL) Loss:

$$\mathcal{L} = \sum_i \left( \frac{(\hat{x}_i - x_i)^2}{2\hat{\sigma}_x^2} + \frac{(\hat{y}_i - y_i)^2}{2\hat{\sigma}_y^2} + \log \hat{\sigma}_x + \log \hat{\sigma}_y \right)$$

This method is inspired by the [LEAP Competition](https://www.kaggle.com/competitions/leap-atmospheric-physics-ai-climsim) writeups.
In my experiments, this method outperformed RMSE, Laplacian NLL, Huber with temporal decay and other multimodal prediction approaches.

 I attribute this success to the model's ability to dynamically encode uncertainty. By adaptively estimating the difficulty of each prediction (assigning higher σ to uncertain situations), this mechanism acts as a robust regularizer, preventing the model from overfitting to highly uncertain data.

## 2. **Model Architecture**

The model takes an input of shape `(BATCH, NUM_PLAYERS, SEQ_LEN, FEATS_NUM)` and outputs `(BATCH, HORIZON, 4)`, where the 4 outputs are 
(x, y, σ_x, σ_y). 

A challenging aspect of this competition is that ground truth post-pass trajectories are provided only for a subset of players in a given play, not everyone.
To address this, I designed the model to perform single-agent prediction per inference. Specifically, I constructed the input such that the target player to be predicted is always fixed at index 0 of the NUM_PLAYERS dimension. The model receives the context of all surrounding players but is trained to output the future trajectory only for the player at the 0-th index.

![Model Architecture](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F6927795%2F48ff6817816724ec508183f63d102731%2Fst_transformer_simplified.png?generation=1764821585721937&alt=media)

### Encoder: ST-Transformer
My main encoder is a Spatio-Temporal Transformer that alternates between Spatial Attention and Temporal Attention. In the Temporal Attention block, I employed a CNN instead of a simple linear layer to capture local temporal dependencies more effectively. The Spatial Attention mechanism incorporates domain biases, which are detailed in the next section.

### Attention Bias
 Inspired by Graph Neural Networks (GNN) for pedestrian prediction, I injected attention biases based on domain knowledge. 
I added biases for
- High collision risk (Dutra et al. 2017, Computer graphics forum,Vol. 36. No. 2.)
- Inverse Distance (Mohamed et al. CVPR, 2020, pp. 14424-14432)
 In their experiments, they compared various kernel functions—including quadratic exponential decay and inverse quadratic decay—and demonstrated that the simple inverse decay (1/d) performed best. I adopted this finding directly.
- Players within the field of view(Zhang et al. Pattern Recognition 142 (2023) 109633 )
- Teammates Players
- Opponents Players 

Most of them are proposed for graph neural network, but I used them as attention bias with learnable intensity.
This allows the model to softly prioritize these interactions.

### Decoder: TXP-CNN
For the decoder, I utilized TXP-CNN (Mohamed et al. CVPR, 2020, pp. 14424-14432).
This architecture treats the time dimension as channels and performs convolution over the feature dimension.While somewhat counterintuitive, I found this worked significantly better than a standard Transformer decoder for this specific task.

## 3. Training Strategy 
#### Preprocessing:
- Passer-Centric Coordinates: I normalized the coordinate system to be relative to the passer. All positions were recalculated with the passer at the origin (0, 0).
- Unify Left: I aligned the field geometry so that all offensive plays attack towards the left. This ensured consistent directionality for the model.

#### Data Augmentation
- Transition Sequences: I trained model to predict the pre-pass and post-pass trajectory (input: [: -N]frames, output: last N frames of pre-pass + post-pass frames)  in the training data, accounting for roughly 30% of the dataset.
- Non-Target Prediction:  I also trained the model to predict the last N frames of the pre-pass trajectory of a random player, accounting for roughly 30% of the dataset. This help the model learn player motion dynamics.

## 4. **Inference Strategy**
To further squeeze out performance during inference, I applied the following techniques:
- Test Time Training (TTT): I updated the model weights on the test data using only the pre-pass prediction auxiliary task (predicting the trajectory before the pass).

  - This resulted in a very marginal improvement of approximately 0.001 RMSE on Public LB.

  - However, observing the loss curve during TTT in CV, which consist of the same year, the optimization behavior resembled a random walk. I am uncertain how well this generalizes to data from different years.

- Test Time Augmentation (TTA): I also employed TTA by applying Player Permutation. Averaging predictions across different player input orders helped stabilize the results.

## 5. **Other Experiments & Missed opportunities**
- Previous Years' Data: Although utilizing data from previous Big Data Bowl competitions might have been possible, I realized this too late to incorporate it into my pipeline.
- Temporal U-Net: I attempted an architecture similar to U-Net that convolves over the time dimension and applies ST-Transformer at various temporal scales. Although the initial experiments showed some promise, I ran out of time to fine-tune it before the submission deadline.
- GNNs & Social-LSTM:RNN-based approaches like Social-LSTM and standard GNNs did not perform well on their own. However, replacing the Spatial Transformer with a GNN yielded decent results, so I used it in an ensemble.
- BERT-like Models:A model receiving (SEQ_LEN+ HORIZON) input performed comparably, so I used it in an ensemble.
- CVAE:Conditional VAEs worked well for minimizing the best of K RMSE but struggled with unimodal (mean) prediction.
- Feature Selection Strategy: I regret relying on an extensive set of features without more rigorous selection. While my experiments indicated that using a rich feature set outperformed a smaller subset, these validations were conducted using a smaller version of the model for efficiency. It is possible that the larger model might have benefited from a more curated feature set, whereas the smaller proxy model simply needed more raw information to learn.
- Cluster-Based Mixture of Experts: Inspired by arXiv:2402.08698, I applied K-Means clustering to the encoder outputs to categorize inputs into distinct tasks. Based on these clusters, I designed the gating mechanism to heavily weight the specific expert assigned to that cluster. Unfortunately, this explicit task routing did not yield better performance than a single model.