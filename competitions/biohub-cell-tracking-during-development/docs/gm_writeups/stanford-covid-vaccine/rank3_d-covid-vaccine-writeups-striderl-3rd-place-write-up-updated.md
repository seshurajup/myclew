# 3rd Place Write-up [UPDATED]

Truly want to thank Eterna and Kaggle for hosting this meaningful competition. Thanks all the kagglers for sharing great ideas and notebooks, I learnt a lot in this competition. And congratulations to all winning teams, especially my former teammate @nullrecurrent for the 1st place!! 

These two days were like a roller coaster.. that I dropped from 4th place on LB to 720th place on the first evaluation, and climbed back to 3rd place today. I think my shake were among the largest in this competition... which means my solution is not that robust in different situations, compare to @youhanlee and @nyanpn . There is still many things to learn. 

Before this competition, I know little about mRNA and degradation. So most techniques I used were from NLP competitions. I will summarize my approaches below: 

### Data
- I filtered out data based on <code> *_error_* </code> columns larger than [6,8,10] and train labels less than -0.5
- The first augmentation of data I did was reversed sequences (you also need to reverse other features and labels too). This is natural, since as an augmentation in NLP I would change the order of sentences in a paragraph. I noticed LB benefitted a lot from data, and correlation of private predictions decreased. Here the correlation is only calculated on the first 91 positions. 
- Then I thought adding more augmented data to train and test would make sense, so I used @its7171's great [notebook](https://www.kaggle.com/its7171/how-to-use-arnie-on-kaggle-notebook) to generate possible structure and predicted loop types for each samples, using them as training and test augmentation. 
- So my final training data includes (1) original data (2) reversed original data (3) Augmented data from Arnie (4) reversed augmented data. My fold prediction is also an average of these four. 

### Model
- I used mostly the AE pretrained GNN notebook by @mrkmakr (thanks so much for this great great notebook, I wouldn't be able to get 3rd without your work). 
- added different layers of LSTM/GRU/wavenet at the end with different units/parameters for diversity.
- In general, 2 x 128 units of LSTM or GRU layer at the end works the best for me. 
- wavenet is having very low correlation with other structures but score is generally worse. good for blending though. 
- I tried to slight increase the units of multi-head-attention, change the dropout of different layers etc. But in general the original structure is very good already. 

### Features
- I again used the backbone of @mrkmakr's code :)
- In additional his original node features, I added bpp max, bpp second max, diff between max and 2nd max, bpp sum, pair type etc. 
- In addition to @mrkmakr's structure adjacency matrix, I also added two matrices to specify the neighbors of each node's pair. This feature alone increased 20bps on LB. 

### 3D distance
- I managed to get 3D structures for all samples from http://rnacomposer.cs.put.poznan.pl/, as discussed by @hengck23 and @shujun717. This website can calculate predicted 3D structures given sequence and original structure. But the problem is that even the batch work only take maximum of 10 sequences. So I wrote a simple [script](https://www.kaggle.com/c/stanford-covid-vaccine/discussion/189574#1041899) using Selenium to call their server 600+ times to get all the 3D structures in .pdb files. 
- Now I published the 3D data in this post, along with a starter notebook: https://www.kaggle.com/c/stanford-covid-vaccine/discussion/189604
- I parsed these pdb files to get 3D distance between all C1 atoms, to replace the original distance matrix from @mrkmakr's notebook. I noticed about 10-15bps increase in LB and decrease in prediction correlation. 
- I feel that I didn't use the full power of these .pdb files, but I didn't have enough time to dig deeper. 

### Training Strategy
- 5 fold stratified CV based on sequence edit distance. 
- Based on @xhlulu's loss function (version v9), I used different weights for each target columns. I tried [0.2, 0.3, 0.3, 0.1, 0.1] and [0.2, 0.4, 0.2, 0.1, 0.1]. 

### Pseudo Labeling
- As also mentioned in @nullrecurrent's great write-up, using the first 91 positions is the key here. After reading her write-up, I have to say that the save and load methodology is genius :). But I found that in most situations, even PL increase the validation score (even 30~50 bps), it helps the next epoch (trained on training samples) to converge much faster than not using the PL.  
- This methodology was tricky because if I add too much PL, CV score became unreliable. So I only added a few epochs of PL at the beginning of each fold to help the model converge. 

### Blending
- Final model is a simple blending of 20+ models. 
- My strongest single model is using all of the above with LB 0.23125 and PB 0.34429
- I have a safe submission, which blended 4 models that only used GNN and very little PL, because I feel that RNN may still be affected by length of the sequence, and using blend PL might cause overfitting too. This blend ended up with PB 0.34540. 

### Final Thoughts
- For me, **the key to climb the LB was 4x augmented data and PL**.  Adding some RNN layers at the end of @mrkmakr's model, with some basic bpp features will help you in 241x range. Reversed sequence will help about 20bps, Arnie augmentation will help about 10bps, and PL will help about ~50bps if using correctly.