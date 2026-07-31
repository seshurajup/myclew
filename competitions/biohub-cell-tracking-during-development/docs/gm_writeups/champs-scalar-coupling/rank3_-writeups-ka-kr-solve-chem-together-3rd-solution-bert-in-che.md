# 3rd solution - BERT in chemistry - End to End  is all you need

Our story

As you know, we need to find the appropriate representation for data. That’s why we’re struggling to do feature engineering.

As CPMP mentioned, two member of our team have domain knowledge on this competition. 
Sunghwan Choi has Ph.D at quantum chemistry and works on the field of quantum chemistry and chemical application of machine learning. 

I’m going on the Ph.D course in chemical engineering and have some experience for quantum calculation and dealing molecules. More important thing is that I’m kaggler :).

We thought that our problem can be solved by conventional graph models whose edge features are distance-derived properties. Those models are quite conventional in machine learning applications on chemical system. Hence, we tried to figure out appropriate hyper parameters and edge features for models

As many kagglers did, we also tested message passing neural networks by adopting many useful kernels. We thank to many kegglers; especially heng cher keng. :) by modification of their solution, we got a silver place 1 month ago. But, the gap between leading group and us was getting large. 

We had to find out breakthrough. We brainstormed a lot. At that time, limerobot, who is an expert of natural language processing suggested to use the raw xyz coordinates, 
I and Sunghwan didn’t agree with that, because if we use xyz coordinates instead of distance, translational and rotational invariances are not satisfied. The model which do not preserve those invariances seems to be ridiculous
 
But, limerobot did on his way. And he showed that his model won my GNN. 
His model was based on BERT model. You can see the big success in toxic competition on his profile. 

Maybe, insights from toxic competition save us :).

Anyway, he don’t have any domain knowledge. He just wanted to make a model to learn the complex representation using xyz coordinates. (end-to-end)

He just input the xyz coordinates of atom1, atom2, coupling type, distance and difference of each xyz coordinates (he thought the model can learn the distance formula based on that...amazing)

Because we had only one month, we decided to do all things based on the transformer.
After that, we did a number of experiment for hyper parameter tuning.

Because BERT is very large model, the performances differed according to the number of hidden layers, type of embedding and learning schedule.

Before 1 week to end of competition, we found our own scheduling and parameters and tiny modification of readout layer ( I will explain model later )  

We made multiples of models. Based on the ensemble of them, we got 3rd place :).

Thanks for reading our stories. We want to share the specific magics below. Please keep going :)

# Overall architecture

Here is our overall architecture for our model. As you can see, our input sequence is transformed into output sequence by BERT-encoder.

•	The number of encoder layer: 8
•	The number of heads for attention 8
•	Dropout ratio is 0.1 which is conventional choice for BERT model. 
•	For each type, different readout networks but having same architecture are used

 ![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F1155353%2F9d825e9c577db8b951a18cb9827e683b%2Ffig_1.png?generation=1567130786795298&amp;alt=media)

# Input features and embedding layer for them

Our float sequences composed of multiple embedding results, are the magic to get the achievement.

 ![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F1155353%2Fac01c25bcce0d2e5e1638b6f6bd2c952%2Ffig_2.png?generation=1567130820868426&amp;alt=media)

Multi-head attention layer itself preserve permutational invariance therefore, order of couplings do not change the results but invariance when atomic_index_0 and atomic_index1 are changed, is not preserved since we use feature vector as concatenation of  the embedding results of atomic charge, xyz coordinate(position), atomic number of two atoms. 

•	Size of embedding for atomic charge: 32
•	Size of embedding for position: 256
•	Size of embedding for atomic number: 64
•	Size of embedding for distance: 64
•	Size of embedding for type: 64

Total feature size for single feature vector is (32+256+64)*2+64+64=832

# Augmentation
In order to impose pseudo-invariance on our model, we use rotational and translational noise when augmenting data.

•	Translational noise: For each axis, Gaussian noise( mean: 0, std: 2) was added
•	Rotational noise: Rotational transformation whose axis is translational noise vector and angle is from Gaussian noise( mean:0, std: 3.14/2 )

# Regression layers for predicting scalar coupling constants
•	As you know that, spin-coupling (sc) value can be decomposed into four different terms (fc, sd, pso, dso)
•	After optimizing architecture and various losses, we found that auxiliary target using contributions gave a high boost.
•	During training, model minimize loss1 + loss2 with AdamW algorithm.
•	There are 8 regression layers to cover 8 different coupling types
 
![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F1155353%2F939a6ff0e4a6848477a655a2bfc6ad54%2Ffig_3.png?generation=1567130847675121&amp;alt=media)

# Specific learning rate
•	We were always using linear learning rate decay.
•	We think that there might be improvement with various learning schedule such as cycle lr. But, we didn’t have time because BERT is very large…(about 75M parameters. It took 1~2 days to get a model using 2~4 V100 machine)

# Pseudo labeling 
•	To get the more results, we needed some magics. With having an insight that there are less probability to be overfitted (Sunghwan choi’s insight) and experiment result from limerobot, we adopted pseudo-labeling.
•	After predicting for test set, we used the pseudo-labeled test dataset for training.
•	The model showed more than -3.4 CV. So, we trained model more with only train data to minigate overfitting. (finally we got ~-3.11 LB single models)
•	Overall learning process is illustrated below.
 
![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F1155353%2F92bc1322f07e876a7e5f1c60684a095d%2Ffig_4.png?generation=1567130868749389&amp;alt=media)

# Final submission
•	We made 14 models with various seed and hidden layers (most have 8 layers, other have 6 layers)
•	After weighted average according to cv score of types, we got -3.16.
•	After multiple procedure of pseudo labeling, we had 8 models. 
•	With simple average, we got -3.19 (our final score :) 2 hours before the end of competition.

# What we’ve learned
•	End-to-End works!!!. Amazing BERT.
•	Learning schedule is very important for modelling of molecular property.