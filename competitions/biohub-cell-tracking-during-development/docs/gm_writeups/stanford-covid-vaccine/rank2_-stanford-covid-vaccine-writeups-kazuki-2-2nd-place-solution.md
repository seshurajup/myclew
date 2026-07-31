# 2nd Place Solution

I was preparing this as 1st place solution but unfortunately I got **5th** 2nd place again.
Who cast the curse???

# LB progress
![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F317344%2F7839efcb3651f2aedaa830116f096133%2F2020-10-19%207.52.26.png?generation=1603061606818239&alt=media)
# Pipeline
![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F317344%2Fb13f002dcf298c871e273638e6930c43%2F2020-10-17%2016.50.40.png?generation=1602921081920687&alt=media)
I made prediction of SN_filter and pseudo label, but otherwise it's almost same methodology as another of my [2nd place solution](https://www.kaggle.com/c/trends-assessment-prediction/discussion/162765) of TReNDS.

# Stacking
I'm really good at stacking with xgboost. However, stacking using oof of pseudo label model is little bit hard due to the drastic oof. So I tried to add gaussian noise to the oof and prediction of test, and then I succeeded to avoid overfitting. 
![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F317344%2F0f51b7b8c34d1a85170217132173549a%2F2020-10-19%2015.42.57.png?generation=1603112091717651&alt=media)
![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F317344%2Ff7d8997d7beac2371a450165963b9980%2F2020-10-19%2015.44.59.png?generation=1603112114562030&alt=media)

# CV vs LB
Also, our CV and LB are super-correlated. We can completely predict LB score using CV, so we didn't submit much.

# LB simulation
Now I think this work
![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F317344%2F42216ecab6efd6b771802b927a2149ae%2F2020-10-19%2021.53.01.png?generation=1603112007528427&alt=media)
https://www.kaggle.com/c/stanford-covid-vaccine/discussion/189196#1040074

# Reference
[AE GNN](https://www.kaggle.com/mrkmakr/covid-ae-pretrain-gnn-attn-cnn)
[Data augmentation](https://www.kaggle.com/its7171/how-to-generate-augmentation-data)
[First penguin who has suspicion of LB](https://www.kaggle.com/c/stanford-covid-vaccine/discussion/189196)