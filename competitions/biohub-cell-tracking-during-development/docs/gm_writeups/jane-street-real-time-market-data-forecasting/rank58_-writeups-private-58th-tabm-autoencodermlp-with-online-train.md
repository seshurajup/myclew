# [Private 58th] TabM, AutoencoderMLP with online training & GBDT offline models

Thanks to this amazing competition and the efforts of every one of my teammates ! @lechengyan @chronoscop
Our final solution of 0.0092 lb is ensembling NN models with online learning , GBDT offline models and a ridge model. My part is mainly for TabM model and some GBDT models, which is what I am gonna to talk about. AutoencoderMLP and online learning are @lechengyan 's part and @chronoscop is in charge of one of XGB.

# 1. Cross-Validation
I simply used the last 120 dates as my validation and it shows good correlations with LB.

# 2. TabM Model
Actually, in the early time of this competition, I noticed TabM and found it is a tabular NN model with great potential. I also public a baseline [notebook](https://www.kaggle.com/code/i2nfinit3y/jane-street-tabm-ft-transformer-inference).
### Model Architecture
The model parts I mainly adjust are feature embed layers and backbone.
For category features, I use onehot encoding for every category tensor, which contains feature_09, feature_10, feature_11, symbol_id and time_id. I remap these category features with ordinal encode and every new category  in the test data will be mapped to the same category. Apart from one-hot encoding, I also use embedding layers but it didn't help much. It's worth noting that taking time_id as category features improve my model al lot.
But for continuous features, I have tried LinearEmbeddings, PeriodicEmbeddings and PiecewiseLinearEmbeddings, which didn't work well. So I don't use any embedding layer for continuous features.
The backbone of TabM is just 3 layers MLP with the same 512 dimensions. I haven't experimented with too many dimensional combinations, but 512 seems to be the most stable.
### Loss function
I have used huber loss, logcosh loss, mae loss, zero-mean R2 loss and mse loss, and R2 loss and mse loss work best. Specifically, mse loss can get higher score in the cv but R2 loss is more robust in every validation epoch. So I choose R2 loss as loss function.
### Hyperparameter
Dropout : 0.25. Higher dropout rate (0.5) will result in slower convergence and lower (0.1) will be overfitting, so 0.25 seems to be a good choice for me.
Learning rate: 1e-3.
Weight decay: 8e-4.
Batch size : 8192
epoch : 5 or 6. Generally, I will submit multiple epoch to confirm the best model checkpoint and most of time 5 or 6 is the best.
optimizer : AdamW
k : 16. k is the number of ensemble in the finial output. I tried 8, 16, 24, 32, and I found 16 can takes both scores and training time into account.
### Datapreprocessing
Simple mean-std standardize and fill zero for nan values.
### Feature Engeneering & Auxiliary targets
I use time_id, symbol_id, 78 original features (besides feature_61, because it just change with dates) and 9 responder lags 1 of last record. Additionally, I also use sin and cos of time_id  with 967 periods or 483 periods and feature_61 with 20 period to catch some periodical change. Because though feature_61 only changes with the date, it also changes in about 20 dates. Also, I tried to use more lags features but it work worse in cv and lb.
Besides responder_6, I use responder_3 as my auxiliary target, because it has high correlations with responder_6. I tried to add more auxiliary targets in my training, but there was no obvious improvement in my model.
### Data Augmentation
I add some gaussian noise in continuous features with 0.02 std.
### Training Sample
I use the data with dates after 252, because there are too many Nan values in the first 252 dates. I have tried use the dates after 750 to train, which improve my cv but decrease my lb.
### Scores
without online learning    cv : 0.0106  lb : 0.0077
with online learning    cv : 0.0116  lb : 0.0083

# 3. AutoencoderMLP
### Datapreprocessing
Fill 3 for Nan values without standardization.
### Feature Engeneering
We use time_id, symbol_id, 79 original features and responder_6 lags 1 of last. We didn't use auxilary target in the AutoencoderMLP
### Model Architecture
Encoder and Decoder : Both are a lieanr layer with 96 hidden dimension.
MLP : 5 layers with hidden dimension 96, 896, 448, 448, 256
### Hyperparameter
Dropout : [0.035, 0.038, 0.424, 0.104, 0.25, 0.32, 0.271, 0.25]
Learning rate: 1e-3
Batch size : 8192
Epoch : about 15
optimizer : Adam
### Loss function
We use mse loss as reconstruction loss, and weighted-mse loss as prediction loss
### Scores
without online learning    cv : 0.0103  lb : 0.0072
with online learning    cv : 0.0110  lb : 0.0078

# 4. My GBDT and Ridge offline model
### Feature Engeneering
It is more simple now. For LGB, XGB and Ridge, I just use time_id, symbol_id, 79 features, responder lags 1 of last, mean, std and max. And I did not specify category features in the GBDT model.
### Training Sample
The dates after 750 for XGB and Ridge and dates after 678 for LGB 
### Hyperparameter
I didn't optimize my hyperparameter just use fixed one.
```
LGB_Params = {
        'learning_rate': 0.05,
        'max_depth': 6,
        'num_leaves': 62,
        'n_estimators': 200,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 1,
        'reg_lambda': 1,
        'random_state': 42,
        'device' : 'gpu',
        'gpu_use_dp': True,
        'objective': 'l2',
    }

XGB_Params = {
    'learning_rate': 0.05,
    'max_depth': 6,
    'n_estimators': 300,
    'subsample': 0.8,
    'colsample_bytree': 0.6,
    'reg_alpha': 1,
    'reg_lambda': 1,
    'random_state': 42,
    'tree_method': 'hist',
    'device' : 'cuda',
    'n_gpu': 1,
    'objective' : 'reg:squarederror',
}
```
### Scores
LGB   cv: 0.0096 lb : 0.0072
XGB   cv: 0.0102  lb : 0.0073
Ridge cv: 0.0035 lb : 0.0044
After ensembling these 3 models, the lb is 0.0076.

# 5. @chronoscop's XGB
### Datapreprocessing
Fill 0 for Nan values without standardization
### Feature Engeneering
use symbol_id, time_id, 79 original feature and responder lags1 of last. If add more lags features, it would be easy to overfit.
### Training Sample
use dates after 917 to train the model.
### Hyperparameter
```
params = {
'objective': 'reg:squarederror',
'random_state': 1212,
'tree_method': 'hist',
'device' : 'cuda',
'learning_rate': 0.02156022412857549,
'max_depth': 8,
'subsample': 0.7697954003310141,
'colsample_bytree': 0.5182134365961873,
'reg_alpha': 0.0032315937370696354,
'reg_lambda': 0.002663721647776419
}
```
### Scores
cv : 0.0102   lb : 0.0072

# 6. Online Learning
Online learning play an important role in this competition.
We collect data  for each previous date and then update model in every date. For every date, we will combine the data of the last date with partial data from the previous dates (using random sampling) for online training. In the training part, instead of batch processing, we input all the data into the model and then train 5 epochs.
```
if previous_test is not None and lags is not None:
    train = previous_test.join(
        lags_.select(["time_id", "symbol_id", pl.col("responder_6_lag_1").alias("responder_6")]),
        on=["time_id", "symbol_id"],
        how="left"
    )

    if pre_train is not None and len(pre_train) > 300000:
        pre_train = pre_train.sample(n=300000, seed=2025)
    if pre_train is None:
        pre_train = train
    else:
        pre_train = pl.concat([pre_train, train])
    X = pre_train.to_pandas()[cols].fillna(3).values
    y = pre_train['responder_6'].to_numpy()
    weights = pre_train['weight'].to_numpy()

    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()

    for epoch in range(5):
        optimizer.zero_grad()
        _, y_pred = model(torch.FloatTensor(X).to(device))
        loss = criterion(y_pred.squeeze(), torch.FloatTensor(y.copy()).to(device))
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch + 1}/5, Loss: {loss.item()}")

if previous_test is None:
    previous_test = test
else:
    previous_test = pl.concat([previous_test, test])
```
We tried to add online learning into GBDT, but it didn't work well and cost too many time.

# Thanks !
I think TabM can get higher score if adjust different kind of backbones, but I have no time to experiment. I also tried GRU, LSTM or Transformer in the early time, but they all failed, so then I focus on tabular model. Hoping top teams can share more trick and I indeed learn a lot in the competition. Thanks to all of you! :)
PS: Our code will be released when it is sorted out.

----------------------------------------------------------------------
UPDATE : 
TabM training code https://www.kaggle.com/code/i2nfinit3y/jane-street-tabm-training?scriptVersionId=217873214
Single TabM online learning : https://www.kaggle.com/code/i2nfinit3y/jane-street-tabm-online-learning

----------------------------------------------------------------------
UPDATE :
we’ve open-sourced both our inference and training code.