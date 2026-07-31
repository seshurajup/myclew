# Brief Summary of 5th Place

First of all I would like to thank Kaggle and host team for organizing this interesting contest.
How great is it that I might be able to contribute to the research of covid-19 vaccine by participating in this contest?
I'm very excited to be a part of this meaningful contest.

## models
I extended the following models.
- AE model based on [AE pretrain + GNN + Attn + CNN](https://www.kaggle.com/mrkmakr/covid-ae-pretrain-gnn-attn-cnn).
- GRU+LSTM model mainly based on [OpenVaccine: Simple GRU Model](https://www.kaggle.com/xhlulu/openvaccine-simple-gru-model).

Final model is weighted average of variation of these 2 models.

I have already shared some techniques [here](https://www.kaggle.com/its7171/gru-lstm-with-feature-engineering-and-augmentation).
- augmentation
- sample_weight with all data
- some feature extraction
- CV strategy with clustering

I mainly focused on data preprocessing.

## additional augmentation
I used eternafold, vienna, nupack, contrafold and rnasoft to extract structure and loop_type.
These backend engines are used to extract additional bpps too.
Especially eternafold and contrafold worked well.

## features
I extracted following features.
For detail information, please refer to the [source code for this](https://www.kaggle.com/its7171/feature-extraction).
・bpps_sum
・bpps_max
・bpps_sum-max
・The value of bpps of the pair - the strength of the pair.
・The type of the pair (CG or GU or AU or None)
・Information on the neighbors of the pair
・entropy

## Some experiments.
・130 length sequence training.
I added dummy 39 length sequence to the training sequence.
I expected this model improves private scores, but did not make significant improvements.
・reversed sequence
I added reversed sequence data as augmentation data and it did not make significant improvements either.But these 2 modelshelped the ensemble a bit.
I tried these 2 models only for GRU+LSTM model.

## Results
|  model  |  private |  public  |
| ---- | ---- | ---- |
|ensemble of AE pretrain + GNN + Attn + CNN|0.34799|0.23260|
|ensemble of GRU+LSTM|0.35477|0.24222|
|ensemble of all models     |0.34471|0.23025|

I was able to get 5th place with private score of 0.3447.
Needless to say, I couldn't have done this without the input from the Kaggle community.
I'd like to thank Kaggle community, especially @xhlulu, @mrkmakr and @hengck23!!!