# 3rd place solution

**Competition:** CAFA 6 Protein Function Prediction

**Team:** yuanllo153

**Final score:** 0.44640, **3rd place** at the end of the competition

**Local validation scores:** BP = 0.3980027260171876, MF = 0.7122192560267626, CC = 0.6098138808955816

Sorry for the somewhat clickbait-style subtitle, but the result of the CAFA6 competition was indeed unexpected and exciting for me. About a month ago, when I opened Kaggle and saw the score, my first reaction was simply: “What happened?” Looking back, five months earlier, my best rank was only around the 300s. My solution may be somewhat disappointing, because overall it is largely a reproduction of the results from the U 900 team‘s [CAFA5-2nd codebase](https://github.com/btbpanda/CAFA5-protein-function-prediction-2nd-place), with only some small modifications. Nevertheless, I still want to follow the open-source spirit of the Kaggle community and share some related thoughts and data. I hope they can be useful to others and make a small contribution to the progress of CAFA.

## Background

I am currently a junior undergraduate student. Three years ago, I started my university studies with a strong interest in life sciences. Around 2024, AlphaFold2 had attracted enormous attention, and many teachers and classmates around me were discussing the new possibilities that AI could bring to life sciences. Influenced by this environment, I also became interested in AI for biology and hoped to take part in this field in some way. Later, I discovered Kaggle. The real-world tasks, detailed discussions, and executable code on the platform helped me understand what it actually means to work on a practical problem. I learned a lot from this process. My connection with Kaggle began in that time. 

CAFA6 is the second competition I have participated in. To be honest, when I first started, I did not have a clear direction. I also overestimated what AI tools could do for me. I thought they might help me quickly find effective solutions, but in practice, the suggestions I received did not lead to any real breakthrough or particularly valuable insight. AI is like an airplane, and clear thinking is its aviation fuel. Without that fuel, the airplane can never take off. I am gradually coming to understand this truth.

During my earliest attempts, A lot of time was wasted on meaningless discussions with AI. Later, I gradually realized that I should start from strong existing work by others. So I found the CAFA5-2nd solution. After reproducing that solution, I tried my best to test my own ideas and also discussed many possible directions with AI. Unfortunately, I was not able to achieve better results beyond the original solution.

In some sense, I feel a bit embarrassed that a largely reproduced solution ended up winning a prize several months later. Here, I would like to express my sincere gratitude to the CAFA5-2nd team. Thank you for making your work open source. Your solution gave me the opportunity to learn about advanced methods and ideas in protein function prediction, and it also strengthened my interest in this field. For a beginner who has just started exploring research, your work has been extremely valuable.

## Feature engineering

The solution I ultimately adopted utilised esm2_t33_650M_UR50D, and the `prot_t5_xl_uniref50` model to encode the proteins. ESM-IF did not work in my attempts. All three use the final layer output, mean-pooled to a fixed dimension.  Initially, I hadn’t realized that participants in the previous competition had already shared relevant protein language model encodings, so I rented GPU to perform the encoding from scratch(so I did some unnecessary work）. The code I used can be found at https://www.kaggle.com/code/yuanlao153/cafa6-encoding-code/edit/run/327666451,  As I did not carry out the encoding on the Kaggle platform, there may be some issues with the code that require debugging. I recommend that you obtain the encoding for the 224,309 protein sequences in this test set directly from https://www.kaggle.com/datasets/yuanlao153/aaaaaaaaa, The training set I actually used  has been merged with the CAFA5 training set; the code for the merge is available at https://www.kaggle.com/code/yuanlao153/cafa6-cafa5 and can be run directly on Kaggle or you can get the train and test embedding I used directly [here](https://www.kaggle.com/datasets/yuanlao153/cafa6-train-embeds).The final training set comprised 145,382 proteins. This is a comparison of the Venn diagrams for the num of training and test proteins in CAFA5 and CAFA6.
![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F20197397%2F8b98b2dca088958c1e03f7e12f077f75%2F1.jpg?generation=1781755766705855&alt=media)
Since my solution is almost entirely a reproduction of the U900 team's CAFA5-2nd solution, I will not present it as a new method. Instead, based on my own limited understanding, I will briefly explain several key steps in their data-processing pipeline and the overall model architecture, so that we can better understand their excellent work. For more details, please refer to their official [docs](https://github.com/btbpanda/CAFA5-protein-function-prediction-2nd-place/blob/main/CAFA5docs.pdf). If there are any mistakes in my explanation, please feel free to point them out.

I will start with `create_helpers.py`, U900 team's CAFA5-2nd solution used to construct the training targets. During my reproduction, I discovered that setting the `propagate` parameter in `create_helpers.py` to True or False has a significant impact on the final score on the local test set.
![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F20197397%2F54fc992fd9f5edb403cbef4ee123aa53%2F2.jpg?generation=1781755854652818&alt=media)
### Converting `train_terms.tsv` into a Label Matrix

 `create_helpers.py` converts `train_terms.tsv` into a protein × GO term matrix: each row is a protein, each column is a GO term, and the value is `1` if the protein has that GO annotation，'0' if not, and `NaN` if the term and its father terms are all '0'.
![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F20197397%2F9c2604d8c581d190c8a9683469c6d331%2F3.jpg?generation=1781755864851295&alt=media)

---

### Propagating Labels through the GO Hierarchy

For protein `P40518`, the original annotation contains child terms such as `GO:0044396` and `GO:0000001`. If `propagate=False`, only these directly annotated terms are set to `1`, while their parent or ancestor terms remain `0`.

If `propagate=True`, the positive label is propagated upward in the GO hierarchy. For example, because `GO:0000001` is labeled as `1`, its parent terms such as `GO:0048308` and `GO:0048311`, and higher ancestor terms such as `GO:0006996` and `GO:0051646`, are also changed from `0` to `1`. This makes the training labels consistent with the GO parent-child structure.
![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F20197397%2F06c6b45c796bd40cbdef513ee38c3190%2F4.jpg?generation=1781755879681162&alt=media)
---

###  Using `NaN` for Unknown Negative Labels

Some missing GO terms should not simply be treated as negative samples. For example, for protein `P40518`, `GO:0000002` is not annotated. However, its related parent or ancestor terms in that branch are also not positively annotated `0`. In this case, it cannot confidently say that `P40518` does not have `GO:0000002`; it is simply unknown.

Therefore, the code sets this position to `NaN` instead of `0`. During training, this `NaN` label is ignored in the loss calculation, which avoids incorrectly using uncertain labels as negative samples.
![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F20197397%2Fd231756edb3e658c48a5610eddf92aac%2F5.jpg?generation=1781755891705161&alt=media)
---

## Model
![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F20197397%2F798cba9e0bc47f427408174cd99d96e3%2F6.jpg?generation=1781755903217736&alt=media)

I would like to use this diagram created for me by Image2 to briefly introduce the model architecture used in the U900 team's CAFA5-2nd solution, so that you can better understand their amazing work. Overall, the model generates base prediction scores with Py-boost, Logistic Regression, and Neural Network models, which are then used as input features for GCN stacking.

---
### 1. Py-boost Models

The Py-boost section contains four GBDT-based models:

- `pb-t54500-cond`
- `pb-t54500-raw`
- `pb-t5esm4500-cond`
- `pb-t5esm4500-raw`

#### Inputs

Each protein is represented by embeddings + taxonomic features:

- **T5 embedding + taxon**  
    → 1024 + 32 = **1056 dimensions**
- **T5 + ESM embedding + taxon**  
    → 1024 + 1280 + 32 = **2336 dimensions**

#### Output space

The prediction targets are GO terms:

- BP: 3000
- MF: 1000
- CC: 500  
    → Total = **4500 GO terms**

---
#### GCN Training and Inference 

##### GCN Training

Only the following models contribute:

- `pb-t54500-cond`
- `pb-t54500-raw`

Training uses:

- **oof_pred.pkl**  
    → 5-fold out-of-fold predictions  
    → shape: **145,382 × N**
- **test_pred.pkl**  
    → used as validation-time evaluation features  
    → shape: **224,309 × N**

In training, GCN learns from OOF predictions, while test predictions are used only for validation scoring.

---
##### GCN Inference

Only inference-stage predictions are used:

- **test_pred.pkl**  
    → predictions for **224,309 test proteins**

---

### 2. Logistic Regression Models

Two linear models are used:

- `lin-t5-cond`
- `lin-t5-raw`

#### Inputs

- T5 embedding + taxon  
    → 1024 + 32 = **1056 dimensions**

#### Outputs

- BP / MF / CC:
    - 10000 / 2000 / 1500  
        → Total = **13,500 GO terms**

#### GCN usage

Same pattern:

- **oof_pred.pkl → GCN training**
- **test_pred.pkl → GCN inference**

---

### 3. Neural Network (nn-serg)

A side neural model:

- trained independently using T5/ESM embeddings
- outputs are saved as `.pkl`

It is used as an additional feature source for:

- GCN training
- GCN inference
---

### GCN Stacking
Image2 has revised my original hand-drawn explanatory diagram, and it looks perfect.
![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F20197397%2Fa5bda878014320687a8f45c56b856a45%2F7.jpg?generation=1781755916997162&alt=media)
my original version is here it's awful
![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F20197397%2F8972004ff105410fff74ad4b8bd92429%2FPasted%20image%2020260617195140.png?generation=1781756323496121&alt=media)
This is a simple explanation of how the GCN stacker constructs its input features from several base model predictions.

At the top, the input is written as:

```
5 × 4 + 1
```

This means that, for each protein and each GO term, the GCN receives features from **5 prediction sources**, each source is encoded into **4 channels**, and an additional **GOA annotation channel** is added. Therefore, the total input dimension per protein-term pair is:

```
5 models × 4 channels + 1 GOA channel = 21 channels
```

---
### 1. One TTA Configuration

The figure shows one TTA configuration as an example. In this configuration, the GCN uses predictions from several base models. These five models were used for GCN training, whilst four models with different configurations were used for prediction to increase diversity.

Model for training：
```
pb-t54500-cond
pb-t54500-raw
lin-t5-cond
lin-t5-raw
side model: nn-serg
```

Model for inference：

| TTA    | Cond models         | Raw models         | Linear Cond   | Linear Raw   |
| ------ | ------------------- | ------------------ | ------------- | ------------ |
| `cfg0` | `pb_t54500_cond`    | `pb_t54500_raw`    | `lin_t5_cond` | `lin_t5_raw` |
| `cfg1` | `pb_t5esm4500_cond` | `pb_t54500_raw`    | `lin_t5_cond` | `lin_t5_raw` |
| `cfg2` | `pb_t54500_cond`    | `pb_t5esm4500_raw` | `lin_t5_cond` | `lin_t5_raw` |
| `cfg3` | `pb_t5esm4500_cond` | `pb_t5esm4500_raw` | `lin_t5_cond` | `lin_t5_raw` |

These models have already predicted probabilities for a subset of GO terms. The GCN then combines these predictions and expands them to the full GO ontology graph.

---
### 2. Four Channels for One Model

For each base model, its prediction is converted into four channels:
#### Channel a: Source indicator
This channel indicates whether a GO term has a real prediction from the model.

```
0 = the term has a model prediction1 = the term is filled by prior
```
If a model does not predict a certain GO term, the code fills that position using a prior value from `prior.pkl`.
#### Channel b: Original prediction value
This channel stores the original model prediction score. If the model does not cover the term, the value is replaced by the prior score.
For example:
```
GO:C = 0.8
```
means the model predicts this protein has GO term `C` with probability `0.8`.
#### Channel c: Strict propagation
This channel applies a strict parent-node constraint along the GO DAG.
If term `C` has two parent terms `A` and `B`, then:
```
C = C × A × B
```
For example:
```
A = 0.9B = 0.5C = 0.8C_after = 0.8 × 0.9 × 0.5 = 0.36
```
This is strict because **all parent terms need to support the child term**. If any parent score is low, the child score will be strongly reduced.
#### Channel d: Relaxed propagation
This channel applies a more relaxed parent-node constraint.
Instead of requiring all parents to be high, it measures whether **at least one parent term supports the child term**:
```
C = C × [1 - (1 - A) × (1 - B)]
```
For example:
```
A = 0.9B = 0.5C = 0.8C_after = 0.8 × [1 - (1 - 0.9) × (1 - 0.5)]  = 0.8 × 0.95 = 0.76
```
This is more relaxed than strict propagation because one strong parent is already enough to keep the child score relatively high.

---
### 3. GOA Channel

Besides the 5 base models, the GCN also receives one GOA channel.
The GOA channel represents external electronic GO annotations. It is a binary feature:
```
1 = this protein has this GOA annotation 0 = no such GOA annotation
```
This provides additional biological evidence beyond the learned base model predictions.

---

### 4. Conditional Models vs. Raw Models

Conditional models and raw models are treated differently.

Conditional models, such as:
```
pb-t54500-condlin-t5-cond
```
keep `NaN` labels during training. Their outputs are closer to conditional probabilities such as:
```
P(term | father term)
```
Therefore, the propagation step helps convert these conditional-style scores into scores that better reflect the full GO hierarchy.
The relationship is:
```
P(term | father term) × P(father term) = P(term)
```

Raw models, such as:
```
pb-t54500-rawlin-t5-raw
```
fill `NaN` labels as `0` during training. Their outputs are closer to direct unconditional probabilities:
```
P(term)
```
Therefore, the directly predicted raw-model terms do not need the same conditional propagation.

## Interesting Finding

Since I did not want to waste the ESM-IF embeddings I had generated, I replaced the original `pb_t5esm4500` group in the Py-boost experiments with `pb_t5if4500`, using ESM-IF features instead of ESM features. After training, I used their predictions in the final GCN training and inference stages. However, the final performance did not show a significant improvement.

Before the competition started, I consulted one of my teachers who studies enzymatic catalysis about how to predict the function of an unknown protein. She said that, in general, sequence determines structure, and structure determines function. I then asked whether providing structural information would make protein function prediction more accurate. Her answer was no, at least not necessarily. If the structural information refers to atomic coordinates, then apart from using those coordinates to compare structural similarity between proteins, it is difficult to extract effective functional information directly from the 3D coordinates themselves.
![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F20197397%2F97859ea8a1f5a9b661f952817f07b703%2F8.jpg?generation=1781755943308333&alt=media)
In many cases, if we only want to compare similarity between proteins, sequence comparison is already very strong, because proteins with similar sequences usually have similar structures. If we want to obtain useful information from structure, the important part may not be the global structure, but the functional domains or local regions that interact with substrates. Of course, this explanation was mainly from the perspective of enzyme proteins.
![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F20197397%2F80ca42504a3a5918a9aba70b56f12653%2F9.jpg?generation=1781755961802146&alt=media)
This also matches my observation in the experiment: global structural representations such as ESM-IF may not serve as a strong complement to sequence-based information such as ESM-650M, or ProtT5. A more promising direction may be to split proteins into functional domains and represent them at the domain or local-structure level to obtain more useful information.
![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F20197397%2F6055e1867f185b0ff972d30c91a7d1d6%2F10.jpg?generation=1781755972043371&alt=media)
### ESM-IF Feature Expansion Experiment

I added two additional GBDT models using T5 + ESM-IF features, where ESM-IF provides 512-dimensional structure embeddings. The two models were `pb_t5if4500_raw` and `pb_t5if4500_cond`. As a result, the number of input models for `GCNStacker` increased from 5 to 7.

| Configuration          | Models |   GCN Input Dim   |             CC Final |         MF Final | Source                                               |
| ---------------------- | :----: | :---------------: | -------------------: | ---------------: | ---------------------------------------------------- |
| Original 5-model setup |   5    | 21 (Linear 21→16) |           **0.6015** |       **0.7048** | `log/train_gcn_cc.log`, `log/train_gcn_mf.log`       |
| + IF (7-model setup)   |   7    | 29 (Linear 29→16) | **0.6069** (+0.0054) | 0.7020 (-0.0028) | `log/train_gcn_if_cc.log`, `log/train_gcn_if_mf.log` |

**Conclusion:**  
The IF features are effective for the smaller ontology, CC, which contains only 4,041 terms, improving the final score by **+0.0054**. However, they do not help the medium-to-large ontology, MF, which contains 10,131 terms, and even lead to a slight decrease of **-0.0028**.

**Possible explanation:**  
The 512-dimensional ESM-IF embeddings may be partially redundant with the sequence-level functional information already captured by T5 and ESM. The additional structural signal only brings a marginal gain for CC, where the term space is relatively sparse.

### ESM-IF Inference Substitution Experiment

After completing the 4-channel TTA inference based on ESM features, I further tested whether structural information could yield benefits during the final GCN inference stage. Specifically, without retraining the GCN, I simply replaced the base model predictions involving ESM in the TTA configuration with those from the ESM-IF model; that is, I substituted the original T5 + ESM model with the two GBDT models—`raw` and `cond`—trained using the T5 + ESM-IF structural embedding.

In the original 4-channel TTA configuration, `cfg0` consists solely of T5 models and therefore remains unchanged; in the remaining three channels, wherever `pb_t5esm4500_raw` or `pb_t5esm4500_cond` is used, it is replaced with the corresponding ESM-IF version:

| cfg  | original ESM TTA                       | IF（ESM-IF）TTA                        |
| ---- | -------------------------------------- | ------------------------------------ |
| cfg0 | `pb_t54500_cond + pb_t54500_raw`       | `pb_t54500_cond + pb_t54500_raw`     |
| cfg1 | `pb_t5esm4500_cond + pb_t54500_raw`    | `pb_t5if4500_cond + pb_t54500_raw`   |
| cfg2 | `pb_t54500_cond + pb_t5esm4500_raw`    | `pb_t54500_cond + pb_t5if4500_raw`   |
| cfg3 | `pb_t5esm4500_cond + pb_t5esm4500_raw` | `pb_t5if4500_cond + pb_t5if4500_raw` |

The experimental results are as follows:

| ontology | original ESM TTA | IF（ESM-IF）TTA | difference |
| -------- | ---------------: | ------------: | ---------: |
| BP       |           0.3980 |        0.3945 |    -0.0035 |
| MF       |           0.7122 |        0.7081 |    -0.0041 |
| CC       |           0.6098 |        0.6090 |    -0.0008 |

The results show that, following the replacement with ESM-IF, the overall CAFA score fell from **0.5733** to **0.5705**, a decrease of **0.0028**. Among the three models, both BP and MF showed a relatively significant decline, whilst CC remained largely unchanged, falling by only **0.0008**. This indicates that, given the GCN has already been trained on the original 5-model predictions, directly replacing the ESM predictions with ESM-IF predictions during the inference stage does not yield any improvement; rather, it causes a certain degree of distribution shift. Therefore, the final solution continues to use the original ESM TTA results as the main submission.

## Acknowledgement
Thanks again to [sergeifironov](https://www.kaggle.com/sergeifironov), [btbpanda](https://www.kaggle.com/btbpanda), and [alexandervc](https://www.kaggle.com/alexandervc) for their outstanding [solution](https://github.com/btbpanda/CAFA5-protein-function-prediction-2nd-place) in the CAFA5 competition. I truly learned a lot from it.

Thanks to [sergeifironov](https://www.kaggle.com/sergeifironov) for providing the protein [embeddings](https://www.kaggle.com/datasets/sergeifironov/t5embeds), and thanks again to [andreylalaley](https://www.kaggle.com/andreylalaley) for sharing the [embeddings](https://www.kaggle.com/andreylalaley). If only I had found them earlier. Thanks to everyone on Kaggle. If only... no, forget the “if only”! I found you all at just the right time!
## Code and Resources
I have upload the code here [https://github.com/yuanlao153/CAFA5-protein-function-prediction-2nd-place/tree/cafa6-adapt](https://github.com/yuanlao153/CAFA5-protein-function-prediction-2nd-place/tree/cafa6-adapt)， related model weights and data are here [https://pan.quark.cn/s/4a4b2b6aae4b](https://pan.quark.cn/s/4a4b2b6aae4b), I will upload the resources to the Kaggle dataset as soon as possible, but the py-boost file is very large and cannot be uploaded to Kaggle; I will find a way to upload it to Google Drive as soon as possible.