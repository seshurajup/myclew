# 1st place - A lot of features, a lot of models, and a little bit of luck

Wow! I have always dreamed of getting 1st place and it has finally come true :D. Congrats to everyone who managed to outperform the blind blending notebooks (which for some reason did quite well on private) and many thanks to the people who shared their code and insights including but not limited to: @cdeotte, @masayakawamata, @yekenot, @yeoyunsianggeremie, @jklol86, @mikhailnaumov.

Now onto my solution:

#FE
This was probably the most important part of the competition. My best single model has CV 0.92818 LB 0.92923 which would have placed 2nd! It uses the following features:

- Combination of base features (pairs) + TE/CE
- Digits of numerical features
- Combination of digits between each numerical feature (pairs, triples, quadruples) + TE/CE. This means I create interaction features between each feature's digits only
- Round features from [here](https://www.kaggle.com/code/masayakawamata/s5e11-single-xgb-add-features)
- Some combination of digits and base features
- Some combination of digits from different base features
- TE/CE of base features + some digits on the original dataset
- TE of base features on the original dataset using 'employment_status' as the target
- TE of base features on the train dataset using 'employment_status' and 'debt_to_income_ratio' as the target

Misc. features that improved CV by very little:

- Quantile binned numericals
- a 'default_risk' feature I found somewhere
- a categorical version of 'credit_score' (I was going to try out doing this with other numerical features but I sort of forgot)

I know that some of these may be hard to understand so I will publish my best single model code soon (UPD: the notebook can be found [here](https://www.kaggle.com/code/mahoganybuttstrings/pg-s5e11-xgb-cv-0-92818-pb-0-92923))

#Models
I will elaborate on this part later as I am a bit busy but in short, my ensemble has a grand total of 100 models :D!

UPD: The models and their scores (of the best model) as follows:

| Model type | CV | Public LB | Private LB |
| --- | --- |
| XGBoost | 0.928175 | 0.92831 | 0.92923 |
| LightGBM | 0.928081 | 0.92818 | 0.92920 |
| RealMLP | 0.927952 | 0.92812 | 0.92906 |
| TabM | 0.927833 | 0.92805 | 0.92895 |
| LightGBM-dart | 0.927772 | 0.92822 | 0.92920 |
| CatBoost | 0.927656 | 0.92806 | 0.92899 |
| XGBoost (reg) | 0.927334 | 0.92789 | 0.92857 |
| DANet | 0.926766 | 0.92776 | 0.92867 |
| Resnet | 0.926557 | 0.92713 | 0.92798 |
| Logistic Regression | 0.926291 | 0.92669 | 0.92757 |
| LAMA-DenseLight | 0.926145 | NA | NA |
| Trompt | 0.926000 | 0.92616 | 0.92756 |
| Gandalf | 0.925989 | 0.92741 | 0.92808 |
| Bartz | 0.925944 | 0.92703 | 0.92802 |
| FTTransformer | 0.925272 | 0.92576 | 0.92704 |
| Random Forest | 0.925079 | 0.92549 | 0.92643 |
| DeepFM | 0.924872 | 0.92537 | 0.92645 |
| TabulaRNN | 0.924809 | 0.92587 | 0.92706 |
| LNN | 0.924702 | 0.92537 | 0.92656 |
| LAMA-Dense | 0.924601 | 0.92535 | 0.92643 |
| ExcelFormer | 0.923880 | 0.92379 | 0.92503 |
| ModernNCA | 0.923170 | 0.92332 | 0.92440 |
| Extra Trees | 0.922104 | 0.92424 | 0.92549 |

You might have noticed that some models's scores are pretty terrible. My strategy for this comp was to train as many diverse models as possible and not spend that much time tuning them. I feel like I should change this strategy a bit next comp

#Ensembling
Ridge and HC were the best ensemblers in this comp, with HC being slightly better on CV but about the same on PB. I tried stacking with non-linear models like LGBM/CB/different NNs but they were much worse than linear methods. 

#Conclusion
The formula for getting in top 3 remains the same: a lot of models and a trick or two (which in this case was a lot of FE). As always, happy Kaggling!