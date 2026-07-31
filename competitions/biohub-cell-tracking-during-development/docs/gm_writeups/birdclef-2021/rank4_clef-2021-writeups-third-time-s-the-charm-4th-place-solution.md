# 4th place solution 

First of all, I would like to thank the organizers of the competition and the Kaggle team for the competition hosting. I would also like to thank all the participants who have shared their knowledge so generously, including the previous Birdcall competitions. Two weeks before the end of this competition, I didn't expect it to be so hard competition.

## Keys of my solution
* Ensemble
  * Max 62model (Best private: 47model)
<!-- * global情報を用いたinference -->
* **Inference with global information for SED model**
* Post-processing

## Training
### Preprocess
<!-- logmelspectrogramはTorchAudioを用いて計算され、それぞれの平均と分散を用いて正規化されました。 -->
The logmelspectrograms were calculated using TorchAudio and normalised using the means and variances per a image.
``` python
logmelspec_extractor = nn.Sequential(
            MelSpectrogram(
                32000,
                n_mels=128,
                f_min=20,
                n_fft=2048,
                hop_length=512,
                normalized=True,
            ),
            AmplitudeToDB(top_db=80.0),
            NormalizeMelSpec(),
        )
```  
<!-- waveformでのAugmentationとして、pink noiseやgaussian noiseを加えました。 -->
For Augmentation in waveform, gaussian and uniform, pink noise are added random.  
<!-- logmelspec上のaugmentationとして、Mixupを用いました。パラメータとして0.2~0.5の確率、alpha=0.8を用いました。 -->
As an augmentation on logmelspec, I used Mixup. As a parameter,  I used a probability of 0.2~0.5 and alpha=0.8.
### Modeling
<!-- araiさんのkernelと同じくSED modelを用いました。 -->
I used the SED model like [@hidehisaarai1213 's kernel](https://www.kaggle.com/hidehisaarai1213/pytorch-training-birdclef2021-starter).
<!-- 様々なbackboneを用い、それぞれbatchsizeが36になるように使用する秒数が調整されました(最大で30s, 最小で10s) -->
Using various backbones, the seconds(max 30s, min 10s) used training has been adjusted so that each has a batchsize of 36.   
### Training
<!-- @hidehisaarai1213のmodelとはloss関数の計算が少し異なり`clipwise_pred`を直接最適化しています。こうすることでbackward時のNaNの発生を抑制することができました。 -->
`clipwise_pred` is optimized directly. By doing this, I was able to suppress the generation of NaN on backward.
``` python
loss = bce_with_logits(torch.logit(clipwise_pred), target) + 0.5 * bce_with_logits((framewise_logit.max(1)[0], target)
```
<!-- loss関数はnn.BCEWithLogitsLossを用いました。 -->
<!-- The loss function used is nn.BCEWithLogitsLoss. -->

Other points are that
* use a secondary label
  * soft labels such as 0.5 did not work
* trained 40~50 epochs 
  
### Pseudo labeling
<!-- 私は以下の手順を組み合わせていくつかのパターンを試してみました。 -->
I have tried some patterns using a combination of the following steps.
<!-- * 単純にaudio file単位のrelabeling
  * clipwise_predを用い、閾値は0.2を用いました -->
* Simple per-audio file relabeling
  * Use clipwise_pred and threshold = 0.2
<!-- * 前回のコンペのaraiさんのsolutionにインスピレーションを受け、audio file全体を入力としたframewise_predとtime_attを保存し、学習に用いる期間分を切り取りclipwise_predを計算しました。 -->
* Inspired by [@hidehisaarai1213 's solution](https://www.kaggle.com/c/birdsong-recognition/discussion/183204) in the last competition, I saved `framewise_pred` and `time_att` with the whole audio file as input, and computed `clipwise_pred` by cropping the period used for training.
<!-- * secondary_labelがないaudio fileだけ2番目の方法でpseudo labelを作成する -->
* Create a pseudo label using the second method only for audio files without a secondary_label
<!-- * 閾値がある一定より下(0.05)だった場合、0.1の確率でprimary labelとsecondary labelを削除しました。 -->
* If the pseudo label was below a certain value (0.05), the probability of 0.1 was used to set the primary and secondary labels to 0.

<!-- どのパターンも個々のモデルの性能は0.01ほど向上し、train_soundscapeでのscoreはensembleに貢献しませんでしたが、public LB上では多少機能しました。 -->
All patterns improved the performance of the single models by about 0.01, and although the score on train_soundscape did not contribute to ensemble, it did work somewhat on the public LB.

### 30s finetuning
<!-- 大きいモデルではbatchsizeを確保するために10sなどの短いsegmentで学習されています。そのため、ラベルがノイジーになります。これを解決するためにAttention moduleのみを30sのsegmentを用いて10epoch追加で学習させました。train_soundscapeでのscoreでわずかな改善があり、ensembleの中に含まれています。 -->
Larger models are trained with shorter segments, such as 10s. Therefore, the labels become noisy. To solve this, only the Attention module was trained with 30s segments on 10 additional epochs. There is a slight improvement in score with train_soundscape, and it is included in ensemble.

### Checkpoint selection
<!-- それぞれvalidationでもtrainingと同じ秒数で切り取ったsegmentを用いout-of-foldでのlossが最も小さいものを選びました。 -->
The checkpoint with the smallest oof loss was selected for validation using the same number of seconds as for training.

## Inference with global information
<!-- 自分のsolutionのなかで最も効果的で気に入っている部分です。   -->
This is my favorite and most effective part of my solution.  
<!-- はじめに、このcompetitionに参加する前から持っていたアイディアは、SEDモデルが正確に"Sound Event Detection"を実行できるのであれば、長い期間に対して行われた予測の中で必要な部分だけを切り取ることで短い期間の予測を改善することができるのではないかということでした。 -->
To begin with, the idea that I had before joining this competition was that if the SED model could accurately perform "Sound Event Detection", then it would be possible to improve predictions for the shorter time segment by cropping from the longer time segment the necessary parts of predictions.

<!-- このようにすることで、より長期間の特徴を推測に用いることができます。 -->
In this way, longer term features can be used for inference.
```python
framewise_pred_5s = self.fix_scale(feat[:, :, start:end])
att_5s = torch.softmax(time_att[:, :, start:end], dim=-1)
clipwise_pred_5s = torch.sum(torch.sigmoid(framewise_pred_5s) * att_5s, dim=-1,)
```
<!-- このアイディアを実行することで0.03ほどtrain_soundscapeでのscoreとpublic LB共に改善することができます。実際には長い期間のsegmentとして30sを用い、予測したい5sの区間が中心にくるように実装しました。このアイディアはpseudo labelを作る2つめのパターンでも用いられています。   -->
By implementing this idea, I can improve both score and public LB in train_soundscape by about 0.03. In practice, I used 30s as the segment for the longer period and implemented it so that the 5s interval I want to predict is in the center. This idea is also used to create a pseudo label in the second pattern.  
<!-- 次に、前回の[Birdcallでの7th解法](https://www.kaggle.com/c/birdsong-recognition/discussion/183571)での観察ではaudio file単位で高い閾値を使い出現することを確認できた鳥は、短いsegmentにおいても高い確率で出現することが期待でき、閾値を下げて感度を上げることができます。 -->
Secondly, in [7th place solution in previous Birdcall competition](https://www.kaggle.com/c/birdsong-recognition/discussion/183571),  the birds that were observed to appear using a high threshold for each audio file can be expected to appear in short segments with a high probability, and the sensitivity can be increased by lowering the threshold.  
<!-- この観察に触発され、`clipwise_pred_30s`では高い閾値(0.05)を用いて考えられる鳥のリストを作成し、`clipwise_pred_5s`では低い閾値(0.025)を用いAND演算を行いました。 -->
Inspired by this observation, I used a high threshold (0.05) in `clipwise_pred_30s` to generate a list of possible birds and a low threshold (0.025) in `clipwise_pred_5s`,  to perform AND operations.
```python
((clipwise_pred_30s > high_threshold) + (clipwise_pred_5s > low_threshold)) >= 2
```
<!-- これらの閾値の見当をつけるために`scipy.optimize.dual_annealing`を用い、train_soundscapeに対して最適化しました。これはかなり危険なように見えますが、後に述べる単純なpublic LB probeによりある程度妥当だと自分は考えました。 -->
To get an idea of these thresholds, I used `scipy.optimize.dual_annealing` to optimize for train_soundscape. This seems rather risky, but I thought it was somewhat reasonable due to the simple public LB probing described below.
<!-- この二重の閾値処理によってpublic LBとtrain_soundscapeのscoreは0.03改善されました。   -->
This double thresholding improved the score of public LB and train_soundscape by 0.03.  
<!-- アンサンブルでは単純な平均を取りました。初期にhard voteも検討しましたが、あまりスコアは変わらないため、単純な手法を選びました。   -->
In the ensemble I took a simple average. Initially, hard voting was considered, but the score did not change much, so the simple method was chosen.  
<!-- ## 場所と日付を用いたpostprocessing -->
## Post-processing with location and date
<!-- それぞれのsite周辺450メートル内で観測された鳥のリストと、それぞれの月ごとに出現する鳥のリストを作成し、出現しないものをsubmissionから削除しました。 -->
A list of birds observed within 450 meters around each site and a list of birds appearing in each month was made, and those not appearing were removed from the submission.
<!-- ただし、"COR"と"COL"の2つのsiteは比較的近くのため、どちらも0の場合のみ削除されました。 -->
Here, because the two sites "COR" and "COL" are relatively close, the birds have only been removed if both are zero.
<!-- また、site周辺で観察された鳥の中でいくつかの希少なクラスに関して閾値を4倍にしました。   -->
I have also quadrupled the thresholds for some rare classes of birds observed in the vicinity of the sites.
<!-- これらのpostprocessingにより、train_soundscape・public LB共に0.01以下の一貫した改善が見られました。 -->
This post-processing resulted in a consistent improvement of less than 0.01 for both train_soundscape and public LB.

## Simple public LB probing
<!-- ある程度train_soundscapeとpublic LBは相関していたため、public LBの分布が極端にtrain_soundscapeと似ているのではないかという危惧を感じました。 -->
Since train_soundscape and public LB were correlated to some extent, I was concerned that the distribution of public LB might be extremely similar to train_soundscape.
<!-- それを検証するために、適当なsubmissionに対して、train_soundscapeに含まれていないsiteと鳥の予測をnobirdに変えてsubmitしました。 -->
To verify this, I changed the site and bird predictions not included in the train_soundscape to nobird.
<!-- ここで変化がなければtrain_soundscapeととpublic LBは同じ分布であると考えられますが、かなりscoreが低下しました。(0.71->0.58) -->
If there is no change in the score, it is assumed that train_soundscape and public LB have the same distribution, but the score has decreased considerably. (bird: 0.71->0.64, site: 0.71->0.58)
<!-- よってpublic LBにはtrain_soundscapeに含まれないsite・鳥が含まれることがわかります。 -->
So I can see that public LB contains sites and birds that are not included in train_soundscape.
<!-- ここから、強引な考えですが、publicとprivate はランダムに分割されていると予測しました。(これは厳密には不確実ではありますが、他に手のうちようがありませんでした) -->
From this, I forcefully predicted that public and private would be randomly split. (This is strictly uncertain, but there was nothing else I could do.)

## Experimental code and the inference notebook for best submission
Code: https://github.com/tattaka/birdclef-2021
Inference notebook: https://www.kaggle.com/tattaka/birdclef2021-submissions-pp-ave?scriptVersionId=64016465
Aggregation of the number of birds for post-processing: https://www.kaggle.com/tattaka/make-month-and-site-mask