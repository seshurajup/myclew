# [1st Place] Quick Solution

Thanks to all the hosts, participants, and teammates  @startjapan , @namakemono .
We are also surprised by this result. We think it is because our solution was quite different from the other teams.

I'm not confident in my English, but I hope you all can understand.

We didn't expect to get the top spot, so we weren't prepared to share our solution.
Here is a brief explanation of the solution. There are three stages.

## 1st stage
We used freefield1010 to classify whether it is a nocall or not from the melspectrogram.
https://academictorrents.com/details/d247b92fa7b606e0914367c0839365499dd20121

## 2nd stage
We used short audio to predict which bird is singing.
kkiller( @kneroma ) 's notebook helped us a lot. I think it would have been difficult for us to win without it.
The short audio has noisy labels, so we used the results of the 1st stage to weight the labels.
We used train sound scapes as validation.
The number of models used in the final submission was 10.

## 3rd stage
We extracted 5 candidates from the results of the 2nd stage, and together with the information from meta data, we trained lightgbm to predict whether the bird would be included in the answer.
Only short audio was used for training, and train sound scapes were used for validation.

## Post-processing
The optimal threshold is determined using Ternary search.
Since the threshold varies depending on the percentage of nocall, we used both the case where the percentage of nocall is not changed and the case where the percentage is reduced to 54% as the final submission. However, it was better not to change it.

## Machine
All three of us used Colab Pro as our main machine.

There are a few other tricks.
I'll share the detailed solution later. Please look forward to it.

## 6/4
My teammate @startjapan wrote about our detailed solution.
If you're interested in our detailed solution, please check it out !

https://www.kaggle.com/c/birdclef-2021/discussion/243927