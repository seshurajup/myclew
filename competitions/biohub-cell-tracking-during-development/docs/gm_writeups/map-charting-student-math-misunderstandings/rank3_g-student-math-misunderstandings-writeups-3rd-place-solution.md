# 3rd place (Public 1st) solution : monsaraida & Masaya

First, sincere thanks to the organizers and the Kaggle community for discussions, public notebooks, and feedback. Our team, monsaraida & Masaya, finished Private 3rd (Public 1st). We are grateful for the support.

## Overview

Here is an overview of our solution. In the following sections, we describe monsaraida’s part and Masaya’s part in turn.  
![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F2668122%2Ffe5d400cf85da3294983f8667f2be251%2FKaggle_MAP_3rd_place_solution_overview.svg?generation=1760620299013691&alt=media)

## monsaraida’s part

### Prompting

#### Choices

For the {Question / Answer / Correct? / Student Explanation} structure used in public notebooks, I changed {Answer} to {Choices / Selected}. This change is intended to help the LLM identify potential misunderstandings by presenting not only the student's chosen Answer but also the other options.This change was significant, improving the CV/Public score by over \+0.001. This idea also significantly improved Masaya's model accuracy, so it may be one of the key tricks in this competition.

#### Hints

In this competition, the types of misunderstandings for each question were predetermined. Therefore, as a hint, I added a list of misunderstandings to the prompt for each question, ordered by their probability of occurrence. This change improved my CV score by \+0.0003.

#### Example of prompt
```
Question: What fraction of the shape is not shaded? Give your answer in its simplest form. \[Image: A triangle split into 9 equal smaller triangles. 6 of them are shaded.\]  
Choices: (A) \\( \\frac{1}{3} \\) (B) \\( \\frac{3}{9} \\) (C) \\( \\frac{3}{6} \\) (D) \\( \\frac{3}{8} \\)  
Selected: \\( \\frac{1}{3} \\)  
Correct? Yes  
Student Explanation: 1 goes into everything and 3 goes into nine  
Common mistakes for this question: Incomplete, WNB
```

### Training

#### Base model

I adopted Qwen/Qwen3-14B as the base model after testing the following alternatives and selecting the best-performing one: google/gemma-2-9b-it, Qwen/Qwen2.5-14B-Instruct, Qwen/Qwen2.5-Math-7B-Instruct, deepseek-ai/deepseek-math-7b-instruct, Qwen/Qwen2.5-Coder-14B, deepseek-ai/DeepSeek-R1-Distill-Qwen-14B, mistralai/Mistral-Nemo-Instruct-2407, mistralai/Mixtral-8x7B-Instruct-v0.1, mistralai/Mathstral-7B-v0.1, AmanPriyanshu/gpt-oss-10.8b-specialized-instruction\_following-pruned-moe-only-15-experts

#### LoRA SFT with multi-task learning

I performed supervised fine-tuning with LoRA and used AutoModelForSequenceClassification for classification. During classification, I used multi-task learning with the main task “Category: Misconception (65 classes)” and the following auxiliary tasks:
• True / False (2 classes)
• Correct / Misconception / Neither (3 classes)
• Misconception (36 classes)
Multitask learning contributed to a slight improvement in accuracy.

#### Improving generalization

Due to the high noise levels in this competition's labels, I introduced R-Drop(Regularized Dropout) as a measure to improve generalization. The effect of R-Drop was significant, improving the CV by over \+0.001. Adding AWP (Adversarial Weight Perturbation) on top of R-Drop led to a further, slight CV improvement. Furthermore, introducing the EMA (Exponential Moving Average) stabilized the CV and improved accuracy.

### Inference

#### Multi-stage inference

To ensemble multiple models within the 9-hour inference time limit, I implemented multi-stage inference. In the first stage, I performed inference on the entire test set and ensemble the results. In the next stage, I sorted the results by confidence in ascending order and re-ran inference only on the lowest-confidence 50%. (I also tried more complex staging, but it did not yield the best Private score.)

#### Tips for faster inference

Some public notebooks used \`torch\_dtype=torch.bfloat16\`, but bfloat16 is unavailable in Kaggle’s T4 execution environment, which adds conversion overhead. Changing bfloat16 to float16 made inference 2 times faster. Furthermore, since this competition often involves short prompts, setting \`padding=False\` instead of \`padding="max\_length"\` achieved another 2 times speedup. Combined, these changes resulted in a 4 times speedup, enabling the use of more models in the ensemble and contributing to improved accuracy.

## Masaya’s part

#### Overview

Overview of my pipeline is here,
![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F2668122%2Fe4b2c5269fee5745de7beb63998ef3da%2FKaggle_MAP_3rd_place_solution_Masaya.svg?generation=1760620542415916&alt=media)

One of the advantages of using a CausalLM is that it can achieve higher accuracy even when the labels are of variable length.  
In this competition, since the labels appearing for each QuestionId were predetermined, I hypothesized that limiting the label set for each QuestionId — instead of performing inference across all 37 or 65 labels — would improve accuracy.  
Therefore, I mainly conducted experiments using CausalLM.

My single best model was Qwen2.5-72B-Instruct-GPTQ, but inference on the test set took about 3 hours.  
Therefore, I designed a system where relatively smaller models such as 14B and 32B were used for inference first, and only samples with low top1 probability were re-inferred using the 72B model.  
This approach reduced the inference time for the 72B model to about 1 hour.

Note: This pipeline is used for our Public LB Best and The Private LB score was slightly lower than the pipeline mentioned above.

#### Preprocess

In this competition, both the train and test datasets had a limited number of QuestionIds (only 15). Additionally, each QuestionId had only 2 to 5 Misconceptions, which was also limited. Therefore, to narrow down the predicted tokens of the CausalLM, the prompt was designed to include only the Misconception, Correct, and Incorrect options that appeared for each specific QuestionId. As discussed [here](https://www.kaggle.com/competitions/map-charting-student-math-misunderstandings/discussion/589400), the `True_` / `False_`  labels could be assigned later using rule-based logic, so they were excluded as well to further reduce the number of prediction tokens. This approach significantly improved accuracy while also reducing inference time. (LB/CV \+0.002\~0.003 and faster about 3 times)  
As mentioned above, adding other options to the prompt helped boost accuracy.(LB/CV \+0.003\~0.004)

Here is prompt example,   
```
\<|im\_start|\>system  
You are Qwen, created by Alibaba Cloud. You are a helpful assistant.\<|im\_end|\>  
\<|im\_start|\>user

You are a Mathematics teacher.   
Your task is to evaluate whether the student’s explanation contains a misconception, and if it does, to identify the specific type of misconception.

Question Text: Calculate \\( \\frac{1}{2} \\div 6 \\)  
Choices: (1) \\( \\frac{1}{12} \\), (2) \\( 3 \\), (3) \\( \\frac{6}{2} \\), (4) \\( \\frac{1}{3} \\)

Student Answer: \\( 3 \\)  
Is Student Answer Correct: No

Student Explanation: Because you do half of 6 \= 3

You must choose the your answer from the following options:

A. Correct:NA   
B. Neither:NA   
C. Misconception:FlipChange   
D. Misconception:Mult   
E. Misconception:SwapDividend 

\<|im\_end|\>  
\<|im\_start|\>assistant  
Your Answer:   
```

#### Training Models

Mainly, I tuned Qwen2.5 as the CausalLM.  
The single best model was Qwen2.5-72B-Instruct trained with EMA, but since its performance on the public leaderboard was lower compared to similar models, I decided not to use it for the final submission.  
And I used LoRA for finetuning, `r=16,lora_alpha=32,lora_dropout=0.05,target_modules=( "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")`.  
Main training config is here,

* epoch=2  
* learning_rate=1e-4  
* per_device_batch_size=4  
* gradient\_accumulation\_steps=4

Tips: High learning rate improved better score, but too high learning rate caused collapse while training.

| model  | task | cv | lb\* | pd | use final | inferrence time | Note (other info) |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| Qwen/Qwen2.5-14B-Instruct-AWQ | CausalLM | 0.9469 | 0.943 | 0.942 | ✅ | 30 min |  |
| Qwen/Qwen2.5-32B-Instruct-AWQ | CausalLM | 0.9498 | 0.949 | 0.944 | ✅ | 60 min |  |
| Qwen/Qwen2.5-72B-Instruct-GPTQ | CausalLM | \- | 0.950 | 0.946 | ✅ | 190 min |  |
| Qwen/Qwen2.5-72B-Instruct-GPTQ | CausalLM | \-  | 0.950 | 0.948 |  | 190 min | use EMA for training (Single Best) |
| Qwen/Qwen2.5-32B-Instruct-AWQ | CausalLM | 0.9474 | 0.945 | 0.941 |  | 60 min | Without choices in prompts |
| Qwen/Qwen2.5-72B-Instruct-GPTQ | CausalLM | \- | 0.946 | 0.940 |  | 190 min | Without choices in prompts |

(\*) CV is calculated from fold 0 (because of lack of GPU). LB is from the submitted full-train model.

#### Inference

The prompt design during inference was the same as before.  
At this stage, since each token’s meaning differs depending on the QuestionId, the outputs are converted back to 65 labels to align with monsaraida’s probability.  
The `True_` / `False_` labels can be inferred from the value of `is_correct`.  
As for the 72B model, since it cannot be inferred using LoRA, I performed LoRA merge and quantization (GPTQ), and executed it while offloading to the CPU. (Reference [here](https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics/discussion/550223))

#### Not Worked for me

* Models other than Qwen2.5 including Llama 3.3-70B and gemma2.  
* Causal LM for pair-wise decision. This is too large a time for inference.  
* Fixing the choices token for each label. For example,

```
A. Correct:NA   
B. Neither:NA   
F. Misconception:FlipChange   
D. Misconception:Mult   
R. Misconception:SwapDividend   
```

* Pretrained other competition data. I used [Eedi competition](https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics/overview) pretrained model but it didn’t work.  
* Considering the label noise as discussed [here](https://www.kaggle.com/competitions/map-charting-student-math-misunderstandings/discussion/589400). For `QuestionId=31778`, I included the reversed `True_` / `False_` version in the top 2 predictions for the Correct Neither samples that had high confidence. However, this slightly decreased accuracy on the final results, although there was a small improvement in CV.

#### Reference

* [vllm](https://docs.vllm.ai/en/latest/)
* [auto-round](https://github.com/intel/auto-round)
* [Qwen](https://huggingface.co/Qwen)
* [Eedi competition](https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics/overview)  
* [Qwen2.5-72B & Llama 3.3-70B on 2xT4(questionable performance)](https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics/discussion/550223)
* [Does Test Data Have Questions Different Than Train Data?](https://www.kaggle.com/competitions/map-charting-student-math-misunderstandings/discussion/589400)