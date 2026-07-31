# NVARC solution

The main components of our solution:
- Multi-stage synthetic data generation pipeline
- Improved version of the ARChitects solution
- Improved version of the Tiny Recursive Model

Our overall workflow can be summarized by this picture. We will review each part in the rest of this writeup.

![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F75976%2Ff4f759b7249fc2c56a507a65fee16fe6%2FScreenshot%202025-12-04%20at%2020-36-51%20ARC-AGI-2%20NVARC%20writeup%20-%20Google%20Docs.png?generation=1764877039515916&alt=media)

## Synthetic Puzzles

We build a Synthetic Data Generation (SDG) pipeline with 4 main stages:
1. Collect puzzle descriptions and generate puzzle summaries
2. Mix two summaries and generate new more complex summaries
3. Generate python code for the input grid logic
4. Generate python code for the output grid logic

The majority of generated data is made with `gpt-oss-120b` model using the NeMo-Skills framework. This framework provides many tools for LLM development, including large-scale SDG pipelines on Slurm cluster. You can find more information how to use this framework for SDG [here](https://nvidia-nemo.github.io/Skills/tutorials/2025/08/29/inference-with-gpt-oss-120b-using-stateful-python-code-execution/#synthetic-data-generation). Running this model on a single node with 8xH100 GPUs gives us 15k tokens/s generation throughput.

### Puzzle summaries

We collect the puzzle descriptions from two main sources. First, we found [H-ARC](https://github.com/Le-Gris/h-arc), which was surprisingly underestimated in the ARC community. This dataset contains solution attempts from over 1700 humans on ARC problems, but more importantly it includes the natural-language solution descriptions. Second, we took 160 human written descriptions from [BARC](https://github.com/xu3kev/BARC). After filtering and combining these descriptions we had descriptions for 716 training puzzles from ARC-AGI-2.

### Mix summaries

We prompted LLM to join two puzzle summaries and produce a new more complex puzzle by combining elements from both. In total, we generated 266593 new puzzle summaries.

### Input grid programs

Instead of generating a full puzzle program with input/output logic, we decided to split this into two stages. First - generate a Python program for the input grid logic, second - generate a program of the output grid logic, i.e. transformation rules. To generate reliable and good input grid logic we prompt the LLM to produce additional unit test for the input grid. In total, we generated and filtered 126901 input grid programs in Python code.

### Output grid programs

We prompt LLM with puzzle description and corresponding input grid program to generate the relevant output grid program. Here we apply a different validation strategy. Instead of validating Python code with unit tests, we prompt the LLM multiple times for each input grid logic and filter out solutions which are not consistent. After filtering we had 103253 puzzles, where each puzzle has 1 input grid program and multiple output grid programs, but all of these output programs produce consistent output grids.

## The ARChitects

For this approach we try to use as many puzzles as possible. In addition to our synthetic data we also used a few real puzzles datasets. Our final dataset includes 3.2M augmented samples, where each sample has up to 7 input/output pairs.

| Source | Unique puzzles | Augmented samples \\ per puzzle | Total samples | Share |
| ------- | ---------------: | ---------------------------------- | --------------: | ----- |
| [MINI-ARC](https://github.com/KSB21ST/MINI-ARC) | 147 | 256 | 37632 | 1.2 |
| [ConceptARC](https://github.com/victorvikram/ConceptARC) | 160 | 256 | 40960 | 1.3 |
| [RE-ARC](https://github.com/michaelhodel/re-arc) | 400 | 256 | 102392 | 3.2 |
| ARC-AGI-2 | 609 | 256 | 155904 | 4.8 |
| NVARC training | 47337 | 24 | 1132633 | 34.8 |
| NVARC full | 55886 | 32 | 1785960 | 54.9 |
| | 104539 | | 3255481 | |

ARC-AGI-2 training subset excludes RE-ARC puzzles. NVARC training subset based only on training puzzles from ARC-AGI-2. NVARC full subset uses both training and evaluation puzzles from ARC-AGI-2.

The ARChitects approach has a simple representation of puzzles as a list of input/output grids. But we made it even simpler and used a dialog style template from Qwen3 model. For example, formatted message represents 1 pair of input/output grids looks like this:

```
<|im_start|>user\n123\n456<|im_end|>
<|im_start|>assistant\n78\n90<|im_end|>
```

This simpler representation requires only 16 tokens: 10 - for digits, 1 - new line, “user” - start of input grid, “assistant” - start of output grid, 2 special tokens (`<|im_start|>`, `<|im_end|>`) and 1 for padding `<|endoftext|>`. We used the [NeMo RL](https://github.com/NVIDIA-NeMo/RL) framework for the post-training stage. We run supervised fine-tuning with Megatron backend which allows us to efficiently utilize memory and computation resources of multiple nodes with H100 GPUs. For example, to do a full fine-tuning of the 4B model we used 4 nodes of 8xH100 for 27 hours.

We used LoRA test-time fine-tuning for each puzzle independently with r = 256 and alpha = 32. We removed gradient checkpointing and we removed 4-bit quantization too. Test-time fine-tuning was run with bfloat16 precision. We also used Flash Attention 2 with Unsloth Framework. You can find all hyperparameters in our [source code](https://github.com/1ytic/NVARC).

The main optimization we made in the ARChitects approach is for the decoding stage. We implemented the batch version of the Depth First Search (DFS) algorithm. However, there is one downside effect of batch implementation. It became nondeterministic. This phenomenon is well explained in a [blog](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference) from Thinking Machines Lab. We used an open-sourced [solution](https://github.com/thinking-machines-lab/batch_invariant_ops) from Thinking Machines Lab and managed to make inference batch invariant. However, despite the better precisions, and better local validations scores, this version runs much longer in the Kaggle environment. About 17% slower, and we didn’t use it in the final submission.

The ARChitects approach uses additional augmentations to rescore the candidates from DFS stage. We made a small change here. We used only 8 augmentations per candidate solution, but we used exactly the same augmentations for each candidate solution. This makes the scores of different solutions more comparable. After the competition deadline we tested the different re-scoring strategies and found the better method to select the right candidate. We calculate how many times the candidate solution was found during the DFS stage and adjust this with the geometric mean ensemble
of log-probabilities from different augmentations.

![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F502311%2F061f66e26ce71349a1fab3a2d9adc778%2Fnvarc_evaluation.png?generation=1764873955755619&alt=media)

During the competition we fine-tuned models on different portions of synthetic data. In the figure you can see the effect from adding more data on the loss function during the pretraining stage. The local validation loss measured on an augmented version of 120 evaluation puzzles, and we saw a good correlation with public leaderboard. Our best model scored 27.64% during the competition.

## Tiny Recursive Model

During the last 10 days or so of the competition we explored the use of the Tiny Recursive Models ([TRM](https://github.com/SamsungSAILMontreal/TinyRecursiveModels)) by Alexia Jolicoeur-Martineau.

The day before last we managed to get a working submission that ensembles TRM with the Qwen3 2B model. We lacked time to tune the ensemble and the TRM model itself, but we were glad to be able to test the idea still. We used a few late submissions to finish testing the ensemble and report on these as well. We only used models pretrained during the competition in these late submissions.

Using the [code](https://github.com/SamsungSAILMontreal/TinyRecursiveModels), the ARC Prize organization got a score of 6.9 percents by adapting it to run on 4 nodes of 8 H100 GPUs. Using that amount of compute resources is impossible in Kaggle. To make it worse, we could only use 2 hours or so for TRM given we had to use the rest for the ARChitects approach. Therefore, we had to use a different method. We first pretrained the TRM model, then fine-tuned the TRM model using test data during submission.

Using a batch size of 3072 instead of 768, and learning rate of 3e-4 instead of 1e-4 we could train TRM to the same accuracy as the original code in 24 hours on 8xH100. Using this in a submission yields a score of 2.08% only.

We then tuned various parameters to get the best score while using at most 2 hours of running time on Kaggle. We ended up with these parameters: 4 H cycles instead of the default 3, and 10 halt max steps instead of 16. We used 2000 epochs and only 200 warmup steps. This makes TRM run in about 2 hours on Kaggle.

In order to assess the strength of the TRM model we trained it twice. We trained it with the 4k puzzles and used that for submission. And we also trained it with the competition evaluation dataset removed. We then scored the latter on the evaluation data. We then looked at how many training steps were used for the checkpoint with best pass@2. We finally selected the checkpoint trained on all 4k samples with the closest number of training steps. Best pass@2 on evaluation with a model not trained on evaluation was 9.44%. When submitted the matching checkpoint trained with evaluation scored 7.5%. This is much better than the 2.08% we had before.

After the competition deadline we submitted our best TRM using 4k epochs instead of 2k. It yield a score of 10.0%, in less than 4 hours.

In order to use TRM with the ARChitects approach we modified TRM so that it generates 10 attempts instead of 2 per puzzle. These attempts were then added to the attempts generated by The ARChitects procedure. They were then scored by the Qwen3 model like the other attempts.

This worked fine but with mixed results. Most of the puzzles solved by TRM were solved by Qwen3, hence TRM added nothing there. However, about 2 or 3 puzzles solved by TRM were not solved by Qwen3. Unfortunately, these were not always picked by Qwen3 scoring.

With two late submissions we found that we could improve the 21.53 score obtained with Qwen3 2B to 22.50 with a TRM ensemble. We also saw that when using a Qwen3 4B submission that uses 10 hours only, with a score 27.22, adding TRM yields the same 27.22 score.

While we think that there is some potential for improvement here, ensembling the Qwen3 4B model and TRM model would require more investigations and experiments than we could perform.

## Conclusion

We presented how we improved The ARChitects and TRM models to get the best score on the public leaderboard of ARC Prize 2025 competition. With limited resources during the evaluation stage we needed a good way to compress learned knowledge and skills. Scaling the pretraining stage with good synthetic puzzles was the right direction to succeed in this Kaggle competition. But we also believe that this synthetic data with python programs and reasoning traces could be used for other research directions.