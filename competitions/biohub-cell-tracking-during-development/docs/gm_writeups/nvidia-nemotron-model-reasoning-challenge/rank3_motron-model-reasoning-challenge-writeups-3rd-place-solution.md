# Overview
My overall solution is SFT on deterministic reasoning traces produced by problem specific solvers. The reasoning traces are designed such that the final answer can be reached from the problem statement step-by-step without any hidden deduction or “jumps”.

It builds on top of the excellent early work and ideas from [llkh0a](https://www.kaggle.com/code/llkh0a/nemotron-unsloth-sft-training-3-30-2), [huikang](https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion/689915), [kemshim](https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion/703479), [youkinasa](https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion/701981) and others, with the following improvements:

* Rewritten numeric solver that is more minimal and with a better guessing logic
* Hui Kang-style bit manipulation trace compacted with hex notation and extended with majority operation search
* Data augmentation for all problem types except gravity, unit conversion and numeral
* Constraint satisfaction problem / CSP-like algorithm that solves overall 24.2% of cryptarithm problems within token budget (27.8% of deduce cases, 9.8% of guess cases) using multiplication patterns and backtracking search
* Two stage training approach that decouples memorization and execution to help model learn the cryptarithm trace better
* Local validation strategy that turns out to track the private test set well

In the following sections, I will cover different aspects of the solution in more details.

# Quick links

* Uploaded final model: [Nemotron Reasoning 3rd Place LoRA](https://www.kaggle.com/models/liauys/nemotron-reasoning-3rd-place-lora)
* Submission notebook using the final model: [Nemotron Reasoning 3rd Place LoRA Submission](https://www.kaggle.com/code/liauys/nemotron-reasoning-3rd-place-lora-submission)
* Combined reasoning trace dataset used to train the final model: [Nemotron Reasoning Traces](https://www.kaggle.com/datasets/liauys/bit-v5-unit2-cpat-v24-a8-cap48-48k-rc3k)
* Source code on GitHub to reproduce the solution end to end: [nvidia-nemotron-reasoning-3rd-place-solution](https://github.com/YS-L/nvidia-nemotron-reasoning-3rd-place-solution)

# Solver and trace design

## Gravity, Unit conversion, Numeral

These are the trivial categories that the model can do very well with minimal training. I reused the traces from llkh0a's [notebook](https://www.kaggle.com/code/llkh0a/nemotron-unsloth-sft-training-3-30-2) directly.

## Cipher

I simplified the trace from the same [notebook](https://www.kaggle.com/code/llkh0a/nemotron-unsloth-sft-training-3-30-2) and removed the elaborate analysis on missing letters. I found that simply adding more augmented samples is sufficient to teach the model to fill in missing letters due to the small vocab size involved.

## Bit manipulation

I modified Hui Kang-style bit manipulation trace slightly with the following changes:

The first change is to use hex-notation with tail bits to more compactly represents output bit columns.

Before:
```
OR-NOT
01 1110001110 6
12 1111110111 9
23 1111101011 8
34 1010011111 7
```

After:
```
OR-NOT
01 E3|10 6
12 FD|11 9
23 FA|11 8
34 A7|11 7
```

I tried encoding the last two bits as as a hex character using either left or right padding, but that seemed to confuse the model and caused problems with 9 or 10 examples to perform worse. 

This brings about 25% reduction in token usage, and allows the second change: search over majority operation (MAJ) to fit under token budget.

Lastly, I changed the solver's rule ordering so OR-NOT candidates are tried before plain OR and XOR:

* Default rule ordering: `Identity`, `NOT`, `Constant`, `AND`, `OR`, `XOR`, `AND-NOT`, `OR-NOT`, `XOR-NOT`, `MAJ`
* New ordering: `Identity`, `NOT`, `Constant`, `AND`, `OR-NOT`, `OR`, `XOR`, `AND-NOT`, `XOR-NOT`, `MAJ`

which empirically improves the solve rate slightly (0.6%).

## Numeric equation

After solving a number of numeric equation puzzles manually, I found that there are two underlying structures governing the equations:

1. "simple": `f(a,b) = fmt(op(a,b))`
1. "reverse": `f(a,b) = fmt(rev(op(rev(a),rev(b))))`

where `op` refers to a hidden operation represented by the operator symbol, `rev` means reversing the digits, and `fmt` means formatting negative answers using the optional pre or post negative markers.

In addition, the "simple" or "reverse" structure has to be applied globally across the equations in the same problem, rather than independently within each equation in the same problem.

I rewrote a solver for numeric equations from scratch, adding operation one by one to identify the operations that are truly needed to solve all the problems. I arrived at this set of operations:

```
[
    "a+b",
    "a-b",
    "abs(a-b)",
    "negated absolute diff",
    "b-a",
    "a*b",
    "a+b+1",
    "a+b-1",
    "a*b+1",
    "a*b-1",
    "concat(a,b)",
    "concat(b,a)",
    "max(a,b) mod min(a,b)",
]
```

I think this is the exact set of operations that should be searched over and nothing more, since if the solver were allowed to "peek" at the answer by including the question as an example, all of the problems can be solved without any ambiguity (i.e. with one unique solution).

Without peeking at the answer, sometimes there can be multiple operations that happen to fit all the examples. This happens only for the subtraction type operations: conflicts occur frequently between `['a-b', 'negated absolute diff']`, `['abs(a-b)', 'a-b']`, etc. I haven't found a way to disambiguate that is better than random guess, so I choose the above deterministic ordering (e.g. preferring `a-b` over `negated absolute diff`) to maximize the solve rate on the training numeric problems.

There are more interesting patterns. For example, this is a tabulated co-occurrence counts between pairs of true operations within each problem:
![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F91259%2F6017e38a46b12b9e8656a08e95ec3bf5%2Fnumeric-op-crosstab.png?generation=1781721624926232&alt=media)

The organized patches of zeros in the table shows a clear pattern: `a*b` never appears together with `a*b-1` or `a*b+1` in the same problem; `a+b` never appears together with `a+b-1` or `a+b+1`, etc. This suggests that in the data generation process, there are broad "families" of operations that the data generator will sample from, and within each sampled family only one operation will be chosen. I used this to improve the guessing logic when the question's operator is unknown: the optimal guess is an operation in a family that has not been mapped yet in the existing examples.

Using the above findings, I implemented the following strategy:

1. Tries two structures in this order
    1. "simple": `f(a,b) = fmt(op(a,b))`
    1. "reverse": `f(a,b) = fmt(rev(op(rev(a),rev(b))))`
2. For each structure
    1. Go through each operator symbol
        1. Test candidate operation in the above order
        1. Find the first candidate operation that fits every example answer pairs with this operator
    1. Exit early if any operator cannot be explained, this means the structure is incorrect
3. Prefer "simple" over "reverse" if both fully explains all observed operators
4. If the question's operator is unseen, guess in this priority: `a-b`, `a+b`, `a*b`, choosing the unseen operation type

With this solver, the solve rate ceiling for the training data is:
* deduce: 559 / 596 = 93.79%
* guess:  53 / 136 = 38.97%
* overall: 612 / 732 = 83.61%

## Cryptarithm

My strategy for cryptarithm is inspired by the multiplication pattern lookup ideas floated around in the discussions by [kemshim](https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion/703479) and [youkinasa](https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion/701981). The idea is to use multiplication examples to propagate constraints and provide a good starting point for backtracking solver with most of the digits already solved.

### Pattern lookup table
An encrypted equation such as `{!*{| = ]!]]` can be normalized to a "signature" that describes how different symbols relate to each other, in this case:
```
{!*{| = ]!]] -> ab-ac-dbdd
```
Assuming the operator represented by `*` is multiplication, and the structure governing the equation is "reverse" (see explanation for Numeric equation above), what combinations of `a`, `b`, `c` and `d` would satisfy the equation? It turns out to be not many:

| operand_1 | operand_2 | answer | equation | a | b | c | d |
|---|---|---|---|---:|---:|---:|---:|
| `26` | `27` | `4644` | `62 * 72 = 4464; reverse -> 4644` | `2` | `6` | `7` | `4` |
| `93` | `92` | `1311` | `39 * 29 = 1131; reverse -> 1311` | `9` | `3` | `2` | `1` |

There are exactly 2 of them. From this table, the raw symbols corresponding to `abcd` must be either `2674` or `9321`. Looking at each individual column gives the possible digits for each symbol (e.g. `b` -> `!` has to be either 6 or 2).

With this lookup table, we can quickly narrow down the set of possible digit mappings based on just the signature. This is especially true if there are multiple equations that need to be satisfied simultaneously, since many digit mappings become invalid due to conflicts.

I pre-computed the lookup table by enumerating all 100 * 100 two-digit operand pairs under both structures (`simple` and `reverse`) and the three multiplication-family operations (`a*b`, `a*b+1`, `a*b-1`), each yielding a signature. Candidate digit mappings are recorded for each `(signature, structure, multiplication type)` combination.

I focus on multiplication-family operations because they constrain more symbols than addition or subtraction. In the multiplication-family lookup, there are 3,678 unique signatures and 11,334 non-empty `(signature, structure, multiplication type)` triples.

### Cryptarithm strategy

Assuming we can recall information from the pattern lookup table perfectly, we can solve a subset of cryptarithm problems using this strategy:

1. Identify multiplication (MUL) examples
1. Each MUL example emits the following from the pattern lookup table:
   1. Possible digits for each symbol
   1. Possible candidate mappings
1. When there are more than one MUL examples, compute their intersection.
   1. Intersection finding requires "joining" candidate mappings from different MUL examples
   1. Two (or three, up to four) candidate mappings can either be merged (no conflict at all) or rejected due to conflict
1. The result of the intersection is a much more constrained set of possible candidate mappings, at this point often exactly 1, since most will be rejected due to conflicts. Let's call each of these merged candidate mapping an anchor mapping.
1. For each anchor mapping, launch a backtracking symbolic solver to solve for the remaining unknowns over the free digits
   1. At each step, choose the example with the least number of unknowns and solve that equation
   1. For non-negative answers, try `a+b`, `a+b+1`, `a+b-1`, `a-b`, `abs(a-b)`
   1. For negative answers, try `a-b`, `-abs(a-b)`, `b-a`
   1. Backtrack when no valid solution can be found
   1. Stop early once all unknowns are solved

The above steps are repeated for each (structure, multiplication type) pairs, where structure can be `simple` or `reverse`, and multiplication type can be `a*b`, `a*b+1`, `a*b-1`. The algorithm greedily tries `reverse` first, then `simple`, and stops early whenever it finds a solution that satisfies all the examples.

My symbolic solver supports up to 2 unknowns per equation for simplicity. Even this is sufficient to solve a non-trivial number of problems as the anchor finding phase is often doing most of the heavy lifting.

### Example
It's best to look at a real trace to see exactly what the algorithm is doing. For this problem:
```
In Alice's Wonderland, a secret set of transformation rules is applied to equations. Below are a few examples:
|"-!{ = ?]
{!*{| = ]!]]
"!*\[ = \[]/
/:-{: = |!"
Now, determine the result for: |/*/!
```

The solution trace is in the attachment (`cryptarithm_example.txt`). Let's look at a the key sections.

Equation 2 and 3 are the MUL examples as they have answers of length 4.

This section in the trace emits the possible digits for the symbols based on the lookup table:

```
Sig E2: sig=ab-ac-dbdd map a->【{】 b->【!】 c->【|】 d->【]】
<RECALL_DOMAINS>
sig=ab-ac-dbdd type=R*
a 29
b 36
c 27
d 14
</RECALL_DOMAINS>
Apply symbol map:
【{】 29
【!】 36
【|】 27
【]】 14
```

It means the following: assuming the MUL operator is `a*b` in reverse form, the symbol `{` must be either 2 or 9, `!` must be either 3 or 6, etc. `R*` is a shorthand to refer to `a*b` in reverse form. A similar section is also emitted for E3, but with different domains.

This section intersects the domains for different MUL examples:

```
Possible digits R*:
【!】 E2=36,E3=23456789>36
【"】 E3=1367>1367
【/】 E3=1234567>1234567
【[】 E3=23456789>23456789
【\】 E3=024568>024568
【]】 E2=14,E3=0123456789>14
【{】 E2=29>29
【|】 E2=27>27
```

For the symbol `!`, E2 says it has to be either 3 or 6, while E3 says it has to be either 2, 3, 4, 5, 6, 7, 8, or 9. To satisfy both, this symbol has to be either 3 or 6. In this problem, the intersection doesn't reduce the domain much because E2 is already constraining the domains quite well.

The next section enumerates candidates mapping for each signature from the lookup table:

```
Emit pattern sequences in raw order for R*.
Patterns E2: sig=ab-ac-dbdd map a->【{】 b->【!】 c->【|】 d->【]】
<RECALL_ROWS>
sig=ab-ac-dbdd type=R*
2674
9321
</RECALL_ROWS>
Filter R* recalled rows with domains:
a 29
b 36
c 27
d 14
Passing rows:
2674 OK E2_1: map 【{】 as 2; 【!】 as 6; 【|】 as 7; 【]】 as 4
9321 OK E2_2: map 【{】 as 9; 【!】 as 3; 【|】 as 2; 【]】 as 1
Patterns E3: sig=ab-cd-cdef map a->【"】 b->【!】 c->【\】 d->【[】 e->【]】 f->【/】
<RECALL_ROWS>
sig=ab-cd-cdef type=R*
130842
130972
140782
140823
140963
145703
145983
152763
152964
154623
156432
156783
156834
156984
158643
158793
160724
160945
165472
170482
170624
170865
170936
180342
180423
180765
180927
185463
185706
185967
190372
190463
190645
190736
190827
340512
375281
380514
390564
670251
672593
674281
680271
690483
720531
730581
740532
780534
</RECALL_ROWS>
Filter R* recalled rows with domains:
a 1367
b 36
c 024568
d 23456789
e 14
f 1234567
Passing rows:
130842 OK E3_1: map 【"】 as 1; 【!】 as 3; 【\】 as 0; 【[】 as 8; 【]】 as 4; 【/】 as 2
160945 OK E3_2: map 【"】 as 1; 【!】 as 6; 【\】 as 0; 【[】 as 9; 【]】 as 4; 【/】 as 5
```

The digits correspond to the normalized symbols in the signature (2674 for `ab-ac-dbdd` means a:2, b:6, c:7, d:4, and abcd can be mapped to the actual encrypted symbols). To preserve token budget, only mappings satisfying the intersected domains are explicitly spelled out as `E2_1, E2_2`, etc. My hypothesis is that spelling out the mappings explicitly provides the context necessary for the model to perform the merge / conflict detection in the next step.

This next section performs joins:

```
Join entry: pathY=A# or pathNS【symbol】oldNew or pathNDigit【old】【new】.
K R* E2=E2_1,E2_2
K R* E3=E3_1,E3_2
J R*=4 tuple(s)
J R*
E2_1xE3_1NS【!】63
E2_1xE3_2Y=A1
E2_2xE3_1NS【]】14
E2_2xE3_2NS【!】36
Intersected anchor examples for R*:
A1: structure reversed; op a*b; constructed from E2_1xE3_2
  E2_1: map 【{】 as 2; 【!】 as 6; 【|】 as 7; 【]】 as 4
  E3_2: map 【"】 as 1; 【!】 as 6; 【\】 as 0; 【[】 as 9; 【]】 as 4; 【/】 as 5
  merged map: 【\】 as 0; 【"】 as 1; 【{】 as 2; 【]】 as 4; 【/】 as 5; 【!】 as 6; 【|】 as 7; 【[】 as 9
```

There are 2 mappings from E2, and 2 mappings from E3 to consider, totaling 2*2 = 4 checks. The meaning of the lines:

* `E2_1xE3_1NS【!】63` means `E2_1` conflicts with `E3_1` because the symbol `!` is mapped to that two different digits (they should be consistent)
* `E2_1xE3_2Y=A1` means `E2_1` can be merged with `E3_2` without any conflicts. That merged mapping is spelled out as `merged map` above. This mapping will be referenced as `A1` in the backtracking solver below.

With the anchor mapping determined, the symbol solver can proceed to solve for the remaining unknowns:

```
Solve linear with A1: structure reversed; op a*b; map 【\】 as 0; 【"】 as 1; 【{】 as 2; 【]】 as 4; 【/】 as 5; 【!】 as 6; 【|】 as 7; 【[】 as 9
Linear mode: symbolic affine, max unknowns 2.
F A1: op 【*】 known a*b
Verifying anchor mapping against known linear examples:
Q A1: free 38; E1 U 【?】; E4 U 【:】 pick E1
S E1: rev-form a+b+c; c in 0,1,-1
17+26+c=40+【?】
3-【?】+c=0
【?】=3+c
c=0: 【?】=3 OK -> 【-】=a+b Y
c=1: 【?】=4 N used
c=-1: 【?】=2 N used
C L1 from A1: op 【-】=a+b; 【?】=3
Q L1: free 8; E4 U 【:】 pick E4
S E4: rev-form a+b+c; c in 0
10*【:】+5+10*【:】+2+c=167
-160+20*【:】+c=0
C L2 from L1: op 【-】=a+b; 【:】=8
Y L2
Y L1
```

Notations:
* `Q`: deciding the next equation to solve for based on the current assumptions and states
* `C`: commit to a new assumption
* `S`: symbolic steps to simplify and solve an equation
* `N <reason>`: conflict detected and will backtrack / proceed to the next branch
* `Y`: a valid solution for an operator or symbol is found

Finally, we can apply the found solution to the question and re-encrypt the answer:
```
Found solution: A1,L1,L2
Question symbols: 【|】【/】【/】【!】
Remaining unmapped question symbols: none
Remaining unmapped digits: none
Final char mapping: 【\】 as 0; 【"】 as 1; 【{】 as 2; 【?】 as 3; 【]】 as 4; 【/】 as 5; 【!】 as 6; 【|】 as 7; 【:】 as 8; 【[】 as 9
Final op mapping: 【*】 as a*b; 【-】 as a+b
Query op 【*】 known as a*b.

Applying to query: 【|】【/】【*】【/】【!】
f(【|】【/】,【/】【!】) -m-> f(75,56) -rev-> f(57,65) = 57*65 = 3705 -rev-> 5073
Numeric result: 5073
Encrypted result: 【/】【\】【|】【?】
Stop pattern search after query-valid solution.
```

### More implementation details
* Candidate mapping enumeration is capped at 48 per signature
* Maximum number of allowed anchor mapping is 8. Increasing this further doesn't improve solve rate under budget.
* If there are examples with length 3 answers that share the same operator as MUL examples, they should be treated as MUL examples as well.
* When there is a single unmapped question symbol and a single unmapped digit, the last step will establish the mapping between the two so the question can be answered.
* For questions with unknown operators, the solver applies the same guessing logic as numeric equations: choose an operation from the unused operation family
* The tags `<RECALL_DOMAINS>` and `<RECALL_ROWS>` are for anticipating sections in the trace that should be derived purely from memory. More on this later in the modeling section.

### Solve rate

I constructed a local dataset consisting of the original cryptarithm problems and numeric equations encrypted as additional cryptarithm problems. The solver's solve rate within token budget on this dataset:

* cryptarithm_deduce: 183 / 659 = 27.8%
* cryptarithm_guess: 16 / 164 = 9.8%
* combined: 199 / 823 = 24.2%

# Data augmentation and overall mix

## Cipher
There are 77 unique words and 4 sentence structures:

```
("subject", "verb", "object")
("subject", "verb", "the", "adj", "object")
("subject", "verb", "prep", "prep_object")
("the", "adj", "subject", "verb")
```

I generated 3K additional augmented samples, sampling from the sentence structures and pool of words uniformly.

## Bit manipulation

I augmented new problems for bit manipulation using two methods:

* Swap-augment: pick an existing problem in the original dataset, change the input values for each example (including question example) and apply the underlying transformation to get new output values. Then, shuffle the new examples and select a new question.
* Formula-augment: Construct completely new bit manipulation problems using two-ops / three-ops / majority / choice structures.

In the final dataset, I constructed 3K swap augmented samples and 3K formula augmented samples. Each sample is duplicated twice.

## Numeric equation

To create a new numeric equation problem, I sample a random original problem from a pool of source problems consisting of the original numeric problems and decrypted cryptarithm problems. For each example in this sampled problem, I change the operands with new random values and compute the updated answer. The operator symbols are changed too. This produces a new numeric equation problem.

The reason for doing this rather than creating completely new problems is that I don't want to accidentally create new artifacts or remove underlying structures hidden in the data. This can happen if my own generator differs too much from the official generator.

My final dataset contains all the original numeric problems, decrypted cryptarithm problems, and 3K augmented samples.

## Cryptarithm

To create a new cryptarithm problem, I first create a new numeric equation problem as per the previous section. Then, I apply a cipher encryption on this new numeric problem, which yields a new cryptarithm problem.

This is the category that needs the most number of augmented samples. My final dataset consists of 48K augmented cryptarithm samples, of which 4K samples are of concat type (that can be answered directly without going through the pattern matching path).

## Overall mix

After accounting for validation split (10% of original dataset are used as validation), the final training dataset is the following mix:

| Problem category | Samples | Token count | % of dataset |
|---|---:|---:|---:|
| cryptarithm | 43,157 | 199,165,000 | 59.63% |
| bit_manipulation | 13,998 | 83,769,094 | 19.34% |
| cipher | 4,418 | 5,896,685 | 6.10% |
| numeric | 4,076 | 6,168,838 | 5.63% |
| cryptarithm_pattern_drill | 3,000 | 2,164,784 | 4.14% |
| gravity | 1,437 | 1,195,592 | 1.99% |
| unit_conversion | 1,434 | 700,982 | 1.98% |
| numeral | 857 | 313,229 | 1.18% |
| **Total** | **72,377** | **299,374,204** | **100.00%** |

# Modeling: Two-Stage Training

I used Hui Kang's [training loop](https://www.kaggle.com/code/huikang/end-to-end-finetuning-for-lb-0-85) with cut cross entropy which reduces memory usage, allows a micro-batch size of 4, and yields better GPU utilization.

I found that:
* Disabling `MoE tie weights` gives better results.
* Effective batch size that is too large tends to under train the model. I started with effective batch size of 16, down to 8 over time and eventually settled on 4.

These are the training parameters I used for my final model:

| Parameter | Value |
|---|---:|
| Epochs | `1` |
| Batch size | `4` |
| Gradient accumulation | `1` |
| Effective batch size | `4` |
| Learning rate | `2e-4` |
| LR schedule | linear decay |
| Max sequence length | `8192` |
| Max grad norm | `10` |
| Seed | `42` |
| LoRA rank | `32` |
| LoRA alpha | `32` |
| LoRA dropout | `0.0` |
| LoRA dtype | `fp32` |
| LoRA target modules | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `up_proj`, `down_proj`, `in_proj`, `out_proj`, `lm_head` |
| Manual LM-head LoRA | `1` |
| MoE tie weights | `0` |

As discussed previously in the solver design section for cryptarithm problems, the success of a cryptarithm solution trace during inference time hinges on:
* Correct recall of possible digits and candidate mappings for any given signature and multiplication type
* Correct execution of the algorithm assuming the memories are recalled correctly.

While I think it's possible for the model to learn both from scratch in a single training run given sufficient amount of data, I found that a two stage training approach works better.

## First stage: learning to memorize

In this approach, I first train the model to memorize: recall possible digits and candidate mappings given any signature. For example, given this:

```
<RECALL_DOMAINS>
sig=aa-ba-acde type=R-
```

The model should learn to complete the request with:

```
a 37
b 48
c 59
d 26
e 15
</RECALL_DOMAINS>
```

Or given this:
```
<RECALL_ROWS>
sig=aa-ba-acde type=R-
```

learn to complete the request with:
```
38521
74965
</RECALL_ROWS>
```

The training dataset for the first stage consists purely of these "drill" style recall requests covering all the multiplication signatures from the lookup table. To speed up convergence, each training sample is constructed by packing 16 randomly selected recall requests. This is an example of a packed training sample:

```
<RECALL_DOMAINS>
sig=ab-cc-dcce type=S+
a 6
b 2
c 7
d 4
e 5
</RECALL_DOMAINS>
<RECALL_DOMAINS>
sig=ab-cc-acbd type=S-
none
</RECALL_DOMAINS>
<RECALL_ROWS>
sig=ab-cd-dbbb type=R-
none
</RECALL_ROWS>
<RECALL_ROWS>
sig=aa-bc-ad type=S+
2013
3014
4015
5016
6017
7018
8019
</RECALL_ROWS>
<RECALL_ROWS>
sig=ab-bc-ccbb type=S+
none
</RECALL_ROWS>
<RECALL_ROWS>
sig=aa-ab-cdeb type=S*
40176
62409
70539
</RECALL_ROWS>
<RECALL_ROWS>
sig=ab-ab-cdbe type=R*
73961
94102
</RECALL_ROWS>
<RECALL_DOMAINS>
sig=ab-ca-bdc type=S+
a 5
b 8
c 1
d 7
</RECALL_DOMAINS>
...
```

The opening special tags and request signatures are excluded from loss computation through masking. The first stage model trains with a learning rate of `2e-4`, effective batch size 8, for about 27.6K steps.

Validating on randomly chosen 1000 signatures shows that the first stage model is able to memorize most of the patterns quite well:

| Metric | Value |
|---|---:|
| Overall exact | 97.1% |
| Domain exact | 98.4% |
| Row ordered exact | 95.9% |
| Row-set recall | 98.0% |
| Row-set precision | 97.9% |

## Second stage: learning to solve

The resulting LoRA adapter becomes the initial LoRA adapter in the second stage training, which uses the actual trace data for the tasks as per the previous section. During second stage training, the model can then focus on learning the execution: how to compose the already memorized patterns to arrive at the final solution. A small number of drill style samples (3K) are injected in the data mix to help the second stage model retain memory.

Two stage training appears to be more sample efficient. I experimented with training on 12K cryptarithm samples and check the solve count in the held out validation set. There are about 11 non-concat cryptarithm problems that are solvable under budget.
* Direct training: solved 0/11
* Two stage training: solved 7/11

This allows the cryptarithm trace to be more or less learned (achieving near the target solve rate) with 40K+ samples. One important observation is that first stage training doesn't seem to impair the model's ability to learn the actual downstream tasks.

Two stage training is also more modular: any unrelated changes to solver logic or trace format can reuse the same first stage model without having to memorize from scratch again.

## Ablation
In a post-competition ablation experiment, I trained a model using the same final dataset mix (40K+ cryptarithm samples; all categories included) but without initializing from stage 1 LoRA. The validation set solve count denominator differs from the earlier 12K experiment because this evaluation used a different solver setting. Under this setup:

* Direct training: solved 12/17, private LB 0.888
* Two stage training: solved 16/17, private LB 0.900

This suggests the stage 1 training transferred to the final mixed-task model, improving both cryptarithm solve rate and private leaderboard score.

## Training durations

Both stages were trained on a single NVIDIA RTX PRO 6000 Blackwell (96 GB VRAM) on a remote Ubuntu 24.04 instance, using PyTorch 2.10.0 with CUDA 12.8.

Summary of training durations for the models:

| Model | Number of examples | Training input tokens | Steps | Duration |
|---|---:|---:|---:|---:|
| First stage | 44,136 | 159,305,950 | 27,585 | 25.2 hr |
| Second stage | 72,377 | 299,374,204 | 18,095 | 32.7 hr |

# Validation strategy

The validation strategy is quite simple: hold out about 10% of data for validation, stratified by problem category. This yields 954 validation rows.

When training, I excluded rows belonging to the validation set from training data. For augmented samples, if the augmented sample is derived from an original problem in the validation set, the augmented sample is also excluded from training data. This reduces the chance of any subtle leakage during validation.

The scatter plots show how well validation, public LB, and private LB scores align across submissions:

![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F91259%2F0a61dd9cdf503b9e2f60b322ab73ee8f%2Fcorrelation.png?generation=1781803925017432&alt=media)

The validation strategy turns out to track the private test scores well, but not the public leaderboard scores. The wall at 0.86 is clearly visible on the public LB. One of my 0.85 public LB submissions has a score of 0.90 on the private LB.

On the other hand, private LB scores continue to improve in line with validation scores. But there was no way of knowing. Luckily, I got a 0.87 submission at the right time and that gave me enough motivation to continue pushing through the end.

## Additional metrics

In addition to the official evaluation metric (% of tasks solved), I also track the following metrics:
* Perfect trace rate:
   * This tracks whether the model is emitting the expected trace exactly. Sometimes the model can get the correct answer by luck but still make mistakes in parts of the trace. This is a leading indicator intended to catch that before it affects solve rate.
   * I try to keep this >85% for bit manipulation, and >95% for cipher and numeric equations.
   * For gravity and unit conversion involving floating point arithmetic, this metric is expected to be low. This is fine since the accuracy for those tasks are very high despite not emitting the exact trace matching the floating point decimals.
* Cryptarithm pattern matching specific metrics: I compute metrics to track how well the model recall the memorized domains and rows, and how self-consistent the trace is assuming the recalled memories are correct.

## Iterating with single-category training

When testing a new solver logic or trace format, I try to train and validate on just the target category to isolate the effects of the change. I treat standalone validation scores as the ceiling that can be achieved because usually there will be a slight accuracy drop when the same variant is trained with all other categories.

Once I confirm that a new variant is indeed better in terms of standalone validation scores, I include it in the final mix. If the metrics in the all-categories run drop too much compared to the standalone run, I treat it as a signal to add more augmented samples.

## Overall scores

My final model scores 0.900 on the private leaderboard. Its local validation scores:

| Category | Actual solve count | Ideal solve count | Total | Accuracy |
|---|---:|---:|---:|---:|
| **Overall** | **863** | **864** | **954** | **90.46%** |
| bit_manipulation | 144 | 144 | 161 | 89.44% |
| cipher | 156 | 158 | 158 | 98.73% |
| cryptarithm_deduce | 20 | 21 | 66 | 30.30% |
| cryptarithm_guess | 1 | 1 | 17 | 5.88% |
| equation_numeric_deduce | 56 | 56 | 60 | 93.33% |
| equation_numeric_guess | 8 | 6 | 14 | 57.14% |
| gravity | 160 | 160 | 160 | 100.00% |
| numeral | 158 | 158 | 158 | 100.00% |
| unit_conversion | 160 | 160 | 160 | 100.00% |

(Actual = problems solved by the trained model; Ideal = problems solvable by the deterministic solver within token budget)

# Acknowledgements
My solution wouldn't have been possible without these existing works:

* [llkh0a's notebook](https://www.kaggle.com/code/llkh0a/nemotron-unsloth-sft-training-3-30-2): I got started in this competition following this work. My final model still uses the traces here for some of the categories.
* [huikang's solution](https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion/689915) and [training notebook](https://www.kaggle.com/code/huikang/end-to-end-finetuning-for-lb-0-85): The great open progress prize winning solution. I think the manual training loop is amazing and is not talked about enough!
* [kemshim's discussion](https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion/703479): I borrowed the ideas here to implement a symbolic equation solver for cryptarithm.
* [youkinasa's discussion](https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion/701981): My cryptarithm strategy that intersects constraints from multiple examples is inspired by this.

# Closing

This competition is quite special to me: 5th gold medal after more than a decade of not competing actively, and getting good placement on an LLM competition as a historically "xgboost" / "tabular ML" person.

Thank you to the organizers and fellow Kagglers for making this competition such a great experience!