# I. Opening
We decided to join this competition because we basically have near-zero machine learning experience, especially those involving fine-tuning LLM, but we know that training LLM means creating CoT, or in other words, transferring the human intelligence on how to solve the puzzle into the machine. Therefore, we knew that the most important part of the competition was finding a way to solve the problem, just as a human would, rather than the engineering part of the training itself. That is why we spend most of our time solving the puzzle as humans, refining CoT formats, and researching LORA, while spending the final weeks engineering the pipeline and training the model itself. For the training part, we mostly asked Gemini and Claude to write the code, but we also wrote much of the code ourselves, especially for important parts such as the CoT for `Bit Manipulation` and `Equation`. This write-up is also part of our attempt to win the Open Contribution Award in all 3 categories.

# II. Data Generator
We did not use any data from `train.csv` for our training method. Instead, we analyzed `train.csv` thoroughly, created an infinite synthetic problem and CoT generator, and used only the data from `train.csv` for validation. Many participants mention the instability when using @huikang's method [here](https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion/689915), and we think it's because his method is to train on top of the `train.csv` data and validate on the `train.csv` data itself, which, in our opinion, might provide an unstable LB even with high CV. But since we did not use any `train.csv` for this, the CV should be more honest, as it validates on a problem it has never seen. For each problem type, the details of its generation are as follows. 

### 1. Gravity
We randomized each problem to have 3-5 examples, the `g` to range from `4.9` to `19.6`, and the `t` to range from `1.0` to `5.0`, since those are the values appearing in the problems in `train.csv`. We also noticed that the `g` in `train.csv` is not perfect, so we mimic that by adding some noise as well. This works because the metric has a relative tolerance of `0.01`. As for the CoT itself, we use the exact same CoT by @huikang, with the example available [here](https://nemotron.huikang.dev/synthetic.html?problem=73f98498), where the model calculates `g` by performing a division tally, taking the median of the lists of `g` constants, and performing inference.

### 2. Conversion
Similar to 'Gravity' problems, each problem is randomized to have 3-5 examples, with the pre-conversion value ranging from `5.0` to `50.0`,  and the multiplier ranging from `0.05` to `20.0`. Just like `Gravity` problems, we add noise, and for the CoT, we use the exact same CoT by @huikang, which can be found [here](https://nemotron.huikang.dev/synthetic.html?problem=8ce54da2), which is also similar to `Gravity` problems with the division tally and the median as the deciding factor when there exist different multipliers.

### 3. Cipher
As for `Cipher` problems, we noticed that the problems in `train.csv` are slightly ambiguous: some encodings are not shown in the examples at all, so the model has to predict the most plausible word in that context. Luckily, the problems are quite constrained: only around 80 vocabularies are used, there is no case where multiple answers fit, and they follow a certain sentence format as below. 

>1. SUBJECT->VERB->OBJECT
2. THE->ADJECTIVE->SUBJECT->VERB
3. SUBJECT->VERB->PREPOSITION->PLACE
4. SUBJECT->VERB->THE->ADJECTIVE->OBJECT

As for the vocabulary, they are limited to the list below.
>- SUBJECT: alice, bird, cat, dragon, hatter, king, knight, mouse, princess, queen, rabbit, student, teacher, turtle, wizard
- VERB: chases, creates, discovers, draws, dreams, explores, follows, found, imagines, reads, sees, studies, watches, writes
- ADJECTIVE: ancient, bright, clever, colorful, curious, dark, golden, hidden, magical, mysterious, secret, silver, strange, wise
- OBJECT: book, castle, crystal, door, forest, garden, key, map, message, mirror, potion, puzzle, secret, story, treasure
- PREPOSITION: above, around, beyond, in, inside, near, through, under
- PLACE: castle, cave, forest, garden, island, library, mountain, ocean, palace, school, tower, valley, village, wonderland
- THE: the

Since the CoT by @huikang, which can be found [here](https://nemotron.huikang.dev/synthetic.html?problem=7d301a45), is already refined so that the model is taught to loop over all the words, our only need is to ensure the generated problems use only the format and vocabulary seen in `train.csv`. For each problem, we randomize the format and vocabulary, using 3-5 examples. We also ensured that when the encoding is incomplete, each problem has only one possible answer.

### 4. Roman
For `Roman` problems, this is pretty straightforward. We just randomize numbers from 1-100 and randomize 3-5 examples. As for the CoT, we used our custom CoT, where the model simply writes a lookup table from 1 to 100, verifies each example, and then pulls the final answer from the table. The example can be found [here](https://raw.githubusercontent.com/GMDA60/nemotron-assets/main/roman.txt).

### 5. Bit Manipulation
This is the problem type we spent the most time on. First, we saw @donaldgalliano's post about the 100% solve rate across everything [here](https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion/688461). His claim was that, for `Bit Manipulation` problems, the rule is per bit, not per bit sequence (8-bit or 1 byte). For example, the rule for the 3rd bit is to perform an `XOR` operation on the 4th and 5th bits. While his claim of 100% solvability is indeed correct, we found that without knowing the answer to the query, it's almost impossible to pin down the final transformation, since so many transformations fit, and they don't always converge to the same answer. While using Occam's razor principle indeed helps pin down some cases, the divergence remains high enough to suggest that this was not the rule the generator intended.

So instead, just like how @huikang generates his trace for bit manipulation [here](https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion/690307), we treat it as a byte-level transformation. For example, the rule might be to do an `XOR` operation on the input shifted left by 1, and the input shifted right by 1. A clear example is as follows.

```
Query: 11100111
Rule: Shift Left 1 XOR Shift Right 1
SL1: 11001110
SR1: 01110011
SL1 XOR SR1: 10111101
Answer: 10111101
```

However, as @huikang stated, his approach did not cover ternary operations such as `MAJORITY` and `CHOICE`, so we figured that if we could implement ternary operations on top of his approach, we could increase the number of solves. So we decided to analyze what kinds of rules actually appeared in `train.csv`. First, we want to define unaries or the base byte level operation here, which are shift left 1-7 (`SL`), shift right 1-7 (`SR`), rotate left 1-7 (`RL`), and their `NOT` versions, which make 42 unaries. Then, we create 3 stages as below.

>1. Unary only (e.g: NOT_SL1, RL_2, etc)
2. Pairwise operators where the ops are XOR, AND, OR (e.g: NOT_SL1 XOR SR5, RL2 OR RL5, etc)
3. Ternary operators with 3 unaries (e.g: MAJORITY(NOT_SL1,SR6,RL2), CHOICE(SR4,SL2,SL5), (RL5 XOR SL1) AND SL7, etc)

By doing this, we found fewer cases of divergence and a solve rate of approximately 95%, which led us to believe this was the rule the generator intended. We also found that limiting the unaries for ternaries to 1 `SL`, 1 `RL`, and 1 `SR` did not reduce the solve rate at all. We also limit the unaries to the positive versions only for `MAJORITY` and `CHOICE`, as this did not change our solvability rate. However, we still found an issue: thinking in byte-level transformations means the model has to search through millions of combinations. That's where we tried to look at what's happening on the bit level. One example is as follows.

```
Rule (SL1 XOR SR1) XOR RL1
Bit 0: Constant 0
Bit 1: Input bit 0
Bit 2: Input bit 1
Bit 3: Input bit 2
Bit 4: Input bit 3
Bit 5: Input bit 4
Bit 6: Input bit 5
Bit 7: Input bit 6 XOR input bit 0
```

This is where we found that most byte-level transformations, even ternary ones, can be reduced to simpler bit-level operators or primitives. Therefore, we conducted a deeper research on `train.csv` with these insights and then we found that without reducing our solvability rate, the only case where the ternaries can't be simplified is when the byte level rules involve `CHOICE`, `MAJORITY`, and `XOR-NOT_OR` (e.g: `(SL1 XOR NOT_SR5) OR SL1`) albeit there are cases where it can be simplified too. This is when we found that, for the byte-level transform, there are basically 4 stages, as shown below.

>1. Unary only
2. Pairwise operators
3. Simplified ternary operators
4. True ternary operators.

What we found out is that @huikang's approach basically only covered `Bit Manipulation` problems until stage 4, as he did not cover any ternary operators at all. His method might work for 85% of the dataset because some problems have multiple byte-level transformations that yield the same result. So even if, in the end, they don't form a perfect byte-level transformation, if we can stitch 2 byte-level operators and mask them into a bit-level transformation, the inference would be correct. For example, let's say a problem has 2 byte-level solutions below.

```
Rule 1: CHOICE(RL6,SR6,SL2)
Bit 0: Input bit 2 AND NOT input bit 6
Bit 1: Input bit 3 AND NOT input bit 7
Bit 2: Input bit 4 AND NOT input bit 0
Bit 3: Input bit 5 AND NOT input bit 1
Bit 4: Input bit 6 AND NOT input bit 2
Bit 5: Input bit 7 AND NOT input bit 3
Bit 6: Input bit 0 AND input bit 4
Bit 7: Input bit 1 AND input bit 5

Rule 2: SR2 XOR RL1
Bit 0: Input bit 1
Bit 1: Input bit 2
Bit 2: Input bit 0 AND input bit 3
Bit 3: Input bit 1 AND input bit 4
Bit 4: Input bit 2 AND input bit 5
Bit 5: Input bit 3 AND input bit 6
Bit 6: Input bit 4 AND input bit 7
Bit 7: Input bit 5 AND input bit 0
```

Since both of the byte-level rules fit all the examples, assuming that both of them are the correct rules the generator intended, technically stitching both rules and creating a new rule without regard to whether it's a valid byte-level transformation or not would still produce the correct answer. For example, we can create new rules as shown below, using the first rule for bits 0 to 4 and the second for bits 5 to 7.

```
Bit 0: Input bit 2 AND NOT input bit 6
Bit 1: Input bit 3 AND NOT input bit 7
Bit 2: Input bit 4 AND NOT input bit 0
Bit 3: Input bit 5 AND NOT input bit 1
Bit 4: Input bit 6 AND NOT input bit 2
Bit 5: Input bit 3 AND input bit 6
Bit 6: Input bit 4 AND input bit 7
Bit 7: Input bit 5 AND input bit 0
```

The issue is that it may produce rules that are not true byte-level transformations, leading to incorrect inferences. Therefore, we need to teach the model how to connect each bit-level primitive into a single valid byte-level transformation. For that, we first define the primitives below, where A, B, and C denote the number of input bits.

>- C0: Constant 0
- C1: Constant 1
- ID A: A
- NOT A: ~A
- AND AB: A&B
- AND-NOT AB: A&~B
- NOT-AND-NOT AB: ~A&~B
- OR AB: A|B
- OR-NOT AB: A|~B
- NOT-OR-NOT AB: ~A|~B
- XOR AB: A^B
- XOR-NOT AB: A^~B
- MAJORITY ABC: (A&B)|(B&C)|(C&A)
- CHOICE ABC: (A&B)|(~A&C)
- XOR-NOT_OR ABC: (A^~B)|C

So, for example, `AND-NOT 56` means taking the `AND` operator of input bit 5 and the `NOT` of input bit 6. While @huikang teaches his model to list all its primitives (without ternaries) and defines stride, we did the opposite. We did not teach the model to connect one bit to another one by one; instead, we want to teach it to connect all 8 bits at once to ensure it's a true byte-level transformation. Hence, our next step was to find all valid byte-level transformations and how the primitives continue to other primitives. The examples are as follows.

```
Rule: SL1 XOR NOT_SL2
bit 0: XOR-NOT 12
bit 1: XOR-NOT 23
bit 2: XOR-NOT 34
bit 3: XOR-NOT 45
bit 4: XOR-NOT 56
bit 5: XOR-NOT 67
bit 6: NOT 7
bit 7: C1
Continuation: XOR-NOT->NOT->C1

Rule: SL1 XOR NOT_SL3
bit 0: XOR-NOT 13
bit 1: XOR-NOT 24
bit 2: XOR-NOT 35
bit 3: XOR-NOT 46
bit 4: XOR-NOT 57
bit 5: NOT 6
bit 6: NOT 7
bit 7: C1
Continuation: XOR-NOT->NOT->C1
```

Looking at the examples above, both have the continuation `XOR-NOT->NOT->C1`, but in different sequences. While we deliberately skip teaching the model to create the stride one by one, we instead have it memorize all possible continuations so it can construct the correct 8-bit continuation in one step. With our 4 stages of byte-level transformations, we then tried to map all possible continuations and sequences into a single pkl file, which we have also attached to our notebook below. We found roughly 140 continuations and approximately 7500 unique sequences, which is much lower than we expected. 

For our first CoT approach, we want to use brute force, including ternary searches, just like @huikang did. Since the token is limited to 7680 tokens, we want to iterate through all possible combinations and mark which bit each matches, while also using an encoding that maps 10 bits to 2 letters. The encoding is as follows.

```
00000=A
00001=B
00010=C
....
11000=Y
11001=Z
11010=a
....
11111=f
```
The next steps are largely the same as in @huikang's approach, where we transpose the inputs and outputs. One big difference is that we copy the first example a few times until the number of examples reaches 10, since the problems range from 7 to 10 examples, and our encoding uses 5 bits. Therefore, by appending everything into 10 examples, the transposed version would have 10 bits, which would be encoded perfectly into 2 letters. We also encoded the primitives, placing them before the 5-bit encodings to save tokens. The format is as follows.

```
MAPPING
g 0=0
h 1=1
i A=ID(A)
j A=NOT(A)
k AB=AND(A,B)
l AB=AND-NOT(A,B)
m AB=NOT-AND-NOT(A,B)
n AB=OR(A,B)
o AB=OR-NOT(A,B)
p AB=NOT-OR-NOT(A,B)
q AB=XOR(A,B)
r AB=XOR-NOT(A,B)
s ABC=MAJORITY(A,B,C)
t ABC=CHOICE(A,B,C)
u ABC=OR(XOR-NOT(A,B),C)
```

After transposing the inputs and outputs, the model would enter the search phase, where, just like @huikang's approach, it would search for every possible primitive combination and mark which output bits it matches to. We also prevent bit-matching when the output is used as input, since we use `SL`, `RL`, or' SR' of at least length 1 on our unaries which prevent pure `IDENTITY` or `NOT` that uses the . To explain better, the search phase example for one single primitive looks as follows.

```
j
0 cX 
1 Nc 
2 dD 1
3 IE 2
4 UH 3
5 YT 4
6 dL 5
7 AE 6
found
1 j 2 3 4 5 6 7
``` 

This may seem a bit hard to read so the letter `j` here signifies the primitive `NOT`, the number `0` to `7` on the left signifies the input that is being used, the 2 letters used are the result of the operators used with the inputs on the left, the numbers on the right signifies what bit output the result matches to. Looking at the one-line example below should be easier to understand.

```
j
...
4 UH 3
...
````

Here, this means that the `NOT` of input bit `4` yields a bit that encodes `UH`, which matches output bit `3`. There would be no number on the right, as it did not match anything. Now, the numbers below `found` mean that the operator `j` matches the output bits starting from bit 1, where, among all the matches here, it uses `2 3 4 5 6 7` as the input bits. We would put `none` instead if the produced sequence is invalid, as explained below. This step is essentially a summary of what the model did in the previous step. In other words, it would read as follows:

```
1 j 2 3 4 5 6 7
->
j 2 matches 1
j 3 matches 2
j 4 matches 3
j 5 matches 4
j 6 matches 5
j 7 matches 7
```

An example for a constant, where the input is 0 for constant 0 and 1 for constant 1; pairs, where it uses 2 inputs; and ternary operators, where it uses 3 inputs, looks as follows.

```
h
1 ff 7
found
7 h 1

m
...
04 UH 3
15 IQ 
26 dD 1
37 AE 6
found
6 m 07

s
067 DY 
017 TL 
...
found
none
```

For some pairs and ternary operators, since the order of the input matters, `01` may have a different meaning than `10`, same as `123` might have a different meaning than `321`. If the order does not matter, we sort it where the smallest number appears first on the left. We also constructed the search so that the input numbers on the left are placed in perfect sequence order; if it did not continue into another primitive, the next number appearing in the search would be the next valid sequence. For example, for `NOT-AND-NOT` or encoded into `m`, `15` would continue to `26`, hence the search that uses input `26` would be put on the next line after `15` as written above. The empty newline indicates that they are different sequences of the same primitives and cannot be connected. For ternary, we also use only inputs and searches that appear in our sequence list, since some did not appear at all.

As for a valid sequence, there are a few filters we applied that are consistent across all sequences. The first filter is for pair operators. To understand them better, 2 examples are provided below.

```
bit 0: C0
bit 1: AND 02
bit 2: AND 13
bit 3: AND 24
bit 4: AND 35
bit 5: AND 46
bit 6: AND 57
bit 7: C0

bit 0: AND 13
bit 1: AND 24
bit 2: AND 35
bit 3: AND 46
bit 4: AND 57
bit 5: C0
bit 6: AND 07
bit 7: AND 01
```

For pairwise operators, it can be seen that for a pair sequence that appears in the middle but does not cover the entire byte, it must have 0 on its leftmost input and 7 in its rightmost input. For the prefix, it must have 7 in its rightmost input, and for the suffix, it must have 0 in its leftmost input. This filter ensures that the `found` part only shows the sequence that can truly be used in the final rule and continuation. A similar filter can be seen on ternary operators in an example below.

```
bit 0: XOR-NOT 12
bit 1: (XOR-NOT)_OR 230
bit 2: (XOR-NOT)_OR 341
bit 3: (XOR-NOT)_OR 452
bit 4: (XOR-NOT)_OR 563
bit 5: (XOR-NOT)_OR 674
bit 6: (XOR-NOT)_OR 075
bit 7: OR-NOT 61
```

Based on the constraints of the 4 stages we have defined before, ternary operators can only appear in the middle, and we can notice that they have the same filter as pairwise operators, where the leftmost part must have `0` as its input and the rightmost part must have `7` in its input. However, we can also see that the pairwise filter we created earlier did not work for prefixes and suffixes in sequences with ternary operators. We used that filter later, but on our first CoT, we only used the middle-part filter. If a valid middle part can be created by trimming the edges, we would add it into `found`; if no valid middle part can be created, we won't add anything or add `none` if no valid prefix, suffix, or a full 8-bit coverage.

After listing all the matches, the model must look for all possible continuations and mark the correct ones with `found`. Unlike @huikang's approach, where the model creates the continuation bit by bit, we used this long list of encoded primitives and let the model decide the 8-bit immediately, whether it's in compliance with the continuations. The long list is as seen below.

```
[DEDUCTION]
g
h
i
...
gi
hj
...
ql
qn
rm
ro
gkg
hph
...
rjh match
...
ksk
ktl
ltk
ruo
```

This time, we ordered it by the number of different primitives used and put the true ternary at the end. If there are multiple matches, the model would choose the first one as an implementation of Occam's razor. Then the rule-making is shown below.

```
first found rjh no fallback
0 r 17
1 j 2 3 4 5 6 7
7 h 1

TRUNCATION
0 r 17
1 j 2 3 4 5 6 7
7 h 1

FINAL RULE
0 r 17
1 j 2
2 j 3
3 j 4
4 j 5
5 j 6
6 j 7
7 h 1
```

So first, the model would take the sequences, even if they're longer than the intended sequence, truncate them, and then construct the final rule if it exists, or fall back to `11111111` if it does not. From there, it would do the inference part, which is almost identical to @huikang's code. The full CoT sample can be seen [here](https://raw.githubusercontent.com/GMDA60/nemotron-assets/main/bitmanipsamplefirst.txt). The performance of the CoT here is very poor, as we noticed that the model hallucinates when creating new rules that do not exist, or consistently outputs `11111111`. We thought it might be due to `AB` not being encoded as `A` and `B`, but rather as a single token `AB`. So we tried fixing it by writing `​ A B` instead of `AB`, same for the primitives on writing `​ r` instead of `r,` which stays until the final CoT as well. We removed fallback, too, and the new CoT can be seen [here](https://raw.githubusercontent.com/GMDA60/nemotron-assets/main/bitmanipsamplesecond.txt). However, it still did not perform well.

Therefore, we concluded that encoding is breaking the model completely. Then we returned to @huikang's approach, using the raw bits instead. But doing so would immediately eat up all the tokens, so we implement 3 major fixes. The first one is: since the problem has only 7-10 examples, we need only around 7 examples to pin down the rules and still achieve around 94% accuracy. That means we cut any examples after the 7th, and we don't pad until every problem uses 10 examples. Since we erased the encoding, we removed the encoding lookup table, and our matching parts would look as follows.

```
 i
0 1100001 
1 1100000 
2 1011110 7
3 0111000 
4 1001100 
5 0100000 
6 0010111 
7 1010100 
```

The next fix is to throw away the ternary searches completely, which is counterintuitive: it turns out that, for ternary operators, we don't even need to search at all. As long as we can pin down the correct prefix and suffix, there can usually be only 1 ternary in the middle. Next, we group the matches and apply the filter sequentially, instead of writing the `found` section we made earlier. So the groups are divided based on shift left lengths. For example, group 1 consists of any rules that constrain `SL1,` hence bit 1 would have 2 as its inputs. For example, group 1 is as follows.

```
[DEDUCTION]
PRE FILTER
0 has 1-> r13
1 has 2-> o42 r24
2 has 3-> o13 o03 o53 r35
3 has 4
4 has 5
5 has 6-> o26 p06
6 has 7-> o37 p47
7 has 0-> o40 o20
```

Then we applied the filter only to pairs, without touching the constants, the `ID`, or `NOT`. After the prefix, middle, and suffix filter, group one would look as follows.

```
FILTER PAIR
Prefix has 7 on right
Middle has 0 on left 7 on right
Suffix has 0 on left

0 has 1
1 has 2
2 has 3
3 has 4
4 has 5
5 has 6
6 has 7
7 has 0-> o20
```

We would then do this for group 1 to group 7. After that, we also implement a final filter that only filters prefix or suffix pairs that can be used for ternary operators. After the filter, it would look like this.

```
TRIPLETS CANDIDATE
0 has 1-> r13
1 has 2-> r24
2 has 3-> r35
3 has 4
4 has 5
5 has 6-> o26
6 has 7-> o37
7 has 0-> o40 o20
```

After this, the model just needs to select and verify whether the group that has completed 8 bits complies with the long list of rules we now place at the beginning of the CoT. If the true rules are not ternary, it would just choose the correct group with their final rules and provide inference. The rule creation example is below.

```
Already found from FILTER no triplet

first found r j h

FINAL RULE
0 r17
1 j2
2 j3
3 j4
4 j5
5 j6
6 j7
7 h1
```
However, if the true rules are ternary, the model would simply take the ternary prefix-suffix pairs and extrapolate from them without applying the prefix and suffix filters. While it is indeed a bit of guesswork, the filter actually helps, and the percentage of multiple prefix-suffix pairs that appear, which may cause different extrapolations, is small. We thought it wouldn't hurt the model at all because the guessing has a logical basis. The guessing logic is slightly different for each ternary operator, but one example looks like this.

```
no found from FILTER try triplet
 u CANDIDATE
0 r13
1 r24
2 r35
3 u?46 u?04-> u460
4 u?57 u?15-> u571
5 o26
6 o37
7 o40

first found r u o
```

For `MAJORITY` and `XOR-NOT_OR`, we could just stitch the extrapolation from the left and right in the correct order. However, for `CHOICE`, since we know that the leftmost part must be `0` and the rightmost part must be `7`, we simply inject the `0` from the left all the way to the right. An example is as follows.

```
no found from FILTER try triplet
 t CANDIDATE
Fill 0 from left
0 l21
1 t?32 t??0-> t203
2 t?43 t??1-> t314
3 t?54 t??2-> t425
4 t?65 t??3-> t536
5 t?76 t??4-> t647
6 t?07 t??5-> t750
7 i1

first found l t i
```

The inference part did not change much for this new CoT, whose full version can be seen [here](https://raw.githubusercontent.com/GMDA60/nemotron-assets/main/bitmanipthirdnormal.txt) for non-ternary problems and [here](https://raw.githubusercontent.com/GMDA60/nemotron-assets/main/bitmanipthirdtriple.txt) for ternary problems. This CoT is the one we use for bit manipulation throughout phase 1. However, even after doing all that, we still can't reach the target bit-manipulation score. For ternary rules, we only used the rules, sequences, and continuations that appear in `train.csv`. However, for non-ternary rules, we did not apply that filter and used any possible problems, even if they are not in `train.csv`. We were afraid that the test set would exhibit different byte-level rules; hence, we held back on our filter. Therefore, we tried removing continuations, which are unnecessary for solving `train.csv`. It would be considered necessary if it were the only rule that could solve even a single problem on `train.csv`. This cut the total continuations to approximately 30, down from approximately 140 before. It also reduced the possible sequences from around 7500 to approximately 1500, making learning smoother by clarifying the signal. After doing this, our bit-manipulation score began to approach our target of 94%, and our final bit-manipulation CoT format can be seen [here](https://raw.githubusercontent.com/GMDA60/nemotron-assets/main/bitmanipforthnormal.txt) for non-ternary problems and [here](https://raw.githubusercontent.com/GMDA60/nemotron-assets/main/bitmanipforthtriple.txt) for ternary problems.

For the problem generator, we generated 7-10 examples each, defined the byte-level rule, and ensured that the rule complied with how the CoT solves the problem, such as through Occam's razor. We also made sure the problem did not have multiple answers or diverging answers.

### 6. Equation
So first we tried tracking all the rules that appear in the `train.csv` and concluded that there are only 2 worlds, 4 families, 13 rules needed to solve 100% of the problems, including the guessing one. We'll start by explaining the concept of the world, which is `NORMAL` or `REVERSE`.

1. NORMAL
    This basically means we don't modify the inputs or outputs; we just apply the operators to the first and second inputs. This means `AB op CD` would just mean taking `op` of `AB` and `CD`.

2. REVERSE
    For REVERSE, we first reverse both inputs, apply the operator, and then flip the final result without any integer casts at any step, which allows `0` to appear as the first digit. As for the final output reversal, there are 2 types: `String Reversal` and `Integer Reversal`. For `String Reversal`, we want to flip the whole string, allowing symbols to appear at the end. For `Integer Reversal`, we want to flip the digits only, without altering the symbols. Hence, `AB op CD` means the reversal of the result of `BA op CD`.

Having described both worlds, we will now explain the 4 families and the 13 rules available.

>- CONCATENATION
    - concatab: concatenate a and b in that order
    - conctaba: concatenate b and a in that order
- ADDITION
    - add: a+b
    - addn: a+b-1
    - addp: a+b+1
- MULTIPLICATION
    - mul: a*b
    - muln: a*b-1
    - mulp: a*b+1
- SUBTRACTION
    - subab: a-b
    - posabs: |a-b|
    - negabs: -|a-b|
    - subba: b-a
    - mod: max(a,b) mod min(a,b)

For both worlds, if a negative appears in the final answer, we would change it using the op being used. For example, with `05&80` and the rule is just subtraction in `NORMAL`, we would replace `-` in `-75` with `&`, which gives `&75`. Also, we notice that each problem can have at most 1 rule from each family, and no family can appear more than once except for `CONCATENATION`. We'd also notice that no world mixing is possible. Even so, we still notice many cases where the answers can diverge. For example, examples showing the rule is `subab,` but the query turns out to be `posabs. Therefore, we tried several combinations, including the world, reversal type, `SUBTRACTION` order (as this is the only case where it can cause diverging answers). We found that the order below had the highest deduction score, at around 95%.

>- REVERSE>NORMAL
- Integer Reversal>String Reversal
- subab>posabs>negabs>subba>mod

But even after that, we still need to solve the guessing problems. Since we know that each family can only appear once, for each guessing problem, we grouped the families based on which families appeared in it. Then, for each group, we analyze which operators are best for guessing. We found that the rules above, with the checking order, gave the best score for the guess at around 40%.

>- no SUBTRACTION->use subab
- no ADDITION->use add
- no MULTIPLICATION->use mul

As for the CoT itself, we just write the entire search where the model tries each world and picks the one that is complete, since there can be no world mixing. If both are complete, as in the priority above, `REVERSE` would be chosen every time. It also picks the first rule on the search based on the priority we made above. For the deduction problem, the CoT can be seen [here](https://raw.githubusercontent.com/GMDA60/nemotron-assets/main/numericfirstdeduction.txt). For guessing problems, it identifies the family that appears across all operators and uses the above priority to determine which rule to apply for inference. The full CoT for the guessing problem can be seen [here](https://raw.githubusercontent.com/GMDA60/nemotron-assets/main/numericfirstguess.txt). 

For the problem we generated, we randomized the number of operators from 1 to 3, used 3-5 examples, and randomized whether it was a deduction or a guess problem, with deduction having a higher weight. We also made sure that the problem's final answer complies with the CoT's problem-solving order and is unambiguous.

When running phases 1 and 2, our equation scores have reached the targeted 85% (combination of deduction and guessing), and it even reached 86%, which should be impossible, but it is possible due to hallucinated answers. However, we still think this is not enough to achieve a competitive score on the leaderboard. Since the hosts mentioned that the test data is generated with the same generator, we thought it best to overfit the `train.csv` as hard as possible. So, we tried to implement operator ordering and a tiebreaker to decide which operators are used in the query when multiple answers exist. Instead of taking the first operator that fits, we append them all and do inference from there. For deduction questions, when only 1 answer exists, the CoT can be seen [here](https://raw.githubusercontent.com/GMDA60/nemotron-assets/main/numericsecondconverge.txt).

As for deduction problems with multiple answers, we taught the model to encode the appearing operator rules based on their order of appearance into a single string and then use the lookup table to determine the true query, as shown [here](https://raw.githubusercontent.com/GMDA60/nemotron-assets/main/numericseconddiverge.txt). We also use the same lookup string for the guessing problem, which can be seen [here](https://raw.githubusercontent.com/GMDA60/nemotron-assets/main/numericsecondguess.txt). That made our solve rate for both deduction and guessing around 96%. While this gave us a slight boost in public LB, it unfortunately did not improve the private LB score of the model we ultimately chose. For phase 3 problem generators, we did not make the problems as strict as in phase 1, but we still made sure that the answers complied with the CoT answers.

### 7. Cryptarithm
We did not have enough time to create a high-quality CoT as we spent most of our time in `Bit Manipulation`. Also, we did not have enough compute to produce a large-scale high-quality `Cryptarithm` CoT, so we created one where the model is taught to guess what each operation means, and based on those operations, what numbers fit them, which can be seen here. However, this failed miserably as there are no logical deductions on the encodings, which made the model only learn problems with `concatab` or `concatba`, so we figured it out that it would be better to just use @huikang's CoT, which can be found [here](https://nemotron.huikang.dev/synthetic.html?problem=8ce54da2), as it yields the same result and is already proven to work. However, for the problem generation itself, we used the first version of `Equation` problems to ensure the problems are well-constrained by a C++-based checker and are true `Cryptarithm` problems rather than just random gibberish.

### Amount of Data
Since we train the model sequentially, each iteration might improve the score on some problem types and hurt others. Therefore, based on the score of each problem type in CV, our data distribution for each iteration is as follows.
<br>

| Iteration | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Gravity  | 1250 | 2000 |  500| 500 | 250 | 1000 | 1000 | 1500 | 1500 | 2000 | 3000 | 3000 | 1500 | 3000 | 3000 |
| Conversion  | 1250 | 2000 |  500| 500 | 250 | 1000 | 1000 | 1500 | 1500 | 2000 | 3000 | 3000 | 1500 | 3000 | 3000 |
| Cipher  | 1250 | 2000 |  500| 500 | 250 | 1000 | 1000 | 1500 | 1500 | 2000 | 6000 | 6000 | 3000 | 6000 | 6000 |
| Roman  | 1250 | 2000 |  500| 500 | 250 | 1000 | 1000 | 1500 | 1500 | 2000 | 3000 | 3000 | 1500 | 3000 | 3000 |
| Bit Manipulation  | 5000 | 8000 | 12000 | 12000 | 12000 | 7000 | 9000 | 16000 | 16000 | 10000 | 24000 | 24000 | 36000 | 30000 | 24000|
| Equation  | 3000 | 4000 | 1000| 1000 | 2000 | 1000 | 1000 | 1500 | 1500 | 2000 | 24000 | 24000 | 2700 | 30000 | 24000 |
| Cryptarithm | 3000 | 4000 | 1000| 1000 | 500 | 1000 | 1000 | 1500 | 1500 | 2000 | 3000 | 3000 | 1500 | 3000 | 3000 |
| Overall | 16000 | 24000 | 16000 | 16000 | 15500 | 13000 | 17000 | 28000 | 28000 | 22000 | 66000 | 66000 | 72000 | 78000 | 66000 |

<br>
As for phase 3, which is conducted in iterations 11-15, the increase in data points in folds is due to the data being split into 3 notebooks, as explained below.

# III. Training Loop
We use a modified version of @huikang's training notebook, which can be accessed [here](https://www.kaggle.com/code/huikang/end-to-end-finetuning-for-lb-0-85), where we only change the LR, use our synthetic data instead of his, and add a stopper due to Kaggle's 12-hour GPU notebook limit. For all of our training phase, the parameters are listed below.

<br>
|Parameter|Value|
|---|---|
|Rank| 32|
|Alpha| 32|
|LR|1.0e-4 to 2.0e-4|
|grad_clip_norm| 1.0|
|Schedulizer| Linear|

<br>
Any parameter not listed in the table above uses the same value as @huikang's trainer loop. Our training loop is divided into 3 phases, as shown below.

### 1. Multi Round SFT
In phase 1, we trained 6 iterations of SFT with a high LR, gradually decreasing it. We never really worried about catastrophic forgetting because, as long as we sampled every type of problem, the model won't forget about them. That is also why we used a high LR, from `1.0e-4` to `2.0e-4`, for all iterations, which allowed the model to copy the formats perfectly. While we have reached 0.87 on the public LB with this approach, we still haven't met our goal for bit manipulation, as shown in the validation section below. It gave us hope that by adding more iterations and fixing the CoT, we could achieve a higher score.

### 2. LORA Averaging Regularization
In phase 2, after slightly altering the bit-manipulation CoT, we introduce LORA average, or model soup, in which, instead of iterating sequentially, we combine the 2 models before and after training. For example, after iteration 6 on phase 1, we start training on top of it and call it level 7. And then, when level 7 is done, we combine iteration 6 and level 7 to create iteration 7 soup. After that, we would train level 8 on top of level 7, then combine level 8 with the iteration 7 soup. This serves as a form of regularization for the models, as they were already very stable in phase 1 but still need to learn the new CoT format. It has also proven to yield a very stable score on the public LB compared to those from phase 1. After we trained until level 10, we once again combined the 2 model soups that scored the highest in public LB, which are iteration 8 and the model soup from iteration 9 and level 10, to create the true iteration 10 that would be used in phase 3, which is also our best score in public LB so far.

### 3. Iterative Model Soup
We started phase 3 because we had a slight change to our CoT equation in the final week, so we needed a way to train the models quickly enough without sacrificing stability. Unlike phase 2, we use the final soup from phase 2, train 3 models on top of it with different problem sets, and then average them. In the end, we managed to cram 5 iterations into a short time. Unlike in phase 2, we trained multiple models from the same checkpoint using different problem sets and randomizers. This increased our score as it prevented the model from stopping at the wrong local minima by canceling out each other's errors. This enabled us to train more `Numeric` problems while maintaining the same convergence speed and achieving higher validation, public, and private LB scores.

### Loss Dynamics
The losses aren't very informative, as they can range from `0.0010` to `0.0023` in some iterations. Lower loss does not necessarily mean better validation or leaderboard score, which made us think that validation and LB score alone should be the metric for how good the model is. In many cases, we have also seen the loss gradually decrease over a few hundred steps, reaching an all-time low before shooting up again and fluctuating around it. We might suspect it's due to LR overshoot, but since validation and leaderboard scores kept going up, we didn't look into it much and kept going with our pipeline.

### Solvability Rate, Validation, and Leaderboard Score
For our oracle solver, which is the solver that the CoT used, we have 2 versions. The first version is the one we used for phase 1 and phase 2. Notice that even if there's a change on `Bit Manipulation` CoT, our oracle stayed the same cause it's just a change of format. As for phase 3, since we changed not only the CoT style but also how the CoT solves the problem itself for `Equation`, the solvability rate went up. Below is the table of our oracle's solvability in the number of problems solved.

| Version | 1 | 2 |
|---|---|---|
| Gravity| 1597 | 1597|
| Conversion| 1594 | 1594|
| CIpher| 1576 | 1576|
| Roman| 1576 | 1576|
| Bit Manipulation| 1511 | 1511|
| Equation| 627 | 704|
| Cryptarithm| 65 | 65|
| Total| 8546 | 8623|

<br>
Below is the table in percentage.
<br>

| Version | 1 | 2 |
|---|---|---|
| Gravity| 100.0 | 100.0|
| Conversion| 100.0 | 100.0|
| CIpher| 100.0 | 100.0|
| Roman| 100.0 | 100.0|
| Bit Manipulation| 94.3 | 94.3|
| Equation| 85.7 | 96.2|
| Cryptarithm| 7.9 | 7.9|
| Total| 90.0 | 90.8|

<br>
Therefore, for phases 1 and 2, we aimed to achieve LB scores of 90.0% and 90.8%, respectively, for phase 3. The table below shows how well the model is on `train.csv` and its LB score. Since we don't use any data from `train.csv` to train, the validation is quite honest and shows a correlation trend with the best LB score (among the 5 submissions) to a certain point. The first table is the raw numbers.

| Iteration | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Gravity  | 1571 | 1584 | 1583 | 1583 | 1588 | 1592 | 1595 | **1596** | 1595 | 1594 | 1594 | **1596** | **1596** | 1593 | 1592 |
| Conversion  | 1543 | 1580 | 1578 | 1589 | 1588 | 1588 | 1588 | 1588 | 1587 | 1590 | 1587 | **1594** | 1586 | 1589 | 1590 |
| Cipher  | 1522 | 1551 | 1548 | 1521 | 1544 | 1556 | 1562 | 1569 | 1566 | 1565 | 1566 | 1570 | 1567 | **1572** | 1567 |
| Roman  | 1575 | **1576** | 1574 | **1576** | **1576** | **1576** | **1576** | **1576** | **1576** | **1576** | **1576** | **1576** | **1576** | 1574 | **1576** |
| Bit Manipulation  | 686 | 935 | 869 | 1219 | 1431 | 1342 | 1443 | 1464 | 1492 | 1499 | 1477 | 1481 | 1511 | **1517** | **1517** |
| Equation  | 591 | 606 | 620 | 622 | 598 | 617 | 624 | 624 | 633 | 632 | 647 | 671 | 675 | 674 | **681** |
| Cryptarithm | 15 | 63 | **65** | 63 | 64 | **65** | **65** | 64 | 63 | 64 | 64 | 64 | **65** | 64 | 63 |
| Overall | 7503 | 7895 | 7837 | 8173 | 8389 | 8336 | 8453 | 8481 | 8512 | 8520 | 8511 | 8552 | 8576 | 8583 | **8586** |

<br>
The second table is the percentage version.
<br>

| Iteration | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Gravity | 98.4 | 99.2 | 99.1 | 99.1 | 99.4 | 99.7 | 99.9 | **99.9** | 99.9 | 99.8 | 99.8 | **99.9** | **99.9** | 99.7 | 99.7 |
| Conversion | 96.8 | 99.1 | 99.0 | 99.7 | 99.6 | 99.6 | 99.6 | 99.6 | 99.6 | 99.7 | 99.6 | **100.0** | 99.5 | 99.7 | 99.7 |
| Cipher | 96.6 | 98.4 | 98.2 | 96.5 | 98.0 | 98.7 | 99.1 | 99.6 | 99.4 | 99.3 | 99.4 | 99.6 | 99.4 | **99.7** | 99.4 |
| Roman | 99.9 | **100.0** | 99.9 | **100.0** | **100.0** | **100.0** | **100.0** | **100.0** | **100.0** | **100.0** | **100.0** | **100.0** | **100.0** | 99.9 | **100.0** |
| Bit Manipulation | 42.8 | 58.4 | 54.2 | 76.1 | 89.3 | 83.8 | 90.1 | 91.4 | 93.1 | 93.6 | 92.2 | 92.4 | 94.3 | **94.7** | **94.7** |
| Equation | 80.7| 82.8 | 84.7 | 85.0 | 81.7 | 84.3 | 85.2 | 85.2 | 86.5 | 86.3 | 88.4 | 91.7 | 92.2 | 92.1 | **93.0** |
| Cryptarithm | 1.8 | 7.7 | **7.9** | 7.7 | 7.8 | **7.9** | **7.9** | 7.8 | 7.7 | 7.8 | 7.8 | 7.8 | **7.9** | 7.8 | 7.7 |
| Overall | 79.0 | 83.1 | 82.5 | 86.0 | 88.3 | 87.7 | 89.0 | 89.3 | 89.6 | 89.7 | 89.6 | 90.0 | 90.3 | 90.3 | **90.4** |

<br>
The third table shows the validation score along with the Public and Private LB scores, with some iterations left as N/A because we did not have time to submit all of them.
<br>

| Iteration | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Validation | 79.0 | 83.1 | 82.5 | 86.0 | 88.3 | 87.7 | 89.0 | 89.3 | 89.6 | 89.7 | 89.6 | 90.0 | 90.3 | 90.3 | **90.4** |
| Public | 78.4 | 84.0 | 83.2 | 86.4 | 86.4 | 87.2 | N/A  | 88.0 | N/A | 88.0 | **88.8** |N/A  | 88.4 | N/A | 88.0 |
| Private | 82.8 | 85.6 | 84.4 | 86.8 | 87.6 | 89.2 | N/A | 88.8 | N/A | 89.6 | 89.6 | N/A | 89.6 | N/A | **90.0** |

<br>
In phase 1, as the validation score increased, the public and private LB scores also increased. However, in phases 2 and 3, even though the validation score increased, the public LB score remained in the 88% range, while the private leaderboard showed a slight upward trend. It is also important to note that the public and private LB scores are based on the 5 submissions with the highest scores. During phase 1, we found that our model remained unstable, which could lead to a 1% drop even on the same adapter. This was not the case for phases 2 and 3, as most of them have their scores fluctuating below the 1% range.

### Training Cost and Schematics
Our training is done 100% on Kaggle via the Kaggle CLI after following the tutorial by @citerne [here](https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion/683172). Since compute is tight, we also bought some Google Colab Pro subscriptions to access Kaggle compute. The exact number of tokens used is unclear, as some were due to the compute limit and Kaggle's notebook time limit. There were many cases where we did not train an iteration until the end, so some tokens weren't used. However, we are sure that our final submission is trained on roughly 2 billion tokens, and our team spent approximately $100 on Google Colab Pro subscriptions.

# IV. Submission
As for our final submission, we ultimately chose the 2 submissions that scored best on the public LB. Due to VLLM indeterministic inference even with greedy decoding and temperature of 0, we'd assumed that even with the same model, the best submission on public would also be the best submission in private since there are no reruns confirmed by the hosts [here](https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion/707596#3471471). Our chosen model is also coincidentally the most stable among our submissions as it score 88.8% on public LB and 89.6% on private LB 3 out of 5 times. However, our chosen submission was not the one with the best private score. We have a submission with the highest CV and the highest private LB of 90.0%, but a lower public score. But, since this model was fairly unstable, there was no way for us to tell which submission to upload. Fortunately, due to the stability of our model, the shakedown did not affect us much, as our rank only dropped from 4 on public LB to 5 on private LB.

# V. Insights
Below are our insights, including failures or experiments we did or planned to do during the competition.

### 1. CoT format
Before creating the final CoT, we also experimented with a short CoT that only shows the final deduction, a brief proof of how it works, and the inference, but again, the model is shown not to be able to guess without any logical steps. We even tried with 250K short CoT examples, but the model failed miserably on LB, which shows that LLMs are not that good at guessing when there is no logical basis at all. We first thought that with enough iterations, it could somehow train its muscle memory and create a good guess. While the problem with Unsloth, as explained below, might also be why it failed, our first short CoT, despite having a high solvability rate from the oracle, did not account for any divergence cases, making the gradient very unstable. Therefore, we chose the more logical approach without any guessing.

### 2. Pseudo GRPO with DPO
We also tried using GRPO, but unfortunately, it is notoriously hard to set up in a Kaggle notebook, since notebooks with the upgraded accelerators are air-gapped and debugging is already super hard. We got mysterious C++ compile errors, OOM, and too many errors. Therefore, we tried to tackle it by using pseudo GRPO, inspired by this paper [1], which essentially argues that GRPO is just DPO in disguise. So we tried creating many problems, ran a few iterations of trace generation with VLLM, collected the successes and failures, did DPO, and repeated. But even with 5 iterations, the model can't learn the easy bit-manipulation problem. It wasn't until we saw @huikang's approach, which uses only SFT, and his insight into what he wanted the model to produce during inference, that we pivoted to SFT as well. 

### 3. ORPO, KTO, and DPOP.
SFT only teaches the model what the right answer is, but that doesn't necessarily mean the model knows what's wrong. Therefore, we planned to use several algorithms, such as ORPO [2], a contrastive algorithm that compares 1 correct trace with 1 incorrect one. KTO was also a candidate [3], similar to ORPO, but instead of comparing 2 traces, it trains on a list of desirable and undesirable traces. Neither of these needs a reference model, so running them on a Kaggle notebook should be feasible. Another interesting algorithm is DPOP [4], which helps address DPO pairs with a tiny difference. Since we had run DPO before, setting it up would not be as hard as GRPO. We hoped that, in the end, we would be able to truly squeeze the model's performance and teach it what is wrong and what the difference is, so the model would be much more confident. However, our SFT approach worked well enough, so we decided not to continue down that path, as creating incorrect traces also requires substantial computation that, unfortunately, we did not have.

### 4. Unsloth, Quantization,Unsloth, Quantization, and Packing. 
At first, we used Unsloth with 4-bit quantization and packing, but even after running around 100K examples, the model still failed to learn, which made us think that 4-bit quantization and packing themselves break Nemotron, since it's not a standard transformer. Setting up in the beginning was also notoriously hard because Nemotron itself uses Mamba SSM instead of a standard attention module, just like any transformer, which made us unable to use Flash Attention or XFormers, which we tried to fix but failed miserably. Not only that, GPU RTX PRO 6000 uses sm_120, so causal conv1d and other pip is so hard to set up. It was only after switching to @huikang's public trainer that the model seemed to work, as it did not use any quantization or packing and used a more refined training loop.

### 5. DORA
We actually tried training one iteration with DORA [5], as it's said to be comparable with LORA on higher rank by splitting the magnitude and the direction vector. While it indeed shows lower loss in our experience, our submission returned an error since VLLM does not support it. We could actually try converting the DORA adaptor into a normal LORA adapter so it could be submitted, but we found out that the transformation itself is lossy, so we decided not to risk it and go with normal LORA instead. 

### 6. Ideas on Cryptarithm
We knew that improving `Cryptarithm` is the key to winning this competition. Some ideas we planned were to first convert the symbols into letters with spaces to create distinct tokens, and then train a small model on GRPO to generate the CoT needed for `Cryptarithm`. However, since we can't make the GRPO work, this idea went to waste. Also, we tried to create CoT only for problems involving `add`, `mul`, and/or `sub`, but we didn't have time to design the CoT to demonstrate sufficient deduction. In the end, our model failed on `Cryptarithm`, but fortunately, it performed better than expected on private LB.

### 7. Unclear Source of Improvement. 
Since we did many changes at once, it's actually unclear what improved our score. It's unclear whether our new CoT format or @huikang's trainer was the one that increased our score. We only knew that after applying those 2 changes, the model reached the same score with less data. It's also unclear what our breakthrough LB score from phase 1 was, as in phase 2, we changed the CoT and implemented LORA soup at the same time. Same as phase 3, where we implemented the new CoT for `Equation` and iterative model soup. The score indeed improved dramatically, but it's unclear whether the improvement is due to CoT quality or to the LORA soup itself. Normal multi-round SFT and LORA soup indeed gave a score improvement, but the score improvement on phases 2 and 3 is unclear whether it's from the CoT itself or from the LORA soup. However, we did not have the time to check this and can only conclude that LORA soup made our model very stable. Even without averaging LORA, the same score could arguably be achieved with our CoT format and standard SFT, possibly even on a quantized model.

# VI. Final Notebook
Our pipeline consists of many notebooks that we run with the Kaggle CLI in multiple loops. But instead of attaching all the notebooks as separate files, we have compiled all the important pieces of the notebooks we used into 1 master notebook. Also, we would not attach our other attempts, but only the notebooks we used from phases 1 to 3, with their changes explained. The notebook can be accessed [here](https://www.kaggle.com/code/darrenamadeusmartin/domdolus-tolus-nemotron). Our final submission model is also available in the dataset [here](https://www.kaggle.com/datasets/darrenamadeusmartin/domdolus-tolus-nemotron-final). 

# VII. Message for the Host and Staff
We'd like to thank the NVIDIA team and Kaggle staff for the amazing competition and sincerely hope that a similar competition will be hosted again in the future. A competition with hardware prizes is very encouraging, as computing is not easily accessible to most Kaggle participants. We also hope that Kaggle staff might be able to look into our suggestions for future competitions.

- If similar competitions were done in the future, we hope that the inference part itself could be near deterministic or use more passes to reduce variance and show the true performance of the trained model itself instead of luck.
- When pushing some notebooks through Kaggle CLI, our team experienced an error where it could not detect datasets or notebooks, despite them being attached in `kernel-metadata.json`. When we click the save version button, the notebook shows no errors. However, when we push via the CLI, an error appears stating that the dataset does not exist, even though most of the time the directories are changed from `kaggle/input/notebooks` to `kaggle/input` only. The only problem is that when we fixed the formatting, the same fix did not apply to all notebooks. Some notebooks might work with `kaggle/input/notebooks`; others might work with `kaggle/input`. That's why we put many file searchers in our notebooks. We hope that this will be fixed in the near future.

# VIII. Closing
We'd also like to thank @huikang for his amazing Open Progress Prize write-up, which gave us a great CoT generator and many insights, especially on bit manipulation. We also want to thank @donaldgalliano for his insights into solving the problems and @citerne for guidance on using the Kaggle CLI, which is a core part of our training method. We are also very grateful for the community discussion; we posted a few there, and everyone was very supportive. We also want to congratulate everyone on their achievements and hope this competition has advanced the field of machine learning, especially in LLM fine-tuning. Lastly, we are very grateful as a team to have secured this position and are certainly hoping for further achievements. If there are any unclear points about our approach, feel free to ask in the comments.

# IX. REFERENCE
[1] Y. Wu et al., "It takes two: Your GRPO is secretly DPO," arXiv preprint arXiv:2510.00977, 2025. [Online]. Available: [https://arxiv.org/abs/2510.00977](https://arxiv.org/abs/2510.00977).<br>
[2] J. Hong, N. Lee, and J. Thorne, "ORPO: Monolithic Preference Optimization without Reference Model," arXiv preprint arXiv:2403.07691, 2024. [Online]. Available: [https://arxiv.org/abs/2403.07691](https://arxiv.org/abs/2403.07691)<br>
[3] K. Ethayarajh, W. Xu, N. Muennighoff, D. Jurafsky, and D. Kiela, "KTO: Model Alignment as Prospect Theoretic Optimization," in Proc. Int. Conf. Mach. Learn. (ICML), 2024. [Online]. Available: [https://arxiv.org/abs/2402.01306](https://arxiv.org/abs/2402.01306)<br>
[4] A. Pal et al., "Smaug: Fixing Failure Modes of Preference Optimisation with DPO-Positive," arXiv preprint arXiv:2402.13228, 2024. [Online]. Available: [https://arxiv.org/abs/2402.13228](https://arxiv.org/abs/2402.13228)<br>
[5] S.-Y. Liu et al., "DoRA: Weight-Decomposed Low-Rank Adaptation," in Proc. Int. Conf. Mach. Learn. (ICML), 2024. [Online]. Available: [https://arxiv.org/abs/2402.09353](https://arxiv.org/abs/2402.09353)<br>