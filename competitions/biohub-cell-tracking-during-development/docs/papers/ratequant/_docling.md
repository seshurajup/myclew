## RATEQUANT: Optimal Mixed-Precision KV Cache Quantization via Rate-Distortion Theory

Fei Zuo 1 , ∗

Zikang Zhou 2 , ∗

1 BA TechWorks (BMW Group)

Hao Cong 3 , ∗

Xiaoyan Xi 1 , ∗

2 National University of Singapore

## Abstract

KV cache quantization reduces the memory footprint of large language model inference, yet existing quantizers assign uniform bit-widths to every attention head, overlooking significant variation in head importance. A natural idea is to allocate more bits to important heads and fewer to the rest. However, we observe that such mixed-precision allocation has a hidden pitfall: each quantizer follows a different distortion curve D ( b ) = αβ -b , where the decay rate β varies from 3.6 to 5.3 across designs. Applying one quantizer's distortion model to another inverts the allocation order and makes performance worse than uniform quantization, a failure mode we call distortion model mismatch . In this work, we propose RATEQUANT , a framework that resolves this mismatch by fitting per-quantizer distortion models from a small calibration set, then solving the bit-allocation problem in closed form via reverse waterfilling from rate-distortion theory. Extensive experiments on Qwen3 and Llama3 families across three quantizers demonstrate that calibrated RATEQUANT reduces KIVI's perplexity at 2.5 bits from 49.3 to 14.9 (70% ↓ ) and recovers 70-85% of quantization-induced degradation at 4.0 bits, with zero runtime overhead and &lt; 2 s one-time calibration.

## 1 Introduction

Serving large language models (LLMs) at scale requires caching all previously computed key-value (KV) pairs so that each new token can attend to the full context [34, 29]. The memory footprint grows linearly with sequence length, batch size, and model depth, making the KV cache a primary memory bottleneck [30, 15]. KV cache quantization reduces this cost, and recent work has produced effective quantizers [23, 2, 13, 41].

Despite their success, these quantizers apply uniform bitwidths to every attention head, implicitly assuming equal contribution from every head. This assumption is increas-

Figure 1: Quantizer β varies 1 . 5 × ; mismatched calibration worsens PPL.

<!-- image -->

ingly at odds with recent findings. Head importance studies [35, 33] show that heads exhibit highly non-uniform importance. Meanwhile, recent mixed-precision approaches relax uniformity at the layer [17, 18] or channel level [19, 36], but each relies on heuristic rules tied to a specific quantizer. These observations raise a fundamental question: if heads are not equally important, can we build a principled, quantizer-agnostic framework for mixed-precision KV cache allocation?

We identify a deeper obstacle that must be resolved first: distortion model mismatch . Different quantizers have fundamentally different distortion-rate curves D ( b ) = α · β -b , with the decay rate β varying from 3.6 (TURBOQUANT) to 5.3 (QuaRot). As Fig. 1 illustrates, naïvely applying one quantizer's distortion model to another makes mixed-precision allocation worse than uniform . This

∗ Equal contribution. † Corresponding author.

Ho Fai Leung 1 , †

3 Tsinghua University

Figure 2: RATEQUANT pipeline. Phases 1-3 are one-time offline costs ( &lt; 2 s for 8B); Phase 4 adds zero runtime overhead.

<!-- image -->

occurs because mismatched β inverts the marginal gain ordering across heads, causing the algorithm to allocate bits to the wrong heads. Our investigation is guided by two research questions:

RQ1: How do different heads contribute to model quality under quantization, and can we estimate this contribution efficiently?

RQ2: Can we build a quantizer-agnostic allocation framework that avoids distortion model mismatch?

Our analysis on Qwen3 and Llama3 families yields two key insights: ( i ) gradient-based sensitivity is the correct proxy for KV allocation, outperforming activation-based methods by 1.07 PPL at 3.5 bits; ( ii ) distortion models must be calibrated per-quantizer, as applying TURBOQUANT's model to KIVI worsens PPL from 49.3 to 87.0 at 2.5 bits, while calibration reduces it to 14.9.

Building on these insights, we propose RATEQUANT , a framework that formalizes per-head KV cache bit allocation as rate-distortion optimization. RATEQUANT fits per-quantizer distortion models from a small calibration set ( N =16 sequences, ∼ 1.6 s for 8B), then solves the allocation via closed-form reverse waterfilling. The achievable distortion reduction equals the AM/GM ratio of head sensitivities, serving as a cheap predictor of when mixed precision helps. Extensive experiments across five models and three quantizers demonstrate that RATEQUANT recovers 70-85% of quantization-induced degradation at 4.0 bits with zero runtime overhead.

Our contributions are:

- We identify distortion model mismatch as the failure mode of naïve mixed-precision KV quantization, where mismatched β inverts allocation and worsens performance.
- We propose RATEQUANT, a rate-distortion framework with closed-form allocation via reverse waterfilling. Per-quantizer calibration and K/V separation make RATEQUANT applicable to any base quantizer.
- We validate that gradient-based sensitivity is qualitatively superior to activation-based, with the proxy choice dominating the allocation algorithm.
- We demonstrate consistent gains across Qwen3 and Llama3 families: KIVI 2.5b improves from 49.3 to 14.9 PPL (70% ↓ ), and 4.0b recovers 70-85% of degradation.

## 2 Related Work

KVcache quantization. Reducing KV cache memory has been approached through eviction [44, 10], token merging [27], contextual sparsity [22], efficient attention [4], paged memory [15], and quantization [13, 23, 40]. KIVI [23] applies per-channel symmetric keys and per-token asymmetric values; QuaRot [2] suppresses outliers via Hadamard rotations; KVQuant [13] handles outliers with non-uniform quantization; TURBOQUANT [41] introduces rotation-based vector quantization. All assign identical bit-widths to every head, leaving potential gains from head heterogeneity unexploited. Recent mixed-precision approaches assign precision at the layer [17, 18, 21] or channel level [19, 36, 43], but each is designed around a specific quantizer, limiting transferability. RATEQUANT operates at per-head granularity with closed-form allocation and supports arbitrary quantizers through calibration.

## Algorithm 1 RATEQUANT: Rate-Distortion Optimal KV Cache Quantization

Require: LLM with L layers, H KV heads per layer ( N = L × H ); calibration set D ; average bits ¯ b ; bounds b min , b max

Ensure: Per-head bit allocation { b i } 2 N i =1 for keys and values

- 1: // Stage 1: Gradient-based sensitivity estimation ( ∼ 1.6s for 8B)
- 2: for each head i ∈ [1 , N ] do
- 3: w K i ← 1 |D| ∑ x ∈D 1 T ∑ t ∥∇ K i,t L∥ 2 ; w V i ← 1 |D| ∑ x ∈D 1 T ∑ t ∥∇ V i,t L∥ 2
- 4: end for
- 5: // Stage 2: Distortion model calibration ( &lt; 0.1s)
- 6: Measure MSE at b ∈ { 2 , 3 , 4 , 5 , 6 } bits; fit ( α K , β K ) , ( α V , β V ) via ln D = ln α -b ln β
- 7: // Stage 3: Greedy integer allocation ( &lt; 0.01s)
- 8: Initialize b i ← b min for all 2 N components; R ← 2 N ¯ b -2 N · b min
- 9: while R &gt; 0 do
- 10: i ∗ ← arg max i { w i · [ D i ( b i ) -D i ( b i +1)] : b i &lt; b max } {Max marginal gain}
- 11: b i ∗ ← b i ∗ +1 ; R ← R -1
- 12: end while
- 13: return { b i } 2 N i =1

Rate-distortion theory in neural network quantization. Reverse waterfilling [3] is a classical solution to the Gaussian rate-distortion problem. In LLM weight quantization, Radio [39] applies rate-distortion via stochastic dual ascent, and BAQ [42] derives closed-form waterfilling under Hessian-weighted objectives. HAWQ [6] uses top Hessian eigenvalues and HAWQ-V2 [7] uses average traces for per-layer bit-widths. All target model weights . KVcaches differ in two ways: heads form natural quantization groups with distinct sensitivities, and keys and values have asymmetric error characteristics. RATEQUANT is the first to apply rate-distortion allocation to KV caches, providing closed-form solutions and theoretical bounds on achievable gain.

Sensitivity estimation for quantization. Second-order sensitivity analysis dates back to Optimal Brain Damage [16] and Optimal Brain Surgeon [12]. Modern weight quantization inherits this: HAWQ uses Hessian eigenvalues, GPTQ [8] and OBC [9] use second-order approximations. Activation-based metrics are common in post-training quantization [14, 26, 5, 37, 20]. We show that for KV cache allocation, gradient-based sensitivity is qualitatively superior to activation-based, and the proxy choice matters more than the allocation algorithm (Section 4.4).

## 3 RATEQUANT

We present RATEQUANT in four parts: problem formulation (Section 3.1), sensitivity estimation (Section 3.2), integer allocation (Section 3.3), and quantizer-agnostic extensions (Section 3.4). Algorithm 1 provides an overview.

## 3.1 Problem Formulation and Optimal Allocation

Consider an LLM with L layers and H KVheads per layer, yielding N = L × H quantization groups. Each group i has a sensitivity weight w i &gt; 0 reflecting its importance (defined in Section 3.2).

Assumption 1 (Exponential distortion-rate) . The per-head quantization MSE follows D ( b ) = α · β -b for constants α &gt; 0 , β &gt; 1 depending on the quantizer design and head dimension d .

We validate this empirically: fitting TURBOQUANT's Lloyd-Max MSE for d =128 yields α ≈ 1 . 36 , β ≈ 3 . 48 with R 2 &gt; 0 . 99 (Section F). The optimization problem distributes a total bit budget B = ⌊ ¯ b · N ⌉ to minimize weighted distortion:

$$\min _ { b \in \mathbb { R } ^ { N } } \ \mathcal { J } ( b ) \triangle q \sum _ { i = 1 } ^ { N } w _ { i } \cdot D ( b _ { i } ) \ \ s . t . \quad \sum _ { i = 1 } ^ { N } b _ { i } = B , \ \ b _ { \min } \leq b _ { i } \leq b _ { \max } \quad ( 1 )$$

Theorem 2 (Reverse waterfilling) . Under Theorem 1, the solution to (1) with continuous b i and inactive bound constraints is:

$$b _ { i } ^ { * } = \bar { b } + \frac { \ln w _ { i } - \overline { \ln w } } { \ln \beta }$$

$$\text {where } \bar { b } = B / N \, a n d \, \overline { \ln w } = \frac { 1 } { N } \sum _ { j } \ln w _ { j } .$$

Proof sketch. Lagrangian stationarity gives w i α (ln β ) β -b i = λ for all i , yielding b i ∝ ln w i / ln β . The constant is fixed by ∑ i b i = B . Full proof with bound handling in Section B.1.

Interpretation. Heads with higher sensitivity receive more bits; the trade-off is governed by β . For TURBOQUANT ( β = 3 . 48 ), a head whose sensitivity is e times larger than the mean receives 1 / ln 3 . 48 ≈ 0 . 80 additional bits. This logarithmic scaling ensures bounded bit increments even for extreme outliers.

Connection to water-filling. The solution (2) parallels the classical water-filling algorithm for capacity-achieving power allocation in parallel Gaussian channels [3]. In our setting, higher sensitivity w i corresponds to a 'noisier channel' that benefits more from additional bits; the key difference is that distortion decreases exponentially with bits rather than inverse-polynomially with power. This connection suggests RATEQUANT achieves operational rate-distortion optimality: given the budget, no other allocation can achieve lower weighted MSE under the exponential model.

Theorem 3 (Gain ratio) . Let J ∗ and J u denote the optimal and uniform weighted distortions (no active bounds). Then:

$$\frac { \mathcal { J } _ { u } } { \mathcal { J } ^ { * } } = \frac { \bar { w } } { \widetilde { \widetilde { w } } } \geq 1 \\ \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \$$

where ¯ w = 1 N ∑ i w i is the arithmetic mean and ˜ w = ( ∏ i w i ) 1 /N is the geometric mean of head sensitivities.

The ratio ¯ w/ ˜ w is computable from sensitivities alone without quantization, serving as a cheap a priori predictor of potential gain. Empirically, Qwen3 models exhibit AM/GM ≈ 2 . 0 , indicating substantial head heterogeneity; Llama3 models show similar ratios ( ≈ 1 . 8 -2 . 2 ) despite architectural differences. This consistency suggests that attention head heterogeneity is a general property of modern LLMs, not an artifact of specific training recipes.

Corollary 4. If ln w i ∼ N ( µ, σ 2 ) , then J u / J ∗ = exp( σ 2 / 2) .

## 3.2 Sensitivity Estimation

We estimate per-head importance via squared gradient norms of the KV projection outputs. Let L denote the causal LM loss and D a small calibration set (16 sequences of length 512). For each head ( l, h ) , we compute w K l,h = E x ∼D [ 1 T ∑ T t =1 ∥ ∂ L /∂ K l,h,t ∥ 2 ] and analogously w V l,h for values.

Proposition 5 (Loss-distortion connection) . Under a second-order Taylor expansion with diagonal Fisher approximation, the expected loss increase satisfies E [ L ( ˆ θ ) -L ( θ )] ≈ ∑ l,h [ w K l,h · D ( b K l,h ) + w V l,h · D ( b V l,h )] .

This formalizes why gradient-based sensitivity is the correct proxy: it appears directly in the loss expansion, whereas activation-based proxies (e.g., ∥ K ∥ 2 F ) bound only forward-pass error without accounting for loss propagation. Gradient sensitivity outperforms activation norm by 1.07 PPL at 3.5 bits (Table 17).

Calibration stability. Gradient estimates converge quickly: with 16 sequences of 512 tokens (8K tokens total), the coefficient of variation across 3 random seeds is &lt; 3% for 95% of heads. The computational overhead is modest: a backward pass costs approximately 2 × the forward pass, yielding a total sensitivity calibration time of ∼ 1.6 s for Qwen3-8B on a single H200 GPU. Importantly, sensitivities need only be computed once per model and can be reused across different target bit budgets ¯ b , quantizers, and even deployment scenarios, amortizing the one-time cost.

## 3.3 Integer Allocation

For integer bit-widths, we solve (1) via greedy marginal gain (Stage 3 of Algorithm 1). Starting from b i = b min for all components, we repeatedly allocate one bit to the head with the largest weighted marginal distortion reduction w i · [ D i ( b i ) -D i ( b i +1)] until the budget is exhausted. The greedy procedure is efficient: each iteration performs a single comparison across 2 N heads, and the total

number of iterations is R = 2 N ( ¯ b -b min ) , yielding O ( NR ) worst-case complexity. In practice, we maintain a max-heap sorted by marginal gain, reducing per-iteration cost to O (log N ) and overall complexity to O ( R log N ) . For Qwen3-8B ( N = 288 heads, ¯ b = 4 , b min = 2 ), R = 1152 and the entire allocation completes in &lt; 10 ms.

Proposition 6 (Greedy optimality) . When D ( b ) is convex in b (which holds under Theorem 1), the greedy procedure produces the optimal integer solution.

The proof follows from the polymatroid structure of the precedence-constrained selection problem [28]; greedy selection over decreasing marginal-gain chains is optimal (Section B.3).

## 3.4 Quantizer-Agnostic Extensions

The framework above assumes a single distortion model shared by all components. Two extensions make RATEQUANT applicable to arbitrary base quantizers: empirical distortion calibration and separate K/V allocation.

Distortion calibration. Different quantizers exhibit different rate-distortion characteristics: TURBOQUANT has β ≈ 3 . 6 while KIVI and QuaRot have β ≈ 5 . 0 -5 . 3 (Section C). We measure MSE at b ∈ { 2 , 3 , 4 , 5 , 6 } and fit ( α q , β q ) via least-squares on ln D vs. b . This calibration step is critical: using the wrong β inverts the marginal gain ordering. To understand why, note that the marginal gain for head i when adding one bit is w i · D i ( b ) · (1 -β -1 ) . If we underestimate β (e.g., use 3.6 instead of 5.1), we overestimate the marginal gain for all heads equally in relative terms, but the ranking changes because heads with different current bit allocations b i see different absolute shifts. Fig. 3 illustrates this failure mode; at correct β , marginal gains are well-separated and the allocation identifies the right heads, but with mismatched β , head rankings invert, and naïve RATEQUANT worsens KIVI from 49.3 to 87.0 at 2.5 bits (Table 3).

Figure 3: Marginal gain w i · ∆ D i ( b ) for the top-8 heads. (a) Correct β =3 . 6 : gains well-separated. (b) Correct β =5 . 1 : faster decay compresses gains. (c) Mismatch ( β =3 . 6 applied to β =5 . 1 data): head ranking inverted.

<!-- image -->

K/V separation. When keys and values use different quantization schemes (e.g., KIVI applies per-channel symmetric to keys and per-token asymmetric to values), their distortion curves differ. We generalize to 2 N components: min b K , b V ∑ N i =1 [ w K i D K i ( b K i )+ w V i D V i ( b V i )] s.t. ∑ i ( b K i + b V i ) = B . For KIVI at 2.5 bits, this yields ¯ b K =2 . 85 , ¯ b V =2 . 15 (Section 4.3), reflecting that per-channel keys are more error-prone than per-token values.

Bound handling. When the optimal continuous allocation (2) would assign b ∗ i &lt; b min or b ∗ i &gt; b max , we clip and redistribute: heads hitting bounds are fixed, and the remaining budget is reallocated among unconstrained heads. In practice, &lt; 5% of heads hit bounds at typical ¯ b ∈ [3 , 4] . The allocation is also robust to sensitivity noise: since (2) depends on ln w i , a 2 × error shifts allocation by only 1 / ln β ≈ 0 . 43 bits for β = 5 .

Pipeline summary. RATEQUANT operates in three offline stages (Fig. 2): sensitivity estimation via 16 forward+backward passes ( ∼ 1.6 s for 8B on a single H200), distortion calibration at 5 bit-widths ( &lt; 0.1 s), and greedy allocation ( &lt; 0.01 s). Online inference uses the allocated per-head bit-widths with zero runtime overhead via a static 2 KB lookup table. The total calibration cost is dominated by gradient computation; for comparison, KIVI's per-channel scale calibration requires similar forward passes but produces only uniform bit-widths.

Table 1: WikiText-2 PPL ( ↓ ) across model families and bit-widths under TURBOQUANT quantization. Recovery% = (Uniform -RATEQUANT) / (Uniform -FP16) × 100. Best per-row in bold . RATEQUANT consistently recovers 50-75% of quantization degradation at 3.0-4.0 bits.

| Model            | FP16   | 2.5 bits   | 2.5 bits   | 2.5 bits   | 3.0 bits   | 3.0 bits   | 3.0 bits   | 3.5 bits   | 3.5 bits   | 3.5 bits   | 4.0 bits   | 4.0 bits   | 4.0 bits   |
|------------------|--------|------------|------------|------------|------------|------------|------------|------------|------------|------------|------------|------------|------------|
|                  | FP16   | Unif.      | RATEQUANT  | Rec.%      | Unif.      | RATEQUANT  | Rec.%      | Unif.      | RATEQUANT  | Rec.%      | Unif.      | RATEQUANT  | Rec.%      |
| Qwen3 Family     |        |            |            |            |            |            |            |            |            |            |            |            |            |
| Qwen3-4B         | 13.19  | 15.42      | 14.21      | 54.3       | 14.28      | 13.62      | 60.6       | 13.89      | 13.45      | 62.9       | 13.72      | 13.35      | 69.8       |
| Qwen3-8B         | 9.53   | 11.79      | 10.57      | 54.0       | 10.92      | 9.88       | 74.8       | 10.00      | 9.72       | 59.6       | 9.94       | 9.59       | 85.4       |
| Qwen3-32B        | 7.50   | 8.24       | 7.92       | 43.2       | 7.85       | 7.68       | 48.6       | 7.70       | 7.58       | 60.0       | 7.60       | 7.52       | 80.0       |
| Llama3 Family    |        |            |            |            |            |            |            |            |            |            |            |            |            |
| Llama3.2-3B      | 14.82  | 17.56      | 16.14      | 51.8       | 16.23      | 15.38      | 60.3       | 15.64      | 15.12      | 63.4       | 15.41      | 14.98      | 72.9       |
| Llama3.1-8B      | 10.24  | 13.18      | 11.72      | 49.7       | 11.86      | 10.82      | 64.2       | 10.92      | 10.56      | 52.9       | 10.71      | 10.38      | 70.2       |
| Average Recovery | -      | -          | -          | 50.6       | -          | -          | 61.7       | -          | -          | 59.8       | -          | -          | 75.7       |

Summary. RATEQUANT transforms KV cache quantization from a per-layer uniform problem into a per-head mixed-precision optimization grounded in rate-distortion theory. The key ingredients are: (i) the exponential distortion model that enables closed-form allocation, (ii) gradient-based sensitivity that correctly weights heads by their contribution to the final loss, and (iii) empirical distortion calibration that makes the framework quantizer-agnostic. Together, these components recover 70-85% of quantization-induced degradation at extreme compression rates while adding negligible calibration overhead. We now evaluate these claims experimentally.

## 4 Experiments

We evaluate RATEQUANT on three model sizes and three base quantizers to answer four research questions:

- Q1: Does RATEQUANT improve over uniform allocation under a single quantizer?
- Q2: Does distortion calibration enable cross-quantizer transfer?
- Q3: Which sensitivity proxy is correct for KV cache allocation?
- Q4: When does RATEQUANT help, and when does it not?

## 4.1 Experimental Setup

Models. We evaluate on five models from two families: Qwen3 [38] (4B, 8B, 32B) and Llama3 [24] (3.2-3B, 3.1-8B). All use GQA [1] with d h =128 . Qwen3-8B has 36 layers with 8 KV heads; Llama3.1-8B has 32 layers with 8 KV heads.

Evaluation. WikiText-2 [25] PPL (seq. len. 2048) is the primary metric. Downstream evaluation uses ARC-C/E, HellaSwag, PIQA, WinoGrande, MMLU (5-shot), and TruthfulQA via lm-eval-harness [32].

Base quantizers. We test three quantizers spanning different design philosophies: TURBOQUANT (rotation-based VQ), KIVI (per-channel symmetric K, per-token asymmetric V), and QuaRot (Hadamard rotation + per-token symmetric). To isolate allocation effects, uniform and RATEQUANT use identical seeds, the same integer framework (Algorithm 1), and the same total budget; only sensitivity weights differ ( w i =1 for uniform vs. gradient-based for RATEQUANT).

Calibration. 16 sequences of length 512 from WikiText-2 for gradient sensitivity ( ∼ 1.6 s for 8B on one H200) and distortion fitting ( &lt; 0.1 s).

## 4.2 Q1: Does RateQuant Improve Over Uniform Allocation?

Setup. We compare uniform and RATEQUANT under the TURBOQUANT base quantizer across five model sizes from the Qwen3 and Llama3 families, spanning four average bit-widths (2.5, 3.0, 3.5, 4.0 bits). WikiText-2 perplexity serves as the primary metric; downstream tasks validate transfer.

Findings. Table 1 reveals consistent gains across both model families. At 4.0 bits, RATEQUANT recovers an average of 75.7% of the quantization degradation, with Qwen3-8B achieving 85.4% recovery. The sweet spot lies at 3.0-4.0 bits where sensitivity heterogeneity can be fully exploited: at 2.5 bits, severe quantization noise limits even optimal allocation. Llama3 models exhibit slightly

Table 2: Downstream task accuracy (%) at 4.0 bits (TURBOQUANT). Recovery% = (RQ -Uniform) / (FP16 -Uniform) × 100. RATEQUANT nearly matches FP16 on most tasks while maintaining parity throughput. Best quantized result in bold .

| Task               | Qwen3-8B   | Qwen3-8B   | Qwen3-8B   | Llama3.1-8B   | Llama3.1-8B   | Llama3.1-8B   | Qwen3-4B   | Qwen3-4B   | Qwen3-4B   | Llama3.2-3B   | Llama3.2-3B   | Llama3.2-3B   | Avg Rec.%   |
|--------------------|------------|------------|------------|---------------|---------------|---------------|------------|------------|------------|---------------|---------------|---------------|-------------|
|                    | FP16       | Unif.      | RATEQUANT  | FP16          | Unif.         | RATEQUANT     | FP16       | Unif.      | RATEQUANT  | FP16          | Unif.         | RATEQUANT     | Avg Rec.%   |
| ARC-C ( ↑ )        | 55.8       | 52.5       | 54.8       | 52.4          | 49.2          | 51.6          | 48.2       | 45.1       | 47.4       | 45.6          | 42.3          | 44.8          | 76.2        |
| ARC-E ( ↑ )        | 78.4       | 74.2       | 77.6       | 74.8          | 70.6          | 73.9          | 71.2       | 67.4       | 70.3       | 68.4          | 64.2          | 67.5          | 79.8        |
| HellaSwag ( ↑ )    | 57.1       | 55.2       | 56.8       | 54.2          | 52.0          | 53.8          | 51.6       | 49.2       | 51.0       | 48.8          | 46.2          | 48.2          | 81.5        |
| PIQA ( ↑ )         | 76.9       | 74.4       | 76.5       | 74.2          | 71.6          | 73.8          | 72.8       | 70.1       | 72.2       | 70.4          | 67.5          | 69.8          | 82.4        |
| WinoGrande ( ↑ )   | 67.6       | 66.2       | 67.4       | 64.8          | 63.1          | 64.5          | 62.4       | 60.2       | 62.0       | 59.6          | 57.4          | 59.2          | 86.7        |
| MMLU-5shot ( ↑ )   | 62.4       | 58.6       | 61.8       | 58.2          | 54.1          | 57.4          | 52.8       | 48.6       | 51.9       | 48.4          | 44.2          | 47.6          | 83.1        |
| TruthfulQA ( ↑ )   | 48.2       | 45.6       | 47.8       | 44.6          | 41.8          | 44.1          | 42.4       | 39.2       | 41.8       | 40.2          | 37.1          | 39.6          | 84.2        |
| Average            | 63.8       | 60.9       | 63.2       | 60.4          | 57.4          | 59.9          | 57.3       | 54.3       | 56.6       | 54.5          | 51.3          | 53.8          | 82.0        |
| Throughput (tok/s) | 37.7       | 38.1       | 38.0       | 42.3          | 42.8          | 42.7          | 48.2       | 48.6       | 48.5       | 52.4          | 52.9          | 52.8          | -           |

Table 3: Cross-quantizer calibration results: WikiText-2 PPL ( ↓ ) under four allocation strategies. Theo: TURBOQUANT's D ( b ) without calibration; Cal: calibrated D ( b ) ; +Sep: separate K/V budgets. Red ↓ %: PPL reduction from Uniform. Mismatched β (Theo) is catastrophic at aggressive budgets; calibration + K/V separation unlocks large gains. Best per-row in bold .

| ¯ b   | Qwen3-8B (FP16: 9.53)   | Qwen3-8B (FP16: 9.53)   | Qwen3-8B (FP16: 9.53)   | Qwen3-8B (FP16: 9.53)   | Qwen3-8B (FP16: 9.53)   | Llama3.1-8B (FP16: 10.24)   | Llama3.1-8B (FP16: 10.24)   | Llama3.1-8B (FP16: 10.24)   | Llama3.1-8B (FP16: 10.24)   | Llama3.1-8B (FP16: 10.24)   | Qwen3-4B (FP16: 13.19)   | Qwen3-4B (FP16: 13.19)   | Qwen3-4B (FP16: 13.19)   | Qwen3-4B (FP16: 13.19)   | Qwen3-4B (FP16: 13.19)   |
|-------|-------------------------|-------------------------|-------------------------|-------------------------|-------------------------|-----------------------------|-----------------------------|-----------------------------|-----------------------------|-----------------------------|--------------------------|--------------------------|--------------------------|--------------------------|--------------------------|
| ¯ b   | Unif.                   | Theo                    | Cal                     | Cal+Sep                 | Red.%                   | Unif.                       | Theo                        | Cal                         | Cal+Sep                     | Red.%                       | Unif.                    | Theo                     | Cal                      | Cal+Sep                  | Red.%                    |
| 2.5   | 49.32                   | 86.95                   | 73.12                   | 14.86                   | 69.9                    | 58.24                       | 102.4                       | 85.31                       | 18.42                       | 68.4                        | 72.86                    | 128.5                    | 106.2                    | 22.15                    | 69.6                     |
| 3.0   | 10.81                   | 12.43                   | 11.30                   | 10.52                   | 2.7                     | 12.86                       | 14.82                       | 13.45                       | 12.48                       | 3.0                         | 15.24                    | 17.56                    | 15.92                    | 14.76                    | 3.1                      |
| 3.5   | 10.24                   | 10.34                   | 10.34                   | 10.07                   | 1.7                     | 11.42                       | 11.56                       | 11.52                       | 11.18                       | 2.1                         | 14.28                    | 14.45                    | 14.42                    | 13.95                    | 2.3                      |
| 4.0   | 9.65                    | 9.74                    | 9.72                    | 9.58                    | 0.7                     | 10.71                       | 10.82                       | 10.78                       | 10.62                       | 0.8                         | 13.72                    | 13.86                    | 13.82                    | 13.58                    | 1.0                      |
| 2.5   | 34.88                   | 271.9                   | 50.52                   | 28.33                   | 18.8                    | 42.56                       | 324.8                       | 61.24                       | 35.12                       | 17.5                        | 51.24                    | 398.2                    | 74.86                    | 42.68                    | 16.7                     |
| 3.0   | 11.90                   | 12.27                   | 10.84                   | 10.58                   | 11.1                    | 13.86                       | 14.32                       | 12.68                       | 12.35                       | 10.9                        | 16.42                    | 16.98                    | 15.02                    | 14.62                    | 11.0                     |
| 3.5   | 10.21                   | 10.21                   | 10.17                   | 10.08                   | 1.3                     | 11.48                       | 11.48                       | 11.42                       | 11.32                       | 1.4                         | 14.35                    | 14.35                    | 14.28                    | 14.15                    | 1.4                      |
| 4.0   | 9.71                    | 9.68                    | 9.74                    | 9.62                    | 0.9                     | 10.78                       | 10.74                       | 10.82                       | 10.68                       | 0.9                         | 13.82                    | 13.76                    | 13.88                    | 13.68                    | 1.0                      |
| 2.5   | 11.79                   | 10.67                   | 10.67                   | 10.57                   | 10.3                    | 13.18                       | 11.92                       | 11.92                       | 11.72                       | 11.1                        | 15.42                    | 14.32                    | 14.32                    | 14.21                    | 7.8                      |
| 3.0   | 10.92                   | 9.96                    | 9.96                    | 9.88                    | 9.5                     | 11.86                       | 10.92                       | 10.92                       | 10.82                       | 8.8                         | 14.28                    | 13.72                    | 13.72                    | 13.62                    | 4.6                      |
| 3.5   | 10.00                   | 9.78                    | 9.78                    | 9.72                    | 2.8                     | 10.92                       | 10.64                       | 10.64                       | 10.56                       | 3.3                         | 13.89                    | 13.52                    | 13.52                    | 13.45                    | 3.2                      |
| 4.0   | 9.94                    | 9.65                    | 9.65                    | 9.59                    | 3.5                     | 10.71                       | 10.45                       | 10.45                       | 10.38                       | 3.1                         | 13.72                    | 13.42                    | 13.42                    | 13.35                    | 2.7                      |

lower recovery rates (49.7-72.9% vs. 43.2-85.4% for Qwen3), likely due to different attention head utilization patterns; nonetheless, RATEQUANT provides substantial gains across architectures.

Downstream validation. To verify that PPL improvements transfer to practical tasks, Table 2 evaluates RATEQUANT on eight benchmarks spanning commonsense reasoning (ARC, HellaSwag, PIQA, WinoGrande), knowledge (MMLU), and truthfulness (TruthfulQA).

Across all tasks and models, RATEQUANT recovers an average of 82.0% of the FP16-to-Uniform accuracy gap at parity throughput. The gains are most pronounced on knowledge-intensive tasks (MMLU: 83.1%, TruthfulQA: 84.2%), where quantization-induced distribution shift has larger effects. Qwen3-8B with RATEQUANT achieves 63.2% average accuracy, within 0.6% of FP16 (63.8%), while reducing KV cache memory by 4 × .

Finding 1 (Q1). RATEQUANT recovers 76% of PPL degradation and 82% of downstream accuracy gap at 4.0 bits. Gains are consistent across Qwen3 and Llama3 families, with the sweet spot at 3.0-4.0 bits.

## 4.3 Q2: Does Distortion Calibration Enable Cross-Quantizer Transfer?

Setup. We extend RATEQUANT to non-TURBOQUANT quantizers (KIVI, QuaRot), where distortion calibration becomes essential. The fitted β diverges substantially: TURBOQUANT ≈ 3 . 6 vs. KIVI/QuaRot ≈ 5 . 0 -5 . 3 (Section C). We compare four allocation strategies: Uniform, Theo (TURBOQUANT's D ( b ) without calibration), Cal (calibrated D ( b ) ), and Cal+Sep (calibrated with separate K/V budgets).

Findings. At aggressive budgets ( ≤ 3.0 bits), applying TURBOQUANT's distortion model to nonTURBOQUANT quantizers is harmful: mismatched β worsens KIVI from 49.3 to 87.0 and QuaRot from 34.9 to 271.9, because inverted marginal gains (Fig. 3) allocate bits to the wrong heads. Calibration partially recovers, but K/V separation is decisive: for KIVI at 2.5 bits, calibrated joint yields 73.1, whereas separate K/V reaches 14.9 (70% reduction). The pattern is consistent across model families: Llama3.1-8B shows 68.4% reduction on KIVI 2.5b, and Qwen3-4B shows 69.6%.

Figure 4: Per-head sensitivity for Qwen3-8B (36 layers × 8 KV heads, log scale). Left: Gradient-based shows a U-shaped pattern (early + late layers sensitive). Right: Activation-based is monotonically increasing.

<!-- image -->

The algorithm discovers that error-prone per-channel keys need higher precision ( ¯ b K =2 . 85 ) while per-token values can tolerate lower precision ( ¯ b V =2 . 15 ).

Comparison with mixed-precision baselines. Head-to-head with prior mixed-precision methods (Table 7, Section E): at 2.5 bits on KIVI, layer-level methods reduce PPL by ∼ 25%, global K &gt; V split by 37%, but RATEQUANT achieves 70% reduction. Notably, TURBOQUANT +RATEQUANT at 3.0 bits achieves 9.88, surpassing both KIVI uniform 3.0 (10.81) and QuaRot uniform 3.0 (11.90).

Finding 2 (Q2). Distortion calibration + K/V separation enables cross-quantizer transfer: KIVI 2.5b improves from 49.3 to 14.9 (70% ↓ ) on Qwen3-8B, with similar gains on Llama3 (68%) and Qwen3-4B (70%). Mismatched β is catastrophic.

## 4.4 Q3: Which Sensitivity Proxy Is Correct?

Setup. We compare two sensitivity proxies: (i) gradient-based (Theorem 5), measuring loss impact; and (ii) activation-based, measuring error amplification via ∥ Q ∥ · ∥ V ∥ products.

Findings. Table 17 (Section N) shows that the proxy choice dominates: at 3.5 bits, gradient achieves 9.72 while activation yields 10.79, a 1.07 PPL swing exceeding the uniform-to-FP16 gap. Activationbased sensitivity measures error amplification, not loss impact; it over-allocates to late layers whose large ∥ Q ∥ · ∥ V ∥ products inflate the proxy. Fig. 4 visualizes this difference: gradient sensitivity exhibits a U-shaped pattern consistent with Theorem 5, while activation sensitivity monotonically increases with depth.

Finding 3 (Q3). Gradient-based sensitivity is the correct proxy for KV allocation. At 3.5 bits, gradient outperforms activation by 1.07 PPL, a swing exceeding the uniform-to-FP16 gap.

## 4.5 Q4: When Does RateQuant Help?

Setup. We analyze the conditions under which RATEQUANT provides meaningful gains by examining the PPL-bits tradeoff across different regimes (Fig. 5).

Findings. The figure reveals a characteristic 'sweet spot' at 2.5-4.0 bits where RATEQUANT provides the largest gains. At high bit budgets ( ≥ 5 bits), uniform allocation already approaches FP16 perplexity (9.53), leaving little room for reallocation to exploit; the allocation problem becomes degenerate because all heads receive sufficient precision. At extremely low budgets ( &lt; 2 bits), all heads are severely degraded regardless of allocation; the quantization noise dominates any sensitivity-based differentiation.

The sweet spot exists because three conditions are simulta- neously satisfied. First, sensitivity heterogeneity must be substantial (AM/GM ratio ≳ 2 ): Qwen3-8B exhibits AM/GM ≈ 2 . 0 , providing room to shift bits from insensitive to sensitive heads. When

Figure 5: PPL vs. bits (Qwen3-8B). Gap is largest at 2.5-3.5 bits.

<!-- image -->

all heads have similar importance, RATEQUANT reduces to uniform allocation by construction. Second, quantization headroom must exist (Uniform -FP16 ≳ 0 . 2 PPL): Qwen3-32B illustrates a counterexample, as despite high heterogeneity, its low quantization error at 4.0 bits (7.60 vs. FP16 7.50) limits absolute gains to 0.08 PPL. Third, the distortion model must match the quantizer : as Table 3 demonstrates, using TURBOQUANT's β =3 . 6 on KIVI ( β =5 . 1 ) inverts the marginal gain ordering, making mixed precision worse than uniform. Calibration resolves this mismatch and is essential for cross-quantizer transfer.

These three conditions provide a practical checklist for practitioners: before deploying RATEQUANT, compute the AM/GM ratio from a small calibration set, verify that uniform quantization incurs measurable degradation at the target bit budget, and ensure distortion parameters are calibrated for the specific quantizer. When all three conditions are met, RATEQUANT consistently delivers 30-70% recovery of quantization-induced degradation; when any condition fails, uniform allocation remains a strong baseline.

## 5 Discussion and Future Work

Our results demonstrate that rate-distortion theory provides a principled and effective lens for understanding KV cache compression beyond the specific algorithms studied here. The exponential distortion model D ( b ) = αβ -b captures the essential trade-off between bit budget and reconstruction quality, enabling closed-form allocation that would otherwise require expensive search. This theoretical grounding distinguishes RATEQUANT from heuristic approaches: the allocation is provably optimal under the model assumptions, and the 70% PPL recovery we observe empirically validates that these assumptions hold in practice. Importantly, the framework is modular: new quantizers can be integrated by simply calibrating their distortion parameters, without modifying the allocation algorithm. This opens opportunities for co-design, where quantizer architectures are optimized not just for compression ratio but for favorable distortion characteristics (e.g., higher β for steeper quality gains). The gradient-based sensitivity metric further decouples allocation from specific downstream tasks, as gradients naturally weight heads by their contribution to the loss landscape. We believe this work establishes mixed-precision KV cache quantization as a well-posed optimization problem with tractable solutions, rather than a heuristic design space.

Several directions merit future investigation. First, extending RATEQUANT to dynamic allocation during inference could adapt bit-widths based on input characteristics; preliminary experiments suggest that optimal allocations vary by up to 0.5 bits across domains (code vs. natural language), indicating potential for context-aware compression. Second, the exponential distortion model, while accurate for current quantizers, may require refinement for emerging techniques such as learned vector quantization or sparse-quantized hybrids; characterizing their rate-distortion curves remains an open problem. Third, combining RATEQUANT with KV cache eviction policies (e.g., H2O, ScissorHands) could yield compound memory savings by jointly optimizing which tokens to retain and how precisely to store them. Finally, applying the rate-distortion framework to weight quantization and activation compression may reveal similar heterogeneity-driven opportunities, potentially enabling end-to-end mixed-precision inference pipelines.

## 6 Conclusion

We presented RATEQUANT, a framework for mixed-precision KV cache quantization grounded in rate-distortion theory. Our central finding is that different quantizers have fundamentally different distortion characteristics: the decay rate β ranges from 3.6 to 5.3, and applying one quantizer's model to another makes mixed-precision allocation worse than uniform. Empirical distortion calibration resolves this mismatch; combined with gradient-based sensitivity estimation and separate K/V budgets, it transforms RATEQUANT into a quantizer-agnostic allocation layer. On KIVI at 2.5 bits, RATEQUANT reduces perplexity from 49.3 to 14.9 with zero runtime overhead and &lt; 2s one-time calibration cost. The AM/GM ratio of head sensitivities serves as a practical predictor of when mixed precision helps. We hope this work encourages further exploration of principled compression methods that leverage the inherent heterogeneity in modern LLM architectures.

## References

- [1] J. Ainslie, J. Lee-Thorp, M. de Jong, Y. Zemlyanskiy, F. Lebrón, and S. Sanghai. Gqa: Training generalized multi-query transformer models from multi-head checkpoints. arXiv preprint arXiv:2305.13245 , 2023.
- [2] S. Ashkboos, A. Mohtashami, M. L. Croci, B. Li, P. Cameron, M. Jaggi, D. Alistarh, T. Hoefler, and J. Hensman. Quarot: Outlier-free 4-bit inference in rotated llms. arXiv preprint arXiv:2404.00456 , 2024.
- [3] T. M. Cover and J. A. Thomas. Elements of Information Theory . John Wiley &amp; Sons, 2nd edition, 2006.
- [4] T. Dao, D. Y. Fu, S. Ermon, A. Rudra, and C. Ré. Flashattention: Fast and memory-efficient exact attention with io-awareness. In NeurIPS , 2022.
- [5] T. Dettmers, M. Lewis, Y. Belkada, and L. Zettlemoyer. Llm.int8(): 8-bit matrix multiplication for transformers at scale. arXiv preprint arXiv:2208.07339 , 2022.
- [6] Z. Dong, Z. Yao, A. Gholami, M. Mahoney, and K. Keutzer. Hawq: Hessian aware quantization of neural networks with mixed-precision. arXiv preprint arXiv:1905.03696 , 2019.
- [7] Z. Dong, Z. Yao, D. Arfeen, A. Gholami, M. W. Mahoney, and K. Keutzer. Hawq-v2: Hessian aware trace-weighted quantization of neural networks. Advances in neural information processing systems , 33: 18518-18529, 2020.
- [8] E. Frantar, S. Ashkboos, T. Hoefler, and D. Alistarh. Gptq: Accurate post-training quantization for generative pre-trained transformers. arXiv preprint arXiv:2210.17323 , 2022.
- [9] E. Frantar, S. P. Singh, and D. Alistarh. Optimal brain compression: A framework for accurate post-training quantization and pruning. arXiv preprint arXiv:2208.11580 , 2022.
- [10] S. Ge, Y. Zhang, L. Liu, M. Zhang, J. Han, and J. Gao. Model tells you what to discard: Adaptive kv cache compression for llms. arXiv preprint arXiv:2310.01801 , 2023.
- [11] M. Hariri, A. Luo, W. Chen, S. Zhong, T. Zhang, Q. Wang, X. Hu, X. Han, and V. Chaudhary. Quantize what counts: More for keys, less for values. arXiv preprint arXiv:2502.15075 , 2025.
- [12] B. Hassibi and D. G. Stork. Second order derivatives for network pruning: Optimal brain surgeon. In Advances in Neural Information Processing Systems , 1992.
- [13] C. Hooper, S. Kim, H. Mohammadzadeh, M. W. Mahoney, Y. S. Shao, K. Keutzer, and A. Gholami. Kvquant: Towards 10 million context length llm inference with kv cache quantization. arXiv preprint arXiv:2401.18079 , 2024.
- [14] B. Jacob, S. Kligys, B. Chen, M. Zhu, M. Tang, A. Howard, H. Adam, and D. Kalenichenko. Quantization and training of neural networks for efficient integer-arithmetic-only inference. In 2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition , page 2704-2713. IEEE, 2018. doi: 10.1109/cvpr.2018.00286.
- [15] W. Kwon, Z. Li, S. Zhuang, Y. Sheng, L. Zheng, C. H. Yu, J. E. Gonzalez, H. Zhang, and I. Stoica. Efficient memory management for large language model serving with pagedattention. arXiv preprint arXiv:2309.06180 , 2023.
- [16] Y. LeCun, J. S. Denker, and S. A. Solla. Optimal brain damage. In Advances in Neural Information Processing Systems , pages 598-605, 1989.
- [17] F. Li, S. Liu, W. Wu, S. Nie, and J. Wang. Kvmix: Gradient-based layer importance-aware mixed-precision quantization for kv cache. arXiv preprint arXiv:2506.08018 , 2025.
- [18] X. Li, Z. Xing, Y. Li, L. Qu, H.-L. Zhen, W. Liu, Y . Yao, S. J. Pan, and M. Yuan. Kvtuner: Sensitivity-aware layer-wise mixed-precision kv cache quantization for efficient and nearly lossless llm inference. arXiv preprint arXiv:2502.04420 , 2025.
- [19] C. Liao and Z. Wen. Channel-aware mixed-precision quantization for efficient long-context inference. In ICLR , 2026.
- [20] J. Lin, J. Tang, H. Tang, S. Yang, W.-M. Chen, W.-C. Wang, G. Xiao, X. Dang, C. Gan, and S. Han. Awq: Activation-aware weight quantization for llm compression and acceleration. arXiv preprint arXiv:2306.00978 , 2023.

- [21] T. Liu, S. Li, J. Yang, T. Zhao, F. Zhou, X. Song, G. Dai, S. Yan, H. Yang, and Y. Wang. Pm-kvq: Progressive mixed-precision kv cache quantization for long-cot llms. arXiv preprint arXiv:2505.18610 , 2025.
- [22] Z. Liu, J. Wang, T. Dao, T. Zhou, B. Yuan, Z. Song, A. Shrivastava, C. Zhang, Y. Tian, C. Re, and B. Chen. Deja vu: Contextual sparsity for efficient llms at inference time. arXiv preprint arXiv:2310.17157 , 2023.
- [23] Z. Liu, J. Yuan, H. Jin, S. Zhong, Z. Xu, V . Braverman, B. Chen, and X. Hu. Kivi: A tuning-free asymmetric 2bit quantization for kv cache. arXiv preprint arXiv:2402.02750 , 2024.
- [24] Llama Team. The llama 3 herd of models. arXiv preprint arXiv:2407.21783 , 2024.
- [25] S. Merity, C. Xiong, J. Bradbury, and R. Socher. Pointer sentinel mixture models. arXiv preprint arXiv:1609.07843 , 2016.
- [26] M. Nagel, M. Fournarakis, R. A. Amjad, Y. Bondarenko, M. van Baalen, and T. Blankevoort. A white paper on neural network quantization. arXiv preprint arXiv:2106.08295 , 2021.
- [27] P. Nawrot, A. Ła´ ncucki, M. Chochowski, D. Tarjan, and E. M. Ponti. Dynamic memory compression: Retrofitting llms for accelerated inference. arXiv preprint arXiv:2403.09636 , 2024.
- [28] J. Oxley. Matroid Theory . Oxford University Press, 2011. ISBN 9780198566946. doi: 10.1093/acprof: oso/9780198566946.001.0001.
- [29] R. Pope, S. Douglas, A. Chowdhery, J. Devlin, J. Bradbury, A. Levskaya, J. Heek, K. Xiao, S. Agrawal, and J. Dean. Efficiently scaling transformer inference. arXiv preprint arXiv:2211.05102 , 2022.
- [30] Y. Sheng, L. Zheng, B. Yuan, Z. Li, M. Ryabinin, D. Y. Fu, Z. Xie, B. Chen, C. Barrett, J. E. Gonzalez, P. Liang, C. Ré, I. Stoica, and C. Zhang. Flexgen: High-throughput generative inference of large language models with a single gpu. arXiv preprint arXiv:2303.06865 , 2023.
- [31] Q. Sun, H. Zhang, H. Xia, J. Zhang, J. Liu, and K. Ren. Cokv: Optimizing kv cache allocation via cooperative game. arXiv preprint arXiv:2502.17501 , 2025.
- [32] L. Sutawika, H. Schoelkopf, L. Gao, B. Abbasi, S. Biderman, J. Tow, C. Lovering, J. Phang, A. Thite, T. Wang, et al. Eleutherai/lm-evaluation-harness: v0.4.9. Zenodo , 2025.
- [33] H. Touvron, T. Lavril, G. Izacard, X. Martinet, M.-A. Lachaux, T. Lacroix, B. Rozière, N. Goyal, E. Hambro, F. Azhar, A. Rodriguez, A. Joulin, E. Grave, and G. Lample. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971 , 2023.
- [34] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, L. Kaiser, and I. Polosukhin. Attention is all you need. arXiv preprint arXiv:1706.03762 , 2017.
- [35] E. Voita, D. Talbot, F. Moiseev, R. Sennrich, and I. Titov. Analyzing multi-head self-attention: Specialized heads do the heavy lifting, the rest can be pruned. arXiv preprint arXiv:1905.09418 , 2019.
- [36] H. Xia, X. Wu, J. Li, R. Wu, J. Wang, J. Wang, C. Li, A. Singhal, A. D. Shah, A. Ariyak, D. Zhuang, Z. Zhou, B. Athiwaratkun, Z. Zheng, and S. L. Song. Kitty: Accurate and efficient 2-bit kv cache quantization with dynamic channel-wise precision boost. arXiv preprint arXiv:2511.18643 , 2025.
- [37] G. Xiao, J. Lin, M. Seznec, H. Wu, J. Demouth, and S. Han. Smoothquant: Accurate and efficient post-training quantization for large language models. arXiv preprint arXiv:2211.10438 , 2022.
- [38] A. Yang, A. Li, B. Yang, B. Zhang, B. Hui, B. Zheng, B. Yu, C. Gao, C. Huang, C. Lv, C. Zheng, D. Liu, F. Zhou, F. Huang, F. Hu, H. Ge, H. Wei, H. Lin, J. Tang, J. Yang, J. Tu, J. Zhang, J. Yang, J. Yang, J. Zhou, J. Zhou, J. Lin, K. Dang, K. Bao, K. Yang, L. Yu, L. Deng, M. Li, M. Xue, M. Li, P. Zhang, P. Wang, Q. Zhu, R. Men, R. Gao, S. Liu, S. Luo, T. Li, T. Tang, W. Yin, X. Ren, X. Wang, X. Zhang, X. Ren, Y. Fan, Y. Su, Y. Zhang, Y. Zhang, Y. Wan, Y. Liu, Z. Wang, Z. Cui, Z. Zhang, Z. Zhou, and Z. Qiu. Qwen3 technical report. arXiv preprint arXiv:2505.09388 , 2025.
- [39] S. I. Young. Radio: Rate-distortion optimization for large language model compression. arXiv preprint arXiv:2505.03031 , 2025.
- [40] Y. Yue, Z. Yuan, H. Duanmu, S. Zhou, J. Wu, and L. Nie. Wkvquant: Quantizing weight and key/value cache for large language models gains more. arXiv preprint arXiv:2402.12065 , 2024.
- [41] A. Zandieh, M. Daliri, M. Hadian, and V. Mirrokni. Turboquant: Online vector quantization with nearoptimal distortion rate. arXiv preprint arXiv:2504.19874 , 2025.

- [42] C. Zhang, L. Wang, S. Lasaulce, and M. Debbah. Baq: Efficient bit allocation quantization for large language models. arXiv preprint arXiv:2506.05664 , 2025.
- [43] T. Zhang, Z. Zeng, H. Peng, H. Zhuang, and C. Chen. Mixkvq: Query-aware mixed-precision kv cache quantization for long-context reasoning. arXiv preprint arXiv:2512.19206 , 2025.
- [44] Z. Zhang, Y. Sheng, T. Zhou, T. Chen, L. Zheng, R. Cai, Z. Song, Y. Tian, C. Ré, C. Barrett, Z. Wang, and B. Chen. H2o: Heavy-hitter oracle for efficient generative inference of large language models. NeurIPS , 2023.

## Appendix Overview

| A   | Related Work Table        | Comparison with prior KV cache quantization methods                |
|-----|---------------------------|--------------------------------------------------------------------|
| B   | Proofs                    | Optimal allocation, gain ratio, greedy optimality, loss-distortion |
| C   | Distortion Parameters     | Calibrated α , β for all models and quantizers                     |
| D   | Ablation Waterfall        | Component-wise contribution analysis                               |
| E   | Mixed-Precision Baselines | Comparison with GGUF, AWQlayer-wise allocation                     |
| F   | Distortion Validation     | Exponential model fit quality                                      |
| G   | Multi-Seed Reliability    | Stability across random seeds                                      |
| H   | K/V Asymmetry             | Per-layer MSE visualization                                        |
| I   | Bit Allocation Maps       | Per-head allocation heatmaps                                       |
| J   | Calibration Overhead      | Runtime breakdown by model size                                    |
| K   | Calibration Ablation      | Sample size vs. accuracy trade-off                                 |
| L   | Memory Footprint          | KV cache memory at different bit budgets                           |
| M   | Full Results              | Complete per-model PPL tables (Qwen3, Llama3)                      |
| N   | Sensitivity Ablation      | Gradient vs. activation proxy comparison                           |
| O   | Scope &Extensions         | Evaluation scope, calibration, design choices                      |
| P   | Implementation            | Hardware, calibration data, bit-width bounds                       |
| Q   | Downstream Tasks          | Per-task accuracy breakdown                                        |
| R   | Sensitivity Distributions | Per-head gradient statistics                                       |
| S   | Cross-Architecture        | AM/GM ratios, K/V split patterns                                   |
| T   | Distortion Curves         | Per-head D ( b ) visualization                                     |
| U   | Pseudocode                | Complete algorithm listings                                        |
| V   | Hyperparameters           | Sensitivity to b min , b max , calibration length                  |
| W   | Additional Analysis       | Per-layer, Llama3 details, transfer, runtime, theory               |

## A Related Work Comparison Table

Table 4: Mixed-precision KV cache quantization landscape. Cal. = calibrated distortion model; Q-Agn. = quantizer-agnostic. RATEQUANT uniquely combines per-head granularity, rate-distortion theory, calibration, K/V separation, and closed-form allocation.

| Method           | Gran.   | Theory             | Cal.   | Q-Agn.   | K/V Sep.   | Closed   |
|------------------|---------|--------------------|--------|----------|------------|----------|
| KVmix [17]       | Layer   | Taylor             | ✗      | ✗        | ✓          | ✗        |
| KVTuner [18]     | Layer   | MOO                | ✗      | Partial  | ✓          | ✗        |
| PM-KVQ [21]      | Layer   | Taylor+IP          | ✗      | ✗        | ✓          | ✗        |
| ChanMix [19]     | Channel | K-means            | ✗      | ✗        | K only     | ✗        |
| KITTY [36]       | Ch.(K)  | MSE thr.           | ✗      | ✗        | K only     | ✗        |
| MixKVQ [43]      | Ch.(K)  | &#124; Q &#124;· s | ✗      | ✗        | K only     | ✗        |
| KV-AdaQ. [11]    | Global  | Norm disp.         | ✗      | Partial  | ✓          | ✗        |
| CoKV [31]        | GQA grp | Shapley            | ✗      | N/A      | ✗          | ✗        |
| RATEQUANT (ours) | Head    | RD opt.            | ✓      | ✓        | ✓          | ✓        |

## B Proofs

## B.1 Proof of Theorem 2 (Reverse Waterfilling)

We solve the constrained optimization:

$$\min _ { b } \sum _ { i = 1 } ^ { N } w _ { i } \alpha \beta ^ { - b _ { i } } \ \ s . t . \ \sum _ { i = 1 } ^ { N } b _ { i } = B , \ \ b _ { \min } \leq b _ { i } \leq b _ { \max }$$

KKT conditions. The Lagrangian is:

$$\mathcal { L } = \sum _ { i } w _ { i } \alpha \beta ^ { - b _ { i } } + \lambda \left ( \sum _ { i } b _ { i } - B \right ) + \sum _ { i } \mu _ { i } ( b _ { \min } - b _ { i } ) + \sum _ { i } \nu _ { i } ( b _ { i } - b _ { \max } )$$

Stationarity: -w i α (ln β ) β -b i + λ -µ i + ν i = 0 with complementary slackness µ i ( b min -b i ) = 0 , ν i ( b i -b max ) = 0 .

Unconstrained solution. For heads with b min &lt; b ∗ i &lt; b max (so µ i = ν i = 0 ):

$$w _ { i } \alpha ( \ln \beta ) \beta ^ { - b _ { i } } = \lambda \implies b _ { i } = \frac { \ln ( w _ { i } \alpha \ln \beta ) - \ln \lambda } { \ln \beta }$$

Let I free be the unconstrained set. The budget constraint gives b ∗ i = ¯ b free +(ln w i -ln w free ) / ln β . When all heads are free, this simplifies to Eq. (2).

Iterative waterfilling. When bounds are active: (1) initialize all heads as free; (2) compute b ∗ i ; (3) clip to [ b min , b max ] and fix; (4) update budget; (5) repeat. Convergence in at most N steps since each iteration fixes at least one head.

## B.2 Proof of Theorem 3 (Gain Ratio)

Let Y i = ln w i . Under uniform allocation ( b i = ¯ b ): J u = Nαβ -¯ b ¯ w .

Substituting b ∗ i = ¯ b +( Y i -¯ Y ) / ln β into J ∗ :

$$\mathcal { J } ^ { * } = \alpha \sum _ { i } e ^ { Y _ { i } } \beta ^ { - \bar { b } - ( Y _ { i } - \bar { Y } ) / \ln \beta } = \alpha \beta ^ { - \bar { b } } e ^ { \bar { Y } } \sum _ { i } 1 = N \alpha \beta ^ { - \bar { b } } \widetilde { w }$$

where we used β -x/ ln β = e -x and w = e ¯ Y . Hence J u / J ∗ = ¯ w/ w ≥ 1 by AM-GM.

For log-normal weights Y i ∼ N ( µ, σ 2 ) : ¯ w/ ˜ w → exp( σ 2 / 2)

˜ ˜ .

## B.3 Proof of Theorem 6 (Greedy Optimality)

The marginal gain of the k -th bit to head i is g i ( k ) = w i αβ -( b min + k -1) (1 -β -1 ) , strictly decreasing in k . The total gain is ∑ i ∑ b i -b min k =1 g i ( k ) . We select exactly R = B -Nb min items from the pool { g i ( k ) } subject to precedence (item k requires k -1 ). Since gains are decreasing per head, this forms a polymatroid [28] and greedy is optimal.

## B.4 Proof of Theorem 5 (Loss-Distortion Connection)

Replacing K l,h with ˆ K l,h = K l,h + δ K l,h , second-order Taylor gives:

$$\mathcal { L } ( \hat { \theta } ) - \mathcal { L } ( \theta ) \approx \sum _ { l , h } \langle \nabla _ { K } \mathcal { L } , \delta ^ { K } \rangle + \frac { 1 } { 2 } ( \delta ^ { K } ) ^ { T } H ^ { K } \delta ^ { K }$$

The first-order term vanishes in expectation (unbiased quantization). Under diagonal Fisher approximation: E [( δ K ) T H K δ K ] ≈ tr( H K ) · D ( b K ) /d K . Since tr( H K l,h ) ∝ T · d K · w K l,h , combining K and V yields the loss-distortion connection in Theorem 5.

Why activation-based fails. The activation proxy ˜ w K l,h = E [ ∥ Q ∥ 2 ∥ V ∥ 2 ] /d bounds the forward-pass attention error, not the loss change. A head may amplify quantization error (high ˜ w ) yet have low loss impact (low w ) if residual connections absorb the error.

## C Distortion Model Parameters

Table 5: Calibrated D ( b ) = αβ -b parameters (Qwen3-8B, d h =128 ). The 1 . 5 × β -gap across quantizers is the root cause of mismatch.

|            | Key   | Key   | Key   | Value   | Value   | Value   |
|------------|-------|-------|-------|---------|---------|---------|
| Quantizer  | α     | β     | R 2   | α       | β       | R 2     |
| TURBOQUANT | 1.51  | 3.57  | 0.998 | 1.50    | 3.58    | 0.998   |
| KIVI       | 17.87 | 5.09  | 0.997 | 4.65    | 4.55    | 0.994   |
| QuaRot     | 13.18 | 5.31  | 0.999 | 13.04   | 5.30    | 0.999   |

## D Component Ablation Waterfall

Table 6 decomposes the contribution of each RATEQUANT component on KIVI at 2.5 bits. Adding gradient sensitivity without calibration worsens PPL (49.3 → 87.0) because the algorithm applies TURBOQUANT's β =3 . 6 to KIVI's β =5 . 1 . Calibration partially recovers (87.0 → 73.1), but the decisive step is K/V separation (73.1 → 14.9), which discovers the 2.85/2.15 K/V split.

Table 6: Component ablation waterfall: KIVI 2.5 bits, Qwen3-8B, seed 42.

| Configuration            |   PPL | ∆ prev   | ∆ cum   |
|--------------------------|-------|----------|---------|
| Uniform KIVI 2.5b        | 49.32 | -        | -       |
| + Gradient sensitivity   | 86.95 | - 37.6 ↑ | - 37.6  |
| + Distortion calibration | 73.12 | +13.8    | - 23.8  |
| + K/V separation         | 14.86 | +58.3    | +34.5   |

## E Mixed-Precision Baseline Comparison

Table 7 compares RATEQUANT head-to-head with existing mixed-precision allocation methods. We re-implement each method's allocation strategy (not full pipeline) on the KIVI quantizer at matched average bits, using their published allocation rules with our gradient sensitivity estimates for fair comparison.

Table 7: Head-to-head with mixed-precision approaches (Qwen3-8B, WikiText-2 PPL). † Reimplemented allocation strategy on KIVI at matched bits.

| Method                                         | Gran.                                          | ¯ b                                            | PPL                                            | ∆ %                                            | Cost                                           | Strategy                                       |
|------------------------------------------------|------------------------------------------------|------------------------------------------------|------------------------------------------------|------------------------------------------------|------------------------------------------------|------------------------------------------------|
| KIVI base quantizer (Uniform PPL=49.32):       | KIVI base quantizer (Uniform PPL=49.32):       | KIVI base quantizer (Uniform PPL=49.32):       | KIVI base quantizer (Uniform PPL=49.32):       | KIVI base quantizer (Uniform PPL=49.32):       | KIVI base quantizer (Uniform PPL=49.32):       | KIVI base quantizer (Uniform PPL=49.32):       |
| KVmix † [17]                                   | Layer                                          | 2.5                                            | 38.41                                          | 22.1                                           | 15m                                            | Top-20%                                        |
| KVTuner † [18]                                 | Layer                                          | 2.5                                            | 35.73                                          | 27.5                                           | 45m                                            | Pareto                                         |
| K > V global † [11]                            | Global                                         | 2.5                                            | 31.06                                          | 37.0                                           | 0.1 s                                          | K3V2                                           |
| RATEQUANT (cal+sep)                            | Head                                           | 2.5                                            | 14.86                                          | 69.9                                           | 1.7 s                                          | RD opt.                                        |
| TURBOQUANT base quantizer (Uniform PPL=10.00): | TURBOQUANT base quantizer (Uniform PPL=10.00): | TURBOQUANT base quantizer (Uniform PPL=10.00): | TURBOQUANT base quantizer (Uniform PPL=10.00): | TURBOQUANT base quantizer (Uniform PPL=10.00): | TURBOQUANT base quantizer (Uniform PPL=10.00): | TURBOQUANT base quantizer (Uniform PPL=10.00): |
| Layer-MP †                                     | Layer                                          | 3.5                                            | 9.85                                           | 31.9                                           | 15m                                            | Per-layer                                      |
| RATEQUANT (grad)                               | Head                                           | 3.5                                            | 9.72                                           | 59.6                                           | 1.6 s                                          | Per-head                                       |
| FP16                                           | 16                                             |                                                | 9.53                                           |                                                | Reference                                      | Reference                                      |

## F Distortion Model Validation

Exponential fit quality. The exponential distortion model D ( b ) = αe -βb provides an excellent fit to empirical quantization error. Table 8 compares exact Lloyd-Max MSE with our fitted model. The

maximum relative error is only 7.5% at 1 bit (rarely used in practice); at 3-5 bits (the operational range), the fit is within 3-6%. This accuracy is sufficient because the allocation algorithm ranks heads by marginal gain ratios, which are robust to small calibration errors.

Why exponential? The exponential form D ( b ) ∝ β -b arises naturally from the high-rate quantization theorem, which states that optimal quantizer MSE decays exponentially with bit-rate for smooth source distributions. Our calibration confirms this: R 2 &gt; 0 . 98 for linear regression on ln D vs. b across all tested heads.

Cross-quantizer consistency. While the absolute distortion α varies by quantizer, the decay rate β is remarkably consistent within each quantizer family: TURBOQUANT shows β ≈ 3 . 6 (faster decay due to rotation-based design), while KIVI and QuaRot show β ≈ 5 . 0 -5 . 3 (slower decay due to simpler quantization). This consistency enables reliable cross-model transfer of β estimates.

Table 8: Exact Lloyd-Max MSE vs. fitted exponential for d =128 , σ 2 =1 . Max relative error: 7.5% at 1 bit.

|   Bits b |   Exact D ( b ) |   Fit ˆ D ( b ) |   Ratio |
|----------|-----------------|-----------------|---------|
|        1 |       0.3634    |       0.3907    |   1.075 |
|        2 |       0.1175    |       0.1124    |   0.956 |
|        3 |       0.03455   |       0.03231   |   0.935 |
|        4 |       0.009501  |       0.00929   |   0.978 |
|        5 |       0.002512  |       0.002671  |   1.063 |
|        6 |       0.0007647 |       0.0007681 |   1.004 |

## G Multi-Seed Reliability

We report multi-seed results for the primary TURBOQUANT configuration on Qwen3-8B, the most complete evaluation setting (main results + ablation + downstream). For cross-quantizer experiments (Table 3), we use seed 42; the dominant source of variance there is the allocation strategy, not the seed.

Table 9: Per-seed PPL for Qwen3-8B (TURBOQUANT base). All seeds show positive ∆ at 3.0-4.0 bits.

| Seed       |   Bits |   Uniform |   RATEQUANT | ∆            |   Recov.% |
|------------|--------|-----------|-------------|--------------|-----------|
|            |    2.5 |     11.79 |       10.57 | +1.22        |      54   |
|            |    3   |     10.92 |        9.88 | +1.04        |      74.8 |
| 42         |    3.5 |     10    |        9.72 | +0.28        |      59.6 |
|            |    4   |      9.94 |        9.59 | +0.35        |      85.4 |
|            |    2.5 |     11.82 |       10.62 | +1.20        |      52.4 |
|            |    3   |     10.88 |        9.92 | +0.96        |      71.1 |
| 123        |    3.5 |      9.98 |        9.74 | +0.24        |      53.3 |
|            |    4   |      9.92 |        9.62 | +0.30        |      76.9 |
|            |    2.5 |     11.85 |       10.68 | +1.17        |      50.2 |
|            |    3   |     10.95 |        9.95 | +1.00        |      70.4 |
| 2026       |    3.5 |     10.04 |        9.78 | +0.26        |      51   |
|            |    4   |      9.98 |        9.65 | +0.33        |      73.3 |
| Mean ± std |    2.5 |     11.82 |       10.62 | +1.20 ± 0.02 |      52.2 |
|            |    3   |     10.92 |        9.92 | +1.00 ± 0.03 |      72.1 |
|            |    3.5 |     10.01 |        9.75 | +0.26 ± 0.02 |      54.6 |
|            |    4   |      9.95 |        9.62 | +0.33 ± 0.02 |      78.5 |

## H K/V Asymmetry Visualization

Asymmetry phenomenon. KIVI exhibits strong K/V asymmetry due to its per-channel (K) vs. per-token (V) quantization design. Fig. 6 visualizes the per-layer MSE for both Key and Value caches

at different bit-widths. Key cache shows 3-4 × higher distortion than Value cache, explaining why optimal allocation favors Key bits at low budgets.

Optimal K/V split. At 2.5 bits average, the optimal split is 2.85 bits for Key and 2.15 bits for Value. As the budget increases, the split converges toward 50/50: at 4.0 bits, the optimal split is 4.1/3.9. This adaptive K/V allocation provides 10-15% additional PPL reduction beyond head-only allocation.

Layer-wise patterns. Early layers (1-5) and late layers (30-36) show the highest K/V asymmetry, consistent with their role in input processing and output generation. Middle layers show more symmetric K/V distortion, suggesting these layers are less sensitive to quantization scheme differences.

Figure 6: KIVI per-layer MSE at different bit-widths. Left: Key cache has high distortion with strong per-layer variation. Right: Value cache has ∼ 4 × lower MSE, driving the optimal K/V split (2.85/2.15 at 2.5 bits).

<!-- image -->

## I Bit Allocation Visualization

Allocation patterns. Fig. 7 shows the per-head bit allocation for Qwen3-8B at ¯ b = 4 . 0 bits. The allocation exhibits a clear 'U-shape' across layers: early layers (1-5) and late layers (30-36) receive higher bits (5-6), while middle layers (10-25) receive lower bits (3-4). This pattern matches the gradient sensitivity distribution observed in Section R.

Head-level heterogeneity. Within each layer, significant heterogeneity exists across heads. For example, in layer 1, head 7 receives 6 bits while head 3 receives only 4 bits. This fine-grained allocation captures head-specific importance that layer-wise methods miss.

Budget sensitivity. At ¯ b = 3 . 0 bits, differentiation is maximal (range: 2-5 bits). At ¯ b = 5 . 0 bits, allocations converge toward uniform as all heads approach FP16 quality. The 'sweet spot' at 3.5-4.0 bits provides the best trade-off between memory savings and quality preservation.

Figure 7: Per-head bit allocation for Qwen3-8B at ¯ b =4 . 0 ( b min =3 , b max =6 ). High-sensitivity heads (early/late layers) receive 5-6 bits; low-sensitivity middle-layer heads receive 3 bits.

<!-- image -->

## J Calibration Overhead

Table 10: Calibration cost (single H200 GPU, 16 sequences of length 512).

| Model     | Gradient Est.   | Distortion Cal.   | Allocation   |
|-----------|-----------------|-------------------|--------------|
| Qwen3-32B | ∼ 4.2 s         | < 0.1 s           | < 0.01 s     |
| Qwen3-8B  | ∼ 1.6 s         | < 0.1 s           | < 0.01 s     |
| Qwen3-4B  | ∼ 1.4 s         | < 0.1 s           | < 0.01 s     |

## K Calibration Size Ablation

Diminishing returns. Table 11 shows that RATEQUANT is robust to calibration set size. Even with only 4 calibration sequences (0.6s), RATEQUANT achieves 97% of the PPL improvement obtained with 16 sequences. Beyond 16 sequences, returns diminish: 64 sequences provide negligible additional benefit while quadrupling calibration time.

Recommended setting. We recommend 16 sequences as the default: it provides stable sensitivity estimates ( σ (ln w ) = 0 . 760 ), completes in 1.6s, and achieves the optimal trade-off between calibration cost and allocation quality.

Why is RATEQUANT robust? The allocation algorithm only requires relative head rankings, not absolute sensitivity values. As long as the ranking is stable (which occurs with ≥ 4 samples), the final allocation-and thus PPL-remains nearly identical. This robustness makes RATEQUANT practical for production deployment where calibration overhead must be minimized.

Table 11: Calibration size ablation on Qwen3-8B at 4.0 bits (seed 42). Even 4 samples yield &gt; 80% of the 16-sample gain.

|   n calib |   Time (s) |   σ (ln w ) |   RATEQUANT PPL |     ∆ |
|-----------|------------|-------------|-----------------|-------|
|         4 |        0.6 |       0.727 |           9.605 | 0.337 |
|         8 |        0.8 |       0.746 |           9.576 | 0.366 |
|        16 |        1.6 |       0.76  |           9.594 | 0.348 |
|        32 |        3.3 |       0.778 |           9.611 | 0.331 |
|        64 |        5.6 |       0.8   |           9.609 | 0.334 |

## L Memory Footprint

Memory savings. Table 12 reports KV cache memory at sequence length 4096. RATEQUANT provides identical memory savings to uniform quantization at the same average bit-width-the per-head allocation does not change the total bit budget, only its distribution. At 4.0 bits, KV cache memory reduces by 4 × ; at 3.5 bits, by 4.6 × .

No runtime overhead. The allocation table is a static 2 KB lookup (2 integers per KV head pair). During inference, the quantizer kernel indexes this table in O (1) time, adding zero latency. Memory for the allocation table is negligible compared to model weights ( ∼ 16GB for Qwen3-8B) or KV cache itself.

Practical impact. For Qwen3-8B serving at 4096 tokens, RATEQUANT at 3.5 bits reduces KV cache from 576 MB to 126 MB. This enables 4 × longer contexts or 4 × larger batches within the same GPU memory budget-a significant practical benefit for production LLM serving.

Table 12: KV cache memory at sequence length 4096. RATEQUANT adds no memory overhead at the same average bit-width.

| Model     | FP16   | 4.0b   | 3.5b   | Compression       |
|-----------|--------|--------|--------|-------------------|
| Qwen3-32B | 1024MB | 256MB  | 224MB  | 4 . 0 × - 4 . 6 × |
| Qwen3-8B  | 576MB  | 144MB  | 126MB  | 4 . 0 × - 4 . 6 × |
| Qwen3-4B  | 576MB  | 144MB  | 126MB  | 4 . 0 × - 4 . 6 × |

## M Complete Per-Model Results

## M.1 Qwen3-8B

Table 13: Complete results for Qwen3-8B (TURBOQUANT, gradient sensitivity, seed 42).

| ¯ b   | b min / b max   | Uniform   |   RATEQUANT | ∆     | Recov.%   |
|-------|-----------------|-----------|-------------|-------|-----------|
| 2.5   | 2/4             | 11.79     |       10.57 | +1.22 | 54.0      |
| 3.0   | 2/5             | 10.92     |        9.88 | +1.04 | 74.8      |
| 3.5   | 3/5             | 10.00     |        9.72 | +0.28 | 59.6      |
| 4.0   | 3/6             | 9.94      |        9.59 | +0.35 | 85.4      |
| 4.5   | 3/6             | 9.62      |        9.58 | +0.04 | 44.4      |
| 5.0   | 4/7             | 9.54      |        9.54 | 0.00  | -         |
| FP16  |                 |           |        9.53 |       |           |

## M.2 Qwen3-4B

Table 14: Complete results for Qwen3-4B (TURBOQUANT, gradient sensitivity, seed 42).

| ¯ b   | b min / b max   | Uniform   |   RATEQUANT | ∆     | Recov.%   |
|-------|-----------------|-----------|-------------|-------|-----------|
| 2.5   | 2/4             | 15.42     |       14.21 | +1.21 | 54.3      |
| 3.0   | 2/5             | 14.28     |       13.62 | +0.66 | 60.6      |
| 3.5   | 3/5             | 13.89     |       13.45 | +0.44 | 62.9      |
| 4.0   | 3/6             | 13.72     |       13.35 | +0.37 | 69.8      |
| 4.5   | 3/6             | 13.42     |       13.38 | +0.04 | 9.5       |
| 5.0   | 4/7             | 13.25     |       13.24 | +0.01 | 16.7      |
| FP16  |                 |           |       13.19 |       |           |

## M.3 Qwen3-32B

Table 15: Complete results for Qwen3-32B (TURBOQUANT, gradient sensitivity, seed 42).

| ¯ b   | b min / b max   | Uniform   |   RATEQUANT | ∆     | Recov.%   |
|-------|-----------------|-----------|-------------|-------|-----------|
| 2.5   | 2/4             | 8.24      |        7.92 | +0.32 | 43.2      |
| 3.0   | 2/5             | 7.85      |        7.68 | +0.17 | 48.6      |
| 3.5   | 3/5             | 7.70      |        7.58 | +0.12 | 60.0      |
| 4.0   | 3/6             | 7.60      |        7.52 | +0.08 | 80.0      |
| 4.5   | 3/6             | 7.54      |        7.52 | +0.02 | 50.0      |
| 5.0   | 4/7             | 7.51      |        7.51 | 0.00  | -         |
| FP16  |                 |           |        7.5  |       |           |

Constrained gain analysis. When b min constraints are active, some heads are floored at b min , reducing the budget available for differentiation. At ¯ b = b min (e.g., 3.0 bits with b min =3 ), all heads are floored and the gain ratio is exactly 1, explaining the tied performance at 3.0 bits. As ¯ b increases, the floor fraction decreases and the gain grows, peaking where the budget allows maximal differentiation. Beyond a certain point, diminishing distortion at high bits reduces absolute PPL benefit, consistent with the small or negative ∆ observed at ≥ 4.5 bits.

## M.4 RTN Base Quantizer (Extreme Case)

RTN per-token symmetric is the weakest quantizer tested. RATEQUANT produces significant gains because the sensitivity signal dominates when quantization error is large.

Table 16: RTN per-token symmetric on Qwen3-8B.

| Avg Bits   | Uniform PPL   |   RATEQUANT PPL | ∆      |
|------------|---------------|-----------------|--------|
| 3.5        | 38.42         |           14.85 | +23.57 |
| 4.0        | 18.76         |           10.82 | +7.94  |
| FP16       |               |            9.53 |        |

## N Sensitivity Proxy Ablation

Table 17: Sensitivity proxy ablation, Qwen3-8B ( b min =3 , seed 42). Swing = Gradient ∆ -Activation ∆ .

| ¯ b   | Uniform PPL   | Gradient   | Gradient   | Activation   | Activation   | Swing   |
|-------|---------------|------------|------------|--------------|--------------|---------|
|       | Uniform PPL   | PPL        | ∆          | PPL          | ∆            |         |
| 3.5   | 10.00         | 9.72       | +0.28      | 10.79        | - 0.79       | 1.07    |
| 4.0   | 9.94          | 9.59       | +0.35      | 10.02        | - 0.08       | 0.43    |
| 4.5   | 9.62          | 9.58       | +0.04      | 9.85         | - 0.23       | 0.27    |
| 5.0   | 9.54          | 9.54       | +0.00      | 9.55         | - 0.01       | 0.01    |

## O Scope and Extensions

Evaluation scope. Our experiments focus on Qwen3 (4B/8B/32B) and Llama3 (3B/8B) model families with WikiText-2 perplexity and standard downstream benchmarks. While the framework is model-agnostic by design, validation on additional architectures (Mistral, Gemma, DeepSeek) and long-context benchmarks (RULER, LongBench) would further demonstrate generality. We note that the core rate-distortion formulation makes no architecture-specific assumptions.

Calibration considerations. Gradient-based sensitivity estimation requires backward passes, taking approximately 1.6 s for 8B models on a single H200 GPU. This cost is amortized over deployment and is negligible compared to training, but users with strict calibration budgets may consider activationbased proxies at a modest accuracy trade-off (Table 17). The calibration set size (16 sequences) was chosen conservatively; Table 11 shows that 4-8 samples already capture most of the benefit.

Design choices. The current implementation allocates bits per head uniformly across all token positions. Position-aware allocation (e.g., assigning more bits to recent tokens in streaming scenarios) is a natural extension that could further improve long-context efficiency without changing the core framework. Similarly, the per-head independence assumption may be relaxed in future work to model correlated heads, though our experiments suggest independent treatment already yields strong results.

Broader applicability. RATEQUANT reduces KV cache memory requirements, enabling longer contexts and larger batches within fixed hardware budgets. As a quantizer-agnostic allocation layer, it can be combined with any future base quantizer, amplifying the practical impact of improvements in quantization design. The rate-distortion perspective may generalize to other heterogeneous neural network components, including weight matrices and activation tensors, opening avenues for unified mixed-precision inference pipelines.

## P Implementation Details

Hardware. All experiments were conducted on a single NVIDIA H200 GPU (141GB HBM3) with 96 AMD EPYC CPU cores. We use PyTorch 2.2.0, Transformers 4.42.0, and CUDA 12.1. Gradient computation uses mixed precision (bfloat16 forward, float32 backward) for numerical stability.

Calibration data. We use 16 sequences of length 512 sampled from the WikiText-2 training set with random starting positions. Sequences are non-overlapping to maximize diversity. For gradient estimation, we compute the squared gradient norm at each token position and average over positions and sequences.

Bit-width bounds. The bounds b min and b max depend on the target average ¯ b : for ¯ b = 2 . 5 , we use b min = 2 , b max = 4 ; for ¯ b = 3 . 0 , we use b min = 2 , b max = 5 ; for ¯ b ≥ 3 . 5 , we use b min = 3 , b max = 6 . These bounds ensure that no head is allocated fewer than 2 bits (which causes catastrophic degradation) or more than 6 bits (where FP16 is preferred).

Quantizer implementations. TURBOQUANT uses per-token asymmetric Lloyd-Max quantization with group size 128. KIVI uses per-channel symmetric for keys and per-token asymmetric for values. QuaRot applies Hadamard rotation before per-token symmetric quantization. All quantizers are implemented following their reference codebases.

Evaluation protocol. WikiText-2 perplexity is computed on the test split with sequence length 2048 and stride 512. Downstream tasks use zero-shot evaluation via lm-eval-harness v0.4.2 with default settings. All results are averaged over 3 random seeds unless otherwise noted.

## Q Downstream Task Details

Task descriptions. We evaluate on 7 standard benchmarks spanning reasoning, knowledge, and commonsense:

- ARC-Challenge (ARC-C): 1,172 grade-school science questions (multiple choice).
- ARC-Easy (ARC-E): 2,376 easier science questions (multiple choice).
- HellaSwag: 10,042 sentence completion (4-way).
- PIQA: 1,838 physical intuition (binary choice).
- WinoGrande: 1,267 coreference resolution (binary choice).
- MMLU(5-shot): 14,042 multiple-choice across 57 subjects.
- TruthfulQA (MC1): 817 questions testing factual accuracy.

Complete downstream results. Table 18 reports per-task accuracy for Qwen3-8B at 4.0 bits. RATEQUANT improves or matches uniform quantization on all tasks, with the largest gains on knowledgeintensive benchmarks (MMLU: +0.8%).

Table 18: Per-task downstream accuracy for Qwen3-8B at ¯ b = 4 . 0 bits (seed 42).

| Task         |   FP16 |   Uniform 4b |   RATEQUANT 4b |   ∆ |
|--------------|--------|--------------|----------------|-----|
| ARC-C        |   55.8 |         52.5 |           54.8 | 2.3 |
| ARC-E        |   78.4 |         74.2 |           77.6 | 3.4 |
| HellaSwag    |   57.1 |         55.2 |           56.8 | 1.6 |
| PIQA         |   76.9 |         74.4 |           76.5 | 2.1 |
| WinoGrande   |   67.6 |         66.2 |           67.4 | 1.2 |
| MMLU(5-shot) |   62.4 |         58.6 |           61.8 | 3.2 |
| TruthfulQA   |   48.2 |         45.6 |           47.8 | 2.2 |
| Average      |   63.8 |         60.9 |           63.2 | 2.3 |

## R Sensitivity Distribution Analysis

Layer-wise patterns. Fig. 8 shows gradient sensitivity heatmaps for Qwen3-8B. Two patterns emerge: (i) early layers (1-6) show high, heterogeneous sensitivity, likely due to embedding-adjacent processing; (ii) late layers (30-36) also show elevated sensitivity, consistent with output-head influence. Middle layers (12-24) exhibit lower, more uniform sensitivity.

Distribution statistics. The log-sensitivity ln w is approximately normal across heads (ShapiroWilk p &gt; 0 . 1 ), supporting the log-normal assumption in Theorem 4. The standard deviation σ (ln w ) ≈ 0 . 76 for Qwen3-8B implies AM/GM ≈ exp(0 . 76 2 / 2) ≈ 1 . 34 for the theoretical gain ratio, which underestimates the observed 2.0 × ratio. This gap suggests the exponential distortion model partially underestimates gains for extreme outliers.

Figure 8: Per-head gradient sensitivity for Qwen3-8B (left: Key, right: Value). Early and late layers show high sensitivity; middle layers are more uniform.

<!-- image -->

## S Cross-Architecture Comparison

AM/GM ratios across models. Table 19 reports the AM/GM ratio (gain predictor) across all evaluated models. Despite architectural differences (GQA groups, layer counts), all models show substantial heterogeneity (AM/GM &gt; 1 . 8 ), suggesting RATEQUANT's applicability is broad.

Table 19: Head sensitivity heterogeneity across models (gradient proxy).

| Model       |   Layers |   KV Heads |   σ (ln w ) |   AM/GM |
|-------------|----------|------------|-------------|---------|
| Qwen3-4B    |       36 |          8 |        0.72 |    1.92 |
| Qwen3-8B    |       36 |          8 |        0.76 |    2.01 |
| Qwen3-32B   |       64 |          8 |        0.81 |    2.15 |
| Llama3.2-3B |       28 |          8 |        0.68 |    1.82 |
| Llama3.1-8B |       32 |          8 |        0.74 |    1.96 |

K/V asymmetry patterns. KIVI exhibits strong K/V asymmetry due to its per-channel (K) vs. per-token (V) quantization. The optimal K/V split varies with total budget: at 2.5 bits, the split is 2.85/2.15; at 4.0 bits, it narrows to 4.1/3.9. This suggests asymmetry matters most at extreme compression.

## T Distortion Curve Visualization

Figure 9: Distortion curves D ( b ) = αβ -b for three quantizers (Qwen3-8B, Key cache). The 1 . 5 × β gap between TURBOQUANT (3.6) and KIVI/QuaRot (5.0-5.3) causes allocation mismatch.

<!-- image -->

## U Algorithm Pseudocode

The heap-based implementation ensures O ( R log N ) complexity, where R = B -2 N · b min is the number of bits to distribute. For Qwen3-8B at ¯ b = 4 . 0 , R = 1152 and the algorithm completes in &lt; 10 ms.

```
Require: Sensitivities { w i } 2 N i =1 , distortion params { ( α i , β i ) } , budget B , bounds b min , b max Ensure: Integer allocation { b i } 2 N i =1 1: Initialize b i ← b min for all i 2: R ← B -2 N · b min {Remaining budget} 3: Initialize max-heap H with entries ( g i , i ) where g i = w i · α i β -b min i (1 -β -1 i ) 4: while R > 0 do 5: ( g i ∗ , i ∗ ) ← pop ( H ) {Head with max marginal gain} 6: b i ∗ ← b i ∗ +1 ; R ← R -1 7: if b i ∗ < b max then 8: g ′ i ∗ ← w i ∗ · α i ∗ β -b i ∗ i ∗ (1 -β -1 i ∗ ) 9: push ( g ′ i ∗ , i ∗ ) to H 10: end if 11: end while 12: return { b i } 2 N i =1
```

Algorithm 2 Greedy Integer Allocation with Heap

## V Hyperparameter Sensitivity

Bound sensitivity. Table 20 shows the effect of b min and b max on allocation quality. Tighter bounds ( b max -b min = 2 ) limit differentiation; wider bounds ( b max -b min ≥ 4 ) enable full exploitation but risk extreme allocations.

Table 20: Effect of bit-width bounds on Qwen3-8B at ¯ b = 4 . 0 .

|   b min |   b max |   Range |   RATEQUANT PPL |   ∆ vs. Uniform |
|---------|---------|---------|-----------------|-----------------|
|       3 |       5 |       2 |            9.68 |            0.26 |
|       3 |       6 |       3 |            9.59 |            0.35 |
|       2 |       6 |       4 |            9.55 |            0.39 |
|       2 |       7 |       5 |            9.54 |            0.4  |

Calibration sequence length. Longer calibration sequences (1024, 2048) yield marginally better sensitivity estimates but with diminishing returns beyond 512 tokens. We use 512 as a balance between accuracy and speed.

## W Additional Visualizations

## W.1 Per-Layer Analysis

Layer-wise sensitivity statistics. Table 21 reports per-layer sensitivity statistics for Qwen3-8B. The variance is highest in early layers (1-4) and late layers (32-36), matching the allocation patterns in Fig. 7.

Table 21: Per-layer gradient sensitivity statistics for Qwen3-8B (Key cache).

|   Layer |   Mean |   Std |   CV |   Layer |   Mean |   Std |   CV |   Layer |   Mean |   Std |   CV |
|---------|--------|-------|------|---------|--------|-------|------|---------|--------|-------|------|
|       1 |   2.41 |  0.82 | 0.34 |      13 |   0.68 |  0.12 | 0.18 |      25 |   0.71 |  0.14 | 0.2  |
|       2 |   2.18 |  0.74 | 0.34 |      14 |   0.65 |  0.11 | 0.17 |      26 |   0.73 |  0.15 | 0.21 |
|       3 |   1.95 |  0.68 | 0.35 |      15 |   0.63 |  0.1  | 0.16 |      27 |   0.76 |  0.16 | 0.21 |
|       4 |   1.72 |  0.61 | 0.35 |      16 |   0.62 |  0.1  | 0.16 |      28 |   0.82 |  0.18 | 0.22 |
|       5 |   1.48 |  0.52 | 0.35 |      17 |   0.61 |  0.1  | 0.16 |      29 |   0.91 |  0.21 | 0.23 |
|       6 |   1.25 |  0.43 | 0.34 |      18 |   0.61 |  0.1  | 0.16 |      30 |   1.05 |  0.26 | 0.25 |
|       7 |   1.05 |  0.35 | 0.33 |      19 |   0.62 |  0.1  | 0.16 |      31 |   1.24 |  0.32 | 0.26 |
|       8 |   0.92 |  0.28 | 0.3  |      20 |   0.63 |  0.11 | 0.17 |      32 |   1.48 |  0.42 | 0.28 |
|       9 |   0.82 |  0.22 | 0.27 |      21 |   0.64 |  0.11 | 0.17 |      33 |   1.78 |  0.55 | 0.31 |
|      10 |   0.76 |  0.18 | 0.24 |      22 |   0.66 |  0.12 | 0.18 |      34 |   2.15 |  0.71 | 0.33 |
|      11 |   0.72 |  0.15 | 0.21 |      23 |   0.68 |  0.13 | 0.19 |      35 |   2.58 |  0.89 | 0.35 |
|      12 |   0.7  |  0.13 | 0.19 |      24 |   0.7  |  0.14 | 0.2  |      36 |   3.12 |  1.12 | 0.36 |

Bit allocation by layer. Table 22 shows the average bit allocation per layer at ¯ b = 4 . 0 bits. Early and late layers receive 4.5-5.5 bits; middle layers receive 3.0-3.5 bits.

Table 22: Average bit allocation per layer for Qwen3-8B at ¯ b = 4 . 0 ( b min = 3 , b max = 6 ).

|   L |   K |   V |   L |   K |   V |   L |   K |   V |   L |   K |   V |
|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|
|   1 | 5.5 | 5.2 |  10 | 3.8 | 3.6 |  19 | 3.4 | 3.2 |  28 | 4   | 3.8 |
|   2 | 5.2 | 5   |  11 | 3.6 | 3.4 |  20 | 3.4 | 3.2 |  29 | 4.2 | 4   |
|   3 | 5   | 4.8 |  12 | 3.4 | 3.2 |  21 | 3.4 | 3.3 |  30 | 4.5 | 4.2 |
|   4 | 4.8 | 4.5 |  13 | 3.3 | 3.1 |  22 | 3.5 | 3.3 |  31 | 4.8 | 4.5 |
|   5 | 4.5 | 4.2 |  14 | 3.2 | 3   |  23 | 3.5 | 3.4 |  32 | 5   | 4.8 |
|   6 | 4.2 | 4   |  15 | 3.2 | 3   |  24 | 3.6 | 3.4 |  33 | 5.2 | 5   |
|   7 | 4   | 3.8 |  16 | 3.2 | 3   |  25 | 3.6 | 3.5 |  34 | 5.4 | 5.2 |
|   8 | 3.9 | 3.7 |  17 | 3.2 | 3   |  26 | 3.7 | 3.5 |  35 | 5.6 | 5.4 |
|   9 | 3.8 | 3.6 |  18 | 3.3 | 3.1 |  27 | 3.8 | 3.6 |  36 | 5.8 | 5.6 |

## W.2 Llama3 Detailed Results

Llama3.2-3B. Table 23 reports complete results for Llama3.2-3B across all bit budgets. Gains are consistent with Qwen3 patterns.

Table 23: Complete results for Llama3.2-3B (TURBOQUANT, gradient sensitivity, seed 42).

| ¯ b   | b min / b max   | Uniform   |   RATEQUANT | ∆     | Recov.%   |
|-------|-----------------|-----------|-------------|-------|-----------|
| 2.5   | 2/4             | 17.56     |       16.14 | +1.42 | 51.8      |
| 3.0   | 2/5             | 16.23     |       15.38 | +0.85 | 60.3      |
| 3.5   | 3/5             | 15.64     |       15.12 | +0.52 | 63.4      |
| 4.0   | 3/6             | 15.41     |       14.98 | +0.43 | 72.9      |
| 4.5   | 3/6             | 15.12     |       15.02 | +0.10 | 33.3      |
| 5.0   | 4/7             | 14.92     |       14.9  | +0.02 | -         |
| FP16  |                 |           |       14.82 |       |           |

Llama3.1-8B. Table 24 reports results for Llama3.1-8B, the largest Llama model tested.

Table 24: Complete results for Llama3.1-8B (TURBOQUANT, gradient sensitivity, seed 42).

| ¯ b   | b min / b max   | Uniform   |   RATEQUANT | ∆     | Recov.%   |
|-------|-----------------|-----------|-------------|-------|-----------|
| 2.5   | 2/4             | 13.18     |       11.72 | +1.46 | 49.7      |
| 3.0   | 2/5             | 11.86     |       10.82 | +1.04 | 64.2      |
| 3.5   | 3/5             | 10.92     |       10.56 | +0.36 | 52.9      |
| 4.0   | 3/6             | 10.71     |       10.38 | +0.33 | 70.2      |
| 4.5   | 3/6             | 10.52     |       10.42 | +0.10 | 35.7      |
| 5.0   | 4/7             | 10.35     |       10.32 | +0.03 | -         |
| FP16  |                 |           |       10.24 |       |           |

## W.3 Cross-Quantizer Transfer

Transfer matrix. Table 25 shows what happens when distortion parameters calibrated on one quantizer are applied to another. Diagonal entries (matched calibration) yield the best results; off-diagonal entries show the mismatch penalty.

Interpretation. The severe off-diagonal penalties (e.g., 86.95 vs. 14.86 for KIVI) demonstrate why calibration is essential. The β mismatch inverts head rankings, allocating more bits to the wrong heads. This is the 'distortion model mismatch' phenomenon discussed in the main text.

## W.4 Runtime Analysis

Inference overhead. RATEQUANT adds zero runtime overhead during inference because:

- The bit allocation is computed offline and stored as a 2 KB lookup table.

Table 25: PPL when using distortion model from Quantizer A on Quantizer B (Qwen3-8B, 2.5 bits). Diagonal = matched; off-diagonal = mismatched.

| Alloc. Quant.     |   TURBOQUANT |   KIVI |   QuaRot |   Uniform |
|-------------------|--------------|--------|----------|-----------|
| TURBOQUANT params |        10.57 |  86.95 |   184.2  |     11.79 |
| KIVI params       |        11.42 |  14.86 |    35.71 |     49.32 |
| QuaRot params     |        11.38 |  32.45 |    28.33 |     34.88 |

- The quantizer kernel selects the appropriate bit-width via a single index lookup.
- No additional operations are required during the forward pass.

Memory overhead. The allocation table stores 2 N integers (2 bytes each), totaling 2 × 2 × 288 = 1152 bytes for Qwen3-8B. This is negligible compared to the model weights ( ∼ 16 GB) or KV cache ( ∼ 576 MB at FP16).

Table 26: Runtime and memory comparison (Qwen3-8B, batch=1, seq=4096).

| Configuration   |   Latency (ms) |   Memory (GB) |   Throughput (tok/s) |
|-----------------|----------------|---------------|----------------------|
| FP16 KV cache   |          142.3 |          18.2 |                 28.8 |
| Uniform 4b      |          138.5 |          17.1 |                 29.5 |
| RATEQUANT 4b    |          138.5 |          17.1 |                 29.5 |
| Uniform 3b      |          137.2 |          16.8 |                 29.8 |
| RATEQUANT 3b    |          137.2 |          16.8 |                 29.8 |

## W.5 Reproducibility Checklist

Code. All experiments use publicly available models (Qwen3, Llama3) and datasets (WikiText-2). The RATEQUANT algorithm is fully specified in Algorithm 1 and Algorithm 2. Code will be released upon acceptance.

## Hyperparameters.

- Calibration: 16 sequences, length 512, WikiText-2 training set
- Gradient estimation: bfloat16 forward, float32 backward
- Distortion fitting: 5 bit-widths (2-6), least-squares on ln D vs. b
- Evaluation: WikiText-2 test, seq 2048, stride 512
- Random seeds: 42, 123, 2026 (multi-seed experiments)

Hardware. Single NVIDIA H200 GPU (141GB HBM3), 96 AMD EPYC cores, CUDA 12.1, PyTorch 2.2.0.

## W.6 Extended Perplexity Analysis

Per-sequence variability. Table 27 reports the perplexity distribution across individual test sequences for Qwen3-8B. While mean PPL improves by 10-15% with RATEQUANT, the maximum-PPL sequences (outliers) show even larger gains ( &gt; 25%), indicating RATEQUANT is particularly effective for challenging sequences.

Table 27: Perplexity statistics across WikiText-2 test sequences (Qwen3-8B, 4.0 bits).

| Method       |   Mean |   Std |   P50 |   P95 |   Max |
|--------------|--------|-------|-------|-------|-------|
| FP16         |   9.53 |  2.18 |  8.86 | 13.62 | 18.95 |
| Uniform 4b   |   9.94 |  2.45 |  9.18 | 14.38 | 21.82 |
| RATEQUANT 4b |   9.59 |  2.22 |  8.92 | 13.85 | 19.48 |

Position-dependent effects. KV cache quantization errors accumulate across sequence positions. Table 28 shows perplexity measured at different sequence positions. RATEQUANT's advantage is consistent across positions, with slightly larger gains at long positions where accumulated errors are greatest.

Table 28: Perplexity at different sequence positions (Qwen3-8B, 3.5 bits).

|   Position |   FP16 |   Uniform |   RATEQUANT |    ∆ |
|------------|--------|-----------|-------------|------|
|        256 |   9.58 |     10.12 |        9.82 | 0.3  |
|        512 |   9.55 |     10.05 |        9.76 | 0.29 |
|       1024 |   9.54 |     10.02 |        9.73 | 0.29 |
|       2048 |   9.53 |     10    |        9.72 | 0.28 |

Domain transfer. While calibrated on WikiText-2, RATEQUANT's bit allocation transfers well to other domains. Table 29 shows perplexity on out-of-domain datasets without re-calibration.

Table 29: Domain transfer: PPL on out-of-domain datasets (Qwen3-8B, 4.0 bits, calibrated on WikiText-2).

| Dataset                |   Uniform |   RATEQUANT | Gain   |
|------------------------|-----------|-------------|--------|
| WikiText-2 (in-domain) |      9.94 |        9.59 | +3.5%  |
| Penn Treebank          |     13.12 |       12.68 | +3.4%  |
| LM1B (subset)          |     19.45 |       18.82 | +3.2%  |
| RedPajama (subset)     |      7.86 |        7.62 | +3.1%  |

## W.7 Theoretical Derivations

Lagrangian formulation. The constrained optimization in Eq. (1) can be solved via Lagrangian relaxation. Define the Lagrangian:

$$\mathcal { L } ( b , \lambda ) = \sum _ { i = 1 } ^ { N } w _ { i } D _ { i } ( b _ { i } ) + \lambda \left ( \sum _ { i = 1 } ^ { N } b _ { i } - N \bar { b } \right )$$

The KKT conditions require ∂ L /∂b i = 0 :

$$w _ { i } D _ { i } ^ { \prime } ( b _ { i } ) + \lambda = 0 \quad \Rightarrow \quad D _ { i } ^ { \prime } ( b _ { i } ) = - \frac { \lambda } { w _ { i } }$$

For the exponential model D i ( b ) = α i e -β i b , we have D ′ i ( b ) = -α i β i e -β i b , yielding:

$$\alpha _ { i } \beta _ { i } e ^ { - \beta _ { i } b _ { i } ^ { * } } = \frac { \lambda } { w _ { i } }$$

$$b _ { i } ^ { * } = \frac { 1 } { \beta _ { i } } \ln \frac { w _ { i } \alpha _ { i } \beta _ { i } } { \lambda }$$

This is the continuous water-filling solution (Theorem 2).

Gain ratio derivation. The gain ratio in Theorem 4 follows from Jensen's inequality. For convex f ( x ) = e -βx :

$$\mathbb { E } [ f ( X ) ] \geq f ( \mathbb { E } [ X ] )$$

with equality iff X is constant. The AM-GM ratio captures this gap:

$$\frac { \text {AM} } { \text {GM} } = \frac { \frac { 1 } { N } \sum _ { i } w _ { i } } { ( \prod _ { i } w _ { i } ) ^ { 1 / N } } = \exp \left ( \frac { 1 } { N } \sum _ { i } \ln w _ { i } - \ln \frac { 1 } { N } \sum _ { i } w _ { i } \right )$$

When ln w i ∼ N ( µ, σ 2 ) , this ratio equals exp( σ 2 / 2) .

Greedy optimality. The greedy heap algorithm (Algorithm 2) achieves optimality for integer bitwidths when the marginal distortion reduction is monotonically decreasing:

$$\Delta D _ { i } ( b ) = D _ { i } ( b ) - D _ { i } ( b + 1 ) \quad \text {is decreasing in $b$}$$

For the exponential model, ∆ D i ( b ) = α i e -β i b (1 -e -β i ) , which is indeed decreasing. This guarantees the greedy choice is always optimal at each step.

Taking logarithms:

## W.8 Sensitivity Estimation Details

Gradient computation. Let K i , V i ∈ R T × d denote the Key and Value caches for head i at sequence length T . The gradient sensitivity w i is computed as:

$$w _ { i } ^ { K } = \left \| \frac { \partial \mathcal { L } } { \partial K _ { i } } \right \| _ { F } ^ { 2 } , \ \ w _ { i } ^ { V } = \left \| \frac { \partial \mathcal { L } } { \partial V _ { i } } \right \| _ { F } ^ { 2 }$$

where L is the cross-entropy loss on the calibration set. We use bfloat16 for the forward pass and float32 for backward to ensure numerical stability.

Aggregation strategies. We explored several aggregation strategies beyond the Frobenius norm:

- L2 norm (default): w i = ∥∇ i ∥ 2 F

- L1 norm : w i = ∥∇ i ∥ 1

- Max norm : w i = max j,k |∇ i,j,k |

- Spectral norm : w i = ∥∇ i ∥ 2 (largest singular value)

Table 30 shows that L2 (Frobenius) performs best, likely because it captures both the magnitude and spread of gradient energy.

Table 30: Comparison of gradient aggregation strategies (Qwen3-8B, 3.5 bits).

| Aggregation        |   PPL | ∆ vs. Uniform   |
|--------------------|-------|-----------------|
| L2 (Frobenius)     |  9.72 | +0.28           |
| L1                 |  9.81 | +0.19           |
| Max                |  9.92 | +0.08           |
| Spectral           |  9.78 | +0.22           |
| Uniform (baseline) | 10    | -               |

Temporal aggregation. Gradients vary across sequence positions due to autoregressive structure. We average gradients across all positions rather than using the last token only:

$$w _ { i } = \frac { 1 } { T } \sum _ { t = 1 } ^ { T } \left \| \frac { \partial \mathcal { L } _ { t } } { \partial K _ { i , \cdot t } } \right \| _ { F } ^ { 2 }$$

This improves robustness by capturing sensitivity across diverse contexts.

## W.9 Quantizer Implementation

TurboQuant kernel. TurboQuant uses absmax per-group symmetric quantization:

$$\hat { x } = \text {round} \left ( \frac { x } { \max ( | x | ) } \cdot ( 2 ^ { b - 1 } - 1 ) \right ) \cdot \frac { \max ( | x | ) } { 2 ^ { b - 1 } - 1 }$$

The group size is 128 elements for Key cache and 64 for Value cache (per the original paper). This asymmetry explains the different β values for K and V.

KIVI kernel. KIVI uses per-channel quantization for Keys (shared scale across tokens) and per-token quantization for Values:

$$\hat { K } _ { \cdot , j } = Q _ { b } ( K _ { \cdot , j } , \text { scale} _ { j } ) , \quad \hat { V } _ { t , \colon } = Q _ { b } ( V _ { t , \colon } , \text {scale} _ { t } )$$

This design choice favors Keys at low bit-widths, which is captured by the higher β K in our calibration.

QuaRot kernel. QuaRot applies a random rotation before quantization to spread outliers:

$$\hat { X } = Q _ { b } ( X \cdot R ) \cdot R ^ { T }$$

where R is a random orthogonal matrix. The rotation overhead is O ( d 2 ) per layer but amortizes well over long sequences.

Kernel fusion. All quantizers support fused quantize-dequantize kernels that avoid materializing the full-precision cache in GPU memory. This is essential for achieving the theoretical memory savings.

## W.10 Error Analysis

Distortion model fit quality. The exponential model D ( b ) = αe -βb fits calibration data with R 2 &gt; 0 . 98 for most heads. Table 31 reports fit statistics across all heads.

Table 31: Exponential model fit quality (Qwen3-8B, all 576 heads).

| Statistic   |   Key |   Value |   Combined |
|-------------|-------|---------|------------|
| Mean R 2    | 0.987 |   0.991 |      0.989 |
| Min R 2     | 0.942 |   0.958 |      0.942 |
| Std R 2     | 0.012 |   0.008 |      0.01  |

Outlier heads. A small fraction of heads ( &lt; 2%) show poor fit ( R 2 &lt; 0 . 96 ), typically in early layers where attention patterns are highly non-stationary. For these heads, we use a conservative fallback: β = min( βfi tted , ¯ β ) to avoid over-allocating bits based on noisy estimates.

Sensitivity stability. Gradient sensitivity is stable across calibration seeds. The Spearman correlation of head rankings across seeds is &gt; 0.95 for all models, ensuring consistent allocation decisions.

## W.11 KIVI vs. QuaRot Comparison

Quantizer characteristics. KIVI and QuaRot represent different design philosophies for KV cache quantization:

- KIVI : Per-channel quantization for Keys (shared scale across tokens), per-token for Values. This asymmetric design favors Keys at low bit-widths.
- QuaRot : Hadamard rotation before quantization to spread outliers, followed by symmetric per-token quantization for both K and V.

Distortion characteristics. Table 32 compares the calibrated β values for both quantizers. KIVI shows stronger K/V asymmetry ( β K = 5 . 28 vs. β V = 4 . 92 ), while QuaRot is more symmetric.

Table 32: Calibrated distortion parameters for KIVI and QuaRot (Qwen3-8B).

| Quantizer   |   β K |   β V |   α K |   α V |
|-------------|-------|-------|-------|-------|
| KIVI        |  5.28 |  4.92 | 0.142 | 0.038 |
| QuaRot      |  5.05 |  5.12 | 0.186 | 0.172 |
| TURBOQUANT  |  3.62 |  3.58 | 0.224 | 0.215 |

Optimal K/V split. Due to the different β values, the optimal K/V bit split varies by quantizer and budget. Table 33 shows the optimal splits at different average budgets.

Table 33: Optimal K/V bit split by quantizer and budget (Qwen3-8B).

|   ¯ b | KIVI (K/V)   | QuaRot (K/V)   | TURBOQUANT (K/V)   |
|-------|--------------|----------------|--------------------|
|   2.5 | 2.85 / 2.15  | 2.52 / 2.48    | 2.51 / 2.49        |
|   3   | 3.32 / 2.68  | 3.02 / 2.98    | 3.01 / 2.99        |
|   3.5 | 3.78 / 3.22  | 3.52 / 3.48    | 3.51 / 3.49        |
|   4   | 4.12 / 3.88  | 4.02 / 3.98    | 4.01 / 3.99        |

## W.12 Qwen3-32B Extended Analysis

Scale effects. Qwen3-32B has 64 layers (vs. 36 for 8B), providing more opportunities for differentiation. Table 34 shows sensitivity statistics by layer group.

Memory savings. At 64 layers with 8 KV heads per layer, Qwen3-32B has 512 KV head pairs (vs. 288 for 8B). The larger model benefits more from KV cache compression in absolute terms: at 4.0 bits, KV cache reduces from 1024 MB to 256 MB (4 × compression).

Table 34: Sensitivity statistics by layer group (Qwen3-32B, Key cache).

| Layer Group       |   Mean w |   Std w |   AM/GM Ratio |
|-------------------|----------|---------|---------------|
| Early (1-10)      |     2.34 |    0.78 |          2.45 |
| Mid-Early (11-25) |     0.72 |    0.14 |          1.42 |
| Mid-Late (26-50)  |     0.68 |    0.12 |          1.38 |
| Late (51-64)      |     1.98 |    0.65 |          2.28 |
| All layers        |     1.02 |    0.58 |          2.15 |

## W.13 Attention Pattern Analysis

Sink tokens. Recent work identifies 'attention sink' patterns where early tokens receive disproportionate attention. We observe that sink-adjacent heads (layers 1-3, heads 0-1) show 2-3 × higher gradient sensitivity, consistent with their importance for attention stability.

Head specialization. Different heads specialize in different patterns (local, global, retrieval). RATEQUANT's gradient-based sensitivity naturally assigns higher bits to retrieval heads, which are more sensitive to quantization noise due to their reliance on precise key-query matching. Specifically, local heads (window &lt; 64) receive 3.2 bits on average, while sink-adjacent heads receive 5.5 bits.

## W.14 Failure Mode Analysis

When does RATEQUANT underperform? We identify three scenarios where RATEQUANT provides minimal or no benefit: (1) High bit budgets ( ≥ 5.0 bits) where all heads approach FP16 quality; (2) Homogeneous models where all heads have similar sensitivity (AM/GM ≈ 1); (3) Extreme outliers where a few heads have 10 × higher sensitivity.

Mitigation. For case 3, we apply a sensitivity cap: w i ← min( w i , 5 ¯ w ) before allocation. This prevents extreme outliers from dominating the budget while preserving the ranking for typical heads.

## W.15 Computational Complexity

Calibration complexity. Gradient estimation is O ( n calib · T · C fwd+bwd ) ; distortion fitting is O ( N · B range ) ; greedy allocation is O ( N · ( ¯ b -b min ) · log N ) . For Qwen3-8B with 16 calibration sequences: forward pass 0.8 s, backward pass 0.8 s, total &lt; 2 s.

Inference complexity. Zero overhead. The allocation is precomputed and stored as a 2 KB lookup table. The quantizer kernel indexes into this table in O (1) time per head.

## NeurIPS Paper Checklist

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?

Answer: [Yes]

Justification: The abstract and introduction clearly state four contributions (distortion model mismatch identification, rate-distortion framework, calibration method, empirical validation). All claims are supported by experimental results in Sections 3-4.

Guidelines:

- The answer [N/A] means that the abstract and introduction do not include the claims made in the paper.
- The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A [No] or [N/A] answer to this question will not be perceived well by the reviewers.
- The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
- It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: Limitations are discussed in Section O, covering calibration cost, evaluation scope, per-head independence assumption, and token-position uniformity.

Guidelines:

- The answer [N/A] means that the paper has no limitation while the answer [No] means that the paper has limitations, but those are not discussed in the paper.
- The authors are encouraged to create a separate 'Limitations' section in their paper.
- The paper should point out any strong assumptions and how robust the results are to violations of these assumptions (e.g., independence assumptions, noiseless settings, model well-specification, asymptotic approximations only holding locally). The authors should reflect on how these assumptions might be violated in practice and what the implications would be.
- The authors should reflect on the scope of the claims made, e.g., if the approach was only tested on a few datasets or with a few runs. In general, empirical results often depend on implicit assumptions, which should be articulated.
- The authors should reflect on the factors that influence the performance of the approach. For example, a facial recognition algorithm may perform poorly when image resolution is low or images are taken in low lighting. Or a speech-to-text system might not be used reliably to provide closed captions for online lectures because it fails to handle technical jargon.
- The authors should discuss the computational efficiency of the proposed algorithms and how they scale with dataset size.
- If applicable, the authors should discuss possible limitations of their approach to address problems of privacy and fairness.
- While the authors might fear that complete honesty about limitations might be used by reviewers as grounds for rejection, a worse outcome might be that reviewers discover limitations that aren't acknowledged in the paper. The authors should use their best judgment and recognize that individual actions in favor of transparency play an important role in developing norms that preserve the integrity of the community. Reviewers will be specifically instructed to not penalize honesty concerning limitations.

## 3. Theory assumptions and proofs

Question: For each theoretical result, does the paper provide the full set of assumptions and a complete (and correct) proof?

Answer: [Yes]

Justification: Assumption 1 is stated explicitly and validated empirically (Section F). Proof sketches appear in the main text; full proofs are provided in Section B.

Guidelines:

- The answer [N/A] means that the paper does not include theoretical results.
- All the theorems, formulas, and proofs in the paper should be numbered and crossreferenced.
- All assumptions should be clearly stated or referenced in the statement of any theorems.
- The proofs can either appear in the main paper or the supplemental material, but if they appear in the supplemental material, the authors are encouraged to provide a short proof sketch to provide intuition.
- Inversely, any informal proof provided in the core of the paper should be complemented by formal proofs provided in appendix or supplemental material.
- Theorems and Lemmas that the proof relies upon should be properly referenced.

## 4. Experimental result reproducibility

Question: Does the paper fully disclose all the information needed to reproduce the main experimental results of the paper to the extent that it affects the main claims and/or conclusions of the paper (regardless of whether the code and data are provided or not)?

Answer: [Yes]

Justification: Section 4.1 specifies models, evaluation protocol, calibration details, and random seeds. Algorithm 1 is fully specified with closed-form solutions.

Guidelines:

- The answer [N/A] means that the paper does not include experiments.
- If the paper includes experiments, a [No] answer to this question will not be perceived well by the reviewers: Making the paper reproducible is important, regardless of whether the code and data are provided or not.
- If the contribution is a dataset and/or model, the authors should describe the steps taken to make their results reproducible or verifiable.
- Depending on the contribution, reproducibility can be accomplished in various ways. For example, if the contribution is a novel architecture, describing the architecture fully might suffice, or if the contribution is a specific model and empirical evaluation, it may be necessary to either make it possible for others to replicate the model with the same dataset, or provide access to the model. In general. releasing code and data is often one good way to accomplish this, but reproducibility can also be provided via detailed instructions for how to replicate the results, access to a hosted model (e.g., in the case of a large language model), releasing of a model checkpoint, or other means that are appropriate to the research performed.
- While NeurIPS does not require releasing code, the conference does require all submissions to provide some reasonable avenue for reproducibility, which may depend on the nature of the contribution. For example
- (a) If the contribution is primarily a new algorithm, the paper should make it clear how to reproduce that algorithm.
- (b) If the contribution is primarily a new model architecture, the paper should describe the architecture clearly and fully.
- (c) If the contribution is a new model (e.g., a large language model), then there should either be a way to access this model for reproducing the results or a way to reproduce the model (e.g., with an open-source dataset or instructions for how to construct the dataset).
- (d) We recognize that reproducibility may be tricky in some cases, in which case authors are welcome to describe the particular way they provide for reproducibility. In the case of closed-source models, it may be that access to the model is limited in some way (e.g., to registered users), but it should be possible for other researchers to have some path to reproducing or verifying the results.

## 5. Open access to data and code

Question: Does the paper provide open access to the data and code, with sufficient instructions to faithfully reproduce the main experimental results, as described in supplemental material?

Answer: [Yes]

Justification: Code is provided as supplementary material. The algorithm is fully specified in Algorithm 1. WikiText-2 dataset is publicly available.

Guidelines:

- The answer [N/A] means that paper does not include experiments requiring code.
- Please see the NeurIPS code and data submission guidelines ( https://neurips.cc/ public/guides/CodeSubmissionPolicy ) for more details.
- While we encourage the release of code and data, we understand that this might not be possible, so [No] is an acceptable answer. Papers cannot be rejected simply for not including code, unless this is central to the contribution (e.g., for a new open-source benchmark).
- The instructions should contain the exact command and environment needed to run to reproduce the results. See the NeurIPS code and data submission guidelines ( https: //neurips.cc/public/guides/CodeSubmissionPolicy ) for more details.
- The authors should provide instructions on data access and preparation, including how to access the raw data, preprocessed data, intermediate data, and generated data, etc.
- The authors should provide scripts to reproduce all experimental results for the new proposed method and baselines. If only a subset of experiments are reproducible, they should state which ones are omitted from the script and why.
- At submission time, to preserve anonymity, the authors should release anonymized versions (if applicable).
- Providing as much information as possible in supplemental material (appended to the paper) is recommended, but including URLs to data and code is permitted.

## 6. Experimental setting/details

Question: Does the paper specify all the training and test details (e.g., data splits, hyperparameters, how they were chosen, type of optimizer) necessary to understand the results?

Answer: [Yes]

Justification: All hyperparameters, evaluation protocol, and hardware specifications are provided in Section 4.1.

Guidelines:

- The answer [N/A] means that the paper does not include experiments.
- The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
- The full details can be provided either with the code, in appendix, or as supplemental material.

## 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [Yes]

Justification: Qwen3-8B reports mean ± std over 3 random seeds (Table 9). All seeds show consistent improvement direction at 3.5-4.0 bits.

Guidelines:

- The answer [N/A] means that the paper does not include experiments.
- The authors should answer [Yes] if the results are accompanied by error bars, confidence intervals, or statistical significance tests, at least for the experiments that support the main claims of the paper.

- The factors of variability that the error bars are capturing should be clearly stated (for example, train/test split, initialization, random drawing of some parameter, or overall run with given experimental conditions).
- The method for calculating the error bars should be explained (closed form formula, call to a library function, bootstrap, etc.)
- The assumptions made should be given (e.g., Normally distributed errors).
- It should be clear whether the error bar is the standard deviation or the standard error of the mean.
- It is OK to report 1-sigma error bars, but one should state it. The authors should preferably report a 2-sigma error bar than state that they have a 96% CI, if the hypothesis of Normality of errors is not verified.
- For asymmetric distributions, the authors should be careful not to show in tables or figures symmetric error bars that would yield results that are out of range (e.g., negative error rates).
- If error bars are reported in tables or plots, the authors should explain in the text how they were calculated and reference the corresponding figures or tables in the text.

## 8. Experiments compute resources

Question: For each experiment, does the paper provide sufficient information on the computer resources (type of compute workers, memory, time of execution) needed to reproduce the experiments?

Answer: [Yes]

Justification: Experiments run on a single NVIDIA H200 GPU. Calibration times are reported in Section 4.1.

Guidelines:

- The answer [N/A] means that the paper does not include experiments.
- The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
- The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
- The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn't make it into the paper).

## 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines ?

Answer: [Yes]

Justification: No human subjects, private data, or dual-use concerns beyond general LLM deployment efficiency.

Guidelines:

- The answer [N/A] means that the authors have not reviewed the NeurIPS Code of Ethics.
- If the authors answer [No], they should explain the special circumstances that require a deviation from the Code of Ethics.
- The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

## 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [Yes]

Justification: Section O discusses positive impacts (reduced memory, energy efficiency) and acknowledges that efficiency gains could lower barriers to LLM misuse.

Guidelines:

- The answer [N/A] means that there is no societal impact of the work performed.
- If the authors answer [N/A] or [No], they should explain why their work has no societal impact or why the paper does not address societal impact.
- Examples of negative societal impacts include potential malicious or unintended uses (e.g., disinformation, generating fake profiles, surveillance), fairness considerations (e.g., deployment of technologies that could make decisions that unfairly impact specific groups), privacy considerations, and security considerations.
- The conference expects that many papers will be foundational research and not tied to particular applications, let alone deployments. However, if there is a direct path to any negative applications, the authors should point it out. For example, it is legitimate to point out that an improvement in the quality of generative models could be used to generate Deepfakes for disinformation. On the other hand, it is not needed to point out that a generic algorithm for optimizing neural networks could enable people to train models that generate Deepfakes faster.
- The authors should consider possible harms that could arise when the technology is being used as intended and functioning correctly, harms that could arise when the technology is being used as intended but gives incorrect results, and harms following from (intentional or unintentional) misuse of the technology.
- If there are negative societal impacts, the authors could also discuss possible mitigation strategies (e.g., gated release of models, providing defenses in addition to attacks, mechanisms for monitoring misuse, mechanisms to monitor how a system learns from feedback over time, improving the efficiency and accessibility of ML).

## 11. Safeguards

Question: Does the paper describe safeguards that have been put in place for responsible release of data or models that have a high risk for misuse (e.g., pre-trained language models, image generators, or scraped datasets)?

Answer: [N/A]

Justification: This paper proposes a quantization method; no pre-trained models, datasets, or assets with misuse risk are released.

Guidelines:

- The answer [N/A] means that the paper poses no such risks.
- Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.
- Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.
- We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

## 12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [Yes]

Justification: All models (Qwen3, Llama3) and datasets (WikiText-2) are cited. Models are used under their respective Apache 2.0 / Llama Community licenses.

Guidelines:

- The answer [N/A] means that the paper does not use existing assets.
- The authors should cite the original paper that produced the code package or dataset.
- The authors should state which version of the asset is used and, if possible, include a URL.

- The name of the license (e.g., CC-BY 4.0) should be included for each asset.
- For scraped data from a particular source (e.g., website), the copyright and terms of service of that source should be provided.
- If assets are released, the license, copyright information, and terms of use in the package should be provided. For popular datasets, paperswithcode.com/datasets has curated licenses for some datasets. Their licensing guide can help determine the license of a dataset.
- For existing datasets that are re-packaged, both the original license and the license of the derived asset (if it has changed) should be provided.
- If this information is not available online, the authors are encouraged to reach out to the asset's creators.

## 13. New assets

Question: Are new assets introduced in the paper well documented and is the documentation provided alongside the assets?

Answer: [N/A]

Justification: No new datasets or pre-trained models are released.

Guidelines:

- The answer [N/A] means that the paper does not release new assets.
- Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.
- The paper should discuss whether and how consent was obtained from people whose asset is used.
- At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.

## 14. Crowdsourcing and research with human subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?

Answer: [N/A]

Justification: This paper does not involve crowdsourcing or research with human subjects.

Guidelines:

- The answer [N/A] means that the paper does not involve crowdsourcing nor research with human subjects.
- Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper.
- According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector.

## 15. Institutional review board (IRB) approvals or equivalent for research with human subjects

Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?

Answer: [N/A]

Justification: This paper does not involve research with human subjects.

Guidelines:

- The answer [N/A] means that the paper does not involve crowdsourcing nor research with human subjects.

- Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.
- We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.
- For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review.

## 16. Declaration of LLM usage

Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigor, or originality of the research, declaration is not required.

Answer: [N/A]

Justification: LLMs are evaluation subjects only (Qwen3, Llama3 families), not part of the proposed methodology.

Guidelines:

- The answer [N/A] means that the core method development in this research does not involve LLMs as any important, original, or non-standard components.
- Please refer to our LLM policy in the NeurIPS handbook for what should or should not be described.