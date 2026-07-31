## Routing-Free Mixture-of-Experts

Yilun Liu ∗ , † , 1 , 3 , Jinru Han ∗ , 2 , Sikuan Yan 1 , Volker Tresp 1 , 3 , Yunpu Ma † , 1 , 3 1 Ludwig Maximilian University of Munich 2 University of California, Los Angeles 3 Munich Center for Machine Learning ∗ Equal contribution. † yilun.liu@tum.de cognitive.yunpu@gmail.com

## Abstract

Standard Mixture-of-Experts (MoE) models rely on centralized routing mechanisms that introduce rigid inductive biases. We propose Routing-Free MoE which eliminates any hard-coded centralized designs including external routers, Softmax, TopK and load balancing, instead encapsulating all activation functionalities within individual experts and are directly optimized through continuous gradient flow, enabling each expert to determine its activation entirely on its own. We introduce a unified adaptive load-balancing framework to simultaneously optimize both expert-balancing and token-balancing objectives through a configurable interpolation, allowing flexible and customizable resource allocation. Extensive experiments show that Routing-Free MoE can consistently outperform baselines with better scalability and robustness. We analyze its behavior in detail and offer insights that may facilitate future MoE design and optimization. Code is available at https://github.com/liuyilun2000/RoutingFreeMoE/tree/release .

Figure 1: Standard MoE relies on routing to orchestrate expert activations. Routing-Free MoE let each expert purely-independently determine its own activation. Green indicates activated components; red for inactive components; yellow for trainable components.

<!-- image -->

## 1 Introduction

The scalability of transformer-based Large Language Models (LLMs) (Vaswani et al., 2017) is constrained by the substantial computational resources required (Kaplan et al., 2020). To efficiently expand model capacity without proportional computation cost growth, Mixture-of-Experts (MoE) designs focus on activating subsets of parameters for each input (Shazeer et al., 2017; Fedus et al., 2022). This approach presents a fundamental challenge: how to optimally distribute inputs to experts while satisfying sparsity and balancing considerations.

Existing MoE designs are hindered by structural limitations across multiple dimensions. Standard MoE relies on small, external routers that lack sufficient capacity for storing expert capabilities to

Figure 2: Routing-Free MoE consistently outperforms standard MoE, AoE (Lv et al., 2025), and ReMoE (Wang et al., 2024b) in language modeling. All models are trained on OpenWebText (Gokaslan et al., 2019) under identical environment conditions and best-performing configurations, as described in Section 4.1. FLOPs are estimated for one epoch.

<!-- image -->

determine expert preference for each input, forcing them to learn their prediction via indirect trial-anderror optimization, which inevitably leads to suboptimal routing and unstable early training dynamics (Lv et al., 2025). For computational efficiency, conventional MoE enforces rigid, global constraints on expert activation that ignore input-specific dynamics. The fixed TopK selection imposes uniform sparsity, regardless of varying input complexity and expertise of experts, overlooking potential gains from fewer or more activations (Zhou et al., 2022). The Softmax operation forces a competitive probability distribution that sacrifices the absolute magnitude information of expert activations (Wang et al., 2024b). For load-balancing, existing strategies employ different balancing targets that force predetermined activation patterns, which are often mutually exclusive yet exhibit varying performance under different configurations (Fedus et al., 2022; Zhou et al., 2022; Muennighoff et al., 2025). Rigidly adhering to either may constrain the model's ability to adaptively discover potentially better resource allocation patterns (Do et al., 2025).

To bridge these gaps, we introduce Routing-Free MoE, a bottom-up architecture without any centralized routers, Softmax, TopK, or rigid load-balancing designs, where each expert individually and directly determines its own activation. Our design enables each expert to activate itself purely when its internal confidence score surpasses a configurable threshold. To satisfy efficiency requirements, we design a dynamic, configurable framework to adaptively achieve the sparsity and load-balancing objectives during training, allowing the optimal activation pattern to emerge spontaneously. We utilize an auxiliary loss function that seamlessly integrates both token and expert balancing, providing flexibility to adaptively exploit both depending on training dynamics and workload requirements.

We validate Routing-Free MoE across three scales up to 0.8B with extensive experiments in comparison against standard MoE and strong baselines. Across all settings, Routing-Free MoE consistently achieves better language modeling quality and downstream performance averaged across 9 evaluation benchmarks. It also demonstrates notably improved scalability and robustness. We analyze its training behavior in detail and further document intriguing phenomena for density and load-balancing, offering insights that may guide future improvements in MoE design and optimization.

In summary, our key contributions are:

- A Routing-Free MoE architecture that eliminates routers, Softmax, TopK, and hard-coded load balancing mechanisms.
- A unified, adaptive load-balancing framework that jointly optimizes token-balancing and expertbalancing, allowing customizable application.
- Experiments and analyses demonstrating the improvements of Routing-Free MoE over baselines.

## 2 Preliminaries

Consider a standard MoE LLM(Fedus et al., 2022; Jiang et al., 2024) comprising L transformer blocks. For layer ℓ ∈ { 1 , · · · , L } and token sequence of length T , the forward pass 1 can be formulated as:

1 LayerNorms and dropout are omitted for clarity.

$$x _ { 1 \colon T } ^ { \ell } = \text {SeftAttn} ( h _ { 1 \colon T } ^ { \ell - 1 } ) + h _ { 1 \colon T } ^ { \ell - 1 } ,$$

$$h _ { t } ^ { \ell } = M o E ( x _ { t } ^ { \ell } ) + x _ { t } ^ { \ell } ,$$

where x ℓ 1: T denotes the attention module output with residual connection added. The MoE block performs a token-wise mapping, yielding output h ℓ t at token t ∈ { 1 , · · · , T } with residual added.

The mainstream form of MoE LLMs incorporates multiple structurally identical FFN experts E i ( · ) , i ∈ { 1 , · · · , N } within the MoE layer. This is implemented via a token-wise routing mechanism among all N experts at layer ℓ , with a gating network G ( · ) assigning each token to a designated number K of top-activated experts:

$$h = \sum _ { i = 1 } ^ { N } \left ( G \left ( x \right ) _ { i } E _ { i } \left ( x \right ) \right ) + x ,$$

$$G \left ( \mathbf x \right ) = \text {Softmax} \left ( \text {Top} K \left ( \mathbf x G , K \right ) \right ) ,$$

where G ( · ) : R D ↦→ R N denotes the routing mechanism, whose output serves as the weights of the weighted sum for outputs among all N experts, and only K of N experts receive nonzero values. The router learns its weight matrix G ∈ R D × N that can be interpreted as a set of N individual D -dimensional expert vectors g i , each responding to a characteristic hidden state h i that should activate the corresponding expert E i , with s i = xg i ∈ R N denoting the token-to-expert affinity scores (Zhou et al., 2022; Liu et al., 2025). TopK( s , K ) retains the topK scores and masks the rest to -∞ .

Each individual expert E i is implemented as a Feed-Forward Network (FFN). Modern FFN typically takes the form of a Gated Linear Unit (GLU, Dauphin et al. (2017); Shazeer et al. (2017)):

$$\ F F N ( x ) = [ \sigma ( x W _ { u p } ) \odot ( x W _ { g a t e } ) ] W _ { d o w n } ,$$

where σ is the activation function and ⊙ denotes element-wise multiplication. In MoE LLMs, this design enables a second level of input-dependent information filtering, leading to potentially more effective representations.

The routing in standard MoE relies on several strong inductive biases. With merely N expert vectors g i ∈ R D , the router's capacity is orders of magnitude smaller than the experts themselves, yet must compress the knowledge-intensive activation preferences of all N experts into single dot-product scores without any direct signal about expert capabilities, improving only through indirect trial-anderror (Lv et al., 2025). TopK hard-codes a fixed sparsity ratio K/N regardless of input complexity, preventing input-adaptive activation patterns (Zhou et al., 2022). Softmax discards absolute activation magnitudes by forcing a competitive probability distribution, suppressing the residual contribution of highly suitable experts when others also happen to score higher (Wang et al., 2024b).

## 3 Methodology

## 3.1 Architecture

To alleviate the router bottleneck, Lv et al. (2025) introduce Autonomy-of-Experts (AoE), with

$$\ F F N ( x ) = [ \sigma ( x A _ { g a t e } B _ { g a t e } ) \odot ( x W _ { u p } ) ] W _ { d o w n } ,$$

where A gate ∈ R D × r projects the input hidden state x from hidden dimension D to a lowerdimensional rank r ≪ D , and B gate ∈ R r × D act projects it back. This low-rank representation provides an alternative indicator of expert suitability that originates from within the expert itself rather than from an external router. Each expert can therefore directly produce its own scalar activation score by applying a norm to its own internal representation ∥ xA gate ,i ∥ 2 . However, AoE feeds the internal scores ∥ xA gate ∥ 2 back to the standard centralized TopK and Softmax routing pipeline:

$$G ( \mathbf x ) = \text {Softmax} ( \text {Top} K ( \| \mathbf x \mathbf A _ { g a t e } \| _ { 2 } , K ) ) ,$$

thereby retaining the structural constraints and inductive biases of conventional routing.

Meanwhile, addressing the constraints of TopK and Softmax, Wang et al. (2024b) propose ReMoE, which replaces TopK and Softmax with a single ReLU function applied directly to router's output:

$$G ( x ) = R e L U ( x G ) ,$$

<!-- image -->

- (a) Expert choice (EC) ensures hard expert-balancing and optimizes token-balancing via training.
- (b) Token choice ensures hard tokenbalancing and optimizes expertbalancing via training.
- (c) Routing-Free MoE adaptively optimizes both token-balancing and expert-balancing via training.

Figure 3: Load-balancing for tokens and experts. Routing-Free MoE introduces a unified loadbalancing framework that simultaneously optimizes both expert-balancing and token-balancing through a configurable interpolation.

The sparse activation arises naturally from ReLU without any explicit TopK selection or comparative normalization. The absolute magnitude of router scores is preserved, allowing each expert's residual contribution to be linearly weighted by router's prediction, rather than a normalized relative preference. Nevertheless, ReMoE still retains a centralized external router, preserving the information bottleneck and indirect optimization dynamics.

Building on these insights, our Routing-Free MoE seeks to eliminate all constraints of routing mechanisms. We adopt AoE's FFN design (Equation 6) using each expert's internal norm ∥ xA gate ∥ 2 as the initial expert preference score, grounding the activation decision in the expert's own response to the input. Since ∥ xA gate ∥ 2 is strictly non-negative unlike router's xG , we introduce a learnable per-expert bias term before activation upon ReMoE's design, yielding

$$G _ { i } ( \mathbf x ) = R e L U ( \| \mathbf x A _ { g a t e , i } \| _ { 2 } - b _ { i } ) .$$

With the per-expert bias term introduced, experts whose weighted norm falls below their own bias threshold contribute zero and are effectively deactivated, allowing each expert to jointly adapt both its A gate ,i matrix and the b i parameter to effectively adjust its own activation ratio. We further introduce a global post-activation threshold θ as a configurable hyperparameter for external control over the overall sparsity level, giving the final binary activation decision of each expert:

$$f _ { i } ( x ) = 1 \left \{ G _ { i } ( x ) - \theta \geq 0 \right \} .$$

The result is a fully decentralized architecture without any external router, TopK or Softmax, making each expert independently determine its own activation from within, allowing the global activation pattern to emerge bottom-up from collective self-adjustment of all experts. Figure 1 visualizes the architecture of Routing-Free MoE and its experts.

## 3.2 Training

Training MoE models requires simultaneously maintaining the activation ratio and balanced expert and token distribution. Standard practice hard-codes TopK for the activation ratio, and addresses balancing via either Token-Choice (TC) (Shazeer et al., 2017; Fedus et al., 2022) that guarantees per-token compute but not expert balance, or Expert-Choice (EC) (Zhou et al., 2022) that hardcodes uniform expert utilization but not per-token compute, as illustrated in Figure 3. Both enforce one balanced dimension as a hard constraint and optimizes the other via soft auxiliary loss. As Routing-Free MoE has eliminated all centralized routing mechanisms, standard activation ratio and load-balancing strategies that rely on hard-coded TopK (Do et al., 2025) no longer apply. We introduce a unified load-balancing framework by extending the auxiliary loss of Fedus et al. (2022) to jointly encourage both balanced token distribution across experts and balanced expert activation per token, without requiring any centralized mechanisms.

As the binary activation decision f i in Equation 10 is non-differentiable, we directly use the prethreshold activation weight G i ( x ) as a differentiable activation proxy of expert E i . We define the

mean activation density over a set of experts E and token batch B and its differentiable proxy as:

$$\rho ( \mathcal { E } , \mathcal { B } ) = \frac { 1 } { | \mathcal { E } | | \mathcal { B } | } \sum _ { e _ { i } \in \mathcal { E } } \sum _ { x \in \mathcal { B } } f _ { i } ( x ) ,$$

$$\tilde { \rho } ( \mathcal { E } , \mathcal { B } ) = \frac { 1 } { | \mathcal { E } | | \mathcal { B } | } \sum _ { e _ { i } \in \mathcal { E } } \sum _ { x \in \mathcal { B } } G _ { i } ( x ) .$$

Both quantities equal the target activation density ρ ∞ under perfectly uniform load.

We decompose the load-balancing objective into two complementary terms 2 . The expert-balancing loss L EB encourages uniform distribution of tokens across experts by penalizing experts that consistently receive more or fewer tokens than average:

$$\mathcal { L } _ { E B } = \frac { 1 } { | \mathcal { E } | } \sum _ { e _ { i } \in \mathcal { E } } \left ( \frac { 1 } { | \mathcal { B } | } \sum _ { x \in \mathcal { B } } f _ { i } ( x ) \right ) \left ( \frac { 1 } { | \mathcal { B } | } \sum _ { x \in \mathcal { B } } G _ { i } ( x ) \right ) .$$

The token-balancing loss L TB encourages uniform distribution of activated experts per token by penalizing tokens that consistently activate more or fewer experts than average:

$$\mathcal { L } _ { T B } = \frac { 1 } { | \mathcal { B } | } \sum _ { x \in \mathcal { B } } \left ( \frac { 1 } { | \mathcal { E } | } \sum _ { e _ { i } \in \mathcal { E } } f _ { i } ( x ) \right ) \left ( \frac { 1 } { | \mathcal { E } | } \sum _ { e _ { j } \in \mathcal { E } } G _ { j } ( x ) \right ) .$$

Each loss is a dot product between a binary non-differentiable term and a differentiable proxy, and is minimized when both factors equal ρ ∞ uniformly 3 . The two objectives are combined via a configurable interpolation parameter µ ∈ [0 , 1] :

$$\mathcal { L } _ { L B } = \mu \, \mathcal { L } _ { E B } + ( 1 - \mu ) \, \mathcal { L } _ { T B } .$$

Setting µ = 1 recovers a pure expert-balancing, and µ = 0 recovers pure token-balancing. This provides a single unified framework that interpolates both routing paradigms, allowing load-balancing to be tailored to specific deployment needs.

Rather than fixing the auxiliary loss coefficient λ as a static hyperparameter, we follow Wang et al. (2024b) and adopt an adaptive schedule that drives the empirical activation ratio toward the target ρ ∞ throughout training. The total training objective is

$$\mathcal { L } = \mathcal { L } _ { L M } + \lambda _ { t } \, \mathcal { L } _ { L B } ,$$

with λ t updated at each training step t as:

$$\lambda _ { t + 1 } = \lambda _ { t } \cdot ( 1 + \eta ) ^ { \text {sign} ( \rho _ { t } ( \mathcal { E } , \mathcal { B } ) - \rho _ { \circ } ) } .$$

When the current density ρ t exceeds the target ρ ∞ , λ t increases to exert stronger load-balancing pressure; when below, it decreases to allow more expert activations. The step size η controls the responsiveness of this feedback loop. This formulation avoids the need to manually tune λ while ensuring the model converges to the desired computational budget, and is compatible with both training from scratch and adaptation of pretrained MoE models into Routing-Free MoE.

To encourage all experts to participate during the early warm-up phase of training, each expert's bias is initialized as 1e -6 , allowing all experts to become activated, jointly exploring the representation space and establishing initial specialization before sparsity is enforced. As training progresses and λ increases, the sparsity regularization gradually strengthens, driving the activation density toward target ρ ∞ . By this stage, experts have already developed distinct roles, avoiding expert collapse and ensuring a stable and effective phase transition.

## 4 Experiments

We conduct comprehensive experiments to validate the performance of Routing-Free MoE.

2 We avoid using the terms expert choice and token choice in Routing-Free MoE because the notion of 'choice' implies a centralized comparison and selection process, which contradicts our principle that all decisions should emerge locally and independently at individual experts and tokens.

3 See Appendix A.

Table 1: Evaluation results for standard MoE baseline and Routing-Free MoE across scales and benchmarks. Each model is trained on OpenWebText (Gokaslan et al., 2019) from scratch for one epoch. Details in Appendix E.

|    | Arch.   | Size   | FLOPs   |   PPL ↓ |   PIQA |   HellaS |   WinoG |   ARCe |   ARCc |   OBQA |   QQP |   QNLI |   SST-2 |   Avg. |
|----|---------|--------|---------|---------|--------|----------|---------|--------|--------|--------|-------|--------|---------|--------|
| S  | MoE     | 92.44M | 90.93M  |   31.22 |  57.56 |    27.19 |   51.22 |  33.42 |  21.33 |   24.6 | 36.82 |  49.46 |   49.08 |  38.96 |
| S  | AoE     | 93.85M | 88.57M  |   30    |  56.09 |    27.01 |   50.2  |  33.84 |  21.93 |   25   | 36.82 |  49.46 |   49.08 |  38.82 |
| S  | ReMoE   | 92.44M | 90.93M  |   29.6  |  56.58 |    26.69 |   51.62 |  33.38 |  22.27 |   26   | 36.83 |  49.48 |   49.08 |  39.1  |
| S  | RFMoE   | 95.32M | 91.08M  |   27.42 |  58.49 |    27.09 |   50.59 |  35.77 |  21.59 |   24.4 | 36.98 |  49.44 |   53.56 |  39.77 |
| M  | MoE     | 289.9M | 248.0M  |   25    |  58.32 |    28.26 |   52.33 |  36.2  |  21.42 |   24.6 | 36.85 |  49.46 |   49.31 |  39.64 |
| M  | RFMoE   | 307.3M | 249.2M  |   22.08 |  58.92 |    27.85 |   49.72 |  35.27 |  21.5  |   26.4 | 37.06 |  49.51 |   57.34 |  40.4  |
| L  | MoE     | 808.4M | 608.4M  |   24.58 |  59.19 |    29.37 |   50.83 |  36.41 |  23.29 |   25   | 37.57 |  49.28 |   49.08 |  40    |
| L  | RFMoE   | 870.6M | 613.2M  |   19.97 |  58.87 |    28.27 |   50.51 |  37.46 |  21.67 |   26.6 | 39.93 |  49.86 |   53.67 |  40.76 |

Table 2: Ablation performed at scale S. Detailed configurations for each run are provided in Table E.

| Config.                   | Size   | FLOPs   |   PPL ↓ |
|---------------------------|--------|---------|---------|
| Standard MoE              | 92.44M | 90.93M  |   31.22 |
| w/o router (AoE, r =16)   | 93.85M | 88.57M  |   30    |
| w/o router (AoE, r =32)   | 95.32M | 91.08M  |   30.31 |
| w/o TopK&Softmax (ReMoE)  | 92.44M | 90.93M  |   29.6  |
| Routing-Free MoE ( r =16) | 93.85M | 88.57M  |   28.73 |
| Routing-Free MoE ( r =32) | 95.32M | 91.08M  |   28.33 |

## 4.1 Experimental Setup

Weimplement Routing-Free MoE upon the HuggingFace implementation 4 for the Mixtral architecture (Jiang et al., 2024), which includes attention designed as Grouped Query Attention (GQA, Ainslie et al. (2023)) with Rotary Position Embeddings (RoPE, Su et al. (2024)), FFN using SwiGLU activation (Shazeer, 2020), and Root Mean Square Layer Normalization (RMSNorm, Zhang and Sennrich (2019)) applied to the residual stream prior to each attention and MoE FFN (Xiong et al., 2020).

We conduct experiments across three scales (S/M/L) up to 0.8B, and compare against a standard MoE baseline with identical structure except for the routing mechanism. AoE (Lv et al., 2025) and ReMoE (Wang et al., 2024b) are also examined as both additional baselines and ablated variants for comparison. Detailed hyperparameters and training configurations are in Table 8 in Appendix.

Following standard practices, all models are trained on OpenWebText (Gokaslan et al., 2019) for one epoch. Zero-shot evaluation is conducted across 9 benchmarks (Gao et al., 2024b). We include PIQA (Bisk et al., 2020), HellaSwag (Zellers et al., 2019), WinoGrande (Sakaguchi et al., 2021), ARC-Easy and ARC-Challenge (Clark et al., 2018), OpenBookQA (Mihaylov et al., 2018), and QQP, QNLI, and SST-2 from GLUE (Wang et al., 2018). Accuracy is reported on WinoGrande, QQP, QNLI, and SST-2, and normalized accuracy on others.

## 4.2 Main Results

Table 1 presents language modeling perplexity and downstream benchmark evaluation results across all three scales under an iso-compute comparison, with FLOPs matched within ∼ 1% between MoE and Routing-Free MoE at each scale. Figure 2 shows the detailed validation perplexity evolution with total FLOPs throughout training. Routing-Free MoE achieves consistently better validation perplexity than standard MoE across all scales. On downstream benchmarks, Routing-Free MoE also consistently improves the average performance, with a detailed statistical analysis (Appendix C) supporting that Routing-Free MoE can produce a statistically significant improvement over the standard MoE baseline. Notably, the gain of Routing-Free MoE does not diminish with scale, suggesting that the benefits of our approach can continue to provide gains as models scale up.

[4 https://github.com/huggingface/transformers/tree/v4.57.6/src/transformers/models/mixtral](https://github.com/huggingface/transformers/tree/v4.57.6/src/transformers/models/mixtral)

̃

̃

<!-- image -->

- (a) Density ρ , density target ρ ∞ , and density proxy ˜ ρ
- (b) Adaptive coefficient λ balancing loss L
- and loadLB (c) Training L , language modeling L LM , and regularization λ L LB

Figure 4: Training dynamics of Routing-Free MoE at scale S, with r = 16 , λ 0 = 1e -10 , η = 0 . 02 , and α = 1e -3 .

Table 3: Effect of low-rank projection dimension at scale S, with learning rate α = 1e -3 .

|   r | Size   | FLOPs   |   PPL ↓ |
|-----|--------|---------|---------|
|   8 | 93.11M | 87.32M  |   29.16 |
|  16 | 93.85M | 88.57M  |   28.74 |
|  32 | 95.32M | 91.08M  |   28.34 |
|  64 | 98.27M | 96.09M  |   28.24 |

## 4.3 Architectural Comparison

We experiment at scale S, decomposing the contribution of each architectural change over the standard MoE via incremental ablation, with main results shown in Table 2. Routing-Free MoE outperforms either by a substantial margin under matching size and FLOPs. A visualization of perplexity curves during training for AoE and ReMoE can also be found in Figure 2.

Table 3 reports the effect of r . Increasing r consistently improves perplexity, yet with diminishing returns. Therefore, we set 32 as the default r and scale it proportionally with the hidden dimension D across experiments on other scales for the optimal efficiency-quality trade-off.

A detailed analysis about Routing-Free MoE at deployment is provided in Appendix B, including efficiency improvements by expert parallelism and the effects of adapting threshold θ at inference.

## 4.4 Training Dynamics

Figure 4 visualizes the training dynamics of Routing-Free MoE. The empirical activation density ρ begins near 1 at initialization, drops sharply to the target ρ ∞ as λ grows, and remains close to the target thereafter. The differentiable proxy ˜ ρ also settles slightly above ρ ∞ at convergence. λ rises steeply during the initial warm-up phase, plateaus as ρ converges to ρ ∞ , and collapses sharply near the end as density falls marginally below the target. The regularization term λ L LB exhibits a transient spike caused by the exponential growth of λ , with its magnitude rising above 10 -1 , making it a non-negligible contributor to the total loss L alongside L LM , and steering gradient descent toward directions that also reduce L LB . As training progresses, λ L LB decays to a negligible level, demonstrating how load-balancing pressure is applied precisely when needed without persistently distorting the language-modeling optimization objectives.

Figure 5 further examines training stability by sweeping the learning rate α . Under conservative α , both models converge smoothly with Routing-Free MoE consistently achieving lower perplexity. As α increases, the standard MoE baseline begins to collapse much earlier than Routing-Free MoE. This is evident at scale S, where MoE fails at α = 2e -3 while Routing-Free MoE remains stable throughout training. In addition, tuning α for the standard MoE at scale L yields only marginal gains over its scale-M performance (as in Figure 2); whereas Routing-Free MoE continues to improve as scale increases, highlighting its enhanced scalability. Routing-Free MoE not only outperforms the baseline, but also exhibits better training stability and less sensitivity to learning rates, tolerating a wider range of hyperparameter choices at scale.

<!-- image -->

- (a) Training curves with different α at scale S

(b) Training curves with different

α

at scale L

Figure 5: Training dynamics of Routing-Free MoE by α , with r = 16 , λ 0 = 1e -10 , and η = 0 . 02 .

<!-- image -->

Figure 6: Evolution of expert activation across layers during training on OpenWebText under global density target. Left panel shows the per-layer mean activation as lines, along with IQR as shaded band regions using 1,000-step moving average smoothing. Right panel shows activation distribution at the final training step. Model at scale S.

## 5 Discussion

Beyond its superior performance compared with the baselines, here we analyze some intriguing properties of Routing-Free MoE.

## 5.1 Per-Layer and Global Density

With Top-K at each layer eliminated, a natural question is whether the target activation density ρ ∞ should still be enforced within each layer, or only globally across all experts. Per-layer enforcement computes L LB separately at each layer, continuing to impose the inductive bias for a uniform sparsity at every depth. A global ρ ∞ , in contrast, relaxes the constraint and allows individual layers to deviate from ρ ∞ so long as aggregate activation matches the target and improves the overall loss L . Our experiment on scale S finds that relaxing this inductive bias leads to a significant improvement with perplexity drops from 39.44 to 28.74. In Figure 6 we illustrate the emergent expert activation pattern that Routing-Free MoE develops by itself during training when this per-layer bias is removed. A detailed analysis on this matter is provided in Appendix D.2. Enforcing identical sparsity at every depth suppresses the compute-hungry layers that naturally benefit from activating more experts, while simultaneously forcing unnecessary activations in layers where sparse representations suffice. Once this bias is lifted, the model is free to self-organize into a more effective, functionally aligned activation structure.

## 5.2 Token and Expert Balancing

A key contribution of this work is demonstrating empirically that token- and expert-balancing strategies can be combined in a complementary manner via soft interpolation rather than being treated as mutually exclusive. Prior work (Zhou et al., 2022; Muennighoff et al., 2025) documented that EC and TC perform better under different configurations, but our unified framework provides a mechanism to adaptively balance both objectives.

Table 4 shows that µ = 0 . 5 achieves the lowest perplexity and highest throughput, which degrades toward either direction. Token-balancing enables flexible compute allocation based on input characteristics, while expert-balancing ensures uniform expert utilization and encourages specialization; the two objectives address orthogonal failure modes and jointly produce better outcomes than either

Table 4: Effect of load balancing interpolation µ . Throughput is measured as samples per second.

|          | µ       | PPL ↓       | Eval. Throughput ↑   |
|----------|---------|-------------|----------------------|
| only TB  | 0.0     | 28.41       | 645.7                |
| balanced | 0.2 0.5 | 28.35 28.34 | 648.3 662.3          |
|          | 0.8     | 28.38       | 643.9                |
|          | 1.0     |             |                      |
| only EB  |         | 28.43       | 648.8                |

Figure 7: Expert activation heatmaps on different input for representative layers at scale S. Each row shows the average expert activation ratios on a given benchmark. Darker colors indicate more frequent activation.

<!-- image -->

alone. This complementarity is illustrated in Figure 7. Under token-only balancing (Figure 7a), the absence of expert balancing leads to larger activation probability differences across the expert axis, with a few experts activated far more frequently than others. Conversely, under expert-only balancing (Figure 7c), expert loads are more horizontally uniform, but activation patterns vary notably across benchmarks, indicating that without token-level regularization, expert activation becomes overly sensitive to the input domain distribution. When both objectives are active ( µ = 0 . 5 , Figure 7b), the activation pattern is more uniform along both axes and more consistent across inputs.

## 6 Related Work

## 6.1 MoE Foundations

Mixture-of-Experts was introduced as adaptive mixtures of local experts (Jacobs et al., 1991; Jordan and Jacobs, 1994) and later scaled to deep networks (Eigen et al., 2013; Shazeer et al., 2017). Interpretability studies reveal that FFNs in transformers capture knowledge with sparse activations (Geva et al., 2021; Dai et al., 2022; Dalvi et al., 2019; Durrani et al., 2020; Gurnee et al., 2023), motivating MoE designs that activate only a subset of parameters (Liu et al., 2023). Modern MoE architectures have since been deployed at scale (Lepikhin et al., 2020; Fedus et al., 2022; Zoph, 2022; Komatsuzaki et al., 2022; Rajbhandari et al., 2022; Du et al., 2022), with frontier models with over billions of parameters (Jiang et al., 2024; Dai et al., 2024; Grok, 2024). Structural enhancements such as shared experts (Gou et al., 2023; Dai et al., 2024) further help reduce parameter redundancy.

## 6.2 Routing Mechanisms

Traditional MoE relies on centralized routers, learned linear projections followed by TopK selection (Lepikhin et al., 2020; Fedus et al., 2022). Recent work has moved toward relaxing the routing mechanisms. Lv et al. (2025) replaces the router with expert-internal scoring; Wang et al. (2024b) replaces Softmax and TopK with ReLU gating. Other approaches include using vector quantization for expert assignment (Do et al., 2024), virtual shared experts (Wu et al.), using pretrained language models as routers (Liu and Lo, 2025), and with ternary expert expansion (Yan et al., 2025). Huang

et al. (2024) propose adjusting expert counts based on input difficulty. Do et al. (2025) utilize global TopK selection over a combined similarity score to unify token- and expert-choice routing.

## 6.3 Load Balancing and Training

Token Choice (Shazeer et al., 2017; Fedus et al., 2022) guarantees per-token compute but not expert balance. Expert Choice (Zhou et al., 2022) guarantees expert load balance, but may potentially yield suboptimal matching. Studies has shown that Token Choice and Expert Choice balancing are complementary rather than mutually exclusive (Muennighoff et al., 2025). Recent advances include auxiliary-loss-free balancing via dynamic bias (Wang et al., 2024a; DeepSeek-AI et al., 2025), similarity-preserving routers (Omi et al., 2025), expert specialization through orthogonality losses (Guo et al., 2025b; Feng et al., 2025), and infrastructure-level scheduling (Zhao et al., 2025; Yu et al., 2025). For training stability, TopK selection's discontinuity prevents gradient flow to non-selected experts; Wang et al. (2024b) address this with differentiable ReLU gating and adaptive regularization. Qiu et al. (2025) provide practical guidance on auxiliary losses, while He et al. (2024) and Pan et al. (2024) address auxiliary losses at inference. Do et al. (2025) propose a unified scoring function with global TopK selection that linearly combines TC and EC similarity scores, which addresses representation collapse and token dropping simultaneously, but is not applicable without routing mechanisms. Our approach, instead, seeks unifying token-balancing and expert-balancing under fully decentralized expert self-activation.

## 7 Conclusion

We present Routing-Free Mixture-of-Experts, an MoE architecture that entirely eliminates centralized routing mechanisms, along with a unified adaptive load-balancing framework that jointly optimizes token- and expert-balancing objectives during training. Experiments across 3 scales and 9 benchmarks validate that Routing-Free MoE consistently outperforms baselines, with improved scalability and robustness. We further analyze the load-balancing behavior throughout training and the expert activation patterns. Routing-Free MoE opens a path toward more flexible and efficient architectures. We hope this work encourages future exploration of MoE design and optimization.

## Limitations

Our experiments are conducted exclusively on models trained from scratch on OpenWebText at scales up to 0.8B parameters. While results are consistent across three scales, it remains an open question whether the findings generalize to even larger-scale pretraining regimes, which lie beyond our current resource limits. Similarly, we only evaluate on 9 English benchmarks for commonsense reasoning and natural language inference which are still limited in scope and may not capture the full range of applications present in real-world applications. Given that our method follows standard training practices common among a broad body of peer-reviewed research in this field, it is highly plausible that the observed trends will carry over to broader settings.

Another practical direction we have not explored is converting an existing pretrained standard MoE model into a Routing-Free MoE via continued pretraining or training-free adaptation, rather than training from scratch. Such a conversion could possibly further reduce the compute cost of adopting our approach, making it a promising avenue for future work; yet it introduces a vast set of distinct challenges that lies beyond the scope of this work.

## Ethics Consideration

Routing-Free MoE is proposed as a general purpose language model architecture and does not introduce domain-specific risks beyond those inherent to language model pretraining. However, as with any method that improves model efficiency and scalability, it may lower the barrier to training more capable models, with corresponding societal implications that merit careful consideration in downstream applications.

The OpenWebText dataset used in our experiments may also reflect demographic or geographic biases present in large-scale web corpora. We did not conduct dedicated bias or safety audits, and the absence of such analyses means that potential fairness, privacy, or safety issues may persist. It is therefore incumbent upon downstream users and deployers to perform appropriate task-specific fairness, privacy, and safety evaluations before any real-world deployment. We disclaim responsibility for unintended consequences arising from downstream use that involve applying models trained using our approach.

## Acknowledgments

The authors gratefully acknowledge the scientific support and HPC resources provided by the Karlsruhe Institute of Technology National High Performance Computing Center (NHR@KIT) under the NHR projects 22189, 22560, and 24767. The HoreKa supercomputer at NHR@KIT is funded by the Ministry of Science, Research and the Arts Baden-Württemberg and by the Federal Ministry of Education and Research of Germany. This work is also partially supported by funding from Munich Center for Machine Learning (MCML). We also acknowledge the EuroHPC Joint Undertaking for awarding this project access to the EuroHPC supercomputer LEONARDO under project EHPC-AI2024A06-060, hosted by CINECA (Italy) and the LEONARDO consortium through a EuroHPC Regular Access call. In addition, the authors thank Xu He and Alinur Kozhanov from Technical University of Munich for their contributions during the early stages of this work.

## References

Joshua Ainslie, James Lee-Thorp, Michiel De Jong, Yury Zemlyanskiy, Federico Lebrón, and Sumit Sanghai. 2023. Gqa: Training generalized multi-query transformer models from multi-head checkpoints. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing , pages 4895-4901.

Yonatan Bisk, Rowan Zellers, Ronan Le bras, Jianfeng Gao, and Yejin Choi. 2020. Piqa: Reasoning about physical commonsense in natural language. In Proceedings of the AAAI conference on artificial intelligence , volume 34, pages 7432-7439.

Peter Clark, Isaac Cowhey, Oren Etzioni, Tushar Khot, Ashish Sabharwal, Carissa Schoenick, and Oyvind Tafjord. 2018. Think you have solved question answering? try arc, the ai2 reasoning challenge. ArXiv , abs/1803.05457.

Damai Dai, Chengqi Deng, Chenggang Zhao, R.x. Xu, Huazuo Gao, Deli Chen, Jiashi Li, Wangding Zeng, Xingkai Yu, Y. Wu, Zhenda Xie, Y.k. Li, Panpan Huang, Fuli Luo, Chong Ruan, Zhifang Sui, and Wenfeng Liang. 2024. DeepSeekMoE: Towards ultimate expert specialization in mixtureof-experts language models. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers) , pages 1280-1297, Bangkok, Thailand. Association for Computational Linguistics.

Damai Dai, Li Dong, Yaru Hao, Zhifang Sui, Baobao Chang, and Furu Wei. 2022. Knowledge neurons in pretrained transformers. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers) , pages 8493-8502.

Fahim Dalvi, Nadir Durrani, Hassan Sajjad, Yonatan Belinkov, Anthony Bau, and James Glass. 2019. What is one grain of sand in the desert? analyzing individual neurons in deep nlp models. In Proceedings of the AAAI Conference on Artificial Intelligence , volume 33, pages 6309-6317.

Yann N Dauphin, Angela Fan, Michael Auli, and David Grangier. 2017. Language modeling with gated convolutional networks. In International conference on machine learning , pages 933-941. PMLR.

DeepSeek-AI, Aixin Liu, Bei Feng, Bing Xue, Bingxuan Wang, Bochao Wu, Chengda Lu, Chenggang Zhao, Chengqi Deng, Chenyu Zhang, Chong Ruan, Damai Dai, Daya Guo, Dejian Yang, Deli Chen, Dongjie Ji, Erhang Li, Fangyun Lin, Fucong Dai, and 181 others. 2025. Deepseek-v3 technical report.

Giang Do, Hung Le, and Truyen Tran. 2025. Unified sparse mixture of experts. arXiv preprint arXiv:2503.22996 .

Giang Do, Kha Pham, Hung Le, and Truyen Tran. 2024. On the role of discrete representation in sparse mixture of experts. arXiv preprint arXiv:2411.19402 .

Nan Du, Yanping Huang, Andrew M Dai, Simon Tong, Dmitry Lepikhin, Yuanzhong Xu, Maxim Krikun, Yanqi Zhou, Adams Wei Yu, Orhan Firat, Barret Zoph, Liam Fedus, Maarten P Bosma, Zongwei Zhou, Tao Wang, Emma Wang, Kellie Webster, Marie Pellat, Kevin Robinson, and 8 others. 2022. GLaM: Efficient scaling of language models with mixture-of-experts. In Proceedings of the 39th International Conference on Machine Learning , volume 162 of Proceedings of Machine Learning Research , pages 5547-5569. PMLR.

Nadir Durrani, Hassan Sajjad, Fahim Dalvi, and Yonatan Belinkov. 2020. Analyzing individual neurons in pre-trained language models. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP) , pages 4865-4880.

David Eigen, Marc'Aurelio Ranzato, and Ilya Sutskever. 2013. Learning factored representations in a deep mixture of experts. arXiv preprint arXiv:1312.4314 .

William Fedus, Barret Zoph, and Noam Shazeer. 2022. Switch transformers: Scaling to trillion parameter models with simple and efficient sparsity. Journal of Machine Learning Research , 23(120):1-39.

Yuchen Feng, Bowen Shen, Naibin Gu, Jiaxuan Zhao, Peng Fu, Zheng Lin, and Weiping Wang. 2025. Dive into moe: Diversity-enhanced reconstruction of large language models from dense into mixture-of-experts. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers) , pages 19375-19394.

Chongyang Gao, Kezhen Chen, Jinmeng Rao, Baochen Sun, Ruibo Liu, Daiyi Peng, Yawen Zhang, Xiaoyuan Guo, Jie Yang, and VS Subrahmanian. 2024a. Higher layers need more lora experts. arXiv preprint arXiv:2402.08562 .

Leo Gao, Jonathan Tow, Baber Abbasi, Stella Biderman, Sid Black, Anthony DiPofi, Charles Foster, Laurence Golding, Jeffrey Hsu, Alain Le Noac'h, Haonan Li, Kyle McDonell, Niklas Muennighoff, Chris Ociepa, Jason Phang, Laria Reynolds, Hailey Schoelkopf, Aviya Skowron, Lintang Sutawika, and 5 others. 2024b. The language model evaluation harness.

Mor Geva, Roei Schuster, Jonathan Berant, and Omer Levy. 2021. Transformer feed-forward layers are key-value memories. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing , pages 5484-5495.

Aaron Gokaslan, Vanya Cohen, Ellie Pavlick, and Stefanie Tellex. 2019. Openwebtext corpus. http://Skylion007.github.io/OpenWebTextCorpus .

Yunhao Gou, Zhili Liu, Kai Chen, Lanqing Hong, Hang Xu, Aoxue Li, Dit-Yan Yeung, James T Kwok, and Yu Zhang. 2023. Mixture of cluster-conditional lora experts for vision-language instruction tuning. arXiv preprint arXiv:2312.12379 .

[Grok. 2024. Open release of grok-1.](https://x.ai/blog/grok-os)

Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Peiyi Wang, Qihao Zhu, Runxin Xu, Ruoyu Zhang, Shirong Ma, Xiao Bi, Xiaokang Zhang, Xingkai Yu, Yu Wu, ZF Wu, Zhibin Gou, Zhihong Shao, Zhuoshu Li, Ziyi Gao, Aixin Liu, and 175 others. 2025a. Deepseek-r1 incentivizes reasoning in llms through reinforcement learning. Nature , 645(8081):633-638.

Hongcan Guo, Haolang Lu, Guoshun Nan, Bolun Chu, Jialin Zhuang, Yuan Yang, Wenhao Che, Xinye Cao, Sicong Leng, Qimei Cui, and Xudong Jiang. 2025b. Advancing expert specialization for better moe. In The Thirty-ninth Annual Conference on Neural Information Processing Systems .

Wes Gurnee, Neel Nanda, Matthew Pauly, Katherine Harvey, Dmitrii Troitskii, and Dimitris Bertsimas. 2023. Finding neurons in a haystack: Case studies with sparse probing. arXiv preprint arXiv:2305.01610 .

Xin He, Shunkang Zhang, Yuxin Wang, Haiyan Yin, Zihao Zeng, Shaohuai Shi, Zhenheng Tang, Xiaowen Chu, Ivor Tsang, and Ong Yew Soon. 2024. Expertflow: Optimized expert activation and token allocation for efficient mixture-of-experts inference. arXiv preprint arXiv:2410.17954 .

Roger W. Hockney. 1994. The communication challenge for mpp: Intel paragon and meiko cs-2. Parallel Comput. , 20(3):389-398.

Quzhe Huang, Zhenwei An, Nan Zhuang, Mingxu Tao, Chen Zhang, Yang Jin, Kun Xu, Liwei Chen, Songfang Huang, and Yansong Feng. 2024. Harder task needs more experts: Dynamic routing in moe models. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers) , pages 12883-12895.

Robert A Jacobs, Michael I Jordan, Steven J Nowlan, and Geoffrey E Hinton. 1991. Adaptive mixtures of local experts. Neural computation , 3(1):79-87.

Albert Q. Jiang, Alexandre Sablayrolles, Antoine Roux, Arthur Mensch, Blanche Savary, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Emma Bou Hanna, Florian Bressand, Gianna Lengyel, Guillaume Bour, Guillaume Lample, Lélio Renard Lavaud, Lucile Saulnier, Marie-Anne Lachaux, Pierre Stock, Sandeep Subramanian, Sophia Yang, and 7 others. 2024. Mixtral of experts. arXiv preprint arXiv:2401.04088 .

Difan Jiao, Yilun Liu, Zhenwei Tang, Daniel Matter, Jürgen Pfeffer, and Ashton Anderson. 2024. Spin: Sparsifying and integrating internal neurons in large language models for text classification. In Findings of the Association for Computational Linguistics: ACL 2024 , pages 4666-4682.

Michael I Jordan and Robert A Jacobs. 1994. Hierarchical mixtures of experts and the em algorithm. Neural computation , 6(2):181-214.

Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B Brown, Benjamin Chess, Rewon Child, Scott Gray, Alec Radford, Jeffrey Wu, and Dario Amodei. 2020. Scaling laws for neural language models. arXiv preprint arXiv:2001.08361 .

Aran Komatsuzaki, Joan Puigcerver, James Lee-Thorp, Carlos Riquelme Ruiz, Basil Mustafa, Joshua Ainslie, Yi Tay, Mostafa Dehghani, and Neil Houlsby. 2022. Sparse upcycling: Training mixture-ofexperts from dense checkpoints. arXiv preprint arXiv:2212.05055 .

Dmitry Lepikhin, HyoukJoong Lee, Yuanzhong Xu, Dehao Chen, Orhan Firat, Yanping Huang, Maxim Krikun, Noam Shazeer, and Zhifeng Chen. 2020. Gshard: Scaling giant models with conditional computation and automatic sharding. arXiv preprint arXiv:2006.16668 .

Kuan-Ming Liu and Ming-Chih Lo. 2025. Llm-based routing in mixture of experts: A novel framework for trading. arXiv preprint arXiv:2501.09636 .

Yilun Liu, Yunpu Ma, Yuetian Lu, Shuo Chen, Zifeng Ding, and Volker Tresp. 2025. Parameterefficient routed fine-tuning: Mixture-of-experts demands mixture of adaptation modules. arXiv preprint arXiv:2508.02587 .

Zeyu Liu, Tim Dettmers, Xi Lin, Veselin Stoyanov, and Xian Li. 2023. Towards a unified view of sparse feed-forward network in pretraining large language model. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing , pages 15038-15061.

Ang Lv, Ruobing Xie, Yining Qian, Songhao Wu, Xingwu Sun, Zhanhui Kang, Di Wang, and Rui Yan. 2025. Autonomy-of-experts models. arXiv preprint arXiv:2501.13074 .

Todor Mihaylov, Peter Clark, Tushar Khot, and Ashish Sabharwal. 2018. Can a suit of armor conduct electricity? a new dataset for open book question answering. In Proceedings of the 2018 conference on empirical methods in natural language processing , pages 2381-2391.

Niklas Muennighoff, Luca Soldaini, Dirk Groeneveld, Kyle Lo, Jacob Morrison, Sewon Min, Weijia Shi, Evan Pete Walsh, Oyvind Tafjord, Nathan Lambert, Yuling Gu, Shane Arora, Akshita Bhagia, Dustin Schwenk, David Wadden, Alexander Wettig, Binyuan Hui, Tim Dettmers, Douwe Kiela, and 5 others. 2025. OLMoe: Open mixture-of-experts language models. In The Thirteenth International Conference on Learning Representations .

Nabil Omi, Siddhartha Sen, and Ali Farhadi. 2025. Load balancing mixture of experts with similarity preserving routers. arXiv preprint arXiv:2506.14038 .

Bowen Pan, Yikang Shen, Haokun Liu, Mayank Mishra, Gaoyuan Zhang, Aude Oliva, Colin Raffel, and Rameswar Panda. 2024. Dense training, sparse inference: Rethinking training of mixture-ofexperts language models. arXiv preprint arXiv:2404.05567 .

Zihan Qiu, Zeyu Huang, Bo Zheng, Kaiyue Wen, Zekun Wang, Rui Men, Ivan Titov, Dayiheng Liu, Jingren Zhou, and Junyang Lin. 2025. Demons in the detail: On implementing load balancing loss for training specialized mixture-of-expert models. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers) , pages 5005-5018.

Samyam Rajbhandari, Conglong Li, Zhewei Yao, Minjia Zhang, Reza Yazdani Aminabadi, Ammar Ahmad Awan, Jeff Rasley, and Yuxiong He. 2022. Deepspeed-moe: Advancing mixture-ofexperts inference and training to power next-generation ai scale. In International conference on machine learning , pages 18332-18346. PMLR.

Keisuke Sakaguchi, Ronan Le Bras, Chandra Bhagavatula, and Yejin Choi. 2021. Winogrande: An adversarial winograd schema challenge at scale. Communications of the ACM , 64(9):99-106.

Noam Shazeer. 2020. Glu variants improve transformer. arXiv preprint arXiv:2002.05202 .

Noam Shazeer, Azalia Mirhoseini, Krzysztof Maziarz, Andy Davis, Quoc Le, Geoffrey Hinton, and Jeff Dean. 2017. Outrageously large neural networks: The sparsely-gated mixture-of-experts layer. In International Conference on Learning Representations .

Jianlin Su, Murtadha Ahmed, Yu Lu, Shengfeng Pan, Wen Bo, and Yunfeng Liu. 2024. Roformer: Enhanced transformer with rotary position embedding. Neurocomputing , 568:127063.

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. Advances in neural information processing systems , 30.

Alex Wang, Amanpreet Singh, Julian Michael, Felix Hill, Omer Levy, and Samuel Bowman. 2018. Glue: A multi-task benchmark and analysis platform for natural language understanding. In Proceedings of the 2018 EMNLP workshop BlackboxNLP: Analyzing and interpreting neural networks for NLP , pages 353-355.

Lean Wang, Huazuo Gao, Chenggang Zhao, Xu Sun, and Damai Dai. 2024a. Auxiliary-loss-free load balancing strategy for mixture-of-experts. arXiv preprint arXiv:2408.15664 .

Ziteng Wang, Jun Zhu, and Jianfei Chen. 2024b. Remoe: Fully differentiable mixture-of-experts with relu routing. In The Thirteenth International Conference on Learning Representations .

Chris Wendler, Veniamin Veselovsky, Giovanni Monea, and Robert West. 2024. Do llamas work in english? on the latent language of multilingual transformers. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers) , pages 1536615394.

Songhao Wu, Ang Lv, Ruobing Xie, Xingwu Sun, Di Wang, Rui Yan, and Yankai Lin. Union-ofexperts: Experts in mixture-of-experts are secretly routers.

Ruibin Xiong, Yunchang Yang, Di He, Kai Zheng, Shuxin Zheng, Chen Xing, Huishuai Zhang, Yanyan Lan, Liwei Wang, and Tieyan Liu. 2020. On layer normalization in the transformer architecture. In International conference on machine learning , pages 10524-10533. PMLR.

Shen Yan, Xingyan Bin, Sijun Zhang, Yisen Wang, and Zhouchen Lin. 2025. Tc-moe: Augmenting mixture of experts with ternary expert choice. In The Thirteenth International Conference on Learning Representations .

Yanpeng Yu, Haiyue Ma, Krish Agarwal, Nicolai Oswald, Qijing Huang, Hugo Linsenmaier, Chunhui Mei, Ritchie Zhao, Ritika Borkar, Bita Darvish Rouhani, David Nellans, Ronny Krashinsky, and Anurag Khandelwal. 2025. Efficient moe serving in the memory-bound regime: Balance activated experts, not tokens. arXiv preprint arXiv:2512.09277 .

Rowan Zellers, Ari Holtzman, Yonatan Bisk, Ali Farhadi, and Yejin Choi. 2019. Hellaswag: Can a machine really finish your sentence? In Proceedings of the 57th annual meeting of the association for computational linguistics , pages 4791-4800.

Biao Zhang and Rico Sennrich. 2019. Root mean square layer normalization. Advances in neural information processing systems , 32.

Chenqi Zhao, Wenfei Wu, Linhai Song, and Yuchen Xu. 2025. Micromoe: Fine-grained load balancing for mixture-of-experts with token scheduling. arXiv preprint arXiv:2511.16947 .

Yanqi Zhou, Tao Lei, Hanxiao Liu, Nan Du, Yanping Huang, Vincent Y Zhao, Andrew Dai, Zhifeng Chen, Quoc Le, and James Laudon. 2022. Mixture-of-experts with expert choice routing. Advances in Neural Information Processing Systems , 35:7103-7114.

Barret Zoph. 2022. Designing effective sparse expert models. In 2022 IEEE International Parallel and Distributed Processing Symposium Workshops (IPDPSW) , pages 1044-1044. IEEE.

## A Load-Balancing Losses

Fedus et al. (2022) first introduced a differentiable auxiliary load-balancing loss to MoE, defined as the scaled dot-product

$$\mathcal { L } _ { L B } = \alpha \cdot N \cdot \sum _ { i = 1 } ^ { N } f _ { i } \cdot P _ { i } ,$$

where f i is the fraction of tokens dispatched to expert i and P i is the fraction of router probability allocated to expert i . Since f i is non-differentiable, the gradient flows through P i alone, while f i serves as a fixed per-step weight. The loss is minimized under uniform routing, as ∑ i f i P i = 1 /N when both vectors are uniform. Our L EB and L TB follow the same product-of-averages structure and gradient strategy, extending it to simultaneously encourage uniformity along both the expert and token axes without any centralized routing mechanism. We show that L EB and L TB are minimized towards the desired uniform activation state at ρ ∞ . Let

$$f _ { i } = \frac { 1 } { | \mathcal { B } | } \sum _ { x \in \mathcal { B } } f _ { i } ( x ) , \ \tilde { g } _ { i } = \frac { 1 } { | \mathcal { B } | } \sum _ { x \in \mathcal { B } } G _ { i } ( x )$$

denote the per-expert mean binary activation and its differentiable proxy. Under the constraint that the adaptive coefficient λ t pins the mean density 1 |E| ∑ i f i = ρ ∞ , the sum ∑ i f i ˜ g i is minimized when all terms are equal by the rearrangement inequality, achieved precisely at f i = ˜ g i = ρ ∞ for all i . While the losses are not written as explicit variance terms, any imbalance where over-activated experts co-occur with larger proxy values increases ∑ i f i ˜ g i above the baseline ρ 2 ∞ attained under perfect balance, since f i and ˜ g i share a common dependence on G i ( x ) . The excess above ρ 2 ∞ is therefore a monotone function of the covariance between binary activations and their proxies, providing implicit variance penalization through a fully differentiable surrogate without requiring additional normalization terms. An identical argument applies to L TB by symmetry over the token axis.

## B Routing-Free MoE at Deployment

## B.1 Expert Parallelism

Deploying MoE at scale often benefits from partitioning experts across multiple devices under expert parallelism (EP) (Lepikhin et al., 2020). In this setting, we trace the full critical path of both architectures and analyze per-regime efficiency.

Consider a single MoE layer distributed to M devices, each hosting N/M experts, and processing a batch of T tokens with hidden dimension D and b bytes per element. We adopt the standard α -β model (Hockney, 1994) that transferring n bytes over one hop incurs latency α + n/B , where α is the startup per-hop latency and B is the per-link bandwidth. In standard MoE each token is routed to K experts, and Routing-Free MoE activates an expected K eff = ρ ∞ · N experts per token.

For a standard MoE layer, the router matmul, Softmax, and TopK are strictly sequential and centralized, with costs

$$t _ { \text {routing} } = t _ { \text {router} } + t _ { \text {Softmax} } + t _ { \text {TopK} } \\ \infty \, T \cdot ( D + 2 ) \cdot N .$$

Softmax requires all N logits before any normalization can proceed; TopK requires a full selection pass over N values. Neither can be parallelized across devices, and dispatch cannot begin until both complete. Afterwards it produces a token-to-expert index tensor as dispatch assignment used to pack token buffers, which are sent to multiple target devices via blocking All-to-All:

$$t _ { A 2 A } = ( M - 1 ) \, \alpha + \frac { K \cdot T \cdot D \cdot b } { M \cdot B } .$$

Expert FFNs run on the received K · T/N tokens per expert with no inter-device communication, with the parallelized per-device cost

$$t _ { \text {expert} } \infty \, 3 K \cdot \frac { T } { M } \cdot D \cdot D _ { \text {act} } .$$

Their outputs are then scattered back to originating devices with the same buffer geometry and All-to-All t A2A , and the receiving device uses its locally-stored assignment mask to scatter outputs into the correct positions, which is a cheap local memory operation. Total per-layer cost is

$$T _ { M o E } & = t _ { r o u t i n g } + t _ { e x p e r t } \\ & + 2 ( M - 1 ) \, \alpha + 2 \, \frac { K \cdot T \cdot D \cdot b } { M \cdot B } .$$

Routing-Free MoE starts from a non-blocking All-Gather to broadcast the full token batch:

$$t _ { A G } = ( M - 1 ) \, \alpha + \frac { ( M - 1 ) \cdot T \cdot D \cdot b } { M \cdot B } .$$

Each device can immediately begin scoring its N/M local experts on incoming chunks, pipelining scoring with communication. The total scoring cost per device, including applying xA gate , norm, bias, activation and thresholding, is

$$t _ { s c o r i n g } \, \infty \, T \cdot D \cdot r \cdot \frac { N } { M } ,$$

For activated token-expert pairs, xA gate ,i computed during scoring is directly reused in the FFN forward pass, incurring zero marginal cost. For non-activated tokens computation terminates after the rankr projection, costing only T · D · r rather than a full FFN pass. The remaining per-device FFN cost for activated pairs is

$$t _ { \text {expert} } ^ { * } \circ X _ { \text {eff} } \cdot \frac { T } { M } \cdot ( r + 2 D ) \cdot D _ { \text {act} } . \\ \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \$$

Activated outputs are immediately returned as asynchronous point-to-point messages upon completion, with no barrier required:

$$t _ { \text {combine} } = \alpha + \frac { K _ { \text {eff} } \cdot T \cdot D \cdot b } { M \cdot B } . \\$$

Receiving devices accumulate partial sums as results arrive. Total per-layer cost is

$$T _ { R F M O E } = t _ { s c o r i n g } + t _ { e x p e r t } ^ { * } + M \alpha \\ + \frac { ( M - 1 + K _ { e f f } ) \cdot T \cdot D \cdot b } { M \cdot B } .$$

Setting K = K eff for an iso-compute comparison, the ratio of computation cost for routing and expert processing becomes

$$\frac { t _ { s c o r i n g } + t _ { e x p e r t } ^ { * } } { t _ { r o t i n g } + t _ { e x p e r t } } = \frac { r D + \frac { K } { N } ( r + 2 D ) D _ { a c t } } { ( D + 2 ) M + \frac { K } { N } ( 3 D ) D _ { a c t } } .$$

Since r ≪ D and M ≪ D act, this ratio is strictly less than 1 , and gets smaller when M grows. Therefore, under expert parallelism, Routing-Free MoE always requires less computation per layer than standard MoE.

For communication, the barrier synchronization saving ∆ α = ( M -2) α is strictly positive for all M ≥ 3 , independent of input size or activation ratio. The bandwidth terms differ as

$$\Delta _ { B } = \frac { ( K + 1 - M ) \cdot T \cdot D \cdot b } { M \cdot B } ,$$

which favors Routing-Free MoE when M &lt; K +1 . In the prefill stage where T is large, the ∆ B term dominates the communication overhead, and Routing-Free MoE is advantageous when the number of devices M is no greater than the number of activated experts K for each layer, which typically holds in practice for advanced large-scale MoE models (Guo et al., 2025a) under expert-parallel setting at inference. In the decode stage where T =1 , the bandwidth term becomes negligible and the critical path is instead dominated by α and sequential computational bottlenecks, and Routing-Free MoE becomes exceptionally well-suited for such high-throughput, low-latency streaming inference scenarios.

To corroborate the theoretical analysis, we evaluate Routing-Free MoE against standard MoE under expert-parallel deployment across varying sequence lengths T , device counts M , and expert activations K for our trained model at scale S in production-representative settings. We isolate the prefill

Table 5: Prefill and autoregressive decode tokens-per-second throughput per device under expert parallelism. Prefill process T =1,024 tokens in a single forward pass; decode generates 128 tokens autoregressively with T =1 per step. K =3 denotes TopK for standard MoE and K eff=3 for RoutingFree MoE. All runs use batch size 1 with input tokens sampled randomly from OpenWebText, bfloat16 precision, and report the mean over 10 repetitions after 3 warmup ones.

| Setting       | T             | M             | K             | Throughput ↑   |
|---------------|---------------|---------------|---------------|----------------|
| Prefill Stage | Prefill Stage | Prefill Stage | Prefill Stage | Prefill Stage  |
| MoE           | 1,024         | 1             | 3             | 22,346.66      |
| MoE           | 1,024         | 2             | 3             | 18,034.84      |
| MoE           | 1,024         | 3             | 3             | 18,558.08      |
| MoE           | 1,024         | 4             | 3             | 18,355.43      |
| RFMoE         | 1,024         | 1             | 3             | 22,277.77      |
| RFMoE         | 1,024         | 2             | 3             | 22,268.24      |
| RFMoE         | 1,024         | 3             | 3             | 21,822.29      |
| RFMoE         | 1,024         | 4             | 3             | 21,784.77      |
| Decode Stage  | Decode Stage  | Decode Stage  | Decode Stage  | Decode Stage   |
| MoE           | 1             | 1             | 3             | 57.90          |
| MoE           | 1             | 2             | 3             | 45.65          |
| MoE           | 1             | 3             | 3             | 45.61          |
| MoE           | 1             | 4             | 3             | 44.67          |
| RFMoE         | 1             | 1             | 3             | 52.14          |
| RFMoE         | 1             | 2             | 3             | 52.05          |
| RFMoE         | 1             | 3             | 3             | 50.11          |
| RFMoE         | 1             | 4             | 3             | 49.84          |

and the decode stages. For each configuration, we measured end-to-end processing time, breaking down contributions from computation, communication, and synchronization overhead.

Table 5 reports throughput under expert parallelism for models at scale S. Standard MoE throughput drops sharply when moving from M =1 to M =2 , with Prefill falls by ∼ 19% , and decode falls by ∼ 21% , with little recovery at M =3 or M =4 . This reflects the blocking All-to-All dispatches and the sequential, non-parallelizable routing pipeline that must complete before any expert computation begins. In contrast, Routing-Free MoE retains ≥ 97 . 8% of its single-device prefill throughput and ≥ 95 . 6% of its decode throughput at M =4 , consistent with the asynchronous scoring and point-topoint communication pattern described above. Although Routing-Free MoE is slightly slower than standard MoE on a single device due to the rankr gating projection applied to all N experts rather than a single D × N router matmul, the crossover occurs already at M =2 , where Routing-Free MoE surpasses MoE by 23% in prefill and 14% in decode throughput. The advantage grows with M . At M =4 , Routing-Free MoE delivers 1 . 19 × higher prefill throughput and 1 . 12 × higher decode throughput. This confirms the theoretical prediction that the communication and synchronization savings of Routing-Free MoE compound with device count, while its computational cost ratio remains strictly below unity and decreases monotonically in M . The decode stage exhibits particularly graceful scaling because the per-step bandwidth term T · D · b/ ( M · B ) is negligible at T =1 , leaving the critical path dominated by startup latency α and sequential compute. Here, Routing-Free MoE's elimination of the centralized routing barrier translates almost entirely into wall-clock savings, highlighting its performance for latency-sensitive autoregressive serving.

## B.2 Threshold Adaptation

The design of global post-activation threshold θ provides a lightweight, inference-time mechanism for trading computation against model quality, beyond its role in controlling overall sparsity during training. The routing-free training dynamics naturally encourage each expert to commit decisively to its activation state, which pushes its internal score either well above or well below θ rather than hovering near the boundary. As a result, moderate perturbations to θ at inference time displace only low-confidence, marginally contributing activations, conferring inherent robustness to threshold miscalibration.

Table 6: Effect of the global threshold θ on downstream benchmark average (Avg.) and estimated FLOPs at scale S. ρ eff denotes the empirical mean activation density at θ . † Averages at θ ≥ 1 . 5 are driven by the anomalous SST-2 and OBQA spikes (see Figure 8).

|   θ |   ρ eff (%) | FLOPs   | Avg. ↑   |
|-----|-------------|---------|----------|
| 0.1 |       100   | 120.46M | 39.41    |
| 0.5 |        94.8 | 118.41M | 39.12    |
| 0.8 |        61.4 | 105.40M | 39.55    |
| 0.9 |        48.9 | 100.32M | 39.73    |
| 1   |        37.8 | 96.21M  | 39.80    |
| 1.1 |        29.3 | 92.97M  | 39.98    |
| 1.2 |        22.9 | 90.37M  | 39.77    |
| 1.5 |        11.1 | 85.80M  | 40.77 †  |
| 1.9 |         4.3 | 83.14M  | 40.23 †  |

<!-- image -->

θ

Figure 8: Per-benchmark accuracy across θ at scale S. HellaSwag, QQP, QNLI are nearly invariant to the threshold, while PIQA, ARC-easy, ARC-challenge and Winogrande achieve their best performance around ρ eff ≈ ρ ∞ . SST-2 and OBQA spikes sharply at larger θ , driving the elevated averages.

Table 6 reports the downstream benchmark average across a sweep of θ at scale S. The most striking observation is the stability of performance despite changes in activation density. Reducing ρ eff from 100% to 4.3%, a more than 20 × reduction in expert activations and a 31% drop in FLOPs, degrades the average score by less than two absolute points. This robustness is a direct consequence of the decisive activation patterns learned during routing-free training. Because most expert scores lie far from the decision boundary, sweeping θ over a wide range affects only a small number of uncertain, low-impact activations. Notably, even at extremely low thresholds where nearly all experts are activated, as seen at ρ eff ≈ 100% with θ =0 . 1 , the model does not suffer from gradient explosion or output instability. Since the activation scores G i ( x ) directly scale each expert's output before aggregation, experts that pass the threshold with low scores contribute proportionally little to the final representation. Lowering θ therefore admits only marginal, low-weight activations rather than introducing equally weighted noise from all experts.

The per-benchmark breakdown in Figure 8 reveals that this aggregate stability is not an artifact of compensating trends. The majority of benchmarks, including HellaSwag, QQP, and QNLI are effectively invariant to θ , confirming that the core representations remain intact even under aggressive sparsification. A small number of tasks exhibit larger sensitivity. SST-2 in particular spikes at θ =1 . 5 , which accounts for most of the elevated aggregate averages at high thresholds. We attribute this to the interaction between task-specific input distributions and the activation patterns of individual experts, rather than a systematic benefit of sparser computation. These results highlight a practical advantage of the routing-free design. The threshold θ serves as a single, transparent knob that allows

Table 7: Paired statistical tests comparing Routing-Free MoE against the standard MoE baseline across 9 benchmarks. p -values are one-sided.

| Scale   | W/L     |   ∆ Avg. |   t -stat |   p t |   Cohen's d |
|---------|---------|----------|-----------|-------|-------------|
| S       | 5 / 4   |     0.8  |     1.481 | 0.089 |       0.494 |
| M       | 6 / 3   |     0.76 |     0.764 | 0.233 |       0.255 |
| L       | 5 / 4   |     0.76 |     1.184 | 0.135 |       0.395 |
| Overall | 16 / 11 |     0.77 |     1.858 | 0.037 |       0.358 |

practitioners to balance accuracy and efficiency at deployment time without retraining, with assurance that moderate adjustments will not disrupt the expert specialization patterns learned during training.

## C Statistical Significance Analysis

To support the claim that Routing-Free MoE consistently outperforms the standard MoE baseline, we conduct a formal statistical analysis over the benchmark results reported in Table 1. Each model variant is evaluated on nine benchmarks across three scales, yielding 9 × 3 = 27 paired observations, where each pair consists of the accuracy score of Routing-Free MoE and MoE on the same benchmark at the same scale.

We treat each (scale, benchmark) pair as a matched observation. For each pair we compute the signed improvement ∆ i = RFMoE i -MoE i in percentage points (pp). We test the one-sided null hypothesis H 0 : µ ∆ ≤ 0 against H 1 : µ ∆ &gt; 0 . We apply a one-sided paired t -test, which evaluates whether the mean improvement is significantly positive under an approximate normality assumption. Effect size is quantified via Cohen's d computed on the paired differences, and win rate is reported as a descriptive statistic.

Table 7 reports per-scale statistics. The mean improvement is positive at every scale (+0.80 pp at S, +0.76 pp at both M and L), and Cohen's d ranges from 0.26 to 0.49, indicating a consistent small-to-medium effect. When observations are pooled across all three scales ( n = 27 ), the one-sided paired t -test yields t = 1 . 858 , p = 0 . 037 , rejecting H 0 at the α = 0 . 05 level. Routing-Free MoE achieves a positive improvement in 16 out of 27 pairs, with a mean gain of +0 . 77 pp and Cohen's d = 0 . 36 . Beyond downstream task accuracy, Routing-Free MoE also achieves consistently lower perplexity than the MoE baseline at every scale, with -12 . 2% at S, -11 . 7% at M, and -18 . 8% at L, suggesting that the quality of language modeling improves consistently with scale under Routing-Free MoE's design, independent of the downstream accuracy results.

Taken together, the statistical analysis supports the conclusion that Routing-Free MoE produces a reliable improvement over the standard MoE baseline. The effect is consistent in direction across all three scales and nine benchmarks, reaches statistical significance when observations are pooled with paired t -test p = 0 . 037 , and is accompanied by a moderate effect size with Cohen's d = 0 . 36 , along with substantial perplexity reduction at every scale.

## D Additional Discussion

## D.1 Load-Balancing

A recent line of work explored auxiliary-loss-free load balancing (DeepSeek-AI et al., 2025; Guo et al., 2025a). Our baselines and Routing-Free MoE all employ auxiliary losses following standard practice, ensuring a controlled comparison across routing architectures. Since auxiliary-loss-free balancing is orthogonal to the routing mechanism itself, any such technique could in principle be applied to standard MoE, AoE, ReMoE, and Routing-Free MoE alike. Introducing it for one method but not others would confound the comparison, while applying it uniformly across all baselines would constitute a substantial engineering effort that lies beyond the scope of this work. We therefore focus on the architectural distinction between routing-based and routing-free expert selection, which is the central contribution of this paper, and leave the integration of auxiliary-loss-free balancing as beyond the scope of this work.

Similarly, since all architectures compared in this work operate under the Token Choice load balancing paradigm, which is the dominant setting in the MoE literature (Muennighoff et al., 2025), we choose not to include Expert Choice routing as a direct baseline. Instead, our configurable µ -interpolation between token-balancing and expert-balancing losses allows Routing-Free MoE to smoothly interpolate the spectrum between token-balancing and expert-balancing behavior within a unified framework. This design choice enables a thorough internal comparison across balancing strategies, as demonstrated in our ablation studies, without introducing the confound of an entirely different assignment paradigm.

## D.2 Per-Layer and Global Density

In Figure 6, without any explicit supervision, the model develops a striking three-stage structure, with early layers showing high but rapidly decreasing activation with large variance, middle layers exhibiting a slow monotonic climb with low variance, and late layers displaying a sharp rise in both activation level and variance. This emergent pattern coincides closely with interpretability findings on the heterogeneous functional roles that layers naturally develop in LLMs (Geva et al., 2021; Gao et al., 2024a; Wendler et al., 2024; Jiao et al., 2024), with early layers converting input from token space to concept space, middle layers processing them, and late layers preparing input-dependent outputs back to token space. We argue that the performance gain from global density constraints stems precisely from removing the per-layer inductive bias. Enforcing identical sparsity at every depth suppresses the compute-hungry layers that naturally benefit from activating more experts, while simultaneously forcing unnecessary activations in layers where sparse representations suffice. Once this bias is lifted, the model is free to self-organize into a more effective, functionally aligned activation structure.

## E Additional Experiment Results

Table 8: Performance of standard MoE, AoE, ReMoE, and Routing-Free MoE across configurations. Each model was trained on OpenWebText (Gokaslan et al., 2019). Entries marked as n/a indicate fields not applicable to the respective architectural design. Results underlined denote experiments with gradient explosion.

| Arch.   | Config.   | r   | L   | D    | D act   | Size    | FLOPs   | λ       | η     | µ   | α      | Loss   | Val. Loss   | PPL   |
|---------|-----------|-----|-----|------|---------|---------|---------|---------|-------|-----|--------|--------|-------------|-------|
| MoE     | Top3/12   | n/a | 12  | 512  | 128     | 92.44M  | 90.93M  | n/a     | n/a   | n/a | 5e - 4 | 3.804  | 3.667       | 39.13 |
| MoE     | Top3/12   | n/a | 12  | 512  | 128     | 92.44M  | 90.93M  | n/a     | n/a   | n/a | 1e - 3 | 3.595  | 3.441       | 31.22 |
| MoE     | Top3/12   | n/a | 12  | 512  | 128     | 92.44M  | 90.93M  | n/a     | n/a   | n/a | 2e - 3 | 5.584  | 5.366       | 214.0 |
| AoE     | Top3/12   | 16  | 12  | 512  | 128     | 93.85M  | 88.57M  | n/a     | n/a   | n/a | 1e - 3 | 3.440  | 3.401       | 30.00 |
| AoE     | Top3/12   | 32  | 12  | 512  | 128     | 95.32M  | 91.08M  | n/a     | n/a   | n/a | 1e - 3 | 3.450  | 3.411       | 30.31 |
| ReMoE   | Top3/12   | n/a | 12  | 512  | 128     | 92.44M  | 90.93M  | 1e - 8  | 0.2   | n/a | 1e - 3 | 3.521  | 3.389       | 29.60 |
| RFMoE   | ρ ∞ =1/4  | 16  | 12  | 512  | 128     | 93.85M  | 88.57M  | 1e - 10 | 0.005 | 0.5 | 5e - 4 | 3.579  | 3.620       | 37.34 |
| RFMoE   | ρ ∞ =1/4  | 16  | 12  | 512  | 128     | 93.85M  | 88.57M  | 1e - 10 | 0.05  | 0.5 | 5e - 4 | 3.552  | 3.600       | 36.60 |
| RFMoE   | ρ ∞ =1/4  | 16  | 12  | 512  | 128     | 93.85M  | 88.57M  | 1e - 10 | 0.08  | 0.5 | 5e - 4 | 3.537  | 3.644       | 38.24 |
| RFMoE   | ρ ∞ =1/4  | 16  | 12  | 512  | 128     | 93.85M  | 88.57M  | 1e - 10 | 0.1   | 0.5 | 5e - 4 | 3.571  | 3.612       | 37.04 |
| RFMoE   | ρ ∞ =1/4  | 16  | 12  | 512  | 128     | 93.85M  | 88.57M  | 1e - 10 | 0.02  | 0.5 | 1e - 3 | 3.309  | 3.358       | 28.73 |
| RFMoE   | ρ ∞ =1/4  | 16  | 12  | 512  | 128     | 93.85M  | 88.57M  | 1e - 10 | 0.02  | 0.5 | 2e - 3 | 3.261  | 3.311       | 27.42 |
| RFMoE   | ρ ∞ =1/4  | 16  | 12  | 512  | 128     | 93.85M  | 88.57M  | 1e - 10 | 0.02  | 0.5 | 5e - 3 | 7.097  | 6.971       | 1065  |
| RFMoE   | ρ ∞ =1/4  | 8   | 12  | 512  | 128     | 93.11M  | 87.32M  | 1e - 10 | 0.02  | 0.5 | 1e - 3 | 3.332  | 3.373       | 29.16 |
| RFMoE   | ρ ∞ =1/4  | 16  | 12  | 512  | 128     | 93.85M  | 88.57M  | 1e - 10 | 0.02  | 0.5 | 1e - 3 | 3.309  | 3.358       | 28.74 |
| RFMoE   | ρ ∞ =1/4  | 32  | 12  | 512  | 128     | 95.32M  | 91.08M  | 1e - 10 | 0.02  | 0.5 | 1e - 3 | 3.290  | 3.344       | 28.34 |
| RFMoE   | ρ ∞ =1/4  | 64  | 12  | 512  | 128     | 98.27M  | 96.09M  | 1e - 10 | 0.02  | 0.5 | 1e - 3 | 3.290  | 3.341       | 28.24 |
| RFMoE   | ρ ∞ =1/4  | 8   | 12  | 512  | 128     | 93.11M  | 87.32M  | 1e - 10 | 0.02  | 0.5 | 2e - 3 | 3.717  | 3.699       | 40.41 |
| RFMoE   | ρ ∞ =1/4  | 16  | 12  | 512  | 128     | 93.85M  | 88.57M  | 1e - 10 | 0.02  | 0.5 | 2e - 3 | 3.311  | 3.261       | 26.08 |
| RFMoE   | ρ ∞ =1/4  | 32  | 12  | 512  | 128     | 95.32M  | 91.08M  | 1e - 10 | 0.02  | 0.5 | 2e - 3 | 6.435  | 6.564       | 709.1 |
| RFMoE   | ρ ∞ =1/4  | 64  | 12  | 512  | 128     | 98.27M  | 96.09M  | 1e - 10 | 0.02  | 0.5 | 2e - 3 | 3.548  | 3.516       | 33.65 |
| RFMoE   | ρ ∞ =1/4  | 32  | 12  | 512  | 128     | 95.32M  | 91.08M  | 1e - 10 | 0.02  | 0.0 | 1e - 3 | 3.481  | 3.347       | 28.41 |
| RFMoE   | ρ ∞ =1/4  | 32  | 12  | 512  | 128     | 95.32M  | 91.08M  | 1e - 10 | 0.02  | 0.2 | 1e - 3 | 3.483  | 3.345       | 28.35 |
| RFMoE   | ρ ∞ =1/4  | 32  | 12  | 512  | 128     | 95.32M  | 91.08M  | 1e - 10 | 0.02  | 0.8 | 1e - 3 | 3.488  | 3.346       | 28.38 |
| RFMoE   | ρ ∞ =1/4  | 32  | 12  | 512  | 128     | 95.32M  | 91.08M  | 1e - 10 | 0.02  | 1.0 | 1e - 3 | 3.485  | 3.347       | 28.43 |
| MoE     | Top4/16   | n/a | 24  | 768  | 192     | 289.92M | 247.95M | n/a     | n/a   | n/a | 5e - 4 | 3.808  | 3.300       | 27.11 |
| MoE     | Top4/16   | n/a | 24  | 768  | 192     | 289.92M | 247.95M | n/a     | n/a   | n/a | 1e - 3 | 3.726  | 3.219       | 25.00 |
| RFMoE   | ρ ∞ =1/4  | 48  | 24  | 768  | 192     | 307.30M | 249.17M | 1e - 10 | 0.02  | 0.5 | 5e - 4 | 3.360  | 3.292       | 26.90 |
| RFMoE   | ρ ∞ =1/4  | 48  | 24  | 768  | 192     | 307.30M | 249.17M | 1e - 10 | 0.02  | 0.5 | 1e - 3 | 3.166  | 3.095       | 22.08 |
| MoE     | Top6/24   | n/a | 32  | 1024 | 256     | 808.42M | 608.38M | n/a     | n/a   | n/a | 5e - 4 | 3.947  | 3.202       | 24.58 |
| MoE     | Top6/24   | n/a | 32  | 1024 | 256     | 808.42M | 608.38M | n/a     | n/a   | n/a | 8e - 4 | 4.493  | 3.768       | 43.29 |
|         |           |     |     |      |         |         | 613.19M | 1e - 10 | 0.02  | 0.5 | 5e - 4 | 3.179  | 3.107       |       |
| RFMoE   | ρ ∞ =1/4  | 64  | 32  | 1024 | 256     | 870.60M |         |         |       |     |        |        |             | 22.36 |
| RFMoE   | ρ ∞ =1/4  | 64  | 32  | 1024 | 256     | 870.60M | 613.19M | 1e - 10 | 0.02  | 0.5 | 8e - 4 | 3.076  | 2.994       | 19.97 |
| RFMoE   | ρ ∞ =1/4  | 64  | 32  | 1024 | 256     | 870.60M | 613.19M | 1e - 10 | 0.02  | 0.5 | 1e - 3 | 5.448  | 3.548       | 34.73 |