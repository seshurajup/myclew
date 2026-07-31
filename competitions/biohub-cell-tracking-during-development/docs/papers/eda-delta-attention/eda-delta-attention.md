> **Source:** `source.pdf` · 14 pages · 1 figures · 80 display equations · 4 tables · converted by fleet `paper-md` (backend=**docling+pymupdf-assets**)

<!-- image -->

## Erase-then-Delta Attention:

## Decoupling Erase and Write Addresses in Delta-Rule Linear Attention

Xiao Li 1,2 , Chengruidong Zhang 1 , Hao Luo 1 , Xi Lin 1,3 , Zekun Wang 1 , Zihan Qiu 1 , Yunfei Mao 1 , Langshi Chen 1 , Man Yuan 1 , Minmin Sun 1 , Huiqiang Jiang 1 , Siqi Zhang 1 , Rui Men 1 , Wei Hu 2 , Gong Cheng 2 , Bo Zheng 1† , Dayiheng Liu 1† , Jingren Zhou 1 1 Qwen Team 2 Nanjing University 3 Zhejiang University † Corresponding authors.

## Abstract

Delta-rule linear attention improves recurrent memory updates by correcting what is already stored at the current write address before writing new content. However, the active correction is still anchored to that same write address. As a result, stale information stored at a different address cannot be actively removed before new content is written elsewhere. We propose Erase-then-Delta Attention (EDA), a memory update rule that decouples where to erase from where to write. The key insight is that recurrent memory models should not only correct the current write, but also selectively suppress outdated memory at an independently chosen address. Concretely, our method first applies a targeted erase step along a learned erase direction, and then performs the standard delta-style corrective write along the current write direction. This preserves the corrective behavior of delta-rule updates while expanding their memory-management capacity. Language-model pretraining experiments across dense 2.5B and MoE 25BA2.8B model families show that EDA performs best in both settings. The gain persists after 80B-token long-context midtraining of the MoE models, where EDA also performs best in long-context evaluations from 4k to 128k contexts. A compact update analysis and memory-state probes suggest why: EDA keeps the delta-rule corrective write intact while allocating an additional cleanup path most strongly when passive decay is weak. These results suggest that recurrent memory models should decide not only what to write, but also what stale information to erase and where.

## 1 Introduction

Autoregressive Transformers (Vaswani et al., 2017) have become the foundation of modern language modeling, in part because softmax-based self-attention enables efficient parallel computation. This mechanism achieves strong performance on in-context learning and long-context retrieval by maintaining an explicit key-value cache. However, it also introduces fundamental bottlenecks at inference time: quadratic time complexity and linearly growing memory overhead that limit scalability for long-sequence tasks and agentic reasoning trajectories. To address these constraints, a growing body of work has explored efficient alternatives that maintain constant memory and O ( 1 ) inference time while preserving the expressive power of attention.

Recurrent models based on linear attention (Katharopoulos et al., 2020) and state space models (Gu et al., 2022; Gu &amp; Dao, 2024) offer a principled solution: they compress contextual information into a fixed-size state, enabling constant memory and linear-time training. Early variants such as Linformer (Wang et al., 2020) and RetNet (Sun et al., 2023) lacked data-dependent memory control and underperformed softmax attention. Subsequent models introduced dynamic gating mechanisms (Yang et al., 2025; Dao &amp; Gu, 2024; Beck et al., 2024), allowing selective forgetting and significantly narrowing the performance gap. However, additive gated updates still write new content into a finite state without explicitly correcting the association currently stored at the write address.

Amore recent line of work replaces additive updates with the delta rule (Schlag et al., 2021), which treats the recurrent state as a learnable associative memory that corrects itself toward the current key-value mapping. Gated DeltaNet (GDN) (Yang et al., 2025) combines this corrective write with a head-wise forget gate, and recent channel-wise variants further refine this gate into a diagonal decay that gives each key feature its own retention rate (Team et al., 2025). GDN-2 further separates the scalar delta gate into key-side erase and value-side write gates, but the active edit remains organized around the current write key (Hatamizadeh et al., 2026). We build on this channel-wise gated delta setting, also known as diagonal-plus-low-rank (DPLR), which combines GDN's hardware-efficient delta-rule structure with finer-grained channel-wise forgetting. Despite this progress, a structural limitation remains unaddressed:

the active delta correction still uses the current write direction k t as its only address. This coupling means the model can only suppress memory at the address it is currently writing to; stale information stored elsewhere must either persist or decay through channel-wise but address-agnostic forgetting.

This limitation has tangible consequences. In language modeling and state-tracking tasks, useful memory updates require not only writing new content but also removing obsolete information that would otherwise interfere with future reads and writes. When the model encounters a situation where earlier information must be invalidated-for example, a variable reassignment, a fact correction, or a context shift-it has no direct mechanism to remove the old content before committing the new one. The core missing capability is therefore not stronger forgetting, but targeted deletion of outdated memory at an address chosen independently of the current write .

We address this problem with Erase-then-Delta Attention (EDA), a memory update rule that decouples erasure from writing. Instead of tying memory suppression to the current write address, EDA first removes stale content at an independently selected address and then performs the usual delta-style corrective write at the current write address. Intuitively, the erase step actively clears obsolete memory, while the delta step preserves the corrective writing behavior that makes delta-rule models effective. This yields a strictly richer update rule: the model can erase at one address and write at another within the same recurrent step.

We show that this simple modification has three important consequences. First, it provides a cleaner memory-management view of channel-wise gated delta recurrence by separating diagonal decay, independently addressed erasure, and write-coupled correction. Second, empirical analysis reveals that the model learns a near-orthogonal separation between erase and write addressing, indicating that the two operations serve genuinely different roles. Third, language-model pretraining experiments show that EDA improves over a DPLR-style gated delta baseline and compares favorably with several strong update-rule variants.

In summary, we introduce EDA, a gated delta-rule linear-attention update that decouples erase and write addresses while preserving the standard delta corrective write. We analyze the resulting erase-then-delta update and evaluate it through language-model pretraining, long-context evaluation, and memory-state probes, showing that the extra address acts as a conditional cleanup path rather than merely stronger forgetting.

## 2 Preliminary

We briefly introduce the recurrent memory notation and the channel-wise gated delta update most relevant to our method. The key point is that a diagonal forget gate already provides fine-grained decay, but the active correction and writing remain tied to the same address.

## 2.1 Notation and Linear Associative Memory

We consider a recurrent memory state S t ∈ R d k × dv updated at each step t . The key k t ∈ R d k serves as a write address, the value v t ∈ R dv is the content to store, and the query q t ∈ R d k reads from memory through S ⊤ t q t ∈ R dv .

Standard linear attention updates memory additively:

$$S _ { t } = S _ { t - 1 } + k _ { t } v _ { t } ^ { \top } , \quad o _ { t } = S _ { t } ^ { \top } q _ { t } .$$

This rule is efficient but does not explicitly decide what stale information to suppress.

## 2.2 Coupled Erasure and Corrective Writing

DeltaNet (Schlag et al., 2021; Yang et al., 2024) replaces additive writing with a corrective update derived from the reconstruction loss

$$\mathcal { L } _ { t } ^ { \text {delta} } ( S ) & = \frac { 1 } { 2 } \| S ^ { \top } k _ { t } - v _ { t } \| ^ { 2 } . \\$$

Taking a gradient step with learning rate β t gives

$$S _ { t } = ( \mathbf I - \beta _ { t } k _ { t } k _ { t } ^ { \top } ) S _ { t - 1 } + \beta _ { t } k _ { t } v _ { t } ^ { \top } .$$

Rather than simply accumulating k t v ⊤ t , DeltaNet first corrects what memory currently returns at address k t and then writes the new content at that same address.

Gated DeltaNet (GDN) (Yang et al., 2025) augments this rule with a head-wise scalar forget gate α t ∈ ( 0, 1 ) :

$$S _ { t } = \alpha _ { t } ( \mathbf I - \beta _ { t } k _ { t } k _ { t } ^ { \top } ) S _ { t - 1 } + \beta _ { t } k _ { t } v _ { t } ^ { \top } .$$

Here α t provides uniform decay within a head, while ( I -β t k t k ⊤ t ) provides address-specific correction. However, the erase-and-write behavior is still coupled: the same key k t determines both where memory is strongly modified and where new content is written. As a result, GDN can strongly suppress only the address it is currently writing to.

Following Kimi Delta Attention (KDA) (Team et al., 2025), we use the channel-wise version of this GDN design, replacing the head-wise scalar forget gate with a diagonal decay D t = Diag ( α t ) :

$$S _ { t } = ( \mathbf I - \beta _ { t } k _ { t } k _ { t } ^ { \top } ) D _ { t } S _ { t - 1 } + \beta _ { t } k _ { t } v _ { t } ^ { \top } .$$

The diagonal gate gives each key channel its own retention rate and makes the transition compatible with a diagonal-plus-low-rank view. This improves how strongly different channels are preserved or decayed, but it does not change the addressing structure of the delta update itself: the corrective modification is still anchored to the current write key. Therefore, even with channel-wise gating, stale information stored at a different address cannot be explicitly erased before writing new content elsewhere.

GDN-2 addresses a closely related coupling by separating the scalar delta gate into key-side erase and value-side write gates (Hatamizadeh et al., 2026):

$$S _ { t } = \left ( I - k _ { t } \widetilde { e } _ { t } ^ { \top } \right ) D _ { t } S _ { t - 1 } + k _ { t } z _ { t } ^ { \top } , \quad \widetilde { e } _ { t } = b _ { t } \odot k _ { t } , \ \ z _ { t } = w _ { t } \odot v _ { t } .$$

This decouples the channel-wise erase and write strengths inside the delta residual. However, the erase/read direction ˜ e t is still constructed from the current write key k t , and the correction is still committed along k t . Thus GDN-2 relaxes the gate-level coupling, while the address-level coupling between erasure and writing remains.

This coupling is the limitation we target. If stale information is stored at an address different from the current write address, the diagonal gate can decay feature channels but cannot selectively remove that stale association before writing elsewhere.

## 2.3 Relation to Recent Delta-Style Variants

Recent linear-recurrent models often improve performance by enriching the transition rule or embedding delta-style memory updates inside stronger architectures. DeltaProduct (Siems et al., 2026) increases transition expressivity through multiple Householder-like factors per step, while RWKV-7 (Peng et al., 2025) and Comba (Hu et al., 2026) adopt richer structured transition parameterizations. Recent hybrid architectures further demonstrate that strong designs built around expressive channel-wise gated delta components can be highly competitive with full attention (Team et al., 2025).

Our goal is different. We do not primarily seek a globally richer transition; instead, we introduce a missing memory-management capability: erasing stale memory at one address before performing the standard delta-style corrective write at another. In that sense, our method is best viewed as orthogonal to transition-enrichment approaches and potentially compatible with stronger channel-wise gated delta backbones.

## 3 Method

## 3.1 Overview

Our goal is to extend gated delta-rule linear attention with a missing memory-management capability: selectively deleting stale memory at an address different from the current write address. To do this, we revisit the DPLR-style update rule and identify a structural coupling between active correction and writing. We then introduce Erase-then-Delta Attention (EDA), a sequential update rule that adds an independently addressed erase step before the standard delta-style corrective write. This section first formalizes the limitation of the decay-gated delta baseline, then derives the new rule, and finally discusses its algebraic structure and stability properties.

## 3.2 Erase-Write Coupling in Gated Delta Updates

We consider a recurrent memory state S t updated by a gated delta rule with diagonal decay:

$$S _ { t } = ( \mathbf I - \beta _ { t } k _ { t } k _ { t } ^ { \top } ) D _ { t } S _ { t - 1 } + \beta _ { t } k _ { t } v _ { t } ^ { \top } , \quad D _ { t } = D i a g ( \alpha _ { t } ) .$$

Here D t is a diagonal decay matrix with retention factors α t , β t controls the delta-style correction strength, k t is the current write direction, and v t is the value vector written into memory. This update is effective because it is not a naive additive write: after diagonal decay, the factor ( I -β t k t k ⊤ t ) corrects the memory response along the current write direction, and the additive term β t k t v ⊤ t writes the new content at the same address.

Equation (7) already contains both fine-grained decay and address-specific correction. The diagonal gate D t decides which key channels persist, while the rank-1 term ( I -β t k t k ⊤ t ) induces stronger correction along the current write direction. GDN-2 relaxes the scalar-gate version of this coupling by separating key-side erase and value-side write gates (Hatamizadeh et al., 2026). However, the active correction remains structurally coupled to writing: the edit is still constructed from the current write key, and the correction is still committed along k t . Consequently, these updates can only strongly suppress memory through the address they are currently writing to.

This coupling is the core limitation we address. If stale information is stored at an address different from the current write direction, the model has no direct mechanism to remove it selectively before performing the current write. Instead, it must rely on the decay gate D t , which is not tied to a specific stale address, or wait until future writes happen to revisit that address. Our central design question is therefore: can a delta-rule memory model erase at one address and write at another within the same recurrent step?

## 3.3 Erase-then-Delta: Decoupled Erase-Write Addressing

EDA decouples cleanup from writing by inserting an independently addressed erase operator before the standard delta write:

$$S _ { t } = ( I - \beta _ { t } k _ { t } k _ { t } ^ { \top } ) ( I - \gamma _ { t } e _ { t } e _ { t } ^ { \top } ) D _ { t } S _ { t - 1 } + \beta _ { t } k _ { t } v _ { t } ^ { \top } .$$

The factors in Eq. (8) are applied from right to left. The diagonal decay D t first attenuates retained key coordinates, the erase factor ( I -γ t e t e ⊤ t ) contracts the decayed memory along a learned cleanup address e t , and the usual delta factor then performs corrective forgetting and writing at the current write key k t . This order is part of the update rule: for diagonal decay, D t generally does not commute with the rank-1 erase operator unless D t degenerates to a scalar decay or e t lies in an equal-decay subspace.

To see what the new operator actually erases, let

$$\hat { S } _ { t } = D _ { t } S _ { t - 1 }$$

denote the memory after diagonal decay and before address-selective cleanup. EDA defines the erase address through the online objective

$$\mathcal { L } _ { t } ^ { e r a s e } ( \widehat { S } _ { t } ) = \frac { 1 } { 2 } \| \widehat { S } _ { t } ^ { \top } e _ { t } \| ^ { 2 } .$$

This objective penalizes the content currently returned when the decayed memory is queried at e t . A gradient step with learning rate γ t gives

$$\widetilde { S } _ { t } = ( I - \gamma _ { t } e _ { t } e _ { t } ^ { \top } ) \widehat { S } _ { t } ,$$

where e t is L2-normalized. Thus e t is not merely an extra projection: it is the address whose current memory response is explicitly pushed toward zero.

This readout-level view clarifies why the new direction is more targeted than stronger decay. For any query direction q , the erased memory reads out as

$$\widetilde { S } _ { t } ^ { \top } q = \widehat { S } _ { t } ^ { \top } q - \gamma _ { t } ( q ^ { \top } e _ { t } ) \widehat { S } _ { t } ^ { \top } e _ { t } .$$

When q = e t , Eq. (12) suppresses the response at the erase address by a factor of 1 -γ t . When q is orthogonal to e t , the erase step leaves that readout unchanged before the later delta update. The decay gate D t controls retention by key coordinate; in contrast, Eq. (12) subtracts the content currently returned at a learned memory address, scaled by how much the query aligns with that address.

After this cleanup, EDA applies the standard delta-style corrective write to the erased memory:

$$S _ { t } = ( \mathbf I - \beta _ { t } k _ { t } k _ { t } ^ { \top } ) \widetilde { S } _ { t } + \beta _ { t } k _ { t } v _ { t } ^ { \top } .$$

Substituting Eq. (11) into Eq. (13) recovers Eq. (8). The delta correction and write at k t are therefore unchanged; the new degree of freedom is that stale memory can be suppressed at e t before new content is written at k t . If e t collapses to k t , EDA reduces to a stronger same-address correction; when the two directions differ, cleanup and writing are no longer forced to use the same address. This also distinguishes EDA from gate-level erase/write separation, where the residual can be reweighted by gates but remains organized around the current write key.

The resulting rule separates memory management into three levels of specificity: diagonal decay through D t , independent directional erasure through γ t e t , and write-coupled correction through β t k t . In this sense, EDA adds the missing degree of freedom needed to suppress stale memory at one address before performing a corrective write at another. Figure 1 illustrates the full EDA layer architecture.

Figure 1: Architecture of an EDA layer. The input is projected into query, key, value, output gate, erase gate ( γ ), delta gate ( β ), decay parameters ( α ), and erase address ( e ). The query, key, and erase address are L2-normalized; the erase address uses a low-rank projection. All signals feed into the EDA kernel, whose output is normalized and gated before a final linear projection.

<!-- image -->

Safe gate for bounded decay. The diagonal decay in Eq. (8) is parameterized in log space. Let D t = Diag ( exp ( g t )) , A = exp ( A log ) &gt; 0, u t = a t + b ∆ , and ∆ t = softplus ( u t ) , where a t is the decay projection and b ∆ is a learned bias. The Mamba2/GDN-style log-space gate (Dao &amp; Gu, 2024; Yang et al., 2025) uses

$$g _ { t } ^ { \log } = - A \odot \Delta _ { t } ,$$

which guarantees exp ( g t ) ≤ 1 but leaves the log-decay unbounded below. KDA computes its safe gate as g KDA t = ℓ σ ( A ⊙ u t ) with ℓ &lt; 0, mapping each log-decay coordinate into ( ℓ , 0 ) (Team et al., 2025). EDA instead uses a bounded safe gate with the same lower log-decay limit and maximum value 0:

$$g _ { t } = \ell + ( - \ell ) \exp \left ( - \frac { A } { | \ell | } \odot \Delta _ { t } \right ) ,$$

where the exponential is applied elementwise. Since exp ( -x ) ∈ ( 0, 1 ] for x ≥ 0, this parameterization keeps g t ∈ ( ℓ , 0 ] and therefore bounds each decay coordinate by exp ( ℓ ) &lt; α t , i ≤ 1.

Comparing Eq. (14) and Eq. (15) shows why this bounded form is useful beyond numerical clipping. The Mamba2/GDN-style log-space gate separates two roles: A controls the decay magnitude, while ∆ t = softplus ( u t ) acts as a ReLU-like nonnegative switch for whether a coordinate should decay. Our safe gate preserves this amplitude-switch decomposition near the active region: elementwise, a Taylor expansion around ∆ t , i = 0 gives gt , i = -Ai ∆ t , i + O ( A 2 i ∆ 2 t , i / | ℓ | ) . It therefore behaves like the log-space gate for small decay inputs, but saturates for large inputs instead of driving the log-decay toward -∞ . By contrast, the KDA sigmoid gate bounds the log-decay by applying a sigmoid directly to the affine decay signal, so A mainly changes the sigmoid slope and saturation rather than acting as a separate decay-amplitude parameter. In practice we set ℓ = -5, making the smallest per-step decay factor exp ( ℓ ) ≈ 6.7 × 10 -3 , well within the normal range of half-precision formats. This prevents decay factors from becoming subnormal or zero, allowing decay-weighted chunk tensors to remain in half precision and preserving Tensor-Core-friendly dense matrix multiplications.

Cross-term structure and update order. The order in Eq. (8)-erase first, then delta-is essential. Expanding the product of the two rank-1 operators reveals why:

$$( I - \beta _ { t } k _ { t } k _ { t } ^ { \top } ) ( I - \gamma _ { t } e _ { t } e _ { t } ^ { \top } ) = I - \gamma _ { t } e _ { t } e _ { t } ^ { \top } - \beta _ { t } k _ { t } k _ { t } ^ { \top } + \gamma _ { t } \beta _ { t } ( k _ { t } ^ { \top } e _ { t } ) \, k _ { t } e _ { t } ^ { \top } .$$

The final term-the cross-term-is proportional to the cosine similarity ct = e ⊤ t k t between the erase and write directions. It quantifies the 'leakage' that occurs when the two directions are not orthogonal:

the erase operation can influence the subsequent write-address correction through k t e ⊤ t . Reversing the order (delta first, then erase) would apply the erase operator after the write, allowing memory cleanup to suppress newly written content. By applying erasure first, our rule ensures that cleanup acts on old content before the new corrective write is committed.

Whenthe model learns a near-orthogonal separation between e t and k t (mean | ct | ≈ 0.105, see Figure 2(c)), the cross-term becomes small and the update is well-approximated by two independent corrections acting on orthogonal subspaces. In this regime, the sequential rule is stable and the first-order effects dominate.

## 3.4 EDA with Chunk-wise Parallel

Referring to Eq. (8), the EDA state is multiplied by two rank-1 correction factors per step. To reuse existing DPLR chunk-wise kernels, we interleave the erase and delta sub-steps into a doubled sequence of length 2 t . Let

$$[ q _ { \tau } ^ { \prime } , k _ { \tau } ^ { \prime } , v _ { \tau } ^ { \prime } , \beta _ { \tau } ^ { \prime } , \alpha _ { \tau } ^ { \prime } ] = \begin{cases} [ 0 , e _ { t } , 0 , \gamma _ { t } , \alpha _ { t } ] , & \tau = 2 t - 1 \\ [ q _ { t } , k _ { t } , v _ { t } , \beta _ { t } , 1 ] , & \tau = 2 t \end{cases}$$

Each original step t maps to two sub-steps in the doubled sequence: the odd sub-step applies the erase operator with decay, and the even sub-step applies the delta correction with identity decay. This reduces EDA to a standard DPLR recurrence over twice as many steps. We can rewrite Eq. (8) as:

$$S _ { t } & = ( I - \beta _ { t } | k _ { t } ^ { \prime } | ) ( I - \gamma _ { t } e _ { t } \bar { t } ^ { \top } ) D _ { t } S _ { t - 1 } + \beta _ { t } k _ { t } v _ { t } ^ { \top } \\ & = ( I - \beta _ { t } | k _ { t } ^ { \top } | ) \left ( ( I - \gamma _ { t } e _ { t } \bar { t } ^ { \top } ) D _ { t } S _ { t - 1 } + 0 \right ) + \beta _ { t } | k _ { t } v _ { t } ^ { \top } \\ & = ( I - \beta _ { 2 t } ^ { \prime } k _ { 2 t } ^ { \prime } k _ { 2 t } ^ { \prime \top } ) D _ { 2 t } ^ { \prime } \left ( ( I - \beta _ { 2 t - 1 } ^ { \prime } k _ { 2 t - 1 } ^ { \prime } k _ { 2 t - 1 } ^ { \prime \top } ) D _ { 2 t - 1 } ^ { \prime } S _ { t - 1 } + \beta _ { 2 t - 1 } ^ { \prime } k _ { 2 t - 1 } ^ { \prime } v _ { 2 t - 1 } ^ { \top } \right ) + \beta _ { 2 } ^ { \prime } k _ { 2 t } ^ { \prime } v _ { 2 t } ^ { \prime \top }$$

By partially expanding the recurrence for Eq. (18) into a chunk-wise formulation, we have:

$$S _ { t } = \underbrace { \left ( \prod _ { i = 1 } ^ { 2 t } \left ( I - \beta _ { i } ^ { \prime } k _ { i } ^ { \prime } k _ { i } ^ { T } \right ) D _ { i } ^ { \prime } \right ) } _ { \colon = \text {P} } S _ { 0 } + \underbrace { \left ( \prod _ { i = 1 } ^ { 2 t } \left ( I - \beta _ { j } ^ { \prime } k _ { j } ^ { \prime } k _ { j } ^ { T } \right ) D _ { j } ^ { \prime } \right ) \beta _ { i } ^ { \prime } k _ { i } ^ { \prime } v _ { i } ^ { T } } _ { \colon = \text {H} }$$

Following the chunk-wise algorithm of KDA (Team et al., 2025), we apply WYrepresentation to pack a series of updates into a single compact representation:

$$updates into a single compact representation: \\ w _ { 2 t } = \beta _ { 2 t } ^ { \prime } \left ( \underbrace { \left ( \prod _ { i = 1 } ^ { 2 t } D _ { i } ^ { \prime } \right ) k _ { 2 t } ^ { \prime } - \sum _ { i = 1 } ^ { 2 t - 1 } w _ { i } } _ { \colon = D _ { i } ^ { \prime } \rightarrow 2 t } \left ( k _ { i } ^ { \prime } \underbrace { \left ( \prod _ { j = i } ^ { 2 t } D _ { j } ^ { \prime } \right ) k _ { 2 t } ^ { \prime } } _ { \colon = D _ { i } ^ { \prime } \rightarrow 2 t } } \right ) \right ) \\ u _ { 2 t } = \beta _ { 2 t } \left ( v _ { 2 t } - \sum _ { i = 1 } ^ { 2 t - 1 } u _ { i } \left ( k _ { i } ^ { \prime } D _ { i \rightarrow 2 t } k _ { 2 t } \right ) \right ) \\ P = D _ { 1 \rightarrow 2 t } ^ { \prime } - \sum _ { i = 1 } ^ { 2 t } D _ { i \rightarrow 2 t } ^ { \prime } k _ { i } ^ { \prime } w _ { i } ^ { T } \\ H = \sum _ { i = 1 } ^ { 2 t } D _ { i \rightarrow 2 t } ^ { \prime } k _ { i } ^ { \prime } u _ { i } ^ { T } \\ \intertext { transform } to reduce non-matmul FLOPs }$$

And UT transform to reduce non-matmul FLOPs:

$$A _ { 1 \to 2 t } & = [ \, \diag { d } ( D _ { 1 \to 1 } ^ { \prime } ) \, | \, \diag { d } ( D _ { 1 \to 2 } ^ { \prime } ) \, | \, \cdots \, | \, \diag { d } ( D _ { 1 \to 2 t } ^ { \prime } ) \, ] \\ A _ { i \to 2 t } & = [ \, \diag { d } ( D _ { 1 \to 2 t } ^ { \prime } ) \, | \, \diag { d } ( D _ { 2 \to 2 t } ^ { \prime } ) \, | \, \cdots \, | \, \diag { d } ( D _ { 2 t \to 2 t } ^ { \prime } ) \, ] \\ & \quad M = \left ( I + \text {StrictTril} \left ( \, \diag { D } ( \beta ^ { \prime } ) \, ( \mathbf A _ { 1 \to 2 t } \, \mathbf K ^ { \prime } ) \left ( \frac { K ^ { \prime } } { \mathbf A _ { 1 \to 2 t } } \right ) \right ) \right ) \\ & \quad W = M \left ( \mathbf A _ { 1 \to 2 t } \, \mathbf K ^ { \prime } \right ) \\ & \quad U = M V ^ { \prime }$$

Finally, the state and output can be computed in a chunk-wise manner using the matrix form:

$$S _ { t } & = D _ { 1 \to 2 t } ^ { \prime } S _ { 0 } + ( \mathbf A _ { i \to 2 t } \odot \mathbf K ^ { \prime } ) ^ { T } \left ( U - \mathbf W _ { 0 } \right ) \\ O & = \left ( \mathbf A _ { 1 \to 2 t } \odot \mathbf Q ^ { \prime } \right ) S _ { 0 } + \text {Tril} \left ( \left ( \mathbf A _ { 1 \to 2 t } \odot \mathbf Q ^ { \prime } \right ) \left ( \frac { \mathbf K ^ { \prime } } { \mathbf A _ { 1 \to 2 t } } \right ) ^ { T } \right ) \left ( U - \mathbf W _ { 0 } \right )$$

This formulation reduces EDA's two-factor update to the standard DPLR chunk-wise recurrence.

## 3.5 Efficiency Analysis

The chunk-wise parallel formulation above increases the per-chunk sequence length, which raises the compute workload during prefill. However, the only additional inputs to the kernel are the erase address e and the scalar gate γ , so the increase in HBM traffic remains modest after kernel fusion. Since the chunk-forward pass of channel-wise gated delta models is inherently memory-bound, the wall-clock overhead remains moderate in practice. During autoregressive decoding the effect is smaller still, as the dominant cost is reading and writing the recurrent state rather than computing the rank-1 updates. Moreover, linear-attention layers typically account for a minor fraction of end-to-end model latency, further limiting the overall impact. Optimized kernel implementations will be released at https://github.com/QwenLM/FlashQLA .

## 4 Experiments

## 4.1 Experimental Setup

We evaluate EDA under two matched pretraining scales: a dense 2.5B model family and a larger MoE 25B-A2.8B family. The goal is to test whether the proposed erase-then-delta update improves the recurrent component in both a standard dense setting and a sparse-activated large-model setting. Within each scale, the compared models share the same training setup; detailed architecture hyperparameters and parameter counts are listed in Appendix A.

Compared models. For the dense comparison, we compare a full-attention Transformer baseline with GDN, GDN-2, KDA, and EDA. For the MoE comparison, we compare GDN, KDA, and EDA under the same sparse-activation backbone. Except for the Transformer baseline, all compared linear attention models are hybrid architectures with three linear-attention layers followed by one full-attention Transformer layer, corresponding to a 3:1 linear-to-full attention ratio. This ratio is not tuned specifically for EDA; it follows the common hybrid configuration used in Qwen3.5-style and Kimi Linear architectures (Team, 2025; Team et al., 2025).

Training setup. All models were pretrained for 400B tokens with sequence length 4096 and global batch size 1024. The dense models used a learning rate decayed from 4 × 10 -3 to 3 × 10 -5 , while the MoE 25B-A2.8B models used a learning rate decayed from 2 × 10 -3 to 3 × 10 -5 . We additionally report MoE checkpoints after an 80B-token midtraining stage initialized from the 400B-token pretrained MoE checkpoints. The midtraining stage used sequence length 32k.

Evaluation setup. For downstream evaluation, we report MMLU, MMLU-Pro, GSM8K, MATH, BBH, and EvalPlus. Unless otherwise stated, entries are percentages averaged over two evaluation runs of the same checkpoint, and the Avg. column denotes the unweighted mean over the displayed benchmarks. Brief descriptions of the downstream benchmarks are provided in Appendix B.

Table 1: Evaluation results after 400B-token pretraining. Values are percentages averaged over two evaluation runs; Avg. is the unweighted mean over the six benchmark columns. Within each model family, best results are bold and second-best results are underlined.

| Model         | MMLU   | MMLU-Pro   | GSM8K   | MATH   | BBH   | EvalPlus   | Avg.   |
|---------------|--------|------------|---------|--------|-------|------------|--------|
| Dense 2.5B    |        |            |         |        |       |            |        |
| Transformer   | 50.11  | 18.11      | 20.26   | 12.16  | 35.01 | 30.82      | 27.75  |
| GDN           | 49.99  | 15.76      | 20.83   | 12.85  | 34.99 | 31.08      | 27.58  |
| GDN-2         | 49.90  | 15.84      | 21.40   | 13.56  | 35.18 | 32.95      | 28.14  |
| KDA           | 50.03  | 16.37      | 21.30   | 13.09  | 35.31 | 30.73      | 27.81  |
| EDA           | 49.27  | 16.04      | 23.90   | 13.54  | 35.42 | 32.50      | 28.44  |
| MoE 25B-A2.8B |        |            |         |        |       |            |        |
| GDN           | 64.75  | 31.46      | 58.89   | 31.87  | 51.88 | 50.09      | 48.16  |
| KDA           | 63.84  | 33.14      | 58.61   | 30.78  | 51.87 | 51.60      | 48.31  |
| EDA           | 65.31  | 33.61      | 57.71   | 33.57  | 52.72 | 53.37      | 49.38  |

Table 2: Evaluation results for MoE 25B-A2.8B checkpoints after 400B-token pretraining followed by 80B-token midtraining at 32k sequence length. Values are percentages averaged over two evaluation runs; Avg. is the unweighted mean over the six benchmark columns. Best results are bold and second-best results are underlined.

| Model   |   MMLU |   MMLU-Pro |   GSM8K |   MATH |   BBH |   EvalPlus |   Avg. |
|---------|--------|------------|---------|--------|-------|------------|--------|
| GDN     |  67.43 |      40.55 |   75.93 |  45.94 | 64.25 |      50.09 |  57.37 |
| KDA     |  67.32 |      40.6  |   76.21 |  46.87 | 65.04 |      53.82 |  58.31 |
| EDA     |  68.12 |      41.71 |   75.99 |  49.28 | 65.81 |      51.78 |  58.45 |

## 4.2 Model Results

At the dense 2.5B scale, EDA achieves the strongest average score among all dense models. Compared with KDA, which shares the same channel-wise gated delta backbone but lacks the independent erase address, EDA improves the Avg. score by 0.63 points.

The larger MoE 25B-A2.8B setting gives a clearer picture of the scaling behavior. EDA performs best on most benchmarks and improves the overall evaluation performance across knowledge-heavy, reasoningheavy, and code-oriented tasks. This larger-scale result suggests that address-level erase/write decoupling provides a broadly useful memory-management degree of freedom: the model can preserve the delta-rule correction at the current write key while using a separate learned address to suppress stale content elsewhere.

## 4.3 Midtraining Results

Table 2 reports the same benchmark suite after the MoE 25B-A2.8B checkpoints were further trained for 80B tokens at 32k sequence length.

Midtraining tests whether the pretraining-stage advantage survives a harder adaptation setting rather than only appearing at the original 4k training length. After the 80B-token long-context stage, EDA continues to provide the strongest overall performance, with especially clear gains on knowledge and reasoning benchmarks such as MMLU, MMLU-Pro, MATH, and BBH. This persistence is important because long-context midtraining changes the operating regime of the recurrent state: the model must maintain useful information over longer spans while still removing outdated content that can interfere with later reads.

Combined with the 400B-token pretraining results, the midtraining result strengthens the main conclusion: decoupling erase and write addresses remains useful after the model is further trained for longer contexts, suggesting that the erase path is compatible with, rather than fragile under, subsequent sequence-length adaptation.

## 4.4 Long-Context Evaluation

We evaluate the midtrained MoE checkpoints on the RULER task from 4k to 128k context length. Since midtraining used 32k sequences, the 64k and 128k settings evaluate length extrapolation beyond the training context. Table 3 reports the RULER score at each context length, aggregated over all sub-tasks and four evaluation runs. EDA outperforms both GDN and KDA in the short-context regime from 4k to

Table 3: RULER (Hsieh et al., 2024) long-context results for MoE 25B-A2.8B checkpoints after 400B-token pretraining and 80B-token midtraining at 32k sequence length. Values are percentages averaged over four evaluation runs; 64k and 128k are length-extrapolation settings. Avg. is the unweighted mean over the six displayed context lengths. Best results are bold and second-best results are underlined.

| Model   |    4k |    8k |   16k |   32k |   64k |   128k |   Avg. |
|---------|-------|-------|-------|-------|-------|--------|--------|
| GDN     | 92.4  | 90.09 | 87.28 | 82.15 | 67.62 |  45.16 |  77.45 |
| KDA     | 93.33 | 89.29 | 85.13 | 79.66 | 72    |  42.7  |  77.02 |
| EDA     | 93.84 | 90.88 | 87.55 | 81.52 | 71.9  |  44.22 |  78.32 |

16k, and remains close to the two baselines from 32k to 128k.

## 4.5 Memory-State Analysis

The benchmark gains above do not by themselves explain why an additional erase address helps, since the delta update already has two ways to reduce old content: diagonal decay D t = Diag ( α t ) and writecoupled correction ( I -β t k t k ⊤ t ) . Throughout this subsection we analyze a fixed layer and attention head unless stated otherwise: k t ∈ R d k is the L2-normalized write key at token t , dk is the per-head key dimension, I ∈ R d k × d k is the identity matrix, α t ∈ ( 0, 1 ] d k is the per-channel retention vector, and β t ∈ ( 0, 1 ) is the delta correction gate. We therefore ask a narrower mechanistic question: when the recurrent state must remove stale content, does the model use the new erase address in a way that cannot be explained by these two existing contraction paths alone?

We first measure a gate-strength allocation , not exact removed state energy. Recall that the diagonal decay is parameterized in log space as D t = Diag ( exp ( g t )) , where g t ∈ R d k is the per-channel log-retention vector. Therefore α t = exp ( g t ) is the per-channel retention factor applied before the erase and delta operators. For token t and head h , let

$$\bar { \alpha } _ { t , h } = \frac { 1 } { d _ { k } } \sum _ { j = 1 } ^ { d _ { k } } \alpha _ { t , h , j }$$

be the mean retention factor of the diagonal decay within that head, where α t , h , j is the j -th key-channel retention value and the sum averages over all dk key channels. Below we write α = ¯ α t , h for compactness. This averaging deliberately collapses the diagonal operator to a scalar summary, so it should not be used to compare the full operator rank or total energy removed by D t against the two rank-1 contractions. Its purpose is narrower: under the average retained scale of a head, we ask how the learned gates allocate contraction strength among the decay path, the write-key correction, and the independent erase path. Since both rank-1 operators act after D t , we define the unnormalized scores

$$b _ { D } = 1 - \alpha , \quad b _ { \Delta } = \alpha \beta _ { t } , \quad b _ { E } = \alpha \gamma _ { t } ,$$

for diagonal decay, same-address correction, and independent erase, respectively; here γ t ∈ ( 0, 1 ) is the erase gate. For readability, after fixing head h , we omit the head index on β t , h and γ t , h and write them as β t and γ t . Here 1 -α is the average decay removal fraction obtained after summarizing the diagonal retention vector by its mean, rather than an exact operator-level decomposition of D t . We plot the normalized share qm = bm / ( bD + b ∆ + bE ) for mechanism m ∈ { D , ∆ , E } . This definition is appropriate for the allocation question because β t and γ t are exactly the contraction factors of the two rank-1 readouts, while the multiplier α accounts for the fact that both contractions operate on the state retained after decay. It should not be read as an exact or fully fair state-energy decomposition: the actual content removed also depends on the anisotropic diagonal decay, the current state projections onto e t and k t , and the overlap between the two addresses.

As a boundary check, we also evaluate raw write-key recall, which asks whether older hidden values can be read back from their original write keys. KDA performs better under this strict probe, so EDA's advantage should not be interpreted as uniformly better historical recall. This motivates focusing on cleanup allocation and erase-address structure rather than raw recall alone.

To test whether the learned erase direction is structured, we use two address-level diagnostics. First, we compare the readout-level control induced by the actual erase address e t ∈ R d k with counterfactual directions: random unit vectors, head-shuffled learned erase directions, and the degenerate same-address choice e t = k t . For each direction strategy, we replay the recurrent state sequence with the same gates and measure the local effect of the erase step: o -t is the readout just before erase at token t , and δ o t is the readout change caused by that erase step. We compute the collateral perturbation score ∥ δ o t ∥ 2 / ∥ o -t ∥ 2 over layers; the raw means are 0.064 for Actual, 0.143 for Random, 0.115 for Shuffle, and 0.223 for

e t = k t . Figure 2(b) plots the same data as a layerwise fold change relative to Actual, with the raw means annotated. A smaller score does not mean 'no erase'; rather, under the same erase-gate budget, it means that the chosen address changes the currently readable state less than an alternative address. This probe therefore does not measure task benefit directly, but asks whether replacing the learned erase address causes larger collateral changes to the current readout. Second, we measure | cos ( e t , k t ) | , the absolute cosine similarity between the L2-normalized erase address and write key, on GSM8K few-shot prompts. The independent reference for this geometry check is the analytic mean of | u ⊤ r | for two independent random unit directions u , r ∈ R 128 , where 128 is the per-head key dimension in this model.

̄

Figure 2: EDA uses an independent cleanup path. (a) Gate-strength allocation by mean-retention bin. Independent erase becomes dominant when decay is weak (¯ α close to one); red percentages above bars denote the erase share. (b) Under the same erase gates, counterfactual erase directions cause larger local readout perturbations than the learned direction; bars show layerwise fold change relative to Actual, and µ denotes the raw mean perturbation score. (c) The erase address stays close to the independent-direction reference; same-address collapse would give | cos ( e , k ) | = 1.

<!-- image -->

Figure 2 shows where the new erase degree of freedom is used. The allocation in Figure 2(a) shows that diagonal decay through D t , same-address correction through ( I -β t k t k ⊤ t ) , and independent erase through ( I -γ t e t e ⊤ t ) account for 35.4%, 31.8%, and 32.8% of the global share, respectively. More importantly, the allocation shifts with decay speed in the expected direction: when ¯ α &lt; 0.3, decay already supplies most of the contraction strength, while for nearly persistent heads with ¯ α ≥ 0.9, independent erase contributes 69.1% and is about 3.0 × the same-address correction contribution. This high-retention regime is exactly where stale content would otherwise survive D t , so the model assigns the extra cleanup budget to γ t e t rather than forcing it through the current write key.

The learned erase direction is also controlled at the readout level rather than arbitrary. Figure 2(b) shows that replacing e t with random, shuffled, or same-address alternatives increases local readout perturbation by about 2.4 × , 1.9 × , and 3.4 × , respectively, after normalizing each analyzed layer by its Actual score. Thus, the learned erase address is not just an additional direction for removing state; under the same erase gates, it changes the current readout less than alternative directions, suggesting a more controlled cleanup operation. Figure 2(c) provides a complementary geometry check: the observed mean | cos ( e t , k t ) | stays around 0.105 across layers, close to the independent-direction reference and far from the value near one expected under same-address collapse. Together with the raw-recall boundary check above, these probes support the address-decoupling interpretation of EDA while clarifying its limitation: independent erase is a conditional cleanup mechanism, not a uniformly better historical-recall mechanism.

## 5 Related Work

Delta-rule and gated linear memory models. DeltaNet (Schlag et al., 2021; Yang et al., 2024) reinterprets recurrent state updates as online gradient descent on a reconstruction loss, replacing naive additive writes with corrective writes that depend on what is already stored at the current address. Gated DeltaNet (GDN) (Yang et al., 2025) extends this with a head-wise forget gate, and recent channel-wise gated variants further replace that head-wise gate with diagonal decay for finer retention control (Team et al., 2025). GDN-2 is the closest motivation-side comparison: it also argues that erase and write should be decoupled in delta-rule memory, but it targets a different axis of coupling (Hatamizadeh et al., 2026). Specifically, GDN-2 separates key-side erase and value-side write gates, allowing the model to assign different strengths to erasing and writing inside the delta residual. The active edit, however, remains organized around the current write key. EDA targets the complementary address-level coupling: it keeps the corrective delta write at k t while adding an independently addressed erase direction before the write. The two designs are therefore orthogonal in spirit: GDN-2 decouples how strongly erase and write are

applied, while EDA decouples where erasing and writing are applied.

Expressive state-transition mechanisms for linear RNNs. A growing body of work seeks to enrich the state transition in linear RNNs beyond the single-step delta correction. DeltaProduct (Siems et al., 2026) applies a sequence of Householder reflections per step, enabling smooth interpolation between diagonal and dense transitions. RWKV-7 (Peng et al., 2025) adopts a diagonal-plus-low-rank (DPLR) parameterization with vector-valued gating, improving state-tracking capacity. Comba (Hu et al., 2026) proposes a scalar-plus-low-rank (SPLR) form motivated by closed-loop control theory, adding output correction alongside state feedback. These approaches increase the expressive power of state evolution globally. Our method is complementary but different in purpose: rather than enriching the transition matrix, we introduce a specific memory-management capability-selectively deleting stale memory at one address before performing a corrective write at another-while preserving the delta-rule structure.

Hybrid architectures and inference efficiency. The computational bottleneck of softmax attention at inference time has motivated hybrid architectures that combine full attention with linear recurrent layers. Models such as Jamba (Lieber et al., 2024) and Nemotron (Gu et al., 2025) interleave sparse full-attention layers with predominantly linear recurrent layers, achieving a practical trade-off between quality and efficiency. Recent channel-wise gated delta hybrids demonstrate that this design can match or exceed full-attention quality while reducing KV cache usage substantially (Team et al., 2025). EDA is orthogonal to these architectural choices: it improves the recurrent component itself, making it a candidate drop-in replacement for channel-wise gated delta layers in hybrid designs.

## 6 Conclusion

We introduced Erase-then-Delta Attention, an address-level modification to delta-rule linear attention that separates where the model erases from where it writes. Instead of relying only on diagonal decay or same-address delta correction to remove stale content, EDA first applies a learned erase operation at an independent address and then performs the corrective delta write at the current write key. This keeps the core delta-rule update intact while giving the recurrent state a more direct way to clean up memory that is not aligned with the current write.

Across dense 2.5B and MoE 25B-A2.8B pretraining, EDA achieves the strongest average performance among the compared models, and the advantage persists after long-context midtraining of the MoE checkpoints. The memory-state analysis further supports the intended mechanism: the learned erase path is used most strongly when passive decay is weak, and counterfactual erase directions cause larger readout changes under the same erase gates. These results suggest that recurrent memory models benefit from deciding not only what to write, but also where stale information should be removed.

Limitations. Our work has several limitations. Introducing the independent erase step reduces raw write-key recall, so the erase path should be understood as a conditional cleanup mechanism rather than a uniform improvement to memory fidelity. Additionally, the current probes measure gate allocation and readout perturbation but do not directly trace individual erase events to specific downstream prediction improvements.

## References

- Maximilian Beck, Korbinian Pöppel, Markus Spanring, Andreas Auer, Oleksandra Prudnikova, Michael K Kopp, Günter Klambauer, Johannes Brandstetter, and Sepp Hochreiter. xLSTM: Extended long shortterm memory. In The Thirty-eighth Annual Conference on Neural Information Processing Systems , 2024. URL https://openreview.net/forum?id=ARAxPPIAhq .
- Karl Cobbe, Vineet Kosaraju, Mo Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, Christopher Hesse, and John Schulman. Training verifiers to solve math word problems. ArXiv , abs/2110.14168, 2021. URL https://api.semanticscholar.org/ CorpusID:239998651 .
- Tri Dao and Albert Gu. Transformers are SSMs: Generalized models and efficient algorithms through structured state space duality. In Forty-first International Conference on Machine Learning , 2024. URL https://openreview.net/forum?id=ztn8FCR1td .
- Albert Gu and Tri Dao. Mamba: Linear-time sequence modeling with selective state spaces. In First Conference on Language Modeling , 2024. URL https://openreview.net/forum?id=tEYskw1VY2 .

- Albert Gu, Karan Goel, and Christopher Re. Efficiently modeling long sequences with structured state spaces. In International Conference on Learning Representations , 2022. URL https://openreview.net/ forum?id=uYLFoz1vlAC .
- Yuxian Gu, Qinghao Hu, Haocheng Xi, Junyu Chen, Shang Yang, Song Han, and Han Cai. Jet-nemotron: Efficient language model with post neural architecture search. In The Thirty-ninth Annual Conference on Neural Information Processing Systems , 2025. URL https://openreview.net/forum?id=WZQXaTNYEB .
- Ali Hatamizadeh, Yejin Choi, and Jan Kautz. Gated deltanet-2: Decoupling erase and write in linear attention, 2026. URL https://arxiv.org/abs/2605.22791 .
- Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Xiaodong Song, and Jacob Steinhardt. Measuring massive multitask language understanding. ArXiv , abs/2009.03300, 2020. URL https://api.semanticscholar.org/CorpusID:221516475 .
- Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. Measuring mathematical problem solving with the MATH dataset. In Thirty-fifth Conference on Neural Information Processing Systems Datasets and Benchmarks Track (Round 2) , 2021. URL https://openreview.net/forum?id=7Bywt2mQsCe .
- Cheng-Ping Hsieh, Simeng Sun, Samuel Kriman, Shantanu Acharya, Dima Rekesh, Fei Jia, and Boris Ginsburg. RULER: What's the real context size of your long-context language models? In First Conference on Language Modeling , 2024. URL https://openreview.net/forum?id=kIoBbc76Sy .
- Jiaxi Hu, Yongqi Pan, Jusen Du, Disen Lan, Xiaqiang Tang, Qingsong Wen, Yuxuan Liang, and Weigao Sun. Improving bilinear RNN with closed-loop control. In Neural Information Processing Systems , 2026. URL https://openreview.net/forum?id=jlJaRXDzCE .
- Angelos Katharopoulos, Apoorv Vyas, Nikolaos Pappas, and Franccois Fleuret. Transformers are rnns: Fast autoregressive transformers with linear attention. In International Conference on Machine Learning , 2020. URL https://api.semanticscholar.org/CorpusID:220250819 .
- Opher Lieber, Barak Lenz, Hofit Bata, Gal Cohen, Jhonathan Osin, Itay Dalmedigos, Erez Safahi, Shaked Meirom, Yonatan Belinkov, Shai Shalev-Shwartz, Omri Abend, Raz Alon, Tomer Asida, Amir Bergman, Roman Glozman, Michael Gokhman, Avashalom Manevich, Nir Ratner, Noam Rozen, Erez Shwartz, Mor Zusman, and Yoav Shoham. Jamba: A hybrid transformer-mamba language model, 2024. URL https://arxiv.org/abs/2403.19887 .
- Jiawei Liu, Chunqiu Steven Xia, Yuyao Wang, and Lingming Zhang. Is your code generated by chatGPT really correct? rigorous evaluation of large language models for code generation. In Thirty-seventh Conference on Neural Information Processing Systems , 2023. URL https://openreview.net/forum?id= 1qvx610Cu7 .
- Bo Peng, Ruichong Zhang, Daniel Goldstein, Eric Alcaide, Xingjian Du, Haowen Hou, Jiaju Lin, Jiaxing Liu, Janna Lu, William Merrill, Guangyu Song, Kaifeng Tan, Saiteja Utpala, Nathan Wilce, Johan S. Wind, Tianyi Wu, Daniel Wuttke, and Christian Zhou-Zheng. Rwkv-7 "goose" with expressive dynamic state evolution, 2025. URL https://arxiv.org/abs/2503.14456 .
- Zihan Qiu, Zekun Wang, Bo Zheng, Zeyu Huang, Kaiyue Wen, Songlin Yang, Rui Men, Le Yu, Fei Huang, Suozhi Huang, Dayiheng Liu, Jingren Zhou, and Junyang Lin. Gated attention for large language models: Non-linearity, sparsity, and attention-sink-free. In The Thirty-ninth Annual Conference on Neural Information Processing Systems , 2025. URL https://openreview.net/forum?id=1b7whO4SfY .
- Imanol Schlag, Kazuki Irie, and Jürgen Schmidhuber. Linear transformers are secretly fast weight programmers. In International Conference on Machine Learning , 2021. URL https://api.semanticscholar. org/CorpusID:235377069 .
- Julien Siems, Timur Carstensen, Arber Zela, Frank Hutter, Massimiliano Pontil, and Riccardo Grazzi. Deltaproduct: Improving state-tracking in linear RNNs via householder products. In Neural Information Processing Systems , 2026. URL https://openreview.net/forum?id=SoRiaijTGr .
- Yutao Sun, Li Dong, Shaohan Huang, Shuming Ma, Yuqing Xia, Jilong Xue, Jianyong Wang, and Furu Wei. Retentive network: A successor to transformer for large language models. ArXiv , abs/2307.08621, 2023. URL https://api.semanticscholar.org/CorpusID:259937453 .

- Mirac Suzgun, Nathan Scales, Nathanael Schärli, Sebastian Gehrmann, Yi Tay, Hyung Won Chung, Aakanksha Chowdhery, Quoc Le, Ed Chi, Denny Zhou, and Jason Wei. Challenging BIG-bench tasks and whether chain-of-thought can solve them. In Anna Rogers, Jordan Boyd-Graber, and Naoaki Okazaki (eds.), Findings of the Association for Computational Linguistics: ACL 2023 , pp. 1300313051, Toronto, Canada, July 2023. Association for Computational Linguistics. doi: 10.18653/v1/2023. findings-acl.824. URL https://aclanthology.org/2023.findings-acl.824/ .
- Kimi Team, Yu Zhang, Zongyu Lin, Xingcheng Yao, Jiaxi Hu, Fanqing Meng, Chengyin Liu, Xin Men, Songlin Yang, Zhiyuan Li, Wentao Li, Enzhe Lu, Weizhou Liu, Yanru Chen, Weixin Xu, Longhui Yu, Yejie Wang, Yu Fan, Longguang Zhong, Enming Yuan, Dehao Zhang, Yizhi Zhang, T. Y. Liu, Haiming Wang, Shengjun Fang, Weiran He, Shaowei Liu, Yiwei Li, Jianlin Su, Jiezhong Qiu, Bo Pang, Junjie Yan, Zhejun Jiang, Weixiao Huang, Bohong Yin, Jiacheng You, Chu Wei, Zhengtao Wang, Chao Hong, Yutian Chen, Guanduo Chen, Yucheng Wang, Huabin Zheng, Feng Wang, Yibo Liu, Mengnan Dong, Zheng Zhang, Siyuan Pan, Wenhao Wu, Yuhao Wu, Longyu Guan, Jiawen Tao, Guohong Fu, Xinran Xu, Yuzhi Wang, Guokun Lai, Yuxin Wu, Xinyu Zhou, Zhilin Yang, and Yulun Du. Kimi linear: An expressive, efficient attention architecture, 2025. URL https://arxiv.org/abs/2510.26692 .
- Qwen Team. Qwen3 technical report, 2025. URL https://arxiv.org/abs/2505.09388 .
- Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin. Attention is all you need. In Neural Information Processing Systems , 2017. URL https://api.semanticscholar.org/CorpusID:13756489 .
- Sinong Wang, Belinda Z. Li, Madian Khabsa, Han Fang, and Hao Ma. Linformer: Self-attention with linear complexity. ArXiv , abs/2006.04768, 2020. URL https://api.semanticscholar.org/CorpusID: 219530577 .
- Yubo Wang, Xueguang Ma, Ge Zhang, Yuansheng Ni, Abhranil Chandra, Shiguang Guo, Weiming Ren, Aaran Arulraj, Xuan He, Ziyan Jiang, Tianle Li, Max Ku, Kai Wang, Alex Zhuang, Rongqi Fan, Xiang Yue, and Wenhu Chen. MMLU-pro: A more robust and challenging multi-task language understanding benchmark. In The Thirty-eight Conference on Neural Information Processing Systems Datasets and Benchmarks Track , 2024. URL https://openreview.net/forum?id=y10DM6R2r3 .
- Songlin Yang, Bailin Wang, Yu Zhang, Yikang Shen, and Yoon Kim. Parallelizing linear transformers with the delta rule over sequence length. In The Thirty-eighth Annual Conference on Neural Information Processing Systems , 2024. URL https://openreview.net/forum?id=y8Rm4VNRPH .
- Songlin Yang, Jan Kautz, and Ali Hatamizadeh. Gated delta networks: Improving mamba2 with delta rule. In The Thirteenth International Conference on Learning Representations , 2025. URL https: //openreview.net/forum?id=r8H7xhYPwz .

## A Model Configurations

The model configurations used in the evaluation are summarized in Tables 4 and 5. All evaluated models use the same vocabulary size (248,320); pretraining used 4096-token sequences, and the MoE midtraining stage used 32k-token sequences. Training used bfloat16 with the AdamW optimizer, SiLU activations in the FFN/MoE blocks, and RMSNorm with ϵ = 10 -6 . The hybrid models use one full-attention Transformer layer in every four layers, placed after three linear-attention layers. The full-attention layers in both the Transformer baseline and the hybrid models use Gated Attention (Qiu et al., 2025). For parameter alignment, the dense Transformer baseline uses 8/4/4 query/key/value heads in its full-attention layers.

Table 4: Scale-level architecture hyperparameters. 'Layers' reports total layers with linear/full-attention counts in parentheses. 'Attn/KV' denotes the query and key/value head counts in hybrid full-attention layers. 'LA K/V' denotes the number of key/value heads in the linear-attention layers, and 'LA dim' denotes their per-head dimensions. For the MoE scale, the FFN/expert column expert width.

| Scale         | Layers    |   d model | Attn/KV   | LA K/V   | LA dim   | FFN/expert   | MoE routing                             |
|---------------|-----------|-----------|-----------|----------|----------|--------------|-----------------------------------------|
| Dense 2.5B    | 24 (18/6) |      2048 | 8/2       | 8/16     | 128/128  | 7424-7488    | -                                       |
| MoE 25B-A2.8B | 28 (21/7) |      2048 | 16/2      | 16/32    | 128/128  | 512          | 256 experts, top-8 activated + 1 shared |

For parameter efficiency, variants with channel-wise forget gates use rank-16 (per-head) low-rank projections for the gate generator. The EDA MoE configuration uses a rank-16 (per-head) erase-address projection and a safe gate with lower bound -5.

Table 5: Total and active parameter counts for evaluated model variants. Dense models activate all parameters; MoE models report both total parameters and the parameters active per token.

| Scale         | Model       | Total params   | Active params   |
|---------------|-------------|----------------|-----------------|
| Dense 2.5B    | Transformer | 2.5052B        | 2.5052B         |
| Dense 2.5B    | GDN-2       | 2.5353B        | 2.5353B         |
| Dense 2.5B    | GDN         | 2.5035B        | 2.5035B         |
| Dense 2.5B    | KDA         | 2.5218B        | 2.5218B         |
| Dense 2.5B    | EDA         | 2.5295B        | 2.5295B         |
| MoE 25B-A2.8B | GDN         | 24.5676B       | 2.7236B         |
| MoE 25B-A2.8B | KDA         | 24.6324B       | 2.7885B         |
| MoE 25B-A2.8B | EDA         | 24.6558B       | 2.8119B         |

## B Evaluation Benchmarks

We evaluate the pretrained checkpoints on a compact set of standard language-model benchmarks. MMLU measures broad multitask knowledge across academic and professional subjects (Hendrycks et al., 2020), while MMLU-Pro increases the difficulty with more challenging questions and larger answer sets (Wang et al., 2024). GSM8K evaluates grade-school mathematical reasoning with word problems (Cobbe et al., 2021), and MATH evaluates more advanced competition-style mathematical problem solving (Hendrycks et al., 2021). BBH covers difficult reasoning tasks selected from BIG-Bench (Suzgun et al., 2023). EvalPlus evaluates code generation with stricter test cases beyond the original HumanEval/MBPP-style checks (Liu et al., 2023).

---

## Assets extracted losslessly (paper-md/pymupdf)

![page 1](assets/fig/p01_0.png)

*page 1*

### Equation crops — every numbered formula as rendered (6)

**(2)** p2 ![eq](assets/eq/eq_p02_000.png)

**(18)** p6 ![eq](assets/eq/eq_p06_013.png)

**(19)** p6 ![eq](assets/eq/eq_p06_028.png)

**(20)** p6 ![eq](assets/eq/eq_p06_055.png)

**(21)** p7 ![eq](assets/eq/eq_p07_071.png)

**(22)** p7 ![eq](assets/eq/eq_p07_075.png)
