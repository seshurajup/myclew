# XAI survey (2023–2026) — full landscape for adoption

Goal: adopt the modern interpretability toolbox so we NEVER trust a model blindly (the count_change lesson).
Organised by family, with **✅ = implemented in our `xai` agent**, **⭐ = high-value to adopt for OUR task**
(3D cell **detection** + graph **linking** + a tabular **division** head).

## 1. CAM family (spatial saliency for the detector CNN)
The `pytorch-grad-cam` library standardises 14 variants — the canonical set:
- **Grad-CAM** ✅, **Grad-CAM++** ✅, **Score-CAM** ✅ (gradient-free)
- HiResCAM, **XGrad-CAM**, **LayerCAM** (better fine localisation), AblationCAM, EigenCAM, FullGrad,
  ShapleyCAM (2024), FinerCAM — ⭐ LayerCAM + XGrad-CAM are the cheap high-value adds.
- Source: jacobgil/pytorch-grad-cam; comparative CAM eval (PMC12350829, 2025).

## 2. Object-DETECTION-specific XAI ⭐⭐ (most relevant — we detect cells)
Classifier CAMs don't fit detectors; 2023–2024 produced detector-native methods:
- **D-RISE** — perturbation saliency for detectors (mask → measure detection-vector change).
- **G-CAME** (2023) — Gaussian Class Activation Mapping *for object detectors*.
- **ODSmoothGrad** (2023) — saliency maps for detectors.
- **BSED** (2023) — Baseline Shapley-based explainable detector.
- **ProtoP-OD** (2024) — prototypical-parts explainable detection.
→ These explain WHY our detector fires/misses a cell — directly tied to node-recall, our #1 lever.

## 3. Backprop / attribution
- **Integrated Gradients** ✅, LRP + **EVO-LRP** (2025, evolutionary-tuned LRP), Guided-IG, Blur-IG,
  DeepSHAP, GradientSHAP.

## 4. Perturbation / surrogate
- **Occlusion** ✅, **RISE** ✅, D-RISE (above), Extremal/Meaningful perturbations.

## 5. Feature attribution (our tabular division head)
- **SHAP** ✅, **LIME** ✅, **permutation** ✅, KernelSHAP/TreeSHAP/FastSHAP.
  (This family already delivered — it caught the fragile `count_change` feature.)

## 6. Concept-based (global, "what concept drives it")
- **TCAV** — concept-activation-vector sensitivity; used in 2024 to *remove* undesirable learned concepts.
- Concept Bottleneck Models, ACE. ⭐ TCAV could test whether the linker keys on "sister-symmetry" concept.

## 7. Prototype-based (case-based "this looks like that")
- ProtoPNet, ProtoTree, **PIP-Net**, LucidPPN (2024), ProtoP-OD. Comprehensive eval: arXiv 2507.06819 (2025).

## 8. Mechanistic interpretability — the 2024–2025 frontier ⭐
- **Sparse Autoencoders (SAEs)** — now for VISION: archetypal SAE (stable dictionary learning), hierarchical
  SAE for CLIP, *monosemantic* features in vision-language models; "selective remapping of visual concepts
  during adaptation" in ViTs. Decompose a layer into interpretable features.
- **Activation patching / circuits / logit lens** — trace causal information flow.
- Causal interpretation of SAE features in vision (arXiv 2509.00749, 2025).

## 9. Evaluation of explanations (don't trust an explanation blindly either)
- Faithfulness metrics + **sanity checks**; "Inpainting the Gaps" framework (2024) for evaluating ViT
  explanations. ⭐ Always pair an attribution with a faithfulness check.

## 10. The 2025–2026 frontier (mechanistic interpretability = MIT 2026 Breakthrough Tech)
The field hit its tipping point in 2025–2026; a 29-author/18-org consensus paper set the open problems.
Core workflow now: *build causal hypotheses about internal behaviour → test with interventions → decompose
representations into interpretable features → validate against benchmarks.* Highlights for us:
- **Prisma** (arXiv 2504.19475, 2025) — ⭐⭐ open-source toolkit for mechanistic interpretability in **VISION
  AND VIDEO** — directly fits our video cell-tracking; SAEs + circuit tools for ViT/video models.
- **Visual Sparse Steering / VS2** (2506.01247) — label-free top-k SAE on frozen CLIP activations; *steer*
  a frozen vision model via sparse features (no labels — matches our <1%-labelled setting).
- **Interpretable & Testable Vision Features via SAEs** (2502.06755); **Concept-Bottleneck SAEs**
  (2512.10805) — steerable concept features; **Sparse CLIP** (2601.20075) co-optimises interpretability+perf.
- **Circuit tracing / causal interventions** (Anthropic 2025–2026) — surface multi-step mechanisms; evaluate
  SAEs with concept annotations (2606.24716); geometric view of SAE concept learning (2606.07007).
- Survey: "Bridging the Black Box" (ACM Computing Surveys 2025, 10.1145/3787104).

Takeaway: **SAEs are now the central method for BOTH vision and video** — decompose the detector/linker's
internal features into monosemantic, testable, *steerable* concepts. Prisma is the ready-made vision+video
entry point. This is the highest-ceiling adoption (harder, but it's where the field converged).

## Adoption priority for THIS competition
1. ⭐⭐ **Detection XAI** (D-RISE, G-CAME) — explain detector recall (our dominant lever).
2. ⭐ **LayerCAM + XGrad-CAM** — cheap CAM upgrades, sharper maps.
3. ⭐ **TCAV** — test the linker/division concepts (sister-symmetry).
4. ⭐ **Sparse Autoencoders** — decompose the detector/linker features (frontier, higher effort).
5. Always attach a **faithfulness sanity-check** to any adopted explanation.

Sources: pytorch-grad-cam (jacobgil); G-CAME arXiv 2306.03400; BSED 2308.07490; ODSmoothGrad 2304.07609;
ProtoP-OD 2402.19142; SAE survey 2503.05613; causal SAE in vision 2509.00749; EVO-LRP 2509.23585;
Inpainting-the-Gaps 2406.11534; CAM eval PMC12350829.
