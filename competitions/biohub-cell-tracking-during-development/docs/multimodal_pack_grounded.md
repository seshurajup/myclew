# Multimodal pack — grounded in top solutions of real multimodal Kaggle competitions

**Multimodal** (Kaggle data-type "multimodal data") = a *single model* consuming ≥2 modalities
(image+text+tabular / signal+metadata) and fusing them *inside* the network. This pack is the
genuinely-missing **FEATURE/MODEL-level FUSION** layer. Late/prediction-level fusion (averaging model
outputs) is already covered by `ensemble` / `blend-optimize` / `infer-cascade` / `tab-stack` /
`checkpoint-merger` and is only *referenced* here, not rebuilt.

Mined with the fleet's `gm-writeup-mine` path via the **nvidia-kaggle bearer** scripts
(`discussion_ingest.py` / `discussion_read.py` / `discussion_query.py`, KGAT token in `.env` +
`~/.kaggle/access_token`, `PROJECT_ROOT=<comp>`). The bearer token WORKED for all four competitions
(discussions ingested into `data/discussions.db`). Kaggle's SPA blocked `WebFetch` of individual
discussion HTML, so the **petfinder winner fusion detail** was cross-checked via `WebSearch`
(PETS-SWINF paper + winner recaps) — flagged inline below.

---

## Writeups mined (provenance + real links)

### 1. petfinder-pawpularity-score — **image + tabular metadata** (the canonical image+tabular fusion comp)
- **All-Rank solution collection** (Saiyan Warrior) — index of every rank's writeup:
  https://www.kaggle.com/competitions/petfinder-pawpularity-score/discussion/300952
- **1st place** (link index → discussion/301686 / 300938 family):
  https://www.kaggle.com/competitions/petfinder-pawpularity-score/discussion/301686
- **PETS-SWINF** (peer-reviewed writeup of the image+metadata NN), arXiv 2201.06061:
  https://arxiv.org/abs/2201.06061
- **Recurring fusion techniques (grounded):**
  - Per-modality **image encoder** = Swin Transformer (global context beat CNNs); the last feature map
    → a fixed image **embedding**.
  - **EARLY / feature-level fusion**: concatenate the 12 dense metadata features onto the image
    embedding, then a small regression head (**SVR / MLP head on the fused vector**), and *blend* the
    NN prediction with the SVR prediction.
  - **Smoothing / BCE-on-normalized-target** to handle noisy labels.
  - **KEY NEGATIVE FINDING (drives modality-dropout):** the 12 metadata features were only *marginally*
    informative — most top teams found **image-only nearly matched image+metadata**. → a robust
    multimodal model must not *depend* on any one modality: **modality-dropout + missing-modality
    imputation** is exactly the winners' robustness insurance. (Cross-checked via WebSearch:
    PETS-SWINF reports metadata gives a small edge over image-only.)

### 2. shopee-product-matching — **image + text** (the image+text fusion comp)
- **Matching tricks from the winners of Shopee** (The Devastator) — distilled winner writeups:
  https://www.kaggle.com/competitions/shopee-product-matching/discussion/329472
- **2nd Place Solution Code** (tkm2261):
  https://www.kaggle.com/competitions/shopee-product-matching/discussion/238362
- **Solution Compilation Thread** (Tensor Girl):
  https://www.kaggle.com/competitions/shopee-product-matching/discussion/238016
- **Recurring fusion techniques (grounded, verbatim from the writeup):**
  - Per-modality encoders → a **shared embedding**: **NFNet-F0 / ViT** (image) + **Indonesian-BERT /
    Multilingual-BERT / Paraphrase-XLM** (text).
  - **EARLY / feature-level fusion**: 2nd place *"trained model with NFNet-F0 and Indonesian BERT
    (concatenated at final feature layers)"* — the two encoder outputs concatenated **inside one model**
    → a "multimodal (image+text) similarity".
  - **Per-modality L2-normalize then concat** (shared-space projection):
    `F.normalize(torch.cat([F.normalize(emb1), F.normalize(emb2)], axis=1))` — the exact
    encoder-adapter pattern (normalize each modality to the unit sphere before fusing).
  - **Metric-learning per-modality heads**: **ArcFace / CurricularFace** (CurricularFace beat ArcFace),
    optimizer **SAM**.
  - **LATE fusion (already covered by `ensemble`/`blend-optimize`)**: combine image-match and text-match
    candidate sets; 1st-place **Iterative Neighborhood Blending (INB)** (QE/DBA — embedding refinement
    on the kNN graph); 2nd place stacked a **GAT + LGB** over cross-modal similarity features.

### 3 & 4. ariel-data-challenge-2024 / 2025 — **spectroscopic signal + detector/star metadata** (the tagged-multimodal signal comps)
- **1st place 2024 recap** (c-number): https://www.kaggle.com/competitions/ariel-data-challenge-2024/discussion/544316
- **15th 2024** (takaito): https://www.kaggle.com/competitions/ariel-data-challenge-2024/discussion/543681
- **Top-3 2024 solutions summarized** (Athar Sayed, in the 2025 comp):
  https://www.kaggle.com/competitions/ariel-data-challenge-2025/discussion/586581
- **Winning Solutions index** (AC): https://www.kaggle.com/competitions/ariel-data-challenge-2025/discussion/586519
- **Recurring techniques (grounded) — HONEST NOTE:** Ariel is dominated by **physics/simulator-based
  signal modeling + analytical fitting + ensembles (GPR / AutoEncoder / NMF / Bayesian inference)**, not
  deep cross-attention fusion. Its "multimodal" character is **signal (spectral time-series) + auxiliary
  detector/star metadata**, where metadata is used to:
  - **per-modality / per-instance normalization + auxiliary conditioning**: normalize the input waveform
    by a per-planet `mean_s`; predict the *difference* to `mean_s` (zero-centred target) — 15th place.
  - **auxiliary per-modality uncertainty head**: `sigma` = std of an ensemble's predictions (a
    per-modality/predictive uncertainty estimate) — grounds the optional auxiliary head.
  - So Ariel grounds **per-modality normalization + projection** and **auxiliary heads**, *not* the
    heavy attention fusion (which comes from petfinder/shopee).

---

## Recurring FUSION techniques → what to build

| Technique (recurring across winners) | Where grounded | Covered already? |
|---|---|---|
| Per-modality encoder → shared embedding | shopee (NFNet+BERT), petfinder (Swin) | encoders exist per-modality; **alignment/projection missing** → build |
| **EARLY / feature-level CONCAT fusion** (concat encoder outputs inside one model) | shopee "concatenated at final feature layers", petfinder embed⊕metadata | **MISSING** → build `multimodal-fusion` |
| Per-modality **L2-norm + projection to a shared dim** | shopee `F.normalize(cat(F.normalize(...)))`, ariel per-planet norm | **MISSING** → build `modality-encoder-adapter` |
| **Gated / FiLM / bilinear / cross-attention** fusion | shopee 2nd GAT-over-similarity; standard VQA co-attention | **MISSING** → build (strategies in `multimodal-fusion`) |
| **Modality-dropout + missing-modality imputation** (null token) | petfinder metadata-marginal → image-only robust | **MISSING** → build `modality-dropout` |
| Auxiliary per-modality heads (ArcFace/SVR/uncertainty) | shopee ArcFace/CurricularFace, petfinder SVR, ariel sigma | optional head in `multimodal-fusion` |
| **LATE / decision fusion** (avg outputs, blend candidate sets, INB) | shopee INB, petfinder NN+SVR blend | **ALREADY COVERED** (`ensemble`,`blend-optimize`,`infer-cascade`,`tab-stack`,`checkpoint-merger`) → referenced, not rebuilt |
| Graph/neighborhood blending on the similarity graph | shopee INB / GAT | covered by Graph pack + late-fusion → referenced |

---

## Dedup verdict

The fleet already does **LATE / prediction-level** fusion well (`ensemble`, `blend-optimize`,
`infer-cascade`, `tab-stack`, `checkpoint-merger`). The genuine gap is **FEATURE/MODEL-level fusion**:
combining an image embedding + text embedding + tabular vector **inside one model** — projection to a
shared space, configurable fusion (concat / sum / gated / FiLM / cross-attention / bilinear), and
**modality-dropout / missing-modality** robustness. That is exactly this pack, and nothing else.

## Agents built (pure torch, GPU-first, CPU fallback; BaseAgent + module handlers)

- **`multimodal-fusion`** — dict of per-modality feature tensors → project each to a shared dim → fuse
  via a configurable strategy (`concat` / `sum` / `mean` / `gated` / `film` / `cross_attention` /
  `bilinear`) → fused representation (+ optional head). Handles N modalities of variable dims.
- **`modality-encoder-adapter`** — wraps heterogeneous per-modality inputs (image / text / tabular) into
  aligned, **L2-normed, shared-dim** embeddings with per-modality LayerNorm + projection + learnable
  **modality-type embeddings** (the shopee `F.normalize(cat(F.normalize(...)))` + shared-space pattern).
- **`modality-dropout`** — training-time random modality masking + inference **missing-modality
  imputation via a learned per-modality null token** (the petfinder robustness insurance).

Data-wise verifier: `test_fleet_agents/multimodal_pack_test.py` (offline, deterministic, BLAS-pinned).
Registered in `fleet_agents/__init__.py`; new **"Multimodal"** pack in `coverage_audit.py`; tagged
`("multimodal",)` in `agent_routing.py` (multimodal-specific; the cross-cutting packs already carry the
single modalities, so a multimodal comp pulls the union).
