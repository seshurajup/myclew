# Pretrained Transformer 3D Detectors for an fp8-Accelerated Biohub Cell-Center Detector

Research + honest assessment. NOT a build. Date: 2026-07-20.

## Question

Can we fine-tune a *pretrained* transformer-based 3D detection/segmentation model as an
**fp8-accelerated cell-center DETECTOR** for biohub (3D light-sheet fluorescence of developing
zebrafish, SiMView)? To be worth it a candidate must be BOTH:

1. **Transformer-heavy** — fp8 only accelerates matmul/Linear/attention (measured 1.33-1.40x on
   our RTX 5090 sm_120). There is no fp8 conv3d kernel, so a conv encoder runs bf16. A hybrid only
   fp8-accelerates its transformer fraction.
2. **Domain-close** — our history: from-scratch = 0.077 (dead); external->biohub transfer = NO-GO
   ("gap is content not style"; light-sheet zebrafish is far from CT/MRI/natural-image pretraining).

## Ranked table

| # | Model | Arch (pure-tf / hybrid, ~%fp8-able) | Pretrain domain | Domain-fit to light-sheet zebrafish nuclei | Weights + license | Source |
|---|-------|-------------------------------------|-----------------|--------------------------------------------|-------------------|--------|
| 1 | **SpatialDINO** | Native 3D ViT (DINOv2). Near-pure transformer, **~90%+ fp8-able** (only the small patch-embed is conv) | **Fluorescence 3D lattice light-sheet (LLSM)**, live-cell, 78 datasets / 45k volumes / 2.4TB | **Modality MATCH** (fluorescence, 3D, anisotropic, low-contrast, monochromatic = SiMView conditions). **Content gap:** pretrained on *subcellular* structures (clathrin pits, vesicles, endosomes, lysosomes) in cultured single cells, NOT whole-embryo nuclei. Smallest content gap available; SSL so re-trainable on comp volumes | **YES** — AWS S3 `s3://spatialdino/models/` (`backbone.pth`), `--no-sign-request`. **MIT** | [repo](https://github.com/kirchhausenlab/spatialdino), [paper](https://www.biorxiv.org/content/10.64898/2025.12.31.697247v2), [PubMed](https://pubmed.ncbi.nlm.nih.gov/41509489/) |
| 2 | **LSM-FM** (Scheinfeld et al.) | Hybrid: UNet + **SwinUNETR**. Transformer is a fraction; **partial fp8** (conv stem/decoder = bf16) | **Fluorescence light-sheet**, but **mouse + human BRAIN tissue** (Allen Institute, SELMA3D). 1,023 patches of 96^3 | Modality match; **content far** (adult brain tissue, not developing embryo nuclei). Small pretrain corpus | YES — [GitHub](https://github.com/AdinaScheinfeld/lsm_fm_public_repo); license = arXiv non-exclusive (verify code LICENSE) | [paper](https://arxiv.org/abs/2605.26026), [html](https://arxiv.org/html/2605.26026v1) |
| 3 | **Primus / PrimusV2** | **Pure 3D ViT** (8^3 tokenizer + transformer + detokenizer, 3D axial RoPE, Eva-02 MLP). **~90%+ fp8-able**. (PrimusV3 = conv-heavy patch embed = more hybrid) | **CT/MRI** medical (nnU-Net datasets: ACDC, AMOS, KiTS, LiTS, WORD...) | **Content far** (CT/MRI anatomy). Modality also far. **No released pretrained checkpoint** — train-from-scratch inside nnU-Net | Code in nnU-Net (Apache-2.0). **No downloadable pretrained weights** (needs verification) | [paper](https://arxiv.org/abs/2503.01835), [nnU-Net primus.md](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/primus.md) |
| 4 | **CellSeg3D (SwinUNETR + WNet3D)** | SwinUNETR = hybrid (partial fp8); WNet3D = conv (no fp8) | Fluorescence/confocal nuclei (Platynereis-Nuclei light-sheet, Mouse-Skull-Nuclei) | **Modality + content close-ish** (3D nuclei in fluorescence) but backbone is hybrid/conv, so weak fp8 | Weights available (napari plugin) | [eLife](https://elifesciences.org/reviewed-preprints/99848), [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC12187128/) |
| 5 | **micro-sam (µSAM) `vit_*_lm`** | SAM **2D** ViT encoder (transformer) + conv mask decoder. fp8-able but **2D**, needs 2.5D stacking | Light-microscopy cells/nuclei (2D) | Content close (cells/nuclei) but **2D**, loses z-context of zebrafish stacks | YES — HF/GitHub; Apache-2.0 | [GitHub](https://github.com/computational-cell-analytics/micro-sam), [Nat. Methods](https://www.nature.com/articles/s41592-024-02580-4) |
| 6 | **Cellpose-SAM** | SAM **2D** ViT encoder + transposed-conv flow head. Transformer-heavy but **2D** | Broad 2D cellular microscopy | Content close but **2D** | YES; non-commercial/Cellpose license (verify) | [bioRxiv 2025.04.28.651001](https://www.biorxiv.org/content/10.1101/2025.04.28.651001), [Cellpose repo](https://github.com/MouseLand/cellpose) |
| 7 | **SAM-Med3D** | Fully 3D SAM: ViT-style 3D image encoder + 3D cross-attn mask decoder. Transformer-heavy, **fp8-able**. Promptable (points/boxes) | **CT/MRI/US** (SA-Med3D-140K, 22k images) | **Content far** (medical anatomy); promptable, not a free-running detector | YES — [HF `blueyo0/SAM-Med3D`](https://huggingface.co/blueyo0/SAM-Med3D) (`sam_med3d_turbo.pth`); Apache-2.0 | [GitHub](https://github.com/uni-medical/SAM-Med3D), [paper](https://arxiv.org/abs/2310.15161) |
| 8 | **MedSAM2 / SAM2** | Hiera hierarchical ViT + memory-attention, but **2D-slice + memory** propagation (conv stem). Partial fp8 | **CT/PET/MRI/US/endoscopy** (455k image-mask pairs) | **Content far** + 2.5D promptable video paradigm, not 3D nuclei detection | YES — open source | [MedSAM2 paper](https://arxiv.org/abs/2504.03600), [repo](https://github.com/bowang-lab/MedSAM2) |
| 9 | **Swin-UNETR / SwinUNETR-V2 / SuPreM / VoCo** | Hybrid Swin encoder + **CNN decoder**; V2 adds *more* conv. **Partial fp8** only | **CT** (5,050 CT volumes; SuPreM 673k annotations) | **Content + modality far** (CT abdomen) | YES — MONAI / [VoCo repo](https://github.com/Luffy03/VoCo); Apache-2.0 | [Swin-UNETR CVPR22](https://openaccess.thecvf.com/content/CVPR2022/papers/Tang_Self-Supervised_Pre-Training_of_Swin_Transformers_for_3D_Medical_Image_Analysis_CVPR_2022_paper.pdf), [NVIDIA blog](https://developer.nvidia.com/blog/novel-transformer-model-achieves-state-of-the-art-benchmarks-in-3d-medical-image-analysis/) |
| 10 | **UNETR** | ViT-B/16 encoder (transformer) + **CNN decoder** with conv skips. Partial fp8 | CT/MRI (BTCV, MSD) | Content + modality far | YES — MONAI; Apache-2.0 | [WACV22](https://openaccess.thecvf.com/content/WACV2022/papers/Hatamizadeh_UNETR_Transformers_for_3D_Medical_Image_Segmentation_WACV_2022_paper.pdf) |
| 11 | **nnFormer** | Interleaved conv + transformer (hybrid). Partial fp8 | CT/MRI (ACDC, Synapse) | Content + modality far | YES — GitHub | [paper](https://arxiv.org/abs/2109.03201) |
| — | **Trackastra** | Transformer, but for **LINKING/association**, not detection heatmaps | — | Solves a different sub-problem (edges), not the DETECTION lever | YES | [repo](https://github.com/weigertlab/trackastra) |

Notes: "%fp8-able" is an architectural estimate of the Linear/attention parameter fraction, not a
measured speedup. All models pretrained on non-zebrafish data; "domain-fit" splits **modality**
(imaging physics) from **content** (what is imaged) because our NO-GO lesson was content, not style.

## Direct answer to "is there a pure-transformer 3D detector pretrained on fluorescence/cell microscopy?"

**Almost.** The only model that is BOTH near-pure-transformer AND pretrained on 3D fluorescence
microscopy is **SpatialDINO** (native 3D ViT / DINOv2, trained on live-cell lattice light-sheet).
BUT it is a **self-supervised feature backbone, not a ready-made detector** — it emits dense
volumetric feature maps; detection/segmentation are downstream (a head or clustering on features).
And its pretrain **content is subcellular organelles inside single cultured cells**, not
whole-embryo nuclei. So: a fluorescence-pretrained pure-transformer 3D *backbone* exists; a
fluorescence-pretrained pure-transformer 3D *nucleus-center detector* does **not** exist off-the-shelf.
Everything else is either hybrid (partial fp8), 2D, medical-CT/MRI (content-far), or a linker.

## Honest verdict

**No candidate cleanly satisfies both bars (transformer-heavy fp8 win AND domain-close enough to
plausibly reach 0.88-0.90 given recall saturation + our transfer NO-GO history).** The field forces
a trade:

- **Transformer-heavy + domain-close:** only **SpatialDINO** — but its pretrain content
  (intracellular vesicles/organelles) has a real content gap to embryo nuclei, which is exactly the
  axis our "gap is content not style" NO-GO flagged as the killer. Modality matches perfectly
  (fluorescence, 3D, anisotropic, low-contrast, monochromatic = SiMView), which is more than any
  medical model offers. It is also SSL, so the honest path is to *re-run/continue DINOv2 pretraining
  on the competition's own unlabeled zebrafish volumes*, then attach a light detection head — this
  sidesteps the content gap better than frozen transfer.
- **Domain-close nuclei models** (CellSeg3D, µSAM, Cellpose-SAM) are hybrid or 2D -> weak/no fp8.
- **Pure-transformer 3D** with real pretraining momentum (Primus, SAM-Med3D) is CT/MRI content-far,
  and Primus ships **no downloadable pretrained checkpoint** (train-from-scratch), so it inherits our
  0.077 from-scratch risk (albeit under nnU-Net's much stronger recipe).

### Single best candidate (if forced to name one)

**SpatialDINO** (`s3://spatialdino/models/backbone.pth`, MIT). It is the only option that is both
fp8-favorable (native 3D ViT) and modality-matched to zebrafish light-sheet, and being SSL it can be
continued-pretrained on comp data to attack the content gap. Treat it as a **research experiment,
not a score lever**: the content gap (organelles vs embryo nuclei) is precisely the failure mode our
history warns about, and it needs a detection head we don't have yet.

### Grounded recommendation

**Keep the bf16 conv UNet (pilkwang) as the production detector.** It is at the recall-saturation
structural ceiling (~0.88-0.909) and there is no evidence a transformer detector beats it on score.
Pursue the fp8-transformer detector only as an R&D track, and only through SpatialDINO
continued-pretraining + a detection head — because that is the single path that gives BOTH a real
fp8 training win AND the closest achievable domain fit. Do not expect a score gain; the realistic
upside is training-throughput (fp8) if/when a transformer detector matches the conv UNet, not a jump
past the recall ceiling.

## "SOTA / fast" claims to re-measure on our sm_120 (their numbers are H100/A100 + CT/MRI)

Every performance number below is from the source domain/hardware and MUST be re-measured before it
informs a decision:

- SpatialDINO "enables automated detection and segmentation without retraining" — **needs
  verification** on zebrafish nuclei; it generalizes across *their* LLSM targets, not proven on
  embryo nuclei. Check: run its inference on a few comp frames, measure node-recall vs pilkwang.
- SpatialDINO / DINOv2-3D fp8 speedup — **needs verification**: benchmark fp8 vs bf16 forward+backward
  of the actual 3D ViT config on the 5090 (expect ~1.3-1.4x on the Linear/attention fraction only).
- Primus "on par with SOTA CNNs (ResEnc-L, MedNeXt)" — CT/MRI Dice on A100; **needs verification**
  that a pure-ViT even trains to competitive node-recall on sparse zebrafish GT from scratch.
- SAM-Med3D "+60% over SAM", MedSAM2 SOTA — medical benchmarks; **irrelevant until re-measured** on
  zebrafish, and both are promptable (need point/box prompts), not free-running detectors.
- LSM-FM "few-shot segmentation" gains — measured on mouse/human brain; **content-far**, re-measure
  on zebrafish or discard.
- Weights/license specifics to verify before any use: LSM-FM repo LICENSE file; Cellpose-SAM
  commercial-use terms; whether Primus has ANY released pretrained checkpoint (currently: none found).

## Sources

- SpatialDINO: https://www.biorxiv.org/content/10.64898/2025.12.31.697247v2 · https://github.com/kirchhausenlab/spatialdino · https://pubmed.ncbi.nlm.nih.gov/41509489/
- LSM-FM (light-sheet foundation model): https://arxiv.org/abs/2605.26026 · https://arxiv.org/html/2605.26026v1 · https://github.com/AdinaScheinfeld/lsm_fm_public_repo
- Primus: https://arxiv.org/abs/2503.01835 · https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/primus.md
- CellSeg3D: https://elifesciences.org/reviewed-preprints/99848 · https://pmc.ncbi.nlm.nih.gov/articles/PMC12187128/
- micro-sam: https://github.com/computational-cell-analytics/micro-sam · https://www.nature.com/articles/s41592-024-02580-4
- Cellpose-SAM: https://www.biorxiv.org/content/10.1101/2025.04.28.651001 · https://github.com/MouseLand/cellpose
- SAM-Med3D: https://github.com/uni-medical/SAM-Med3D · https://arxiv.org/abs/2310.15161 · https://huggingface.co/blueyo0/SAM-Med3D
- MedSAM2/SAM2: https://arxiv.org/abs/2504.03600 · https://github.com/bowang-lab/MedSAM2
- Swin-UNETR / SuPreM / VoCo: https://openaccess.thecvf.com/content/CVPR2022/papers/Tang_Self-Supervised_Pre-Training_of_Swin_Transformers_for_3D_Medical_Image_Analysis_CVPR_2022_paper.pdf · https://developer.nvidia.com/blog/novel-transformer-model-achieves-state-of-the-art-benchmarks-in-3d-medical-image-analysis/
- UNETR: https://openaccess.thecvf.com/content/WACV2022/papers/Hatamizadeh_UNETR_Transformers_for_3D_Medical_Image_Segmentation_WACV_2022_paper.pdf
- nnFormer: https://arxiv.org/abs/2109.03201
- Trackastra: https://github.com/weigertlab/trackastra
