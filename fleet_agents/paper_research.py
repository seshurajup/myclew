"""paper-research — mine RECENT architecture innovations from papers, weighing ACCURACY *and* SPEED.

arch-builder designs from data; this agent supplies the modern building-block CANDIDATES it should search
over — the latest innovations (attention variants, SSMs, sparse-MoE, modern norms/activations, efficient
convs), each tagged with its accuracy effect, its SPEED/latency effect (this is a runtime-limited T4
code-comp, so speed is a first-class axis), and applicability to our stage (detector CNN / graph-linker /
attention). Per "decide only from data", nothing is asserted: every innovation is emitted as a HYPOTHESIS
for arch-search to TRAIN and prove on golden-CV — the agent just curates and ranks the search space.

Reusable / spec-driven: {catalog, task, speed_weight}. Pass a different catalog to reuse for any comp.
"""
from __future__ import annotations
import json
from pathlib import Path

COMP = Path(__file__).resolve().parent.parent
STATE = COMP / "config" / "_auto" / "paper_research.json"
OUT = COMP / "config" / "arch" / "innovation_candidates.yml"

# curated recent innovations. name -> (stage, accuracy, speed, note, adopt_status)
# accuracy/speed: "+" helps, "0" neutral, "-" costs. Grounded in the papers; PROVEN only via arch-search.
INNOVATIONS = {
    # attention / sequence
    "FlashAttention-3":   ("attention", "0", "++", "exact attention, ~2x faster + less memory (Shah 2024)", "adopt-if-attn"),
    "GQA grouped-query":  ("attention", "-", "++", "share KV heads → big memory/speed win, tiny acc cost (Ainslie 2023)", "search"),
    "RoPE positions":     ("attention", "+", "0", "rotary positions generalise better; standard now", "search"),
    "sliding-window attn": ("attention", "0", "+", "local window = O(n) for our short temporal horizon", "search"),
    "Mamba / SSM":        ("sequence", "0", "+", "linear-time state-space; strong on long sequences (Gu 2024)", "search"),
    # capacity
    "sparse MoE (top-k)": ("backbone", "+", "-", "more capacity at ~same FLOPs; memory heavy (per data-regime experts)", "search"),
    # norms / activations / blocks (cheap, usually free wins)
    "RMSNorm":            ("backbone", "0", "+", "drop the mean-subtraction → faster than LayerNorm, same acc", "adopt-cheap"),
    "SwiGLU FFN":         ("backbone", "+", "0", "gated FFN beats ReLU/GELU FFN at equal params (Shazeer 2020)", "adopt-cheap"),
    "pre-norm residual":  ("backbone", "+", "0", "stable deep training; standard", "adopt-cheap"),
    # DETECTION HEAD / LOSS — the hard-positive family (directly addresses our XAI finding: we MISS DIM nuclei =
    # hard, low-signal positives under heavy class imbalance). Each grounded in its paper; PROVEN only via arch-search.
    "focal loss":         ("loss", "+", "0", "down-weights easy examples, concentrates gradient on HARD positives → dim nuclei (Lin RetinaNet 2017)", "search"),
    "CenterNet gaussian-focal": ("loss", "+", "0", "keypoint-heatmap focal; penalty-reduced near-peak → sub-pixel dim-peak recall (Zhou 'Objects as Points' 2019)", "search"),
    "Tversky loss":       ("loss", "+", "0", "tunable FP/FN weighting (α,β) → RECALL-tilt for rare positives (Salehi 2017)", "search"),
    "focal-Tversky":      ("loss", "+", "0", "focal × Tversky: focuses on hard imbalanced regions (small/dim) (Abraham 2019)", "search"),
    "OHEM":               ("loss", "+", "-", "online hard-example mining: backprop only top-loss (hardest) samples (Shrivastava 2016)", "search"),
    "varifocal loss":     ("loss", "+", "0", "IoU/quality-aware focal; weights positives by localisation quality (Zhang VarifocalNet 2021)", "search"),
    "deep supervision":   ("detector", "+", "-", "auxiliary losses at multiple decoder depths → stronger gradient to small/dim objects (Lee DSN 2015)", "search"),
    # ★ 2024-2026 STATE OF THE ART — latest in-domain 3D cell/nucleus detection, segmentation & tracking.
    "U-Mamba":            ("backbone", "+", "+", "hybrid CNN+Mamba SSM: long-range context at LINEAR cost for 3D biomedical seg (Ma 2024)", "search"),
    "SegMamba":           ("backbone", "+", "+", "whole-volume Mamba encoder; global 3D receptive field, faster than transformer (Xing MICCAI 2024)", "search"),
    "MedNeXt":            ("backbone", "+", "0", "scalable ConvNeXt for 3D medical with LARGE kernels + deep sup — validates our stem=[7,7,7] (Roy MICCAI 2023)", "search"),
    "nnU-Net ResEnc-L":   ("backbone", "+", "-", "residual-encoder nnU-Net 'revisited' STILL beats transformers on 3D biomedical (Isensee 2024)", "search"),
    "Primus 3D transformer": ("backbone", "0", "--", "pure-ViT 3D medical that finally matches CNNs with the right tricks (Wald 2024)", "search"),
    "VISTA3D":            ("detector", "+", "-", "3D medical foundation segmentation model, promptable + auto (NVIDIA 2024)", "adopt-if-foundation"),
    "SAM2 promptable":    ("detector", "0", "-", "promptable video/volume segmentation; strong zero-shot, needs prompts/finetune (Ravi 2024)", "adopt-if-foundation"),
    "micro-SAM":          ("detector", "+", "-", "SAM specialised for MICROSCOPY: interactive + automatic instance seg (Archit 2024)", "adopt-if-foundation"),
    "Cellpose3 restore":  ("detector", "+", "0", "image-restoration → segmentation; recovers DIM/noisy nuclei before detection — fits our dim-miss (Stringer 2024)", "search"),
    "Cellpose-SAM":       ("detector", "+", "-", "SAM backbone in Cellpose → generalist crowded-cell segmentation (Pachitariu 2025)", "adopt-if-foundation"),
    "Trackastra":         ("linker", "+", "0", "transformer that learns cell-track ASSOCIATION incl. divisions, no tuning — in-domain (Gallusser ECCV 2024)", "adopt-data"),
    # ★★ PIPELINE LEVERS (verified 2026-07-11 by deep-research — where the REAL score is once detector=DoG-parity + edge-linking saturated) ★★
    "global-motion registration": ("linker", "++", "0", "compensate whole-volume setup drift BEFORE linking so true match falls inside gate — ultrack Tribolium TRA 0.443→0.623 (+0.18); OUR 47%-jump embryo; GENERALIZING→LB-transferable", "adopt-now"),
    "gap-closing / track-stitch": ("linker", "+", "0", "2nd-pass LAP connecting track-end→track-start across k=1-3 missed/frozen frames → recovers adjacency edge_jaccard; gate to motion-consistent (ultrack close_tracks_gaps, KIT-GE min-cost-flow)", "adopt-now"),
    "topology-gated division":  ("linker", "+", "0", "propose division ONLY where accepted parent splits into 2 detections + daughter-symmetry + non-overlap gate (NOT synthetic masks — that flooded FP); ultrack division_weight (Salehi/royerlab)", "prototype"),
    "ILP / min-cost-flow link": ("linker", "0", "-", "global optimizer — SKIP as edge lever (our learned-edge Δ0.000, greedy≈ILP once assoc good); use ONLY as vehicle for division+gap constraints", "skip"),
    # 3D BIOMEDICAL BACKBONE — in-domain nucleus/cell detection literature (our task: dim 3D nuclei, light-sheet,
    # 2-embryo generalisation). The from-basics search found stem=[7,7,7] (big RF) + batch-norm won → these papers
    # explain WHY and point to the next backbone axes. PROVEN only via arch-search.
    "nnU-Net recipe":     ("backbone", "+", "0", "self-configuring 3D U-Net; residual enc + deep sup + big patch — SOTA 3D biomedical baseline (Isensee 2021)", "search"),
    "residual blocks":    ("backbone", "+", "0", "identity skips ease deep-net optimisation; our search kept blocks_per_stage=3 → residual matters (He 2016)", "adopt-cheap"),
    "Attention U-Net gates": ("backbone", "+", "-", "attention gates on skip connections suppress irrelevant BG, focus on faint structures → dim nuclei (Oktay 2018)", "search"),
    "SE channel attention": ("backbone", "+", "0", "squeeze-excite recalibrates channels ~free; helps low-contrast feature emphasis (Hu 2018)", "search"),
    "ASPP dilated context": ("backbone", "+", "-", "atrous spatial pyramid = multi-scale context; low-contrast/dim nuclei need surrounding context (Chen DeepLab 2017)", "search"),
    "Swin-UNETR encoder": ("backbone", "+", "--", "shifted-window transformer encoder for 3D medical; strong but heavy (Hatamizadeh 2022)", "search"),
    "StarDist star-convex": ("detector", "+", "0", "star-convex polygon nucleus rep; IN-DOMAIN SOTA for crowded nuclei (Weigert/Schmidt 2020)", "search"),
    "Cellpose flow-field": ("detector", "+", "0", "gradient-flow representation separates touching cells; in-domain (Stringer 2021)", "search"),
    "large-kernel stem":  ("backbone", "+", "0", "big early receptive field captures whole nucleus + context — our search PICKED stem=[7,7,7] (RepLKNet Ding 2022)", "adopt-data"),
    # vision / detection
    "ConvNeXt-V2 block":  ("detector", "+", "0", "modern conv (GRN + depthwise) rivals ViT, conv-fast (Woo 2023)", "search"),
    "EfficientViT":       ("detector", "0", "++", "cascaded group attention for fast dense detection (Liu 2023)", "search"),
    "depthwise-separable": ("detector", "-", "++", "factorised conv → far fewer FLOPs; small acc cost", "search"),
    "anisotropic 3D conv": ("detector", "+", "+", "z-thin kernels for anisotropic voxels (our data) — fewer params", "adopt-data"),
    # training-time (help accuracy, no inference cost)
    "EMA weights":        ("training", "+", "0", "exponential-moving-avg of weights → free generalisation", "adopt-cheap"),
    "cosine + warmup LR": ("training", "+", "0", "standard schedule; stabilises + improves", "adopt-cheap"),
    # PRECISION / quantization — a hardware-dependent SPEED+MEMORY axis (5090=Blackwell FP4/FP8; T4=FP16/INT8)
    "BF16/FP16 mixed":    ("precision", "0", "+", "standard mixed precision; 2x speed/mem, ~no acc loss; works on T4", "adopt-cheap"),
    "FP8 (E4M3/E5M2)":    ("precision", "0", "++", "Hopper/Blackwell native; ~2x over FP16 for training (not on T4)", "search"),
    "NVFP4 microscaling": ("precision", "-", "++", "Blackwell native 4-bit, block-16 scaling; 2-3x over FP8, half mem; stochastic rounding needed (NVIDIA 2026)", "search"),
    "MXFP4":              ("precision", "-", "++", "microscaling FP4 (block-32); FQT weights+acts+grads at 4-bit", "search"),
    "INT8 PTQ":           ("precision", "-", "+", "post-training int8 quant for INFERENCE; T4-friendly, fast", "search"),
    "INT4 (GPTQ/AWQ)":    ("precision", "-", "+", "4-bit weight quant for inference; more acc cost, big mem save", "search"),
    "BitNet b1.58 ternary": ("precision", "-", "++", "1.58-bit ternary {-1,0,1} weights; train-from-scratch native low-bit", "search"),
    "stochastic rounding": ("precision", "+", "0", "essential for stable FP4/low-bit training (removes quant bias)", "adopt-if-fp4"),
}


def report(q, worker):
    spec = (q.get("spec") or {}) if isinstance(q, dict) else {}
    extra = spec.get("catalog")
    cat = {**INNOVATIONS, **(extra if isinstance(extra, dict) else {})}
    try:
        sw = float(spec.get("speed_weight", 1.0))
    except Exception:  # noqa: BLE001
        sw = 1.0
    def score(acc, spd):
        v = {"++": 2, "+": 1, "0": 0, "-": -1, "--": -2}
        return v.get(acc, 0) + sw * v.get(spd, 0)          # unknown symbols in a custom catalog → neutral, no crash
    # only well-formed 5-tuple entries (stage, acc, speed, note, status) are scored — bad rows skipped, not crash
    cat = {n: m for n, m in cat.items() if isinstance(m, (list, tuple)) and len(m) == 5}
    rows = sorted(((n, *m, score(m[1], m[2])) for n, m in cat.items()), key=lambda r: -r[-1])

    # emit the candidates arch-search should PROVE (cheap+data ones flagged to adopt, rest to search)
    adopt = [n for n, st, ac, sp, note, status, sc in rows if status.startswith("adopt")]
    search = [n for n, st, ac, sp, note, status, sc in rows if status == "search"]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    try:
        import yaml
        OUT.write_text(yaml.safe_dump({"adopt_cheap": adopt, "search_candidates": search,
                                       "ranked": [{"name": r[0], "stage": r[1], "acc": r[2], "speed": r[3],
                                                   "note": r[4], "status": r[5]} for r in rows]}, sort_keys=False))
    except Exception:  # noqa: BLE001
        OUT.write_text(json.dumps({"adopt_cheap": adopt, "search_candidates": search}, indent=2))
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps({"n": len(rows), "adopt": adopt, "search": search}, indent=2))

    from . import ledger
    ledger.log("paper-research",
               summary=f"{len(rows)} recent architecture innovations catalogued (acc×speed); {len(adopt)} cheap-wins, {len(search)} to prove via arch-search",
               detail="; ".join(f"{r[0]}(acc{r[2]},spd{r[3]})" for r in rows[:8]), kind="finding",
               recommendation="adopt the cheap free-wins (RMSNorm/SwiGLU/EMA); arch-search must PROVE the rest — none assumed")
    from researchpapers.fleet import post
    top = "\n".join(f"{'✅' if r[5].startswith('adopt') else '🔬'} **{r[0]}** ({r[1]}) acc`{r[2]}` speed`{r[3]}` — {r[4]}"
                    for r in rows[:8])
    msg = (f"[{worker}] **PAPER-RESEARCH** · {len(rows)} recent innovations · acc×speed ranked (speed_weight {sw})\n"
           f"{top}\n"
           f"**Cheap free-wins to adopt:** {', '.join(adopt[:6])}\n"
           f"**To PROVE via arch-search (not assumed):** {', '.join(search[:6])}\n"
           f"→ `config/arch/innovation_candidates.yml`. Speed is first-class (T4 runtime-limited).")
    post.post_thread(worker, "all", msg, routine=False, kind="finding")
    return ("done", {"n": len(rows), "adopt_cheap": adopt, "search_candidates": search,
                     "out": str(OUT)}, "all", msg)
