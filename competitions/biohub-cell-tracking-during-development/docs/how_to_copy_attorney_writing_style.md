# How to Capture and Reproduce a Patent Attorney's Writing Style

*A practical, technical guide for a patent/IP drafting-assistant (ipauthor.com use case).*
*Last updated: 2026-07-16. All citations are real URLs retrieved during research.*

---

## TL;DR

Yes, it is feasible to capture and reproduce a specific patent attorney's (or an assignee's portfolio's) drafting style from their published work, and to **measure** how well a generated draft matches it. The state of the art gives you two independent things you need:

1. **A style signal you can compute** — a learned authorship/style embedding (LUAR / StyleDistance) whose **cosine similarity** is the primary, content-independent match metric, backed by an interpretable classical metric (**Burrows's Delta** over function words).
2. **A patent-specific quality/structure signal** — **PatentScore / PatClaimEval / PatentEval**, which score claim structure, antecedent basis, and legal correctness (things generic style metrics ignore but that *are* the essence of patent "style").

The winning move is not just to measure but to **optimize the drafting prompt against the style metric** (DSPy/GEPA-style reflective optimization) using the attorney's own held-out patents as targets. Our existing fleet (`prompt-dataset` → `prompt-metric` → runner → `dspy-prompt-optimize`/GEPA) maps onto this directly.

---

## 1. Feasibility & Framing

### Is style protected? (short, factual)

Copyright protects the **specific expression** of a work, not the **style, method, or system** behind it. Copying an author's *style* — the manner or approach — without copying protected expressive elements is generally **not** infringement, even when the style is distinctive. So training a model on an attorney's *own* past drafting to assist *their own future* drafting is a legitimate professional use: you are reproducing a manner of expression for the person who authored it, not copying a competitor's protected text.

- U.S. Copyright Office — What writers should know: <https://www.copyright.gov/engage/writers/>
- Copyright Alliance — writers/style: <https://copyrightalliance.org/education/industry/writers/>
- (Research context on stylistic appropriation under EU law, 2026): <https://arxiv.org/pdf/2606.31250>

Two caveats worth keeping in mind operationally (not legal advice): (a) don't reproduce another firm's boilerplate verbatim where it is a substantial original text; (b) the deliverable is a *drafting aid*, and the attorney remains responsible for claim scope and correctness.

### What "writing style" concretely means for patent prose

Patent "style" is far more structured than literary style. Concretely, for an attorney it decomposes into:

| Layer | Style features to capture |
|---|---|
| **Claim architecture** | Preamble phrasing, transitional phrase choice (`comprising` vs `consisting of` vs `consisting essentially of`), single-period claim structure, element indentation/paragraphing, dependency phrasing ("The apparatus of claim 1, wherein…") |
| **§112(f) usage** | Whether the attorney uses `means for` / `configured to` / `a … module` and how consistently they avoid or invoke means-plus-function |
| **Antecedent-basis discipline** | Introducing elements with `a`/`an`, referring back with `the`/`said`, consistency of `said` vs `the` |
| **Boilerplate & hedging** | "In one embodiment", "in some implementations", "by way of example and not limitation", "it should be understood that", "configured to", "operatively coupled to" |
| **Defined-term conventions** | How terms are defined ("As used herein, …"), capitalization of defined terms, lexicography habits |
| **Reference-numeral patterns** | Numbering scheme (10/12/14 vs 100/102), whether numerals appear in claims, figure-to-spec numeral discipline |
| **Specification ordering** | Field / Background / Summary / Brief Description of Drawings / Detailed Description ordering and section headings |
| **Surface stylometry** | Sentence length & burstiness, function-word distribution, readability, passive-voice rate, hedging density |

The practical consequence: a good "style match" metric must reward **surface stylometry** (function words, sentence rhythm) *and* **structural/legal conventions** (claim format, antecedent basis). No single off-the-shelf metric does both — you combine them (Section 4, Section 6).

- Claim structure & transitions (WIPO drafting): <https://www.wipo.int/edocs/mdocs/aspac/en/wipo_ip_kul_17/wipo_ip_kul_17_5.pdf>
- Claim parts / antecedent basis / means-plus-function overview: <https://www.intepat.com/blog/patent-claims-structure-and-types-of-patent-claims>

---

## 2. How to Get the Corpus (the attorney's finished publications)

You need a clean corpus of the target attorney's (or assignee's) granted patents and published applications. Real sources and endpoints:

| Source | What you get | Endpoint / URL | Attorney/agent identifiable? |
|---|---|---|---|
| **USPTO PatentsView (Search API)** | Disambiguated granted-patent + pre-grant data; **`/attorneys`** and **`/assignees`** endpoints | Endpoint dictionary: <https://search.patentsview.org/docs/docs/Search%20API/EndpointDictionary/> · Attorneys endpoint: <https://patentsview.org/apis/api-endpoints/attorneys> | **Yes** — attorney/agent is a first-class disambiguated entity |
| **USPTO Open Data Portal (ODP)** | Bulk + search API for grants & applications, full text; replaces the legacy PatentsView bulk products | Portal: <https://data.uspto.gov/> · Bulk search API: <https://data.uspto.gov/apis/bulk-data/search> · Bulk dir: <https://data.uspto.gov/bulkdata> · PatentsView transition: <https://data.uspto.gov/support/transition-guide/patentsview> | Correspondence / agent fields in bibliographic data |
| **Google Patents Public Data (BigQuery)** | 90M+ publications, US full text, `patents.publications` table; join on assignee / attorney fields; research table adds embeddings & top-terms | Repo: <https://github.com/google/patents-public-data> · Schema table: <https://github.com/google/patents-public-data/blob/master/tables/dataset_Google%20Patents%20Public%20Datasets.md> · Programmatic guide: <https://www.aipla.org/list/innovate-articles/programmatic-patent-searches-using-google-s-bigquery-public-patent-data> | Assignee reliable; attorney via correspondence text |
| **EPO Espacenet OPS (Open Patent Services)** | REST API behind Espacenet: bibliographic, full text, images, legal status; 130M+ | Developers: <https://developers.epo.org> · Docs (PatZilla): <https://docs.ip-tools.org/patzilla/datasource/epo-ops.html> · Python client: <https://pypi.org/project/python-epo-ops-client/> | Applicant/representative fields; weekly 2.5 GB fair-use limit |
| **Lens.org API** | Merged patent + scholarly, 120+ searchable fields, JSON REST, bulk (PatSeq) | About: <https://about.lens.org/lens-apis/> · Docs: <https://docs.api.lens.org/> · Getting started: <https://docs.api.lens.org/getting-started.html> | Applicant/agent fields searchable |

### How the attorney/agent/firm is identifiable — and the key limitation

- **PatentsView** exposes a disambiguated **`attorneys`** entity (like inventors/assignees), so you can query "all patents where attorney = X" or "assignee = CompanyY". This is the cleanest programmatic path to a named-attorney corpus.
- In raw USPTO/EPO bibliographic data the signal lives in **correspondence / agent / representative** fields.
- **Critical limitation: attorney-of-record ≠ drafter.** The person listed as attorney/agent of record is frequently *not* the associate who actually wrote the specification and claims. A partner's name may appear across drafts written by many associates; conversely, a paralegal-filed continuation carries the same name as an original the partner personally drafted. **Treat "attorney of record" as a noisy label**, prefer the assignee/portfolio level when you want stylistic consistency of a *firm/team*, and, when targeting one human, validate that the corpus is stylistically coherent (Section 5, step 3 — run authorship-verification on the corpus itself to detect multi-drafter contamination before you trust it as a single-style target).

---

## 3. Methods to Capture / Reproduce the Style (with tradeoffs)

| Method | How it works | Pros | Cons / when to use |
|---|---|---|---|
| **Few-shot exemplar prompting** | Put 3–20 of the attorney's own claims/spec passages in-context as style exemplars; instruct "match the style of these examples" | Zero training; instant; easy to update; strongest ROI for small corpora | Context-window limited; can copy *content* not just style; brittle to exemplar selection. **Default starting point.** |
| **Stylometric feature conditioning** | Extract function-word distribution, sentence-length/burstiness, readability, hedging rate; feed as an explicit style profile / system-prompt spec, or use to *evaluate* | Interpretable; cheap; great as a metric and as a guardrail | Coarse; doesn't capture claim structure by itself |
| **Style embeddings (LUAR / StyleDistance / STEL-tuned)** | Encode the corpus into a fixed **style vector**; measure cosine of a draft against it; can also condition generation | Content-independent, learned, robust; the best single *automatic* match signal | Needs the embedding model; a vector isn't directly human-readable |
| **RAG over the corpus** | Retrieve the attorney's most similar prior passages/boilerplate at draft time and inject phrasing/defined-term templates | Reproduces *exact boilerplate & defined-term conventions*; grounded; updates as portfolio grows | Retrieval can leak stale content/claim scope; needs dedup |
| **Fine-tuning / LoRA / DoRA** | Train a low-rank adapter on the corpus so style is *persistent* without exemplars in context | Best style fidelity for larger corpora; frees the context window; StyleTunedLM shows LoRA beats prompting/few-shot at capturing training-data style | Needs enough data (rule of thumb ≥ ~50–100 documents); risk of overfitting boilerplate & memorizing claim scope; retrain to update |
| **Prompt optimization against a style metric (DSPy/GEPA)** | Define a style-similarity **reward** (Section 4) and let a reflective optimizer evolve the drafting prompt to maximize it on held-out patents | Directly optimizes the thing you care about; metric-driven; sample-efficient (GEPA beats RL with ~35× fewer rollouts) | Only as good as the metric; needs a held-out target set. **This is the key lever.** |
| **Style-content disentanglement / paraphrase transfer** | Generate content first, then restyle via paraphrase-based style transfer conditioned on the style vector | Cleanly separates "what to claim" from "how to phrase it" | Extra pipeline stage; transfer can degrade legal precision |

Key references:
- LUAR (Learning Universal Authorship Representations): <https://github.com/LLNL/LUAR> · paper: <https://aclanthology.org/2021.emnlp-main.70/> (via <https://www.researchgate.net/publication/357125222_Learning_Universal_Authorship_Representations>)
- StyleDistance (content-independent style embeddings, 2024/25): <https://arxiv.org/html/2410.12757>
- LoRA for style (StyleTunedLM / PEFT generation style, 2024): <https://arxiv.org/html/2409.04574v1>
- Panza (fully-local personalized writing assistant, RAG + fine-tune, 2024): <https://arxiv.org/pdf/2407.10994>
- DSPy GEPA optimizer: <https://dspy.ai/api/optimizers/GEPA/overview/> · GEPA paper (ICLR 2026 oral): <https://arxiv.org/pdf/2507.19457>

---

## 4. Best Metrics (2024–2026) — the core of the ask

> **Lead with the fingerprint, not the embedding.** LUAR cosine and PatentScore are *holistic* — they tell you a draft *feels* like the attorney (or is well-formed), but they do **not** measure his **concrete fingerprints**: the signature phrases he reuses, how he opens each paragraph, his section/claim structure, and the boilerplate he pastes into every patent. A per-attorney assistant has to match *those*, and — crucially — has to turn every mismatch into **feedback an optimizer (GEPA/DSPy) can act on**. So the primary metric here is a **layered, computable fingerprint suite**; LUAR/PatentScore become the secondary/backstop layer (Section 4.2).

### 4.1 The layered fingerprint suite (PRIMARY) — his concrete habits, per layer

Each layer captures one specific "his X", is cheaply computable from his own corpus (no model, no network), returns a score in `[0,1]`, **and** emits a human-readable miss the optimizer can edit the prompt against.

| Layer | The "his X" it captures | Exact computation | Feedback it yields to the optimizer |
|---|---|---|---|
| **Signature phrases** | The distinctive multi-word phrases he reuses ("it will be appreciated that", "operatively coupled to") and *how often* | Distinctive n-grams (n=2..6) ranked by **cross-document frequency × distinctiveness**. Distinctiveness = **Dunning log-likelihood keyness** vs a background corpus of other attorneys (signed, keep over-represented only); with no background, fall back to document-frequency (text-dispersion keyness). Store each phrase's **characteristic per-1k-token rate**. | *"missing signature phrases: 'it will be appreciated that', 'operatively coupled to'"* and *"signature-phrase RATE off vs his characteristic per-1k usage"* |
| **Openers (prefix)** | How he **starts** paragraphs and sentences ("In one embodiment,", "Referring now to"), and his transition-word habits | Distributions over **paragraph-initial and sentence-initial 1–3grams** (headers excluded) + a transition-word profile. Draft scored by **`1 − Jensen–Shannon`** divergence of its opener distribution vs his. | *"sentence-opener skew: he leads with 'In one embodiment', draft leads with 'The system'"* |
| **Closers (postfix)** | How he **ends** paragraphs and sentences ("…without departing from the scope of the invention.", "…as will be appreciated.") | Distributions over **paragraph-final and sentence-final 1–3grams**; draft scored by **`1 − Jensen–Shannon`** of its closer distribution vs his. | *"sentence-closer skew: he ends with 'the appended claims', draft ends with 'the disclosure'"* |
| **Word affixes (prefix/suffix)** | His morphological signature — character **prefixes/suffixes** he over-uses (Latinate `-tion`/`-ment`, gerund `-ing`, `pre-`/`re-`) | Distributions over the first-3 and last-3 characters of content words (len ≥ 5); draft scored by **`1 − Jensen–Shannon`** vs his prefix + suffix distributions. | *"morphology drift: suffix `-tion` under-used vs his rate"* |
| **Section structure** | His canonical spec skeleton (FIELD → BACKGROUND → SUMMARY → DETAILED DESCRIPTION) and claim transitional form | Detect ALL-CAPS / "FIELD OF THE INVENTION"-style **headers**, learn the canonical **ordered sequence** + modal wording. Score = **LCS sequence alignment** to his order **+** claim-transition conformity (comprising/consisting/wherein, "the X of claim N, wherein" rate). | *"missing/wrong-order sections: SUMMARY, DETAILED DESCRIPTION"* ; *"claim transition off: he uses 'comprising', draft uses 'consisting of'"* |
| **Cross-patent boilerplate** | The near-verbatim blocks he pastes into *many* of his patents ("The foregoing description is illustrative and not restrictive…") | Mine reused passages by **k-word shingling + a MinHash signature + greedy Jaccard clustering** (datasketch-style near-dup detection); keep clusters spanning ≥2 of his patents. Draft scored by **max shingle-Jaccard** to each canonical block. | *"missing canonical boilerplate block (in 9 of his patents): 'The foregoing description is illustrative…'"* |
| **Micro-conventions** | His fingerprints below the phrase level | **Reference-numeral scheme** (10/12 vs 100/102, per-1k rate), **hedging density** (may/can/in some embodiments), **passive-voice proxy**, **mean sentence length**, **defined-term conventions** ("As used herein"). Score = `1 − mean relative deviation` from his profile. | *"hedging density too low vs target"*, *"reference-numeral scheme is 100-series; his is 10-series"* |

**Composite** = a weighted sum of the eight layer scores (sensible defaults `phrase 0.22 / phrase-rate 0.08 / opener 0.15 / closer 0.10 / affix 0.07 / structure 0.17 / boilerplate 0.13 / micro 0.08`, overridable). Report **per-layer, never only the composite**, so the attorney sees *which* habit matches or drifts.

**Implemented as: the `style-fingerprint` agent** (`fleet_agents/style_fingerprint.py`, pure-python + numpy, data-wise test `test_fleet_agents/style_fingerprint_test.py`). It exposes `build_profile(corpus, background=None)`, `score(draft, profile)` (per-layer + composite + `feedback` string), `discrimination_auc(profile, positives, negatives)`, and `as_metric(profile) → (score_fn, feedback_fn)` — the last plugs **straight into `dspy-prompt-optimize` / GEPA** as a reference-free style reward (the runner drafts, this scores + feeds back). `gold` is ignored: style is measured against the *profile*, not a gold text.

- Keyness / log-likelihood distinctive n-grams: <https://www.refsmmat.com/notebooks/keyness.html> · LL vs odds-ratio (Pojanapunya & Todd): <https://www.degruyterbrill.com/document/doi/10.1515/cllt-2015-0030/html> · text-dispersion keyness for lexical bundles (2025): <https://www.sciencedirect.com/science/article/abs/pii/S2772766125000667>
- Paragraph/sentence-opener stylometry: <https://aclanthology.org/2025.findings-acl.913.pdf> · PAN'25 style-change (paragraph-initial = the style marker): <https://pan.webis.de/clef25/pan25-web/style-change-detection.html>
- Near-duplicate boilerplate (shingling + MinHash + LSH): <https://yorko.github.io/2023/practical-near-dup-detection/> · <https://mbrenndoerfer.com/writing/minhash-algorithm-jaccard-similarity-lsh-deduplication>

### 4.1.1 Validation — does the fingerprint actually identify HIM? (discrimination-AUC)

A fingerprint is only trustworthy if it scores **his** patents high and **other attorneys'** low. So validate it the way authorship-verification is validated in the literature — with **ROC-AUC of the separation**: `discrimination_auc(profile, positives=held-out-his-patents, negatives=other-attorneys)` scores both sets with `score()` and returns the ROC-AUC (Mann–Whitney, tie-corrected). A good fingerprint → **AUC ≈ 1.0**; on the synthetic two-attorney test it returns **1.000**. Run this *before* trusting the fingerprint as an optimization target — if AUC is near 0.5, the layers/weights aren't capturing him (or the corpus is multi-drafter; see Section 5, step 3). ROC-AUC is the standard AV metric: TDRLM reports AUC 92.56 for stylometric verification; hybrid stylometric-transformer frameworks report ROC-AUC/F1/Brier/C@1.

- Authorship-verification ROC-AUC as the validation metric: <https://www.sciencedirect.com/science/article/abs/pii/S0957417423012472> (TDRLM) · combining style + semantics for robust AV (2025): <https://www.sciencedirect.com/science/article/pii/S266682702500115X>

### 4.2 Holistic embedding / quality metrics (SECONDARY / BACKSTOP)

The fingerprint suite is deliberately surface-level and interpretable; it can be gamed by a draft that copies phrases but reads wrong overall. So keep the holistic metrics as a **backstop layer**: LUAR/StyleDistance cosine catches "feels like a different author even though the phrases match", and PatentScore/PatClaimEval catch "well-styled but legally malformed claims". Use them as co-objectives and gates, not as the primary per-habit signal.

### Metrics comparison table

| Metric | What it measures | Best for | Link |
|---|---|---|---|
| **LUAR / UAR cosine** | Cosine between learned authorship-representation vectors; captures an author's stylistic fingerprint across topics | **Primary automatic style-match** for one attorney; robust, content-tolerant | <https://github.com/LLNL/LUAR> |
| **StyleDistance cosine** | Cosine over *content-independent* style embeddings trained on synthetic parallel data (0.87 STEL, 0.29 STEL-or-Content) | Style match when you must suppress topic leakage (patents share heavy jargon) | <https://arxiv.org/html/2410.12757> |
| **STEL / STEL-or-Content** | Framework score: can the representation match same-style texts while rejecting same-content paraphrases | *Validating* that your style metric is really measuring style, not topic | <https://www.cis.upenn.edu/~ccb/publications/interpretable-style-embeddings.pdf> · STEB benchmark: <https://arxiv.org/pdf/2606.31741> |
| **PAN authorship-verification metrics** | AV shared-task scoring (AUC, c@1, F1, F0.5u, Brier → averaged); "are these two texts by the same author?" | Verifying a *draft* is attributable to the target's style; corpus-coherence check | PAN'24: <https://pan.webis.de/clef24/pan24-web/style-change-detection.html> · PAN'25: <https://pan.webis.de/clef25/pan25-web/style-change-detection.html> |
| **MAUVE** | Distributional gap between generated-text and human-text distributions in embedding space (KL divergence frontiers) | *Corpus-level* realism: does the batch of drafts distribute like the attorney's real patents | Wiki: <https://en.wikipedia.org/wiki/MAUVE_(metric)> · paper: <https://arxiv.org/abs/2102.01454> · pkg: <https://pypi.org/project/mauve-text/> |
| **Burrows's Delta** | Z-scored distance over most-frequent (function) words; classic interpretable stylometric distance | **Interpretable secondary** style metric; explains *why* styles differ | <https://www.tandfonline.com/doi/full/10.1080/09296174.2026.2612931> · variants: <https://arxiv.org/pdf/2604.19499> |
| **Function-word JS divergence** | Jensen–Shannon divergence of function-word/POS distributions | Cheap, transparent guardrail; catches drift in hedging/boilerplate rate | (classical; computed directly — see Delta refs above) |
| **Text-style-transfer triad** (transfer accuracy · content preservation · fluency) | Style-classifier accuracy; BLEU/BERTScore/BLEURT for content; GPT-perplexity/LLM for fluency; often reported as geometric mean | Measuring the *style-vs-content tradeoff* when restyling a fixed technical disclosure | Survey/eval: <https://arxiv.org/html/2502.04718v1> · LLM-based TST eval: <https://aclanthology.org/2024.lrec-main.1373/> |
| **LLM-as-judge rubric** | Prompted model scores claim-structure adherence, antecedent basis, boilerplate match against a rubric | Structure/legal conventions no embedding captures; fast qualitative signal | Bias survey: <https://arxiv.org/html/2410.02736v1> · scoring-bias: <https://arxiv.org/html/2506.22316v1> · reliability: <https://arxiv.org/pdf/2606.19544> |
| **PatentEval** | Human-anchored error typology for patent generation (claims→abstract, next-claim); studies which auto-metrics track expert judgment | Patent-specific error diagnosis; grounds any auto-metric in expert labels | <https://aclanthology.org/2024.naacl-long.147.pdf> · arXiv: <https://arxiv.org/abs/2406.06589> |
| **PatentScore** | Multi-dim (structural / legal / semantic) score of generated claims; **r=0.819 with experts** vs *negative* r for BLEU/ROUGE/BERTScore | **Patent-specific quality/structure** match — antecedent basis, claim formatting, enforceability | <https://arxiv.org/html/2505.19345v2> |
| **PatClaimEval / Patent-CE** | 5-dimension expert-annotated claim eval (feature completeness, conceptual clarity, terminology consistency, logical linkage, overall) | Comprehensive claim-quality benchmark; expert-aligned | <https://arxiv.org/abs/2505.11095> · ACL'25: <https://aclanthology.org/2025.acl-long.190/> |

### Why plain n-gram metrics fail here

The patent-eval literature is unusually clear on this: for generated patent claims, **BLEU, ROUGE-L, and even BERTScore correlate *negatively* with expert judgment** (PatentScore reports r ≈ −0.12 to −0.16), because good claims deliberately differ in surface tokens while preserving structure and scope. Do **not** use BLEU/ROUGE as a style-match objective for patents. Use them only as weak content-preservation checks inside a style-transfer triad.

### Ranking for THIS use case (matching one attorney's style)

1. **Layered fingerprint suite (`style-fingerprint`) — best primary.** The only metric that scores his *concrete* habits per layer (signature phrases, openers, section/claim structure, cross-patent boilerplate, micro-conventions) **and** hands GEPA/DSPy actionable per-layer feedback. Validate it with **discrimination-AUC** (Section 4.1.1) before trusting it. This is the thing you optimize.
2. **PatentScore (and/or PatClaimEval) — best domain complement.** Captures the *patent-specific* legal correctness (antecedent basis, enforceable claim form) that even the structure layer only approximates; expert-validated. Use as a co-objective/gate.
3. **LUAR / StyleDistance cosine — best holistic backstop.** Learned, content-independent scalar that catches "feels like a different author even though the surface phrases match." Use StyleDistance when topic leakage is a worry (one narrow practice area). A backstop co-objective, not the per-habit signal.
4. **Burrows's Delta / function-word JS — interpretable cross-check.** Overlaps the micro layer (function-word distribution, hedging, sentence length); keep as an independent sanity check on the fingerprint's micro-conventions.
5. **PAN-style authorship verification — corpus QA gate.** Use it to vet the corpus (detect multi-drafter contamination) before building the profile; discrimination-AUC then plays the same role *for the fingerprint itself*.
6. **LLM-as-judge rubric — useful but caveated.** Great for legal-structure adherence it can articulate; subject to verbosity/position/self-enhancement bias and 60–68% expert agreement in specialized domains — pin to a concrete checklist, randomize order, calibrate against human scores. Never the *sole* objective.
7. **MAUVE — corpus-level only.** Needs ~1000+ samples for a stable estimate; a batch/portfolio diagnostic, not a per-draft reward.

---

## 5. A Concrete Workflow / Recipe You Can Run

> Goal: produce a drafting assistant whose output, on held-out patents of the target attorney, maximizes a defined style-match score.

1. **Pull the corpus.** Query PatentsView `attorneys`/`assignees` (or ODP bulk search) for all grants + pre-grant publications for the target attorney/assignee. Pull US full text from Google Patents BigQuery `patents.publications`; use EPO OPS / Lens if the target files in EP.
   - PatentsView: <https://search.patentsview.org/docs/docs/Search%20API/EndpointDictionary/> · BigQuery: <https://github.com/google/patents-public-data>
2. **Clean & segment.** Split each document into **claims** (independent vs dependent), **abstract**, and **spec sections** (Field / Background / Summary / Detailed Description). Strip figures/OCR artifacts. Deduplicate continuations/divisionals (near-identical specs will bias both training and metrics). Keep claims and spec as *separate* style targets — they have different registers.
3. **Build the style profile & vet coherence.**
   - Compute a **stylometric profile**: function-word distribution, sentence-length distribution + burstiness, readability, hedging/boilerplate frequencies, transitional-phrase and `means for`/`configured to` rates, `said`-vs-`the` ratio, reference-numeral scheme.
   - Compute the **style-embedding centroid** (LUAR/StyleDistance) over the corpus.
   - **Coherence check:** run PAN-style authorship verification *within* the corpus. If documents don't verify as one author, the "attorney" label is multi-drafter — either cluster into sub-styles or fall back to portfolio-level style. (This is the drafter≠attorney-of-record guard.)
4. **Choose the method** (Section 3): start with **few-shot** (retrieve the k most-recent, most-representative exemplars via RAG). Add **LoRA** only if the coherent corpus is large enough (≥ ~50–100 docs).
5. **Hold out a target set.** Reserve the attorney's most-recent ~15–20 patents as held-out targets (recent = current style). For each, construct a `{prompt-input → their-patent}` example: the input is a neutralized disclosure/summary or a "draft claim 1 for this invention" instruction; the target is their actual claim/spec.
6. **Define the style reward** = weighted combination:
   `reward = w1·cos_style(draft, attorney_centroid)  +  w2·PatentScore(draft)  −  w3·|Delta(draft) − Delta_target|  +  w4·LLMjudge_structure(draft)`
   (start w = [0.4, 0.3, 0.2, 0.1]; tune on held-out). Include a **content-fidelity guard** (BERTScore/entailment vs the intended disclosure) so the optimizer can't win by drifting claim scope.
7. **Optimize the drafting prompt** with DSPy **GEPA** against that reward on the held-out targets. GEPA's reflective `Prediction(score, feedback)` interface lets you feed *textual* feedback ("antecedent basis broken for 'the module'", "hedging density too low vs target") so it edits the prompt intelligently.
   - GEPA: <https://dspy.ai/api/optimizers/GEPA/overview/> · <https://arxiv.org/pdf/2507.19457>
8. **Evaluate on the held-out set** with the full metric panel (LUAR cosine, PatentScore, Delta, PAN-AV verify-rate, MAUVE at corpus level). Report per-metric, not a single blended number, so the attorney sees *what* matches.
9. **Human review (mandatory).** A qualified attorney reviews claim scope, antecedent basis, and §112 compliance. The metrics gate *candidates*; the human gates *correctness*. Log disagreements to recalibrate the LLM-judge rubric.

### Where our fleet already fits (implementation path)

This recipe is a near-exact match for existing fleet components:

- **`style-fingerprint`** → step 6 (the primary reward): `build_profile(corpus, background)` once, then `as_metric(profile)` returns the `(score_fn, feedback_fn)` GEPA needs — reference-free, per-layer, with actionable feedback. Validate with `discrimination_auc` before optimizing. (`fleet_agents/style_fingerprint.py`.)
- **`prompt-dataset`** → step 5: build `{prompt-input, their-patent}` example pairs from the attorney's corpus (claims and spec variants).
- **`prompt-metric`** → step 6 co-objectives/backstop: register the holistic named metrics (LUAR/StyleDistance cosine + PatentScore + Delta) returning `score + feedback` to blend with the fingerprint composite.
- **runner** → the drafting LLM that produces a candidate from `prompt-input`.
- **`dspy-prompt-optimize` / GEPA** → step 7: evolve the drafting prompt to maximize the blended reward on held-out patents.

So the build is: build the attorney profile with `style-fingerprint`, expose it as the primary reward via `as_metric` (plus LUAR/PatentScore backstops through `prompt-metric`), generate the pair dataset with `prompt-dataset`, and run `dspy-prompt-optimize` (GEPA backend) with the attorney's recent patents as the held-out target set. The fingerprint suite is the new core piece; the rest is existing infrastructure.

---

## 6. Recommendations

**Pragmatic best stack for a per-attorney patent-drafting assistant:**

1. **Base method: few-shot from the attorney's ~15–20 most-recent patents**, selected by RAG (retrieve the most stylistically/technically relevant exemplars per draft). Recent-weighted because attorneys' style drifts and current style is what they want reproduced. Add a **RAG boilerplate/defined-term layer** so exact conventions ("As used herein…", numeral scheme) are reproduced verbatim from *their* corpus.
2. **Primary metric: the layered fingerprint suite (`style-fingerprint`).** Build a profile from his corpus and optimize its **composite** (signature phrases + openers + section/claim structure + cross-patent boilerplate + micro-conventions), consuming the per-layer **feedback** in GEPA. It's the only metric that scores his *concrete* habits and tells the optimizer *which* one to fix. **Prove it first** with `discrimination_auc` (his held-out patents vs other attorneys' — should be ≈1.0); if it can't tell him apart, fix layers/weights or check for multi-drafter contamination before optimizing against it.
3. **Domain co-objective: PatentScore** (structural + legal + semantic) — the expert-validated signal for antecedent basis, claim formatting, and enforceability that even the structure layer only approximates. (PatClaimEval/Patent-CE as complement.)
4. **Holistic backstop: LUAR (or StyleDistance) style-embedding cosine** to the attorney's centroid — catches "reads like a different author even though the surface phrases match." Backstop, not the per-habit signal. Add **Burrows's Delta / function-word JS** as an independent cross-check on the micro layer.
5. **Structure QA: an LLM-judge rubric** scoring claim-structure adherence — but pinned to a concrete checklist, order-randomized, and calibrated against a handful of human scores (mitigating known verbosity/position/self-enhancement bias).
6. **Optional LoRA** once the *coherent* corpus exceeds ~50–100 documents: a persistent adapter gives higher fidelity than prompting and frees the context window. Below that, stay few-shot — LoRA on a tiny corpus overfits boilerplate.
7. **Optimize** the drafting prompt (and, if used, select LoRA checkpoints) with **GEPA** against the weighted `style-fingerprint`-composite + PatentScore reward on held-out patents, feeding GEPA the fingerprint's textual feedback so it edits the prompt intelligently.

**Minimum corpus size:**
- Few-shot / metric evaluation: workable at **~10–15 patents** (enough for a stable centroid + a held-out target set).
- Reliable style-embedding centroid + Delta: **~20–40 documents**.
- LoRA fine-tuning: **≥ ~50–100 documents**, and only after the coherence check passes.

**Pitfalls to watch:**
- **Overfitting boilerplate.** The model learns to spam "in one embodiment" and copy stock paragraphs while missing the *reasoning* style. Mitigate: weight the reward toward *claim-structure* and function-word distribution, not verbatim boilerplate; penalize n-gram copying of long spans.
- **Drafter ≠ attorney-of-record.** Always run the corpus-coherence (PAN-AV) check; fall back to portfolio/firm style if the corpus is multi-drafter.
- **Hallucinated claim scope.** A style-optimized model will happily produce beautifully-styled claims that are *wrong on scope*. Always include a content-fidelity/entailment guard against the intended disclosure, and keep human review mandatory.
- **Wrong metrics.** Do not optimize BLEU/ROUGE/BERTScore for patents — they correlate *negatively* with expert quality (PatentScore).
- **LLM-judge over-trust.** 60–68% expert agreement in specialized domains; calibrate and never make it the sole gate.
- **Topic leakage in embeddings.** Same-field patents share jargon; prefer content-independent embeddings (StyleDistance) and validate with STEL-or-Content.

**What to measure to know it's working:**
- **LUAR/StyleDistance cosine** of held-out generated claims to the attorney's centroid should approach the *intra-author* cosine (draft-to-draft similarity *among the attorney's own patents*) — that intra-author band is your ceiling and your target, not 1.0.
- **PAN authorship-verification** should classify generated drafts as "same author" at a rate approaching the attorney's own held-out verify-rate.
- **PatentScore** on generated claims ≥ the attorney's own baseline PatentScore.
- **Burrows's Delta** between generated and target within the attorney's own document-to-document Delta range.
- **MAUVE** (corpus-level, once you have enough drafts) close to the human corpus.
- **Human acceptance rate** and edit-distance-to-final on real drafting jobs — the ultimate metric.

**One-line recommendation:** *Few-shot from their 20 most-recent patents + LUAR (or StyleDistance) style-embedding cosine as the primary metric + PatentScore for claim structure/legal form + Burrows's Delta as an interpretable secondary + a calibrated LLM-judge rubric for claim-structure adherence, all optimized via GEPA against a held-out set of their own patents; add LoRA only once a drafter-coherent corpus exceeds ~50–100 documents.*

---

## Sources

**Layered fingerprint suite (primary metric — `style-fingerprint`)**
- Keyness / log-likelihood distinctive n-grams: <https://www.refsmmat.com/notebooks/keyness.html> · LL vs odds-ratio (Pojanapunya & Todd, *CLLT*): <https://www.degruyterbrill.com/document/doi/10.1515/cllt-2015-0030/html> · text-dispersion keyness for lexical bundles (2025): <https://www.sciencedirect.com/science/article/abs/pii/S2772766125000667>
- Paragraph/sentence-opener stylometry: <https://aclanthology.org/2025.findings-acl.913.pdf> · PAN'25 style-change (paragraph-initial marker): <https://pan.webis.de/clef25/pan25-web/style-change-detection.html>
- Near-duplicate boilerplate (shingling + MinHash + LSH): <https://yorko.github.io/2023/practical-near-dup-detection/> · <https://mbrenndoerfer.com/writing/minhash-algorithm-jaccard-similarity-lsh-deduplication>
- Authorship discrimination ROC-AUC as validation: <https://www.sciencedirect.com/science/article/abs/pii/S0957417423012472> (TDRLM, AUC 92.56) · combining style + semantics for robust AV (2025): <https://www.sciencedirect.com/science/article/pii/S266682702500115X>

**Style / authorship representation & metrics**
- LUAR (repo): <https://github.com/LLNL/LUAR> · few-shot detection ICLR'24 dir: <https://github.com/LLNL/LUAR/tree/main/fewshot_iclr2024>
- StyleDistance: <https://arxiv.org/html/2410.12757> · mStyleDistance: <https://arxiv.org/html/2502.15168>
- STEL / interpretable style embeddings: <https://www.cis.upenn.edu/~ccb/publications/interpretable-style-embeddings.pdf> · STEB benchmark: <https://arxiv.org/pdf/2606.31741> · content-independent style: <https://arxiv.org/pdf/2204.04907>
- Layered/all-layer authorial style (EMNLP'25): <https://aclanthology.org/2025.emnlp-main.521.pdf>
- Cross-genre AV benchmark (AAAI'25): <https://cocoxu.github.io/publications/AAAI_2025_Cross_genre_Authorship.pdf>

**Authorship verification / style-change (PAN @ CLEF)**
- PAN'24 style analysis: <https://pan.webis.de/clef24/pan24-web/style-change-detection.html> · PAN'24 GenAI AV: <https://pan.webis.de/clef24/pan24-web/generated-content-analysis.html>
- PAN'25 style analysis: <https://pan.webis.de/clef25/pan25-web/style-change-detection.html> · PAN'25 overview: <https://link.springer.com/chapter/10.1007/978-3-031-88720-8_64>

**Distributional / stylometric distance**
- MAUVE: <https://en.wikipedia.org/wiki/MAUVE_(metric)> · <https://arxiv.org/abs/2102.01454> · <https://pypi.org/project/mauve-text/>
- Burrows's Delta (2026 validator): <https://www.tandfonline.com/doi/full/10.1080/09296174.2026.2612931> · Delta variants: <https://arxiv.org/pdf/2604.19499>

**Text style transfer evaluation**
- Reliable-metrics survey (2025): <https://arxiv.org/html/2502.04718v1> · LLM-based TST eval (LREC'24): <https://aclanthology.org/2024.lrec-main.1373/> · SC2 long-text TST: <https://arxiv.org/pdf/2406.04578>

**LLM-as-judge reliability**
- Biases survey: <https://arxiv.org/html/2410.02736v1> · scoring bias: <https://arxiv.org/html/2506.22316v1> · reliability-without-validity: <https://arxiv.org/pdf/2606.19544> · position bias: <https://aclanthology.org/2025.ijcnlp-long.18.pdf>

**Patent-NLP eval & generation (2024–26)**
- PatentEval (NAACL'24): <https://aclanthology.org/2024.naacl-long.147.pdf> · <https://arxiv.org/abs/2406.06589>
- PatentScore: <https://arxiv.org/html/2505.19345v2>
- PatClaimEval / Patent-CE (ACL'25): <https://arxiv.org/abs/2505.11095> · <https://aclanthology.org/2025.acl-long.190/>
- PatentWriter benchmark: <https://arxiv.org/html/2507.22387v1> · AutoPatent: <https://arxiv.org/pdf/2412.09796> · EPO-dataset claim gen: <https://arxiv.org/pdf/2505.12568> · LLM4DPCG (fine-tune): <https://github.com/scylj1/LLM4DPCG>

**Corpus sources**
- PatentsView endpoint dict: <https://search.patentsview.org/docs/docs/Search%20API/EndpointDictionary/> · attorneys endpoint: <https://patentsview.org/apis/api-endpoints/attorneys>
- USPTO Open Data Portal: <https://data.uspto.gov/> · bulk search API: <https://data.uspto.gov/apis/bulk-data/search>
- Google Patents Public Data (BigQuery): <https://github.com/google/patents-public-data> · schema: <https://github.com/google/patents-public-data/blob/master/tables/dataset_Google%20Patents%20Public%20Datasets.md>
- EPO OPS: <https://developers.epo.org> · <https://docs.ip-tools.org/patzilla/datasource/epo-ops.html> · <https://pypi.org/project/python-epo-ops-client/>
- Lens.org API: <https://about.lens.org/lens-apis/> · <https://docs.api.lens.org/>

**Methods / optimization / personalization**
- DSPy GEPA: <https://dspy.ai/api/optimizers/GEPA/overview/> · GEPA paper: <https://arxiv.org/pdf/2507.19457>
- LoRA style personalization (PEFT generation style): <https://arxiv.org/html/2409.04574v1>
- Panza local personalized writing assistant: <https://arxiv.org/pdf/2407.10994>

**Style & copyright framing**
- U.S. Copyright Office (writers): <https://www.copyright.gov/engage/writers/> · Copyright Alliance: <https://copyrightalliance.org/education/industry/writers/> · stylistic appropriation & EU law (2026): <https://arxiv.org/pdf/2606.31250>

**Patent claim style conventions**
- WIPO claim drafting: <https://www.wipo.int/edocs/mdocs/aspac/en/wipo_ip_kul_17/wipo_ip_kul_17_5.pdf> · claim structure/types: <https://www.intepat.com/blog/patent-claims-structure-and-types-of-patent-claims>
