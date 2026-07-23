"""style-fingerprint — a REAL, interpretable "attorney writing-style fingerprint" metric suite. Where the
holistic embedding metrics (LUAR cosine, PatentScore) score WHETHER a draft feels like the author, this agent
measures the author's CONCRETE fingerprints — the signature phrases he reuses, how he opens each paragraph and
sentence, his section/claim structure, and the boilerplate he pastes into every patent — and turns every miss
into optimizer feedback (a string GEPA/DSPy can act on). Reference-free: everything is measured against a
PROFILE built from the attorney's own corpus, not against a gold text.

Layers (each interpretable + independently computable):
  • signature_phrases  — distinctive n-grams (n=2..6) ranked by cross-document frequency × distinctiveness vs a
                         background (Dunning log-likelihood keyness; DF-only when no background), each with its
                         characteristic per-1k-token rate. (keyness: Rayson/Dunning log-likelihood; text-
                         dispersion keyness — Egbert & Biber.)
  • opener_dist        — distribution over paragraph-initial and sentence-initial 1-3grams + a transition-word
                         profile (paragraph-initial sentences are the stylometric markers PAN style-change uses).
  • section_template   — the canonical ALL-CAPS / "FIELD OF THE INVENTION"-style header SEQUENCE + modal wording.
  • boilerplate_blocks — near-verbatim passages reused across the corpus, mined by k-word shingling + a MinHash
                         signature and greedy Jaccard clustering (datasketch-style near-dup detection).
  • micro              — reference-numeral scheme, hedging density, passive-voice proxy, mean sentence length,
                         defined-term conventions, claim transitional markers (comprising/consisting/wherein).

score(draft, profile) → per-layer scores in [0,1] + a weighted composite + a human-readable feedback string.
discrimination_auc(profile, positives, negatives) → ROC-AUC of score() separating his held-out patents from
other attorneys' — the VALIDATION that the fingerprint actually identifies HIM (AV is scored by ROC-AUC).
as_metric(profile) → (score_fn, feedback_fn) that plug straight into dspy-prompt-optimize as a style reward.

Pure python + numpy (sklearn optional, unused in the core); a BaseAgent with a data-wise test.

Real references retrieved 2026-07: keyness/log-likelihood — https://www.refsmmat.com/notebooks/keyness.html ,
https://www.degruyterbrill.com/document/doi/10.1515/cllt-2015-0030/html (LL vs odds-ratio, Pojanapunya & Todd),
text-dispersion keyness for lexical bundles — https://www.sciencedirect.com/science/article/abs/pii/S2772766125000667 ;
paragraph/sentence-opener stylometry — https://aclanthology.org/2025.findings-acl.913.pdf ,
https://pan.webis.de/clef25/pan25-web/style-change-detection.html ; near-dup boilerplate (shingling+MinHash+LSH) —
https://yorko.github.io/2023/practical-near-dup-detection/ , https://mbrenndoerfer.com/writing/minhash-algorithm-jaccard-similarity-lsh-deduplication ;
authorship-DISCRIMINATION-AUC as style-metric validation — https://www.sciencedirect.com/science/article/abs/pii/S0957417423012472 (TDRLM, AUC 92.56),
https://www.sciencedirect.com/science/article/pii/S266682702500115X .
"""
from __future__ import annotations
import hashlib
import math
import re
from collections import Counter, defaultdict
from .base import BaseAgent

_WORD = re.compile(r"[A-Za-z][A-Za-z'\-]*")
_NUMERAL = re.compile(r"\b\d{2,4}\b")
_BE = {"is", "are", "was", "were", "be", "been", "being", "am"}
_HEDGE = ["may", "can", "could", "might", "would", "should", "optionally", "preferably", "generally",
          "typically", "substantially", "approximately", "about", "in some embodiments", "in one embodiment",
          "in certain embodiments", "in various embodiments", "by way of example", "it will be appreciated",
          "it should be understood", "for example", "such as"]
_TRANSITIONS = {"however", "therefore", "thus", "moreover", "furthermore", "accordingly", "additionally",
                "consequently", "hence", "meanwhile", "nevertheless", "nonetheless", "similarly", "notably",
                "specifically", "particularly", "alternatively", "optionally", "indeed", "importantly",
                "generally", "typically", "preferably", "first", "second", "finally"}
_KNOWN_HEADERS = ["field of the invention", "field", "background", "background of the invention", "summary",
                  "summary of the invention", "brief description of the drawings", "brief description",
                  "detailed description", "detailed description of the invention", "claims", "abstract",
                  "cross-reference to related applications", "technical field", "what is claimed is"]


# ---------------- tokenisation ----------------
def _norm(s):
    return " ".join(_WORD.findall(str(s).lower()))


def _words(text):
    return _WORD.findall(str(text).lower())


def _sentences(text):
    out = []
    for block in re.split(r"\n+", str(text)):        # split on lines first so headers don't glue onto prose
        block = block.strip()
        if not block or _is_header(block):
            continue
        for p in re.split(r"(?<=[.!?])\s+", block):
            p = p.strip()
            if p:
                out.append(p)
    return out


def _paragraphs(text):
    paras = [p.strip() for p in re.split(r"\n\s*\n+", str(text)) if p.strip()]
    if len(paras) <= 1 and "\n" in str(text):
        paras = [p.strip() for p in str(text).split("\n") if p.strip()]
    return paras or ([str(text).strip()] if str(text).strip() else [])


def _is_header(line):
    s = line.strip().rstrip(":").strip()
    if not s or s.endswith("."):
        return None
    if s.lower() in _KNOWN_HEADERS:
        return s
    letters = [c for c in s if c.isalpha()]
    words = s.split()
    if 1 <= len(words) <= 10 and len(letters) >= 2 and sum(c.isupper() for c in letters) / len(letters) >= 0.8:
        return s
    return None


def _headers(text):
    out = []
    for ln in str(text).split("\n"):
        h = _is_header(ln)
        if h:
            out.append(h.strip())
    return out


# ---------------- keyness (Dunning log-likelihood) ----------------
def _log_likelihood(a, b, n1, n2):
    """Signed Dunning LL: +over-represented in target(1), -over-represented in background(2)."""
    if n1 <= 0 or n2 <= 0 or (a + b) <= 0:
        return 0.0
    e1 = n1 * (a + b) / (n1 + n2)
    e2 = n2 * (a + b) / (n1 + n2)
    ll = 0.0
    if a > 0:
        ll += a * math.log(a / e1)
    if b > 0:
        ll += b * math.log(b / e2)
    ll *= 2.0
    sign = 1.0 if (a / n1) >= (b / n2) else -1.0
    return sign * ll


# ---------------- n-gram signature phrases ----------------
def _doc_ngrams(text, nmin=2, nmax=6):
    """N-grams counted WITHIN prose segments (paragraphs/sentences), so a phrase never spans a section
    header or paragraph break (keeps signature phrases clean + interpretable)."""
    counts, present, total = Counter(), set(), 0
    for seg in _sentences(text):                     # _sentences already drops headers + splits on lines
        toks = _words(seg)
        total += len(toks)
        for n in range(nmin, nmax + 1):
            for i in range(len(toks) - n + 1):
                g = " ".join(toks[i:i + n])
                counts[g] += 1
                present.add(g)
    return counts, present, total


def _signature_phrases(corpus, background=None, nmin=2, nmax=6, top_k=25, min_df=2):
    df = Counter()
    tot = Counter()
    n_tokens_total = 0
    for doc in corpus:
        counts, present, ntok = _doc_ngrams(doc, nmin, nmax)
        n_tokens_total += ntok
        for g in present:
            df[g] += 1
        for g, c in counts.items():
            tot[g] += c
    n_docs = len(corpus)
    min_df = min(min_df, n_docs) if n_docs > 1 else 1

    bg_counts, bg_tokens = Counter(), 0
    if background:
        for doc in background:
            bcounts, _, bntok = _doc_ngrams(doc, nmin, nmax)
            bg_tokens += bntok
            for g, c in bcounts.items():
                bg_counts[g] += c

    cands = []
    for g, dfreq in df.items():
        if dfreq < min_df:
            continue
        rate = 1000.0 * tot[g] / max(1, n_tokens_total)
        df_frac = dfreq / n_docs
        if background:
            ll = _log_likelihood(tot[g], bg_counts.get(g, 0), n_tokens_total, bg_tokens)
            if ll <= 0:                       # keep only phrases OVER-represented in his corpus
                continue
            sc = df_frac * ll
        else:
            ll = 0.0
            sc = df_frac * (1.0 + 0.1 * len(g.split())) * (1.0 + math.log1p(rate))
        cands.append({"phrase": g, "n": len(g.split()), "df": dfreq, "rate_per_1k": round(rate, 3),
                      "ll": round(ll, 3), "score": round(sc, 4)})

    cands.sort(key=lambda d: (-d["score"], -d["n"], -d["df"]))
    kept = []
    for c in cands:
        if any(c["phrase"] in k["phrase"] for k in kept):   # drop a shorter phrase contained in a kept one
            continue
        kept.append(c)
        if len(kept) >= top_k:
            break
    return kept


# ---------------- opener distributions ----------------
def _opener_dist(corpus):
    para_first, sent_first = Counter(), Counter()
    para_3, sent_3 = Counter(), Counter()
    trans = Counter()
    for doc in corpus:
        for p in _paragraphs(doc):
            if _is_header(p):                        # a header is not a prose opener
                continue
            w = _words(p)
            if w:
                para_first[w[0]] += 1
                para_3[" ".join(w[:3])] += 1
        for s in _sentences(doc):
            w = _words(s)
            if w:
                sent_first[w[0]] += 1
                sent_3[" ".join(w[:3])] += 1
                if w[0] in _TRANSITIONS:
                    trans[w[0]] += 1

    def _probs(c):
        tot = sum(c.values()) or 1
        return {k: v / tot for k, v in c.items()}

    return {
        "para_first_token": _probs(para_first),
        "sent_first_token": _probs(sent_first),
        "para_top3grams": para_3.most_common(8),
        "sent_top3grams": sent_3.most_common(8),
        "transition": _probs(trans),
    }


def _draft_opener_dist(draft):
    return _opener_dist([draft])


# ---------------- closer (postfix) distributions — how he ENDS paragraphs/sentences ----------------
def _closer_dist(corpus):
    para_last, sent_last = Counter(), Counter()
    para_3, sent_3 = Counter(), Counter()
    for doc in corpus:
        for p in _paragraphs(doc):
            if _is_header(p):
                continue
            w = _words(p)
            if w:
                para_last[w[-1]] += 1
                para_3[" ".join(w[-3:])] += 1
        for s in _sentences(doc):
            w = _words(s)
            if w:
                sent_last[w[-1]] += 1
                sent_3[" ".join(w[-3:])] += 1

    def _probs(c):
        tot = sum(c.values()) or 1
        return {k: v / tot for k, v in c.items()}

    return {
        "para_last_token": _probs(para_last),
        "sent_last_token": _probs(sent_last),
        "para_close_top3grams": para_3.most_common(8),
        "sent_close_top3grams": sent_3.most_common(8),
    }


# ---------------- word-level character affixes (prefix / suffix morphology signature) ----------------
def _affix_dist(corpus, plen=3, slen=3, min_word=5):
    pre, suf = Counter(), Counter()
    for doc in corpus:
        for w in _words(doc):
            if len(w) >= min_word:
                pre[w[:plen]] += 1
                suf[w[-slen:]] += 1

    def _probs(c):
        tot = sum(c.values()) or 1
        return {k: v / tot for k, v in c.items()}
    return {
        "prefix": _probs(pre),
        "suffix": _probs(suf),
        "top_prefixes": pre.most_common(10),
        "top_suffixes": suf.most_common(10),
    }


def _js_divergence(p, q):
    """Jensen-Shannon divergence (base-2, ∈[0,1]) between two dict distributions over a shared support."""
    keys = set(p) | set(q)
    if not keys:
        return 0.0
    ps = sum(p.values()) or 1.0
    qs = sum(q.values()) or 1.0

    def _kl(a, b):
        s = 0.0
        for k in keys:
            ak = a.get(k, 0.0)
            if ak <= 0:
                continue
            bk = b.get(k, 0.0)
            if bk <= 0:
                continue
            s += ak * math.log2(ak / bk)
        return s
    P = {k: p.get(k, 0.0) / ps for k in keys}
    Q = {k: q.get(k, 0.0) / qs for k in keys}
    M = {k: 0.5 * (P[k] + Q[k]) for k in keys}
    return max(0.0, min(1.0, 0.5 * _kl(P, M) + 0.5 * _kl(Q, M)))


# ---------------- section template ----------------
def _section_template(corpus):
    hdr_df = Counter()
    positions = defaultdict(list)
    seqs = []
    for doc in corpus:
        hs = _headers(doc)
        seqs.append([h.lower() for h in hs])
        seen = set()
        for i, h in enumerate(hs):
            hl = h.lower()
            hdr_df[hl] += 1
            positions[hl].append(i)
            if hl not in seen:
                hdr_df.setdefault(hl, hdr_df[hl])
            seen.add(hl)
    # modal wording per lowercased header
    modal = {}
    for doc in corpus:
        for h in _headers(doc):
            modal.setdefault(h.lower(), Counter())[h] += 1
    headers = []
    for hl, dfreq in hdr_df.items():
        med = sorted(positions[hl])[len(positions[hl]) // 2] if positions[hl] else 0
        wording = modal.get(hl, Counter()).most_common(1)
        headers.append({"text": hl, "df": dfreq, "median_pos": med,
                        "modal": wording[0][0] if wording else hl.upper()})
    headers.sort(key=lambda d: (d["median_pos"], -d["df"]))
    canonical = [h["text"] for h in headers if h["df"] >= max(2, len(corpus) // 2)] or [h["text"] for h in headers]
    return {"headers": headers, "canonical_sequence": canonical,
            "modal_headers": [h["modal"] for h in headers]}


# ---------------- boilerplate mining (shingle + MinHash + Jaccard cluster) ----------------
def _shingles(text, k=5):
    w = _words(text)
    if len(w) < k:
        return frozenset([" ".join(w)]) if w else frozenset()
    return frozenset(" ".join(w[i:i + k]) for i in range(len(w) - k + 1))


def _minhash(shingles, num_perm=48):
    """Deterministic MinHash signature (md5-hashed shingles, affine permutations mod a large prime)."""
    if not shingles:
        return tuple([0] * num_perm)
    prime = (1 << 61) - 1
    hs = [int(hashlib.md5(s.encode()).hexdigest()[:15], 16) for s in shingles]
    sig = []
    for i in range(num_perm):
        a = 2 * i + 1
        b = 2654435761 * (i + 1)
        sig.append(min(((a * h + b) % prime) for h in hs))
    return tuple(sig)


def _mh_jaccard(s1, s2):
    if not s1 or not s2:
        return 0.0
    return sum(1 for x, y in zip(s1, s2) if x == y) / len(s1)


def _passages(text):
    """Candidate reusable passages: paragraphs, plus each individual sentence (catches one-line boilerplate)."""
    out = []
    for p in _paragraphs(text):
        w = _words(p)
        if len(w) >= 4 and not _is_header(p):
            out.append(p.strip())
    for s in _sentences(text):
        w = _words(s)
        if len(w) >= 5:
            out.append(s.strip())
    return out


def _boilerplate_blocks(corpus, k=5, thr=0.6, max_blocks=12):
    items = []                                   # (doc_idx, text, shingleset, minhash)
    for di, doc in enumerate(corpus):
        seen = set()
        for p in _passages(doc):
            key = _norm(p)
            if key in seen:
                continue
            seen.add(key)
            sh = _shingles(p, k)
            if sh:
                items.append((di, p, sh, _minhash(sh)))
    clusters = []                                # each: {docs:set, reps:[(text,shingles)], text}
    for di, text, sh, mh in items:
        best, bj = None, 0.0
        for cl in clusters:
            j = max(_mh_jaccard(mh, rmh) for (_, _, rmh) in cl["members"])
            if j > bj:
                bj, best = j, cl
        if best is not None and bj >= thr:
            best["docs"].add(di)
            best["members"].append((text, sh, mh))
        else:
            clusters.append({"docs": {di}, "members": [(text, sh, mh)]})
    blocks = []
    for cl in clusters:
        if len(cl["docs"]) < 2:                  # boilerplate = reused across ≥2 of his patents
            continue
        # representative = longest member; store its exact shingles (JSON-safe list) for scoring
        rep_text, rep_sh, _ = max(cl["members"], key=lambda m: len(m[1]))
        blocks.append({"text": rep_text, "df": len(cl["docs"]), "shingles": sorted(rep_sh)})
    blocks.sort(key=lambda b: (-b["df"], -len(b["shingles"])))
    return blocks[:max_blocks]


# ---------------- micro conventions ----------------
def _passive_proxy(sents):
    hits, n = 0, 0
    for s in sents:
        w = _words(s)
        n += 1
        for i in range(len(w) - 1):
            if w[i] in _BE and (w[i + 1].endswith("ed") or w[i + 1] in ("made", "given", "shown", "seen",
                                                                        "taken", "known", "coupled", "disposed")):
                hits += 1
                break
    return hits / max(1, n)


def _micro(corpus):
    text = "\n".join(corpus)
    toks = _words(text)
    ntok = max(1, len(toks))
    sents = []
    for doc in corpus:
        sents += _sentences(doc)
    low = text.lower()

    def _rate(pat_count):
        return 1000.0 * pat_count / ntok

    numerals = _NUMERAL.findall(text)
    two = sum(1 for x in numerals if len(x) == 2)
    three = sum(1 for x in numerals if len(x) == 3)
    hedge_ct = sum(low.count(h) for h in _HEDGE)
    defined = (low.count("as used herein") + low.count("refers to") + low.count("defined as")
               + low.count("hereinafter") + len(re.findall(r'\(\s*"[^"]+"\s*\)', text))
               + len(re.findall(r'\bmeans\b', low)))
    dep_claim = len(re.findall(r"of claim\s+\d+", low))
    return {
        "mean_sentence_len": round(sum(len(_words(s)) for s in sents) / max(1, len(sents)), 3),
        "numeral_rate_per_1k": round(_rate(len(numerals)), 3),
        "two_digit_numeral_frac": round(two / max(1, len(numerals)), 3),
        "three_digit_numeral_frac": round(three / max(1, len(numerals)), 3),
        "hedging_rate_per_1k": round(_rate(hedge_ct), 3),
        "passive_proxy": round(_passive_proxy(sents), 3),
        "defined_term_rate_per_1k": round(_rate(defined), 3),
        "comprising_rate_per_1k": round(_rate(low.count("comprising")), 3),
        "consisting_rate_per_1k": round(_rate(low.count("consisting")), 3),
        "wherein_rate_per_1k": round(_rate(low.count("wherein")), 3),
        "dep_claim_rate_per_1k": round(_rate(dep_claim), 3),
    }


# =============================================================================
# PUBLIC API
# =============================================================================
def build_profile(corpus, background=None, nmin=2, nmax=6, top_k=25):
    """Build the attorney's writing-style fingerprint from his patent texts. JSON-safe dict."""
    corpus = [c for c in (corpus or []) if str(c).strip()]
    if not corpus:
        raise ValueError("build_profile needs a non-empty corpus of the attorney's texts")
    return {
        "n_docs": len(corpus),
        "signature_phrases": _signature_phrases(corpus, background, nmin, nmax, top_k),
        "opener_dist": _opener_dist(corpus),
        "closer_dist": _closer_dist(corpus),
        "affix_dist": _affix_dist(corpus),
        "section_template": _section_template(corpus),
        "boilerplate_blocks": _boilerplate_blocks(corpus),
        "micro": _micro(corpus),
        "has_background": bool(background),
    }


_DEFAULT_WEIGHTS = {"phrase_coverage": 0.22, "phrase_rate_match": 0.08, "opener_js": 0.15,
                    "closer_js": 0.10, "affix_match": 0.07, "structure_match": 0.17,
                    "boilerplate_overlap": 0.13, "micro_conformity": 0.08}


def _phrase_scores(draft, profile):
    dnorm = _norm(draft)
    dtok = max(1, len(_words(draft)))
    phrases = profile.get("signature_phrases", [])
    if not phrases:
        return 1.0, 1.0
    present = 0
    rate_terms = []
    for ph in phrases:
        p = ph["phrase"]
        occ = dnorm.count(p)
        if occ > 0:
            present += 1
        pr = ph.get("rate_per_1k", 0.0)
        if pr > 0:
            dr = 1000.0 * occ / dtok
            rate_terms.append(1.0 - min(1.0, abs(dr - pr) / (pr + 1e-9)))
    coverage = present / len(phrases)
    rate_match = sum(rate_terms) / len(rate_terms) if rate_terms else coverage
    return coverage, rate_match


def _structure_score(draft, profile):
    st = profile.get("section_template", {})
    canon = st.get("canonical_sequence", [])
    dh = [h.lower() for h in _headers(draft)]
    if canon:
        # LCS length / len(canonical) — order-aware sequence alignment
        m, n = len(dh), len(canon)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(m):
            for j in range(n):
                dp[i + 1][j + 1] = dp[i][j] + 1 if dh[i] == canon[j] else max(dp[i][j + 1], dp[i + 1][j])
        seq_sim = dp[m][n] / n
    else:
        seq_sim = None
    micro = profile.get("micro", {})
    dmicro = _micro([draft])
    claim_keys = ["comprising_rate_per_1k", "consisting_rate_per_1k", "wherein_rate_per_1k",
                  "dep_claim_rate_per_1k"]
    terms = []
    for k in claim_keys:
        p = micro.get(k, 0.0)
        d = dmicro.get(k, 0.0)
        if p == 0 and d == 0:
            continue
        terms.append(1.0 - min(1.0, abs(d - p) / (abs(p) + abs(d) + 1e-9)))
    claim_sim = sum(terms) / len(terms) if terms else None
    vals = [v for v in (seq_sim, claim_sim) if v is not None]
    return sum(vals) / len(vals) if vals else 0.5


def _boilerplate_score(draft, profile):
    blocks = profile.get("boilerplate_blocks", [])
    if not blocks:
        return 1.0
    dpass = [_shingles(p) for p in _passages(draft)] or [_shingles(draft)]
    per_block = []
    for b in blocks:
        bs = set(b["shingles"])
        best = 0.0
        for ds in dpass:
            if not ds or not bs:
                continue
            j = len(ds & bs) / len(ds | bs)
            best = max(best, j)
        per_block.append(best)
    return sum(per_block) / len(per_block)


def _micro_score(draft, profile):
    micro = profile.get("micro", {})
    dmicro = _micro([draft])
    terms = []
    for k, p in micro.items():
        d = dmicro.get(k, 0.0)
        if p == 0 and d == 0:
            continue
        terms.append(1.0 - min(1.0, abs(d - p) / (abs(p) + abs(d) + 1e-9)))
    return sum(terms) / len(terms) if terms else 0.5


def _feedback(draft, profile, layers):
    dnorm = _norm(draft)
    msgs = []
    miss = [ph["phrase"] for ph in profile.get("signature_phrases", [])[:12] if ph["phrase"] not in dnorm]
    if miss:
        msgs.append("missing signature phrases: " + "; ".join(f'"{m}"' for m in miss[:5]))
    # opener skew
    od = profile.get("opener_dist", {})
    prof_top = od.get("sent_top3grams", [])
    if prof_top:
        want = prof_top[0][0]
        dsent = _sentences(draft)
        dfirst = " ".join(_words(dsent[0])[:3]) if dsent else ""
        if want and want != dfirst:
            msgs.append(f'sentence-opener skew: he leads with "{want}", draft leads with "{dfirst or "(none)"}"')
    # closer (postfix) skew
    prof_close = profile.get("closer_dist", {}).get("sent_close_top3grams", [])
    if prof_close and layers.get("closer_js", 1.0) < 0.6:
        want_c = prof_close[0][0]
        dsent = _sentences(draft)
        dlast = " ".join(_words(dsent[-1])[-3:]) if dsent else ""
        if want_c and want_c != dlast:
            msgs.append(f'sentence-closer skew: he ends with "{want_c}", draft ends with "{dlast or "(none)"}"')
    # missing canonical boilerplate block
    blocks = profile.get("boilerplate_blocks", [])
    if blocks and layers.get("boilerplate_overlap", 1.0) < 0.4:
        b = blocks[0]
        msgs.append(f'missing canonical boilerplate block (in {b["df"]} of his patents): '
                    f'"{b["text"][:80]}..."')
    # section order
    st = profile.get("section_template", {})
    canon = st.get("canonical_sequence", [])
    dh = [h.lower() for h in _headers(draft)]
    if canon and dh != canon:
        missing_h = [h for h in canon if h not in dh]
        if missing_h:
            msgs.append("missing/!wrong-order sections: " + ", ".join(h.upper() for h in missing_h[:4]))
        elif dh[:len(canon)] != canon:
            msgs.append("section order differs from his canonical: " + " → ".join(h.upper() for h in canon))
    if layers.get("phrase_rate_match", 1.0) < 0.6:
        msgs.append("signature-phrase RATE off vs his characteristic per-1k usage")
    return " | ".join(msgs) if msgs else "matches the attorney's fingerprint on all layers"


def score(draft, profile, weights=None):
    """Score a draft against the attorney's profile. Per-layer ∈[0,1] + weighted composite + feedback string."""
    w = dict(_DEFAULT_WEIGHTS)
    if profile.get("weights"):
        w.update(profile["weights"])
    if weights:
        w.update(weights)
    cov, rate = _phrase_scores(draft, profile)
    od = profile.get("opener_dist", {})
    dod = _draft_opener_dist(draft)
    js_sent = _js_divergence(dod["sent_first_token"], od.get("sent_first_token", {}))
    js_para = _js_divergence(dod["para_first_token"], od.get("para_first_token", {}))
    opener_js = 1.0 - 0.5 * (js_sent + js_para)
    # closers (postfix): how he ends sentences/paragraphs
    cd = profile.get("closer_dist", {})
    dcd = _closer_dist([draft])
    jc_sent = _js_divergence(dcd["sent_last_token"], cd.get("sent_last_token", {}))
    jc_para = _js_divergence(dcd["para_last_token"], cd.get("para_last_token", {}))
    closer_js = 1.0 - 0.5 * (jc_sent + jc_para)
    # word-level char affixes (prefix/suffix morphology)
    ad = profile.get("affix_dist", {})
    dad = _affix_dist([draft])
    ja_pre = _js_divergence(dad["prefix"], ad.get("prefix", {}))
    ja_suf = _js_divergence(dad["suffix"], ad.get("suffix", {}))
    affix_match = 1.0 - 0.5 * (ja_pre + ja_suf)
    layers = {
        "phrase_coverage": round(cov, 4),
        "phrase_rate_match": round(rate, 4),
        "opener_js": round(opener_js, 4),
        "closer_js": round(closer_js, 4),
        "affix_match": round(affix_match, 4),
        "structure_match": round(_structure_score(draft, profile), 4),
        "boilerplate_overlap": round(_boilerplate_score(draft, profile), 4),
        "micro_conformity": round(_micro_score(draft, profile), 4),
    }
    composite = sum(w[k] * layers[k] for k in layers) / (sum(w[k] for k in layers) or 1.0)
    out = dict(layers)
    out["composite"] = round(composite, 4)
    out["feedback"] = _feedback(draft, profile, layers)
    return out


def _auc(pos_scores, neg_scores):
    """ROC-AUC via the Mann-Whitney U statistic with average-rank tie handling."""
    if not pos_scores or not neg_scores:
        return float("nan")
    allv = [(s, 1) for s in pos_scores] + [(s, 0) for s in neg_scores]
    allv.sort(key=lambda x: x[0])
    ranks = [0.0] * len(allv)
    i = 0
    while i < len(allv):
        j = i
        while j + 1 < len(allv) and allv[j + 1][0] == allv[i][0]:
            j += 1
        avg = (i + j) / 2.0 + 1.0        # 1-based average rank over the tie block
        for k in range(i, j + 1):
            ranks[k] = avg
        i = j + 1
    rank_pos = sum(r for r, (_, lab) in zip(ranks, allv) if lab == 1)
    n_pos, n_neg = len(pos_scores), len(neg_scores)
    return (rank_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def discrimination_auc(profile, positives, negatives, weights=None):
    """VALIDATION: ROC-AUC of score() separating his held-out patents (positives) from other attorneys'
    (negatives). A real fingerprint scores HIS patents high and others low → AUC → 1.0."""
    pos = [score(d, profile, weights)["composite"] for d in positives if str(d).strip()]
    neg = [score(d, profile, weights)["composite"] for d in negatives if str(d).strip()]
    return _auc(pos, neg)


def as_metric(profile, weights=None):
    """Return (score_fn, feedback_fn) for dspy-prompt-optimize / GEPA. gold is IGNORED (style is reference-free
    against the profile)."""
    def score_fn(draft, gold=None):
        return float(score(draft, profile, weights)["composite"])

    def feedback_fn(draft, gold=None):
        return score(draft, profile, weights)["feedback"]
    return score_fn, feedback_fn


def summarize_profile(profile, k=10):
    """JSON-safe compact view for the board/ledger (drops heavy shingle lists)."""
    return {
        "n_docs": profile.get("n_docs"),
        "has_background": profile.get("has_background"),
        "signature_phrases": [{"phrase": p["phrase"], "df": p["df"], "rate_per_1k": p["rate_per_1k"]}
                              for p in profile.get("signature_phrases", [])[:k]],
        "modal_sent_openers": profile.get("opener_dist", {}).get("sent_top3grams", [])[:5],
        "modal_sent_closers": profile.get("closer_dist", {}).get("sent_close_top3grams", [])[:5],
        "top_word_prefixes": profile.get("affix_dist", {}).get("top_prefixes", [])[:6],
        "top_word_suffixes": profile.get("affix_dist", {}).get("top_suffixes", [])[:6],
        "canonical_sections": [h.upper() for h in profile.get("section_template", {}).get("canonical_sequence", [])],
        "boilerplate_blocks": [{"df": b["df"], "preview": b["text"][:100]}
                               for b in profile.get("boilerplate_blocks", [])[:5]],
        "micro": profile.get("micro", {}),
    }


class StyleFingerprint(BaseAgent):
    name = "style-fingerprint"
    thread = "S"
    kind = "finding"

    def run(self, q, worker):
        spec = self.spec(q)
        corpus = spec.get("corpus") or []
        if not corpus:
            return self.escalate(worker, "researcher",
                                 "style-fingerprint: pass spec['corpus'] (list of the attorney's patent texts). "
                                 "Optional: spec['background'] (other attorneys), spec['draft'] (score it), "
                                 "spec['positives']+spec['negatives'] (discrimination-AUC validation).")
        try:
            profile = build_profile(corpus, background=spec.get("background"),
                                    top_k=int(spec.get("top_k", 25)))
        except ValueError as e:
            return self.escalate(worker, "researcher", f"style-fingerprint: {e}")
        data = {"profile": summarize_profile(profile), "n_signature_phrases": len(profile["signature_phrases"]),
                "n_boilerplate_blocks": len(profile["boilerplate_blocks"]),
                "canonical_sections": [h.upper() for h in profile["section_template"]["canonical_sequence"]]}
        parts = [f"style-fingerprint: profiled {profile['n_docs']} patents → "
                 f"{len(profile['signature_phrases'])} signature phrases, "
                 f"{len(profile['boilerplate_blocks'])} cross-patent boilerplate blocks, "
                 f"{len(profile['section_template']['canonical_sequence'])} canonical sections."]
        if profile["signature_phrases"]:
            parts.append(f'top phrase: "{profile["signature_phrases"][0]["phrase"]}" '
                         f'(df={profile["signature_phrases"][0]["df"]}).')
        if spec.get("draft"):
            sc = score(spec["draft"], profile, weights=spec.get("weights"))
            data["score"] = sc
            parts.append(f"draft composite={sc['composite']} (phrase={sc['phrase_coverage']}, "
                         f"opener={sc['opener_js']}, boilerplate={sc['boilerplate_overlap']}). "
                         f"feedback: {sc['feedback'][:120]}")
        if spec.get("positives") and spec.get("negatives"):
            auc = discrimination_auc(profile, spec["positives"], spec["negatives"], weights=spec.get("weights"))
            data["discrimination_auc"] = round(auc, 4)
            parts.append(f"discrimination-AUC (his vs others) = {round(auc, 4)}.")
        msg = " ".join(parts)
        self.log(msg, kind="finding",
                 recommendation="wire as_metric(profile) into dspy-prompt-optimize as the style reward")
        return self.done(data, msg)


_AGENT = StyleFingerprint()


def run(q, worker):
    return _AGENT.run(q, worker)
