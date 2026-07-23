"""data-wise test for style-fingerprint — the interpretable attorney writing-style fingerprint suite.
Fully offline + deterministic. Builds a synthetic "attorney A" corpus (shared signature phrase
"it will be appreciated that", fixed opener "In one embodiment,", a repeated closing boilerplate block,
canonical section headers) and an "attorney B" corpus (different phrases/openers/structure). Proves the
fingerprint surfaces A's habits, scores A-style drafts above B-style drafts per layer, discriminates his
patents from others by ROC-AUC, and emits concrete optimizer feedback. No LLM, no network, no heavy deps."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fleet_agents import style_fingerprint as SF

fails = []


def check(n, c):
    print(("PASS " if c else "FAIL ") + n)
    if not c:
        fails.append(n)


# ---- synthetic corpora --------------------------------------------------------
_A_CLOSE = ("The foregoing description is illustrative and not restrictive, and the scope shall be "
            "determined by the appended claims.")


def _a_doc(thing, part):
    return f"""FIELD OF THE INVENTION

The present invention relates to {thing}.

BACKGROUND

In one embodiment, the system comprises a {part}. It will be appreciated that the {part} may be configured to operate. In one embodiment, the device includes a memory coupled to the {part}.

SUMMARY

In one embodiment, a method comprises receiving data from the {part}. It will be appreciated that various modifications can be made.

DETAILED DESCRIPTION

In one embodiment, the apparatus of claim 1 wherein the {part} is configured. {_A_CLOSE}"""


def _b_doc(thing, part):
    return f"""TECHNICAL FIELD

This disclosure pertains to {thing}.

DESCRIPTION OF RELATED ART

Referring now to the figures, a {part} receives signals. Various changes may be made without departing from the scope of the disclosure. Referring now to the drawings, a {part} transmits values.

BRIEF SUMMARY

Referring now to the embodiments, a process obtains inputs from the {part}. Numerous alternatives exist without departing from the scope.

DESCRIPTION OF EMBODIMENTS

Referring now to figure 1, the {part} of claim 1 wherein the controller responds. Persons skilled in the art will recognize numerous variations within the spirit of the disclosure."""


A_train = [_a_doc(t, p) for t, p in [("widgets", "processor"), ("gadgets", "controller"),
                                     ("sprockets", "sensor"), ("actuators", "module"),
                                     ("valves", "regulator"), ("couplers", "driver")]]
A_holdout = [_a_doc("turbines", "rotor"), _a_doc("pumps", "impeller")]
B_docs = [_b_doc(t, p) for t, p in [("gizmos", "controller"), ("machines", "sensor"),
                                    ("engines", "module"), ("relays", "switch")]]

A_draft = f"""FIELD OF THE INVENTION

The present invention relates to fasteners.

BACKGROUND

In one embodiment, the system comprises a bracket. It will be appreciated that the bracket may be configured to operate.

SUMMARY

In one embodiment, a method comprises transmitting data. It will be appreciated that various modifications can be made.

DETAILED DESCRIPTION

In one embodiment, the apparatus of claim 1 wherein the bracket is configured. {_A_CLOSE}"""

B_draft = """TECHNICAL FIELD

This disclosure pertains to fasteners.

DESCRIPTION OF RELATED ART

Referring now to the figures, a clamp receives signals. Various changes may be made without departing from the scope of the disclosure.

BRIEF SUMMARY

Referring now to the embodiments, a process obtains inputs.

DESCRIPTION OF EMBODIMENTS

Referring now to figure 2, the clamp of claim 1 wherein the controller responds. Persons skilled in the art will recognize numerous variations."""


# ---- 1. build_profile surfaces A's fingerprints -------------------------------
profA = SF.build_profile(A_train)
phrases = profA["signature_phrases"]
check("signature phrase 'appreciated'", any("appreciated" in p["phrase"] for p in phrases))
check("signature phrase 'in one embodiment'", any("in one embodiment" in p["phrase"] for p in phrases))
check("phrases carry a per-1k rate", all("rate_per_1k" in p and "df" in p for p in phrases))

blocks = profA["boilerplate_blocks"]
check("repeated closing boilerplate mined",
      any("foregoing description is illustrative" in b["text"] for b in blocks))
check("boilerplate block reused across >=2 patents", any(b["df"] >= 2 for b in blocks))

sent_top = [g for g, _ in profA["opener_dist"]["sent_top3grams"]]
check("canonical sentence opener is 'in one embodiment'", sent_top and sent_top[0].startswith("in one embodiment"))
check("canonical section sequence detected",
      "field of the invention" in profA["section_template"]["canonical_sequence"]
      and "detailed description" in profA["section_template"]["canonical_sequence"])

# background-aware keyness path also works (Dunning log-likelihood vs attorney B)
profA_bg = SF.build_profile(A_train, background=B_docs)
check("keyness (LL vs background) keeps A-distinctive phrases",
      any("appreciated" in p["phrase"] for p in profA_bg["signature_phrases"]))


# ---- 2. score: A-style draft beats B-style draft, composite + each layer ------
sA = SF.score(A_draft, profA)
sB = SF.score(B_draft, profA)
check("composite: A-draft > B-draft", sA["composite"] > sB["composite"])
check("phrase layer: A-draft > B-draft", sA["phrase_coverage"] > sB["phrase_coverage"])
check("opener layer: A-draft > B-draft", sA["opener_js"] > sB["opener_js"])
check("boilerplate layer: A-draft > B-draft", sA["boilerplate_overlap"] > sB["boilerplate_overlap"])
check("all layers in [0,1]", all(0.0 <= sA[k] <= 1.0 for k in
      ["phrase_coverage", "phrase_rate_match", "opener_js", "closer_js", "affix_match",
       "structure_match", "boilerplate_overlap", "micro_conformity", "composite"]))
check("A-draft phrase coverage strong", sA["phrase_coverage"] >= 0.3)

# prefix + postfix at BOTH word and paragraph/sentence level are captured
check("profile has paragraph/sentence CLOSERS (postfix)",
      "closer_dist" in profA and profA["closer_dist"]["sent_close_top3grams"])
check("profile has WORD prefix/suffix affixes",
      "affix_dist" in profA and profA["affix_dist"]["top_suffixes"])
check("closer (postfix) layer: A-draft >= B-draft", sA["closer_js"] >= sB["closer_js"])
check("affix (word prefix/suffix) layer: A-draft >= B-draft", sA["affix_match"] >= sB["affix_match"])


# ---- 3. discrimination-AUC: his held-out patents vs other attorneys' ----------
auc = SF.discrimination_auc(profA, positives=A_holdout, negatives=B_docs)
print(f"  discrimination_auc = {auc:.3f}")
check("discrimination_auc >= 0.8 (fingerprint identifies HIM)", auc >= 0.8)


# ---- 4. feedback names concrete misses ---------------------------------------
fb = sB["feedback"]
print("  B-draft feedback:", fb[:160])
check("feedback names a missing signature phrase", "missing signature phrase" in fb.lower())
check("feedback names the missing canonical boilerplate block", "boilerplate" in fb.lower())


# ---- 5. as_metric returns callables usable by dspy-prompt-optimize/GEPA -------
score_fn, feedback_fn = SF.as_metric(profA)
check("as_metric returns callables", callable(score_fn) and callable(feedback_fn))
check("score_fn(A) > score_fn(B) (gold ignored)", score_fn(A_draft) > score_fn(B_draft, gold="anything"))
check("feedback_fn returns a string", isinstance(feedback_fn(B_draft), str) and len(feedback_fn(B_draft)) > 0)


# ---- 6. agent handler: corpus -> done; empty -> escalated --------------------
import fleet_agents as F

VALID = {"done", "escalated", "holding", "error", "failed", "skipped"}
o1 = F._RAW_HANDLERS["style-fingerprint"](
    {"question": "profile", "spec": {"corpus": A_train, "draft": A_draft,
                                     "positives": A_holdout, "negatives": B_docs}}, "t")
check("handler with corpus -> done", o1[0] == "done")
check("handler reports score + auc", isinstance(o1[1], dict) and "score" in o1[1] and "discrimination_auc" in o1[1])
o2 = F._RAW_HANDLERS["style-fingerprint"]({"question": "x", "spec": {}}, "t")
check("empty spec -> escalated (valid status)", o2[0] == "escalated" and o2[0] in VALID)


print("=== style-fingerprint: " + ("PASS" if not fails else "FAIL " + ",".join(fails)) + " ===")
sys.exit(1 if fails else 0)
