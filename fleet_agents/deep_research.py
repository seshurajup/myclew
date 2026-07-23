"""deep-research — a PERSISTENT, reusable, journaled paper-mining fleet agent. Turns the one-off web-research
I ran via ephemeral sub-agents into a first-class fleet capability: dispatch a research question, it mines the
2024-2026 literature grounded in OUR competition domain (baked in below), and returns concrete ADOPT / SEARCH /
SKIP verdicts + a ranked shortlist + verified sources — then RECORDS the synthesis to the journal (ledger).

It shells out to the Claude CLI (`claude -p`, the fleet's LLM, which has WebSearch/WebFetch) with a strict,
domain-grounded prompt, saves the full report to research/deep_research/<slug>.md, and logs a decision-trail
entry so the research story lives in the journal like every experiment ([[researchpapers_pipeline_orchestrator]]).

Reusable/spec-driven: {question, family, timeout, model}. A BaseAgent subclass with its own data-wise test.
"""
from __future__ import annotations
import re
import subprocess
from pathlib import Path
from .base import BaseAgent

COMP = Path(__file__).resolve().parent.parent
OUT = COMP / "research" / "deep_research"

# OUR DOMAIN — baked in so EVERY research query is grounded in this competition (user: "remember our domain 100%").
DOMAIN = """Competition: biohub-cell-tracking-during-development (Kaggle). Biology: zebrafish embryo development
(Zebrahub), gastrula→segmentation ~5-24hpf; histone-mCherry NUCLEAR reporter. Imaging: light-sheet microscopy,
3D+time volumes. Task: (1) detect nuclei per 3D frame, (2) link across time into lineages, (3) detect DIVISIONS.
Metric: edge_jaccard + 0.1*division_jaccard, 7µm matching (royerlab official; runs locally = LB). CV is
EMBRYO-DISJOINT (train/test share no embryo); our 2-CV = 44b6 + 6bba (leave-one-embryo-out). Labels are SPARSE
POINT annotations (few complete tracks/frame; unlabeled != background => positive-unlabeled). We build a
FROM-SCRATCH 3D UNet detector (64x256x256 volume -> center heatmap). KEY XAI-diagnosed failure: we MISS DIM /
low-contrast nuclei (missed-cell intensity 0.62 vs 0.72 for hits); node-recall is the dominant metric lever.
Offline Kaggle CODE comp (no inference internet; ~2xT4; weights must be pre-staged as a Kaggle Dataset)."""

FORMAT = """Output a tight markdown report. For EACH relevant method/paper (EMPHASISE 2024-2026, include seminal
ones only to ground): (1) the ONE concrete mechanism, (2) whether it actually BEATS a well-tuned baseline in
fair 2024 comparisons (be skeptical), (3) a verdict ADOPT-NOW / SEARCH / PROTOTYPE / SKIP with a one-line reason
tied to OUR domain (dim-miss + embryo-disjoint generalization + sparse point/PU labels + offline 2xT4). Verify
claims with WebSearch/WebFetch and give resolvable source URLs. END with: 'SHORTLIST:' then the 2-3 things most
worth trying on our pipeline and exactly what to change. Be concrete and honest; no hype."""


class DeepResearch(BaseAgent):
    name = "deep-research"
    thread = "A"
    kind = "finding"

    def _slug(self, s):
        return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")[:50] or "research"

    def run(self, q, worker):
        spec = self.spec(q)
        question = spec.get("question") or q.get("question") if isinstance(q, dict) else None
        question = question or "Latest 2024-2026 methods relevant to our detector/linker"
        family = spec.get("family", "")
        prompt = (f"You are a research analyst. DOMAIN:\n{DOMAIN}\n\nRESEARCH QUESTION: {question}"
                  + (f"\nFocus family: {family}" if family else "") + f"\n\n{FORMAT}")
        # OPTIONAL dry_run: build+return the prompt WITHOUT spending an LLM call (preview / offline test).
        if spec.get("dry_run"):
            return self.done({"question": question, "family": family, "prompt": prompt, "dry_run": True},
                             f"[{worker}] deep-research [dry-run]: prompt built ({len(prompt)} chars), CLI not called.")
        cmd = ["claude", "-p", prompt, "--dangerously-skip-permissions"]
        if spec.get("model"):
            cmd += ["--model", str(spec["model"])]
        # OPTIONAL timeout (seconds), clamped to a sane [30, 7200] window so a bad value can't hang or 0-out.
        try:
            timeout = max(30, min(7200, int(spec.get("timeout", 900))))
        except Exception:  # noqa: BLE001
            timeout = 900
        try:
            r = subprocess.run(cmd, cwd=str(COMP), capture_output=True, text=True, timeout=timeout)
            report = r.stdout.strip()
        except Exception as e:  # noqa: BLE001
            return self.done({"error": str(e)[:200]}, f"[{worker}] deep-research: claude CLI failed — {str(e)[:120]}")
        if not report:
            return self.done({"error": "empty report"}, f"[{worker}] deep-research: empty report (check claude auth/CLI).")

        OUT.mkdir(parents=True, exist_ok=True)
        path = OUT / f"{self._slug(question)}.md"
        path.write_text(f"# deep-research: {question}\n\n{report}\n")
        # extract the SHORTLIST tail as the recommendation for the journal
        short = report.split("SHORTLIST:", 1)[-1].strip()[:400] if "SHORTLIST:" in report else report[-400:]
        summary = f"deep-research [{family or 'general'}]: {question[:120]}"
        from . import ledger
        ledger.log("deep-research", summary=summary, detail=report[:600], kind="decision",
                   recommendation=short or "see report")
        self.save_state({"question": question, "family": family, "report_path": str(path), "chars": len(report)})
        msg = (f"[{worker}] **DEEP-RESEARCH** · {family or 'general'} · {question[:90]}\n"
               f"→ report saved `{path.relative_to(COMP)}` ({len(report)} chars), journaled as a decision.\n\n"
               f"**SHORTLIST**: {short[:300]}")
        self.post(worker, "leader", msg, routine=False, kind="finding")
        return self.done({"question": question, "report_path": str(path), "shortlist": short}, msg, to="leader")


_AGENT = DeepResearch()


def run(q, worker):
    return _AGENT.run(q, worker)
