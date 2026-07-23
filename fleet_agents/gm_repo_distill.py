"""gm-repo-distill — REUSABLE self-improving loop over Kaggle grandmaster winner GitHub repos. For each repo
it: (1) shallow-clones into ~/kaggle/github/, (2) SCANS the code to extract the reusable ML techniques it
uses (grep a curated technique lexicon over source + README), (3) checks which techniques our fleet already
covers vs which are NEW/missing, (4) records a distillation entry to docs/gm_distill_manifest.json, (5) DELETES
the clone to free disk (unless keep=True). The manifest means a re-run skips finished repos.

This is the deterministic first pass — a fast, cheap technique inventory across all ~215 winner repos that
pinpoints which repos have techniques our agents DON'T yet have, so a Claude deep-dive (the expensive step)
only runs on the genuinely-novel ones. Reusable/spec-driven: {repos, github_dir, keep, limit, batch}.
A BaseAgent with its own data-wise test (clone/claude injectable → tested OFFLINE).
"""
from __future__ import annotations
import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from .base import BaseAgent

COMP = Path(__file__).resolve().parent.parent
GHDIR = Path(os.path.expanduser("~/kaggle/github"))
MANIFEST = COMP / "docs" / "gm_distill_manifest.json"

# curated technique lexicon → (regex, our-fleet-agent that covers it or "" if a gap to build)
TECH = {
    "test-time augmentation (TTA)": (r"\bTTA\b|test.?time.?aug|tta_", "aug-ablation"),
    "exponential moving average (EMA)": (r"\bEMA\b|ModelEma|ema_decay|update_ema", ""),
    "stochastic weight averaging (SWA)": (r"\bSWA\b|swa_|AveragedModel|swa_utils", ""),
    "pseudo-labeling / self-training": (r"pseudo.?label|self.?train|noisy.?student", "pseudo-label"),
    "knowledge distillation": (r"distill|teacher.?student|kd_loss|soft.?target", "distill"),
    "mixup / cutmix": (r"mixup|cutmix|cut.?mix", ""),
    "focal loss": (r"focal.?loss|FocalLoss", ""),
    "label smoothing": (r"label.?smooth", ""),
    "arcface / metric learning": (r"arcface|arc.?margin|adacos|cosface|sub.?center", ""),
    "weighted boxes fusion (WBF)": (r"weighted.?box|wbf|box.?fusion", ""),
    "SAM / sharpness-aware": (r"\bSAM\b.*optim|sharpness.?aware|SAMOptim", ""),
    "gradient accumulation": (r"grad.?accum|accumulation.?steps", ""),
    "AMP / mixed precision": (r"amp|autocast|GradScaler|mixed.?precision", "gpu-best-practices"),
    "OOF stacking / blending": (r"\bOOF\b|out.?of.?fold|stack|blend", "tab-stack"),
    "optimized rounding / threshold": (r"OptimizedRounder|threshold.?opt|coeff.*round", "post-optimize"),
    "LoRA / PEFT finetune": (r"\bLoRA\b|peft|adapter|get_peft", "lora-train"),
    "test-time training (TTT)": (r"test.?time.?train|TTT|transductive", ""),
    "self-consistency / majority vote": (r"self.?consistency|majority.?vote|maj@|airv", "self-consistency-aggregator"),
    "graph neural net / GNN": (r"\bGNN\b|GraphConv|message.?passing|torch_geometric", "gnn-link-train"),
    "temporal / sequence model": (r"transformer|conformer|tcn|wavenet|lstm|gru", ""),
    "diffusion / generative aug": (r"diffusion|ddpm|stable.?diffusion|cyclegan|gan", "gan-train"),
    "nnU-Net / medical seg": (r"nnunet|nnU-Net|batchgenerators", "nnunet-segmentation-runner"),
    "ensemble / snapshot": (r"snapshot.?ensem|ensemble|seed.?average", "ensemble"),
    "quantization / pruning": (r"quantiz|int8|prune|distil.*compress|tome", "quantize"),
    "retrieval / RAG": (r"retriev|\bRAG\b|faiss|bm25|rerank", "llm-retrieve-rerank"),
}


def _load_manifest(path=None):
    p = Path(path) if path else MANIFEST
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:  # noqa: BLE001
            return {}
    return {}


def _save_manifest(m, path=None):
    p = Path(path) if path else MANIFEST
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(m, indent=1))


def scan_repo_dir(path):
    """Grep the technique lexicon over a repo dir's source + README → {technique: our_agent_or_gap}."""
    p = Path(path)
    text = ""
    for f in list(p.rglob("*.py"))[:400] + list(p.rglob("*.md"))[:40] + list(p.rglob("*.ipynb"))[:60]:
        try:
            text += f.read_text(errors="replace")[:20000] + "\n"
        except Exception:  # noqa: BLE001
            continue
    found = {}
    for tech, (rx, agent) in TECH.items():
        if re.search(rx, text, re.I):
            found[tech] = agent  # "" = a GAP (no fleet agent yet)
    return found


def distill(repos, github_dir=None, keep=False, limit=None, clone=None, scan=None, manifest_path=None):
    """Clone→scan→record→cleanup each repo. `clone`/`scan` injectable for offline testing; `manifest_path`
    overridable (tests use a temp one). Returns per-repo {techniques, gaps, status}; updates the manifest;
    deletes each clone unless keep=True."""
    gd = Path(github_dir) if github_dir else GHDIR
    gd.mkdir(parents=True, exist_ok=True)
    clone = clone or _shallow_clone
    scan = scan or scan_repo_dir
    man = _load_manifest(manifest_path)
    out = {}
    todo = [r for r in repos if r not in man][: (limit or len(repos))]
    for repo in todo:
        dest = gd / repo.replace("/", "__")
        try:
            ok = clone(repo, dest)
            if not ok:
                out[repo] = {"status": "clone-failed"}; man[repo] = out[repo]; continue
            techs = scan(dest)
            gaps = sorted(t for t, a in techs.items() if not a)
            covered = sorted(t for t, a in techs.items() if a)
            out[repo] = {"status": "done", "techniques": sorted(techs), "gaps": gaps, "covered_by": {t: techs[t] for t in covered}}
            man[repo] = out[repo]
        except Exception as e:  # noqa: BLE001
            out[repo] = {"status": f"error:{type(e).__name__}"}; man[repo] = out[repo]
        finally:
            if not keep and dest.exists():
                shutil.rmtree(dest, ignore_errors=True)   # free disk immediately
    _save_manifest(man, manifest_path)
    return out


def _shallow_clone(repo, dest, timeout=180):
    if dest.exists():
        shutil.rmtree(dest, ignore_errors=True)
    try:
        r = subprocess.run(["git", "clone", "--depth", "1", "--quiet", f"https://github.com/{repo}.git", str(dest)],
                           capture_output=True, text=True, timeout=timeout)
        return r.returncode == 0
    except Exception:  # noqa: BLE001
        return False


class GmRepoDistill(BaseAgent):
    name = "gm-repo-distill"
    thread = "S"
    kind = "verdict"

    def run(self, q, worker):
        spec = self.spec(q)
        repos = spec.get("repos")
        if not repos:
            wl = COMP / "docs" / "gm_repos.txt"
            src = wl if wl.exists() else Path("/tmp/gm_repos.txt")
            repos = [r.strip() for r in src.read_text().splitlines() if "/" in r] if src.exists() else []
        res = distill(repos, github_dir=spec.get("github_dir"), keep=bool(spec.get("keep", False)),
                      limit=int(spec["limit"]) if spec.get("limit") else None)
        gap_counts = {}
        for r, v in res.items():
            for g in v.get("gaps", []):
                gap_counts[g] = gap_counts.get(g, 0) + 1
        top_gaps = sorted(gap_counts.items(), key=lambda x: -x[1])[:12]
        done = [r for r, v in res.items() if v.get("status") == "done"]
        self.save_state({"distilled": res, "gap_counts": gap_counts})
        rows = "\n".join(f"| {t} | {c} repos |" for t, c in top_gaps)
        msg = (f"[{worker}] **GM-REPO-DISTILL** · scanned {len(done)}/{len(res)} winner repos (cloned→scanned→deleted)\n"
               f"top NEW techniques (no fleet agent yet), by repo frequency:\n| technique GAP | count |\n|:-|:-|\n{rows}\n"
               f"→ manifest {MANIFEST.name}; these gaps are the priority for Claude deep-dive + a new reusable agent.")
        self.log(summary=f"gm-repo-distill: {len(done)} repos scanned, top gaps {[t for t,_ in top_gaps[:5]]}",
                 detail="deterministic technique inventory across winner repos; clones deleted to save disk",
                 kind="verdict", recommendation="deep-dive the top-gap repos with Claude → add each as a reusable fleet agent")
        self.post(worker, "leader", msg, routine=False, kind="verdict")
        return self.done({"distilled": res, "gap_counts": gap_counts, "top_gaps": top_gaps}, msg, to="leader")


_AGENT = GmRepoDistill()


def run(q, worker):
    return _AGENT.run(q, worker)
