"""paper-learn — a paper becomes a TAUGHT lesson series with PyTorch proofs, basics first.

The pipeline, one agent per job (no forks, each reused):

    paper-md   PDF → markdown + assets + `equations.json` (every display formula, with its page and a
               rendered crop) + `sections` (the paper's own heading tree)
        ↓ manifest.json
    paper-learn  (THIS)  plan basics → one lesson per paper section → advanced;
               bind every numbered formula and figure to the lesson that owns its pages;
               attach the paper PACK's runnable PyTorch PROOF for each formula;
               write Pattern-B pairs through `learner.add_lesson` (the existing writer, reused);
               register the section in `learning/curriculum.yml`;
               refresh through `learning/lessonkit.py` so every code cell runs and its REAL output is
               captured; then VERIFY and report COVERAGE — formulas placed, formulas proven, cells run.

Why a PACK and not an LLM here: a lesson is only worth reading if its code *proves* the paper — e.g.
"one gradient-descent step on ⟨M k, v⟩ IS the linear-attention recurrence" must be an assertion that
runs, not prose. Proofs are therefore authored per paper in a pack module and this agent is the
deterministic engine that places, runs, verifies and publishes them. The coverage report is the
guarantee that not one formula was skipped.

PACK CONTRACT — `learning/paper_packs/<slug>.py` (or `spec.pack`):
    SLUG      str                       the paper-md slug (asset paths)
    HEADER    str                       preamble prepended to every code cell (imports/helpers)
    BASICS    [dict]                    prerequisite lessons, authored basics-first:
                                        {id,title,subtitle?,cells:[{note,code?,image?,shape?}]}
    SECTION   {"4.2": {title?, why, before?:[cell], after?:[cell], skip_eqs?:[int]}}
    EQ        {33: {name, latex, why, code?, shape?, figure?}}   per-formula teaching + proof
    ADVANCED  [dict]                    experiments / appendix / "what we steal" lessons
    ORDER_BASE int, SECTION_TITLE str   where the series sits in the curriculum

Spec:
    {"kind": "paper-learn", "spec": {"manifest": "docs/papers/<slug>/manifest.json",
                                     "pack": "learning/paper_packs/<slug>.py",
                                     "action": "build|plan|verify", "refresh": true}}
"""
from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path

from . import learner
from .base import BaseAgent, COMP

LESSONKIT = COMP / "learning" / "lessonkit.py"
PACK_DIR = COMP / "learning" / "paper_packs"


def _python() -> str:
    """The interpreter that RUNS the lesson code cells (real torch, real data)."""
    for c in (COMP / "research" / "cellmot_venv" / "bin" / "python",
              Path("/home/seshu/miniconda3/envs/kaggle_vision/bin/python")):
        if c.exists():
            return str(c)
    import sys
    return sys.executable


def load_pack(path):
    """Import a paper pack module from a file path (no package import side effects)."""
    p = Path(path)
    if not p.is_absolute():
        p = COMP / p
    if not p.exists():
        raise FileNotFoundError(f"paper pack not found: {p}")
    spec = importlib.util.spec_from_file_location(f"paper_pack_{p.stem}", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ------------------------------------------------------------------ pure planning / rendering
# ---------------------------------------------------------------- open weights: the published architecture
CFG_FIELDS = ("model_type", "hidden_size", "num_hidden_layers", "num_attention_heads",
              "num_key_value_heads", "head_dim", "intermediate_size", "moe_intermediate_size",
              "vocab_size", "max_position_embeddings", "num_experts", "n_routed_experts",
              "num_experts_per_tok", "num_experts_per_token", "num_local_experts",
              "n_shared_experts", "num_shared_experts", "torch_dtype", "dtype",
              "tie_word_embeddings", "rope_theta", "sliding_window", "attention_bias",
              # fields a paper's own design choices show up in (K3: SiTU-GLU betas, attn residuals,
              # MLA rank, the KDA/full-attention interleave) — the config CONFIRMS the paper's text
              "hidden_act", "activation_situ_beta", "activation_situ_linear_beta",
              "attn_res_block_size", "kv_lora_rank", "q_lora_rank", "first_k_dense_replace",
              "moe_router_activation_func", "moe_renormalize", "mla_use_nope", "mla_use_output_gate")


def hf_config(repo: str, cache_dir: Path, revision: str = "main", timeout: int = 60) -> dict:
    """Fetch ONLY `config.json` for a Hugging Face repo — the published ARCHITECTURE, no weights.

    A `config.json` is a few kB, so a paper that ships open weights can be compared against previous
    generations and against today's top models for free (`AutoModel.from_config` would then build the
    real graph with random init). Cached under the paper folder so the lesson still runs offline.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    dst = cache_dir / (repo.replace("/", "__") + ".json")
    if dst.exists() and dst.stat().st_size > 20:
        return json.loads(dst.read_text())
    raw = None
    try:
        from huggingface_hub import hf_hub_download
        raw = Path(hf_hub_download(repo, "config.json", revision=revision)).read_text()
    except Exception:  # noqa: BLE001  — plain HTTP works for public repos with no hub installed
        try:
            import urllib.request
            url = f"https://huggingface.co/{repo}/resolve/{revision}/config.json"
            req = urllib.request.Request(url, headers={"User-Agent": "paper-learn"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                raw = r.read().decode()
        except Exception as e:  # noqa: BLE001
            return {"repo": repo, "error": f"{type(e).__name__}: {str(e)[:120]}"}
    cfg = json.loads(raw)
    cfg["repo"] = repo
    dst.write_text(json.dumps(cfg, indent=2))
    return cfg


def config_row(cfg: dict) -> dict:
    """Pure: a config.json → the comparable architecture row (nested `text_config` handled)."""
    src = {**cfg.get("text_config", {}), **{k: v for k, v in cfg.items() if k != "text_config"}}
    row = {"repo": cfg.get("repo", "?")}
    for f in CFG_FIELDS:
        if f in src:
            row[f] = src[f]
    experts = row.get("num_experts") or row.get("n_routed_experts") or row.get("num_local_experts")
    if experts:
        row["experts"] = experts
        row["active_experts"] = row.get("num_experts_per_tok") or row.get("num_experts_per_token")
        row["shared_experts"] = row.get("n_shared_experts") or row.get("num_shared_experts")
    h, n = row.get("hidden_size"), row.get("num_hidden_layers")
    if h and n:                                   # a crude but comparable dense-parameter estimate
        ffn = row.get("intermediate_size") or 4 * h
        per_layer = 4 * h * h + 3 * h * ffn * (experts or 1)
        row["params_est_B"] = round((n * per_layer + 2 * h * (row.get("vocab_size") or 0)) / 1e9, 2)
    return row


def arch_spec(cfg: dict) -> str:
    """Pure: a published `config.json` → a DESIGNED architecture spec module (`<repo>_arch.py`).

    A config.json is a bag of 60 keys; a spec is a dataclass of the ~15 that define the model plus the
    budget arithmetic you actually reason with (total vs active parameters, KV cache per token, FLOPs per
    token). Weights are never needed: `AutoConfig`+`AutoModel.from_config` would build this exact graph.
    """
    r = config_row(cfg)
    name = re.sub(r"\W+", "_", r["repo"].split("/")[-1]).strip("_")
    keep = [(k, r[k]) for k in ("model_type", "hidden_size", "num_hidden_layers", "num_attention_heads",
                                "num_key_value_heads", "intermediate_size", "moe_intermediate_size",
                                "vocab_size", "max_position_embeddings", "experts", "active_experts",
                                "shared_experts", "kv_lora_rank", "attn_res_block_size", "hidden_act",
                                "activation_situ_beta", "activation_situ_linear_beta") if k in r]
    lines = [f'"""{name} — architecture spec derived from {r["repo"]}/config.json (no weights).',
             '',
             'Generated by fleet `paper-learn`. Every number is the published value; the methods are the',
             'budget arithmetic (parameters, KV cache, FLOPs) you need to compare designs.',
             '"""',
             'from dataclasses import dataclass', '', '', '@dataclass', f'class {name}Arch:']
    for k, v in keep:
        lines.append(f'    {k}: {type(v).__name__ if not isinstance(v, str) else "str"} = '
                     f'{v!r}' if not isinstance(v, bool) else f'    {k}: bool = {v!r}')
    lines += [
        '',
        '    @property',
        '    def experts_sparsity(self):',
        '        """experts held / experts used per token (1.0 for a dense model)."""',
        '        return (self.experts / self.active_experts) if getattr(self, "experts", None)'
        ' and getattr(self, "active_experts", None) else 1.0',
        '',
        '    def params_total(self):',
        '        """~parameters, counting EVERY expert (what the checkpoint costs)."""',
        '        d, n = self.hidden_size, self.num_hidden_layers',
        '        attn = 4 * d * d',
        '        ffn = 3 * d * getattr(self, "moe_intermediate_size", None or self.intermediate_size)',
        '        e = getattr(self, "experts", 0) or 1',
        '        emb = 2 * d * getattr(self, "vocab_size", 0)',
        '        return n * (attn + ffn * e) + emb',
        '',
        '    def params_active(self):',
        '        """~parameters touched per token (what a forward pass costs)."""',
        '        d, n = self.hidden_size, self.num_hidden_layers',
        '        a = getattr(self, "active_experts", 0) or 1',
        '        sh = getattr(self, "shared_experts", 0) or 0',
        '        ffn = 3 * d * (getattr(self, "moe_intermediate_size", None) or self.intermediate_size)',
        '        return n * (4 * d * d + ffn * (a + sh)) + 2 * d * getattr(self, "vocab_size", 0)',
        '',
        '    def kv_bytes_per_token(self, dtype_bytes=2):',
        '        """KV cache per token: 2 (K and V) x layers x kv_heads x head_dim x bytes."""',
        '        head_dim = self.hidden_size // self.num_attention_heads',
        '        kv = getattr(self, "num_key_value_heads", None) or self.num_attention_heads',
        '        return 2 * self.num_hidden_layers * kv * head_dim * dtype_bytes',
        '',
        '    def flops_per_token(self):',
        '        """~2 x active parameters (one multiply-add per weight)."""',
        '        return 2 * self.params_active()',
    ]
    return "\n".join(lines) + "\n"


def models_cells(rows: list, cache_rel: str) -> list:
    """Pure: the architecture-comparison cells — a markdown table plus a live DataFrame the lesson
    recomputes from the cached configs (so the numbers on the page are never hand-typed)."""
    if not rows:
        return []
    keys = [k for k in ("repo", "model_type", "hidden_size", "num_hidden_layers", "num_attention_heads",
                        "num_key_value_heads", "experts", "active_experts", "max_position_embeddings",
                        "params_est_B") if any(k in r for r in rows)]
    head = "| " + " | ".join(keys) + " |\n|" + "---|" * len(keys) + "\n"
    body = "".join("| " + " | ".join(str(r.get(k, "–")) for k in keys) + " |\n" for r in rows)
    note = ("### Open weights — the published architecture, fetched without the weights\n\n"
            f"`config.json` only ({len(rows)} model(s), a few kB each; cached in `{cache_rel}`), so the "
            "paper's architecture can be compared against earlier generations and today's top models. "
            "`AutoModel.from_config(cfg)` would build the identical graph with random init.\n\n"
            + head + body)
    code = f'''import json, pathlib, pandas as pd                              # configs cached at build time
rows = [json.loads(p.read_text()) for p in sorted(pathlib.Path("{cache_rel}").glob("*.json"))]
def row(c):                                                       # the comparable fields
    src = {{**c.get("text_config", {{}}), **{{k: v for k, v in c.items() if k != "text_config"}}}}
    e = src.get("num_experts") or src.get("n_routed_experts") or src.get("num_local_experts")
    return dict(repo=c.get("repo", "?"), type=src.get("model_type"), d=src.get("hidden_size"),
                layers=src.get("num_hidden_layers"), heads=src.get("num_attention_heads"),
                kv_heads=src.get("num_key_value_heads"), experts=e,
                active=src.get("num_experts_per_tok"), ctx=src.get("max_position_embeddings"))
df = pd.DataFrame([row(c) for c in rows])
if "kv_heads" in df and "heads" in df:                            # GQA ratio: KV-cache saving per token
    df["gqa_ratio"] = (df["heads"] / df["kv_heads"]).round(1)
if "experts" in df and "active" in df:
    df["sparsity"] = (df["experts"] / df["active"]).round(1)      # experts held / experts used
df'''
    budget = f'''# the budget arithmetic a design is actually compared on (no weights involved)
def budget(c):
    s = {{**c.get("text_config", {{}}), **{{k: v for k, v in c.items() if k != "text_config"}}}}
    d, n = s.get("hidden_size", 0), s.get("num_hidden_layers", 0)
    e = s.get("num_experts") or s.get("n_routed_experts") or s.get("num_local_experts") or 1
    a = s.get("num_experts_per_tok") or s.get("num_experts_per_token") or 1
    sh = s.get("n_shared_experts") or s.get("num_shared_experts") or 0
    ffn = s.get("moe_intermediate_size") or s.get("intermediate_size") or 4 * d
    kvh = s.get("num_key_value_heads") or s.get("num_attention_heads") or 1
    hd = d // max(s.get("num_attention_heads", 1), 1)
    tot = n * (4 * d * d + 3 * d * ffn * e) + 2 * d * s.get("vocab_size", 0)
    act = n * (4 * d * d + 3 * d * ffn * (a + sh)) + 2 * d * s.get("vocab_size", 0)
    return dict(repo=c.get("repo", "?"), total_B=round(tot / 1e9, 1), active_B=round(act / 1e9, 2),
                sparsity=round(tot / max(act, 1), 1), kv_kB_per_token=round(2 * n * kvh * hd * 2 / 1024, 1),
                gflops_per_token=round(2 * act / 1e9, 1), ctx_M=round(s.get("max_position_embeddings", 0) / 1e6, 3))

bud = pd.DataFrame([budget(c) for c in rows]).sort_values("total_B", ascending=False)
print("total_B counts every expert (checkpoint size); active_B is what one token touches")
bud'''
    diagram = f'''import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(9.5, 3.6), constrained_layout=True)
ax.set_axis_off()
for i, c in enumerate(rows):
    s = {{**c.get("text_config", {{}}), **{{k: v for k, v in c.items() if k != "text_config"}}}}
    e = s.get("num_experts") or s.get("n_routed_experts") or s.get("num_local_experts") or 0
    a = s.get("num_experts_per_tok") or s.get("num_experts_per_token") or 0
    x = i * 2.0
    ax.add_patch(plt.Rectangle((x, 0), 1.6, 2.6, fill=False, lw=1.4, ec="#8a8f98"))
    ax.text(x + 0.8, 2.75, c.get("repo", "?").split("/")[-1][:18], ha="center", fontsize=9, weight="bold")
    rows_txt = [f"d = {{s.get('hidden_size')}}", f"layers = {{s.get('num_hidden_layers')}}",
                f"heads {{s.get('num_attention_heads')}} / kv {{s.get('num_key_value_heads')}}",
                (f"MoE {{a}} of {{e}}" if e else "dense FFN"),
                f"ffn {{s.get('moe_intermediate_size') or s.get('intermediate_size')}}",
                f"ctx {{(s.get('max_position_embeddings') or 0) // 1024}}K",
                f"act {{s.get('hidden_act', '?')}}"]
    for j, t in enumerate(rows_txt):
        ax.text(x + 0.8, 2.3 - j * 0.33, t, ha="center", fontsize=8, color="#333")
    if e:                                                          # show the sparsity as a filled bar
        frac = a / e
        ax.add_patch(plt.Rectangle((x + 0.15, -0.35), 1.3, 0.18, color="#e7eaef"))
        ax.add_patch(plt.Rectangle((x + 0.15, -0.35), 1.3 * frac, 0.18, color="#0b6cff"))
        ax.text(x + 0.8, -0.62, f"{{100*frac:.1f}}% of experts active", ha="center", fontsize=7, color="#555")
ax.set_xlim(-0.4, max(2.0 * len(rows), 2)); ax.set_ylim(-0.9, 3.1)
p = pathlib.Path("{cache_rel}/arch_diagram.png"); fig.savefig(p, dpi=150); plt.close(fig)
print("architecture diagram written to", p)'''
    return [{"note": note, "code": code},
            {"note": "### The budget these numbers imply\n\nTotal vs **active** parameters, KV cache per "
                     "token and FLOPs per token — computed from the configs, not quoted.", "code": budget},
            {"note": "### The same architectures, drawn\n\nOne box per model with its real dimensions and "
                     "the fraction of experts a token actually activates.",
             "code": diagram,
             "image": f"{cache_rel}/arch_diagram.png\nPublished architectures side by side (config.json only)"}]


def with_header(cells: list, header: str) -> list:
    """Prepend the pack's HEADER to the FIRST code cell only — lessonkit runs a lesson's cells in one
    shared namespace, so repeating the imports in every cell would just be noise on the page."""
    out, done = [], False
    for c in cells:
        c = dict(c)
        if c.get("code") and header and not done:
            c["code"] = header.rstrip() + "\n\n" + c["code"].lstrip("\n")
            done = True
        out.append(c)
    return out


def eq_pages(equations: list, total: int) -> dict:
    """Pure: equation number → page for ALL 1..total.

    The extractor binds most right-margin numbers but not every one (a `(45)` that shares a block, a
    two-column float…). Equation numbers are monotone in page order, so the gaps interpolate exactly —
    which is what lets the PACK stay the source of truth for formulas while crops stay opportunistic.
    """
    known = {}
    for e in equations:
        n = e.get("n")
        if str(n or "").isdigit() and 1 <= int(n) <= total:
            known.setdefault(int(n), int(e.get("page", 0)))
    if not known:
        return {}
    out, last = {}, min(known.values())
    for n in range(1, total + 1):
        if n in known:
            last = known[n]
        else:                                            # next known number bounds it from above
            nxt = next((known[m] for m in range(n + 1, total + 1) if m in known), last)
            last = max(last, min(last, nxt))
        out[n] = last
    return out


def eq_cells(numbers: list, EQ: dict, crops: dict, header: str, asset_rel: str,
             skip: set | None = None, kind: str = "paper") -> list:
    """Pure: the formulas OWNED BY one section → cells, driven by the pack (not by the extractor).

    Every number in `numbers` gets a cell: the pack's hand-checked `latex` (PDF text cannot express a
    fraction), its `why`, its runnable PROOF `code`, and — when the extractor managed to bind one — the
    PDF crop of that exact formula as printed.

    `kind="repo"` teaches a code base instead of a paper, so the numbered unit is an API + its INVARIANT
    rather than an equation: the pack supplies `sig` (the real signature, rendered as code) instead of
    `latex`, and the proof calls the cloned repo rather than re-deriving algebra.
    """
    skip = skip or set()
    repo = kind == "repo"
    cells = []
    for n in numbers:
        if n in skip:
            continue
        spec = EQ.get(n, {})
        e = crops.get(n, {})
        latex = (spec.get("latex") or e.get("latex") or "").strip()
        name = spec.get("name") or (f"Unit ({n})" if repo else f"Equation ({n})")
        why = spec.get("why") or ("> TODO(author): what every symbol is, what the formula *does*, and "
                                  "why the paper needs it here.")
        if repo:
            sig = (spec.get("sig") or latex).strip()
            body = f"```python\n{sig}\n```" if sig else ""
            note = [f"### {name} — unit ({n})", "", body, "", why]
            cell = {"note": "\n".join(note)}
            if spec.get("code"):
                cell["code"] = (header + "\n" + spec["code"].strip()) if header else spec["code"].strip()
            if spec.get("shape"):
                cell["shape"] = spec["shape"]
            if spec.get("figure"):
                cell["image"] = f"{asset_rel}/{spec['figure']}"
            cells.append(cell)
            continue
        note = [f"### {name} — eq. ({n})", "", f"$$\n{latex}\n$$", "", why]
        cell = {"note": "\n".join(note)}
        if spec.get("code"):
            cell["code"] = (header + "\n" + spec["code"].strip()) if header else spec["code"].strip()
        if spec.get("shape"):
            cell["shape"] = spec["shape"]
        if spec.get("figure"):
            cell["image"] = f"{asset_rel}/{spec['figure']}"
        elif e.get("image"):
            cell["image"] = f"{asset_rel}/eq/{Path(e['image']).name}\nEquation ({n}) as printed in the PDF"
        cells.append(cell)
    return cells


def fig_cells(figs: list, asset_rel: str) -> list:
    """Pure: the section's figures as `--- image` cells with the paper's own captions."""
    out = []
    for f in figs:
        cap = (f.get("caption") or "").strip()
        label = f.get("label") or "Figure"
        out.append({"note": f"### {label} — as printed in the paper\n\n{cap}",
                    "image": f"{asset_rel}/{Path(f['path']).name}\n{cap}"})
    return out


def repo_manifest(pack, out_dir: Path) -> Path:
    """A GitHub clone, taught with the same contract as a paper.

    Papers and code bases differ in what the numbered thing IS — a formula you re-derive vs an API whose
    invariant you assert by calling it — but nothing else differs: both want basics→advanced ordering, a
    coverage gate so no unit is silently skipped, and a proof per unit. So a repo pack declares `REPO`
    (url/commit/local/sections) and this synthesises the manifest.json + equations.json shape that
    `paper-md` would have produced, letting the ENTIRE existing build path run unchanged — asset copying,
    orphan pruning, curriculum registration, the honesty check.

    Sections are ordered `REPO["sections"]`; "pages" are their ordinal, which is all `plan_from_paper`
    needs. There are no PDF crops, because the source of truth is the code — which is stronger: a crop
    shows what was printed, a passing call shows what the library actually does at the pinned commit.
    """
    repo = getattr(pack, "REPO", {})
    slug = getattr(pack, "SLUG", "repo")
    secs = [{"num": num, "title": title, "page": i + 1, "level": 1}
            for i, (num, title) in enumerate(repo.get("sections", []))]
    md = repo.get("md") or ""
    man = {"slug": slug, "title": repo.get("title", slug), "source": repo.get("url", ""),
           "commit": repo.get("commit", ""), "local": repo.get("local", ""), "kind": "repo",
           "sections": secs, "pages": len(secs), "figures": [], "md": md,
           "equations_json": str(out_dir / "units.json")}
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "manifest.json").write_text(json.dumps(man, indent=2))
    (out_dir / "units.json").write_text(json.dumps(
        [{"n": n, "latex": (sp.get("sig") or ""), "name": sp.get("name", "")}
         for n, sp in sorted(getattr(pack, "EQ", {}).items())], indent=2))
    return out_dir / "manifest.json"


def coverage(manifest: dict, equations: list, EQ: dict, plan: list, total: int) -> dict:
    """Pure: the nothing-missed report, judged against the paper's OWN equation count (`TOTAL_EQ`).

    `placed`  = numbers that landed in a lesson         (must equal `total`)
    `taught`  = numbers with hand-checked LaTeX + why    (the pack's job)
    `proven`  = numbers with runnable PyTorch            (the strongest form)
    `cropped` = numbers whose PDF crop the extractor bound (nice-to-have)
    """
    allnums = list(range(1, total + 1))
    placed = sorted({n for it in plan for n in it.get("eqs", [])})
    cropped = sorted({int(e["n"]) for e in equations if str(e.get("n") or "").isdigit()})
    # a paper unit is "taught" when it has hand-checked LaTeX; a repo unit, when it has a real signature
    taught = sorted(n for n in allnums if EQ.get(n, {}).get("latex") or EQ.get(n, {}).get("sig"))
    proven = sorted(n for n in allnums if EQ.get(n, {}).get("code"))
    figs = manifest.get("figures", [])
    return dict(total=total, placed=len(placed), missing=[n for n in allnums if n not in placed],
                taught=len(taught), untaught=[n for n in allnums if n not in taught],
                proven=len(proven), unproven=[n for n in allnums if n not in proven],
                cropped=len(cropped), figures=len(figs),
                figures_placed=sum(len(it.get("figures", [])) for it in plan), lessons=len(plan))


def cells_ran(learning: Path) -> dict:
    """Did every `--- code` cell actually produce a real `--- output`? (the honesty check)"""
    txt = learning.read_text(errors="replace")
    blocks = re.findall(r"(?m)^--- (note|code|output|image|shape)\s*$", txt)
    code = ran = 0
    for i, b in enumerate(blocks):
        if b == "code":
            code += 1
            if i + 1 < len(blocks) and blocks[i + 1] == "output":
                ran += 1
    return {"code_cells": code, "with_output": ran, "silent": code - ran}


# ------------------------------------------------------------------ the agent
class PaperLearn(BaseAgent):
    name = "paper-learn"
    thread = "S"
    kind = "finding"

    def run(self, q, worker):
        spec = self.spec(q)
        mpath = spec.get("manifest")
        # A repo pack carries its own source of truth (the clone), so it needs no paper-md run: it declares
        # REPO and synthesises the manifest shape paper-md would have written.
        if not mpath and spec.get("pack"):
            _p = load_pack(spec["pack"])
            if getattr(_p, "KIND", "paper") == "repo":
                mpath = str(repo_manifest(_p, COMP / "docs" / "repos" / getattr(_p, "SLUG", "repo")))
        if not mpath:
            return self.escalate(worker, "leader",
                                 f"[{worker}] paper-learn needs `spec.manifest` (a paper-md manifest.json) "
                                 f"or a repo pack (KIND='repo', which supplies its own). Run kind=paper-md first.")
        mp = Path(mpath) if Path(mpath).is_absolute() else COMP / mpath
        manifest = json.loads(mp.read_text())
        slug = manifest.get("slug", mp.parent.name)
        eqs_all = json.loads(Path(manifest.get("equations_json", mp.parent / "equations.json")).read_text())

        pack = load_pack(spec.get("pack") or (PACK_DIR / f"{slug.replace('-', '_')}.py"))
        kind = getattr(pack, "KIND", "paper")
        noun = "units" if kind == "repo" else "equations"
        EQ = getattr(pack, "EQ", {})
        SECTION = getattr(pack, "SECTION", {})
        HEADER = getattr(pack, "HEADER", "")
        out_dir = Path(spec.get("out_dir") or (COMP / "learning" / "annotated"))
        prefix = spec.get("prefix") or getattr(pack, "PREFIX", slug[:2])
        order_base = int(spec.get("order_base") or getattr(pack, "ORDER_BASE", 1000))
        section_title = spec.get("section") or getattr(pack, "SECTION_TITLE", manifest.get("title", slug))
        asset_rel = f"learning/assets/{slug}"

        # ---- plan: basics (pack) → one lesson per paper section → advanced (pack)
        total = int(spec.get("total_eq") or getattr(pack, "TOTAL_EQ", 0) or
                    max([int(e["n"]) for e in eqs_all if str(e.get("n") or "").isdigit()] or [0]))
        pages_of = eq_pages(eqs_all, total)
        crops = {}
        for e in eqs_all:                                   # first crop wins per number
            if str(e.get("n") or "").isdigit():
                crops.setdefault(int(e["n"]), e)
        core = learner.plan_from_paper(manifest, prefix, order_base=order_base + 100, level=1)
        drop = {s.lower() for s in (spec.get("skip_sections") or getattr(pack, "SKIP_SECTIONS", ["references"]))}
        core = [it for it in core if it["title"].strip().lower() not in drop]
        # A pack may declare which section owns which equations (EQ_SECTIONS). Prefer it: page
        # interpolation is a heuristic, and putting eq. 29 one lesson early breaks the namespace its
        # later cells share (a cell that says `W2` needs the cell that built `W2` in the SAME lesson).
        ranges = {name: (lo, hi) for name, lo, hi in getattr(pack, "EQ_SECTIONS", [])}
        assigned = set()
        for it in core:
            if ranges:
                lo, hi = ranges.get(it["num"], (0, -1))
                it["eqs"] = [n for n in range(max(lo, 1), hi + 1) if n <= total and n not in assigned]
            else:
                p0, p1 = it["pages"]
                it["eqs"] = [n for n, pg in pages_of.items() if p0 <= pg <= p1 and n not in assigned]
            assigned |= set(it["eqs"])
        leftover = [n for n in range(1, total + 1) if n not in assigned]
        if leftover and core:                               # nothing may fall off the end
            core[-1]["eqs"] = sorted(core[-1]["eqs"] + leftover)
        cov = coverage(manifest, eqs_all, EQ, core, total)

        if spec.get("action") == "plan":
            rows = "; ".join(f"{it['id']} §{it['num']} {it['title'][:28]} p{it['pages'][0]}-{it['pages'][1]} "
                             f"({len(it['eqs'])} eq)" for it in core)
            msg = (f"[{worker}] PAPER-LEARN plan for `{slug}`: {len(getattr(pack,'BASICS',[]))} basics + "
                   f"{len(core)} section lessons + {len(getattr(pack,'ADVANCED',[]))} advanced. "
                   f"Formulas {cov['total']} total / {cov['placed']} placed / {cov['taught']} taught / "
                   f"{cov['proven']} proven. {rows}")
            self.post(worker, "all", msg)
            return self.done({"plan": core, "coverage": cov}, msg)

        # ---- copy the paper's assets under the comp root so the hub serves them
        import shutil
        aroot = COMP / asset_rel
        (aroot / "eq").mkdir(parents=True, exist_ok=True)
        src = mp.parent / "assets"
        for sub, dst in (("fig", aroot), ("eq", aroot / "eq"), ("tab", aroot)):
            for f in sorted((src / sub).glob("*.png")) if (src / sub).exists() else []:
                shutil.copy2(f, dst / f.name)

        # paper-md writes an absolute path; a repo pack states one already relative to the comp root
        _md = Path(manifest["md"]) if manifest.get("md") else None
        md_rel = "" if _md is None else str(_md.relative_to(COMP) if _md.is_absolute() else _md)
        paper = manifest.get("title", slug)
        refresh = bool(spec.get("refresh", True))
        py = spec.get("python") or _python()
        written, ids = [], []

        def emit(lid, title, cells, order, subtitle="", source=md_rel):
            res = learner.add_lesson(lid, title, "", "", order=order, subtitle=subtitle, source=source,
                                     cells=cells, out_dir=out_dir, refresh=False)
            lp = Path(res["learning"])
            ok = learner._refresh(lp, py) if refresh else None
            st = cells_ran(lp) if refresh else {}
            written.append({"id": lid, "title": title, "refreshed": ok, **st})
            ids.append(lid)

        for i, L in enumerate(getattr(pack, "BASICS", [])):
            emit(L["id"], L["title"], with_header(L["cells"], HEADER),
                 order_base + 10 * i, L.get("subtitle", ""), L.get("source", md_rel))

        for it in core:
            s = SECTION.get(it["num"], {})
            title = f"{it['num']} {s.get('title') or it['title']}".strip()
            why = s.get("why") or (f"> TODO(author): the *why* of §{it['num']} in 3–5 lines.")
            eqr = f"({it['eqs'][0]}–{it['eqs'][-1]})" if it["eqs"] else "(none)"
            if kind == "repo":
                head = {"note": f"## {title}\n**Repo:** {paper} · {manifest.get('source','')}"
                                f"{' @ ' + manifest['commit'][:9] if manifest.get('commit') else ''} · "
                                f"{noun} {eqr}.\n\n{why}"}
            else:
                head = {"note": f"## {title}\n**Paper:** {paper} · §{it['num']} · pages {it['pages'][0]}–"
                                f"{it['pages'][1]} · equations {eqr} · {len(it['figures'])} figures.\n\n{why}"}
            cells = [head] + list(s.get("before", [])) + fig_cells(it["figures"], asset_rel)
            cells += eq_cells(it["eqs"], EQ, crops, "", asset_rel, set(s.get("skip_eqs", [])), kind)
            cells += list(s.get("after", []))
            emit(it["id"], title, with_header(cells, HEADER), it["order"], f"{paper} §{it['num']}")

        # ---- open weights: if the paper (or its lineage) ships configs, fetch them and compare
        rows = []
        models = spec.get("models") or getattr(pack, "MODELS", [])
        if models:
            cdir = mp.parent / "models"
            cfgs = [hf_config(r, cdir) for r in models]
            rows = [config_row(c) for c in cfgs if "error" not in c]
            bad = [c["repo"] for c in cfgs if "error" in c]
            for c in cfgs:                       # a designed spec module per model, next to its config
                if "error" not in c:
                    (cdir / (c["repo"].split("/")[-1].replace("-", "_") + "_arch.py")).write_text(arch_spec(c))
            if rows:
                emit(f"{prefix}cfg", "Open weights — published architectures side by side",
                     with_header(models_cells(rows, str(cdir.relative_to(COMP))), HEADER),
                     order_base + 880, f"{paper} · config.json only, no weights")
            if bad:
                self.post(worker, "all", f"[{worker}] paper-learn could not fetch config.json for {bad} "
                                         f"(offline or gated) — the comparison lesson skipped them.",
                          routine=True)

        for i, L in enumerate(getattr(pack, "ADVANCED", [])):
            emit(L["id"], L["title"], with_header(L["cells"], HEADER),
                 order_base + 900 + 10 * i, L.get("subtitle", ""), L.get("source", md_rel))

        # PRUNE our own orphans: if a rebuild renumbers the sections (a heading fix changes the plan),
        # the previous run's files would linger and the hub would show a duplicate lesson. Only files
        # this pack generated are touched — identified by their `@ source:` pointing at this paper.
        pruned = []
        for old in sorted(out_dir.glob(f"{prefix}*.learning")):
            if old.stem in ids:
                continue
            head = old.read_text(errors="replace")[:400]
            if md_rel and f"@ source: {md_rel}" in head:
                old.unlink(); (old.with_suffix(".py")).unlink(missing_ok=True)
                pruned.append(old.stem)
        reg = learner.curriculum_add(section_title, ids)
        silent = [w["id"] for w in written if w.get("silent")]
        code_cells = sum(w.get("code_cells", 0) for w in written)
        with_out = sum(w.get("with_output", 0) for w in written)
        report = dict(slug=slug, lessons=len(written), ids=ids, coverage=cov, curriculum=reg, pruned=pruned,
                      code_cells=code_cells, cells_with_output=with_out, silent_lessons=silent,
                      out_dir=str(out_dir), assets=asset_rel, python=py)
        (COMP / "learning" / f"_paper_learn_{slug}.json").write_text(json.dumps(report, indent=2))

        msg = (f"[{worker}] PAPER-LEARN ✅ `{paper[:60]}` → **{len(written)} Pattern-B lessons** "
               f"({', '.join(ids[:4])}…) in `{out_dir.relative_to(COMP)}`, registered as “{section_title}”. "
               f"{noun.capitalize()}: {cov['placed']}/{cov['total']} placed, {cov['taught']} taught, "
               f"{cov['proven']} PROVEN in PyTorch. Figures {cov['figures_placed']}/{cov['figures']}. "
               f"Code cells {with_out}/{code_cells} produced real output"
               + (f" — SILENT in {silent}" if silent else "")
               + (f" (pruned stale {pruned})" if pruned else "")
               + (f". Untaught {noun}: {cov['untaught'][:12]}" if cov["untaught"]
                  else f". Every {noun[:-1]} taught."))
        self.post(worker, "all", msg)
        self.log(f"paper-learn: {slug}", detail=json.dumps(cov), kind="finding")
        return self.done(report, msg)


_AGENT = PaperLearn()


def run(q, worker):
    return _AGENT.run(q, worker)
