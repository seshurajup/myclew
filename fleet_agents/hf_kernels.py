"""hf-kernels — discover + arch-check Hugging Face Hub compute kernels for THIS box (no local build).

HF `kernels` (get_kernel("kernels-community/<name>")) loads PRE-COMPILED CUDA kernels from the Hub — flash-attn,
fused activation, quantization, fp8/mxfp8 GEMM, MoE — no nvcc, no ABI-risky pip build. This agent answers the one
question that actually matters for us: **does a kernel have a build that runs on OUR box** (torch 2.8 / cu128 /
x86_64 / RTX 5090 sm_120)? — because most kernels ship sm_90 (Hopper) / sm_100 (datacenter Blackwell) only, which are
BINARY-INCOMPATIBLE with the 5090's sm_120 (see fp8_sm120_ecosystem_verdict).

Honest limits it enforces:
  • Build-dir names (torch28-cxx11-cu128-x86_64-linux) encode torch/cuda/abi/platform — NOT compute capability. The
    sm_120 SASS lives INSIDE the fatbin, so a matching variant is necessary-but-not-sufficient → the real proof is a
    load+run smoke test (rule: measure the full thing, never assume — feedback_measure_full_step_not_microbench).
  • HF kernels download from the Hub at runtime → INTERNET required → NOT usable in an offline Kaggle code-comp
    (T4 kernels must be vendored as wheels instead; and T4=sm_75 has no fp8 anyway).
"""
from __future__ import annotations
import os
import re
from .base import BaseAgent

# Curated kernels-community repos worth checking, by NEED. Not exhaustive — discover() also queries the Hub live.
KERNELS_BY_NEED = {
    "fp8-gemm":      ["kernels-community/deep-gemm"],
    "flash-attn":    ["kernels-community/flash-attn", "kernels-community/flash-attn3", "kernels-community/vllm-flash-attn3"],
    "activation":    ["kernels-community/activation"],
    "quantization":  ["kernels-community/quantization", "kernels-community/quantization-eetq"],
    "moe":           ["kernels-community/moe"],
    "attention":     ["kernels-community/flash-attn", "kernels-community/paged-attention"],
    "norm":          ["kernels-community/layer-norm", "kernels-community/rmsnorm"],
}


def box_variant():
    """The build-variant string HF `kernels` would need for THIS box: torch<maj><min>-cxx11-cu<cuda>-<arch>-linux."""
    info = {"torch": None, "cuda": None, "arch": None, "cc": None, "variant": None, "platform": "x86_64-linux"}
    try:
        import platform as _pl
        import torch
        tv = torch.__version__.split("+")[0].split(".")            # 2.8.0 -> ['2','8','0']
        info["torch"] = f"torch{tv[0]}{tv[1]}"                      # torch28
        cu = getattr(torch.version, "cuda", None)                  # '12.8'
        info["cuda"] = ("cu" + cu.replace(".", "")) if cu else None  # cu128
        info["platform"] = f"{_pl.machine()}-linux"                # x86_64-linux
        if torch.cuda.is_available():
            cc = torch.cuda.get_device_capability(0)
            info["cc"] = f"{cc[0]}.{cc[1]}"                         # 12.0
            info["arch"] = f"sm_{cc[0]}{cc[1]}"                     # sm_120
        info["variant"] = f"{info['torch']}-cxx11-{info['cuda']}-{info['platform']}"
    except Exception as e:  # noqa: BLE001
        info["error"] = f"{type(e).__name__}: {e}"
    return info


def _variant_matches(build_variants, box):
    """A build dir matches if its torch tag AND cuda tag AND platform equal ours. (cc is inside the fatbin, not here.)"""
    want_t, want_c, want_p = box.get("torch"), box.get("cuda"), box.get("platform")
    exact = [v for v in build_variants if want_t and want_c and v.startswith(want_t + "-") and want_c in v and v.endswith(want_p)]
    # torch is often pinned to a nearby minor; also surface same-cuda/same-platform builds for other torch as fallbacks
    near = [v for v in build_variants if want_c and want_c in v and (want_p in v) and v not in exact]
    return exact, near


def check(repo_id):
    """Inspect a Hub kernel repo → does it have a build matching our torch/cuda/platform? Returns a structured verdict.
    Honest: a matching variant is necessary-not-sufficient for sm_120 (SASS is inside the fatbin → smoke-test to prove)."""
    box = box_variant()
    out = {"repo": repo_id, "box": box, "ok": False, "exact": [], "near": [], "note": ""}
    try:
        from huggingface_hub import HfApi
        fs = HfApi().list_repo_files(repo_id)
    except Exception as e:  # noqa: BLE001
        out["note"] = f"hub query failed ({type(e).__name__}: {str(e)[:80]}) — need internet + valid repo id"
        return out
    variants = sorted({f.split("/")[1] for f in fs if f.startswith("build/") and f.count("/") >= 2})
    out["all_variants"] = variants
    exact, near = _variant_matches(variants, box)
    out["exact"], out["near"] = exact, near
    if exact:
        out["ok"] = True
        out["note"] = (f"variant present for {box['variant']} → LOADABLE on this box's torch/cuda. "
                       f"BUT {box.get('arch')} SASS is inside the fatbin — LOAD+SMOKE-TEST to confirm it actually runs "
                       f"(most fp8 kernels ship sm_90/sm_100 only). Internet required (no offline Kaggle T4).")
    elif near:
        out["note"] = (f"no exact {box['torch']} build; same-cuda builds exist for other torch ({near[:4]}). "
                       f"May work if torch ABI-compatible — verify by load-test.")
    else:
        out["note"] = (f"NO build for our cuda/platform ({box['cuda']}/{box['platform']}). Not usable here. "
                       f"Variants offered: {variants[:6]}{'…' if len(variants) > 6 else ''}")
    return out


def cuobjdump_check(repo_id, cc=None):
    """DECISIVE sm_120 test (research-identified): download the matching-variant .so and cuobjdump its fatbin for the
    target SASS (e.g. sm_120). Answers definitively 'does this kernel actually have code for our GPU?' — a variant-name
    match cannot. No `kernels` install needed (uses huggingface_hub + the CUDA-toolkit cuobjdump). Returns a verdict."""
    import glob
    import subprocess
    box = box_variant()
    want_cc = (cc or box.get("cc") or "").replace(".", "")          # '120'
    out = {"repo": repo_id, "target": f"sm_{want_cc}", "has_sass": None, "note": ""}
    chk = check(repo_id)
    variant = (chk.get("exact") or chk.get("near") or [None])[0]
    if not variant:
        out["note"] = "no torch/cuda variant for this box → nothing to dump"; return out
    try:
        from huggingface_hub import HfApi, hf_hub_download
        sos = [f for f in HfApi().list_repo_files(repo_id) if f.startswith(f"build/{variant}/") and f.endswith(".so")]
        if not sos:
            out["note"] = f"variant {variant} has no .so listed"; return out
        p = hf_hub_download(repo_id, sos[0])
        cu = subprocess.run(["cuobjdump", "-lelf", p], capture_output=True, text=True, timeout=60)
        if cu.returncode != 0:
            out["note"] = f"cuobjdump unavailable/failed ({cu.stderr[:60]}); load-test on the card instead"; return out
        archs = sorted(set(re.findall(r"sm_(\d+)", cu.stdout)))
        out["archs_in_fatbin"] = [f"sm_{a}" for a in archs]
        out["has_sass"] = want_cc in archs
        out["note"] = (f"fatbin has {out['archs_in_fatbin']} — sm_{want_cc} {'PRESENT ✓ (still smoke-test to be sure)' if out['has_sass'] else 'ABSENT ✗ → will NOT run on this GPU'}")
    except Exception as e:  # noqa: BLE001
        out["note"] = f"cuobjdump_check failed ({type(e).__name__}: {str(e)[:80]})"
    return out


def discover(need=None, use_mcp_hint=True):
    """List candidate Hub kernels for a NEED and check each against our box. need∈KERNELS_BY_NEED or None=all.
    DISCOVERY can be widened with the Hugging Face MCP (semantic Hub search over kernels-community) when it's available
    interactively — the MCP finds candidates; this agent's check()/cuobjdump_check() prove usability on our sm_120
    (the MCP cannot run CUDA). Headless/offline fleet runs fall back to the curated KERNELS_BY_NEED map below."""
    needs = [need] if need in KERNELS_BY_NEED else list(KERNELS_BY_NEED)
    results = {}
    for nd in needs:
        for repo in KERNELS_BY_NEED[nd]:
            if repo not in results:
                results[repo] = check(repo)
    hint = ("widen discovery via HF MCP semantic Hub search (kernels-community); then cuobjdump_check each for sm_120"
            if use_mcp_hint else None)
    return {"need": need, "box": box_variant(), "kernels": results, "mcp_discovery_hint": hint}


class _HfKernels(BaseAgent):
    name = "hf-kernels"

    def run(self, q, worker):
        spec = (q.get("spec") or {}) if isinstance(q, dict) else {}
        repo = spec.get("repo"); need = spec.get("need")
        box = box_variant()
        if repo:
            r = check(repo)
            verdict = "USABLE (load-test to confirm sm_120)" if r["ok"] else ("maybe (near variant)" if r.get("near") else "NOT on this box")
            msg = (f"[{worker}] **HF-KERNELS** {repo} on {box.get('arch')}/{box.get('variant')}: {verdict}\n  {r['note']}")
            payload = r
        else:
            d = discover(need)
            usable = [k for k, v in d["kernels"].items() if v["ok"]]
            nope = [k for k, v in d["kernels"].items() if not v["ok"]]
            msg = (f"[{worker}] **HF-KERNELS** box={box.get('arch')}/{box.get('variant')} · need={need or 'all'}\n"
                   f"  variant-present: {usable or 'none'}\n  no-build-here: {nope}\n"
                   f"  NOTE: variant-present ≠ sm_120 runs (SASS in fatbin → load-test). Internet-only (not offline T4). "
                   f"Most fp8 kernels ship sm_90/sm_100 → dead on 5090 sm_120.")
            payload = d
        try:
            from researchpapers.fleet import post
            post.post_thread(worker, "all", msg, routine=False, kind="finding")
        except Exception:  # noqa: BLE001
            pass
        return ("done", payload, "all", msg)


_AGENT = _HfKernels()


def run(q, worker):
    return _AGENT.run(q, worker)
