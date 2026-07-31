"""setup-env — REUSABLE dependency manager for lib-gated fleet agents. Many agents escalate with
"needs X (not installed)" (foundation-3d-matcher→mast3r/dust3r, automl-oof-factory→autogluon, molecular→rdkit,
gp-symbolic→gplearn, nnunet→nnunetv2, chess→python-chess…). This agent RESOLVES those cleanly:

  • REQUIRES registry maps each gated agent → its pip package(s) + the import used to probe presence.
  • `check` reports which gated agents are ready vs missing (probes imports; never installs).
  • `install` pip-installs the missing ones SAFELY — the project pins numpy 2.4.6 + torch 2.8.0+cu128 (sm_120),
    so ABI-sensitive installs go with a numpy CONSTRAINT and torch is re-verified/repaired (+cu128) afterward so
    a heavy dep (autogluon/nnunet) can never silently break cuda.is_available(). DRY-RUN by default.

Comp-agnostic and safe: dry_run default, never raises, reports per-package status. A BaseAgent with a
data-wise test (registry + check + dry-run planning; no heavy install in the test).
"""
from __future__ import annotations
import importlib
import subprocess
import sys
from pathlib import Path
from .base import BaseAgent

COMP = Path(__file__).resolve().parent.parent

# agent-name → {"pip": [pip specs], "probe": import-name to test presence, "abi": pulls numpy/torch?}
REQUIRES = {
    "foundation-3d-matcher": {"pip": ["mast3r", "dust3r"], "probe": "mast3r", "abi": True},
    "automl-oof-factory":    {"pip": ["autogluon.tabular"], "probe": "autogluon.tabular", "abi": True},
    "molecular-featurizer":  {"pip": ["rdkit"], "probe": "rdkit", "abi": False},
    "gp-symbolic-feature":   {"pip": ["gplearn"], "probe": "gplearn", "abi": False},
    "nnunet-segmentation-runner": {"pip": ["nnunetv2"], "probe": "nnunetv2", "abi": True},
    "chess-search-engine":   {"pip": ["python-chess"], "probe": "chess", "abi": False},
    "nnue-trainer":          {"pip": ["python-chess"], "probe": "chess", "abi": False},
}
NUMPY_PIN = "numpy==2.4.6"                      # project ABI anchor (biohub); never let a dep move it
TORCH_REPAIR = ["torch", "--index-url", "https://download.pytorch.org/whl/cu128"]

# ── VERIFIED cu12 / CUDA-12.8 NVIDIA stack for the RTX 5090 (sm_120) ────────────────────────────────
# The 5090 driver is 12.8; sm_120 needs a COHERENT cu12/12.8 stack. A single cu13/cu130 wheel poisons
# it: cupy loads the cu13 nvrtc (PTX ISA too new) → CUDA_ERROR_INVALID_IMAGE, or cu13 curand →
# CURAND_STATUS_INITIALIZATION_FAILED. These exact versions were PROVEN to run sm_120 kernels (elementwise,
# gather, cumsum, interp, curand) on 2026-07-23. Rule: every accelerated lib (torch+cu128, cupy-cuda12x,
# cuDF/RAPIDS, TensorRT) must sit on THIS stack; never install a bare nvidia-* (cu13) or torch +cu130.
CU128_STACK = {
    "nvidia-cuda-nvrtc-cu12": "12.8.93",   # JIT: emits PTX ISA ≤8.7 the 12.8 driver loads (the key fix)
    "nvidia-cublas-cu12":     "12.8.4.1",
    "nvidia-curand-cu12":     "10.3.9.90", # cu13 curand → INIT_FAILED on 12.8 driver; this one initialises
    "nvidia-cusolver-cu12":   "11.7.3.90",
    "nvidia-cusparse-cu12":   "12.5.8.93",
    "nvidia-cufft-cu12":      "11.3.3.83",
    "nvidia-nvjitlink-cu12":  "12.8.93",
    "nvidia-nccl-cu12":       "2.29.3",
    "nvidia-cudnn-cu12":      "9.8.0.87",    # DL primitives (torch/onnx); loads on sm_120
}
# RAPIDS / cuDF GPU-dataframe stack — VERIFIED groupby on sm_120 (2026-07-23). Same cu12 family;
# gives GPU pandas for the big tabular assembles (Track A: 3.78M rows). Install from pypi.nvidia.com.
RAPIDS_STACK = {
    "cudf-cu12":              "25.8.0",
    "rmm-cu12":               "25.8.0",
    "pylibcudf-cu12":         "25.8.0",
    "libcudf-cu12":           "25.8.0",
    "librmm-cu12":            "25.8.0",
    "nvidia-cuda-nvcc-cu12":  "12.9.86",
    "nvidia-cuda-runtime-cu12": "12.9.79",
}
RAPIDS_INDEX = "https://pypi.nvidia.com"   # pip install --extra-index-url=RAPIDS_INDEX ...
# cu13/cu130 packages that POISON the sm_120 stack (bare names = cu13 wheels dragged in by torch+cu130).
CU13_POISON = ("torch==*cu130*", "nvidia-cuda-nvrtc", "nvidia-cuda-runtime", "nvidia-cublas",
               "nvidia-curand", "nvidia-cusolver", "nvidia-cusparse", "nvidia-cufft", "nvidia-cuda-cupti",
               "nvidia-cudnn-cu13", "nvidia-cusparselt-cu13", "nvidia-nccl-cu13", "nvidia-nvjitlink",
               "nvidia-nvshmem-cu13", "nvidia-nvtx")


def scan_cuda_stack():
    """Detect cu13/cu130 CONTAMINATION of the sm_120 stack. Returns {clean, cu13_found, torch_build, cupy}.
    A single bare `nvidia-*` (cu13) or torch+cu130 is enough to break cupy/cuDF on the 5090."""
    import importlib.metadata as _md
    installed = {d.metadata["Name"].lower(): d.version for d in _md.distributions()
                 if d.metadata.get("Name")}
    # bare cu13 names (no -cu12/-cu13 suffix on the CUDA-13 wheels except a couple)
    cu13 = []
    for n in ("nvidia-cuda-nvrtc", "nvidia-cuda-runtime", "nvidia-cublas", "nvidia-curand",
              "nvidia-cusolver", "nvidia-cusparse", "nvidia-cufft", "nvidia-cuda-cupti", "nvidia-nvjitlink",
              "nvidia-nvtx", "nvidia-cudnn-cu13", "nvidia-cusparselt-cu13", "nvidia-nccl-cu13",
              "nvidia-nvshmem-cu13"):
        if n in installed:
            cu13.append(f"{n}=={installed[n]}")
    tb = None
    try:
        import torch; tb = torch.__version__
    except Exception:  # noqa: BLE001
        pass
    cupy_v = installed.get("cupy-cuda12x")
    clean = not cu13 and (tb is None or "cu130" not in (tb or "") and "+cu13" not in (tb or ""))
    return {"clean": clean, "cu13_found": cu13, "torch_build": tb, "cupy": cupy_v,
            "reason": "sm_120 stack clean (cu12/12.8 only)" if clean
                      else f"cu13/cu130 CONTAMINATION: {cu13 or tb} — poisons cupy/cuDF on the 5090"}


def repair_cuda_stack(dry_run=True):
    """Purge cu13/cu130 contamination and pin the VERIFIED cu12/12.8 wheels for sm_120. dry_run only plans.
    Order: uninstall bare cu13 nvidia-* (+torch cu130 if present) → install CU128_STACK pins → clear cupy
    kernel cache. Torch itself is re-installed +cu128 separately via repair_torch()."""
    scan = scan_cuda_stack()
    purge = [p.split("==")[0] for p in scan["cu13_found"]]
    if scan["torch_build"] and ("cu130" in scan["torch_build"] or "+cu13" in scan["torch_build"]):
        purge.append("torch")
    install = [f"{k}=={v}" for k, v in CU128_STACK.items()]
    plan = {"purge": purge, "install": install,
            "post": "rm -rf ~/.cupy/kernel_cache/*  &&  repair_torch(dry_run=False) if torch needed"}
    if dry_run:
        return {"dry_run": True, **plan, "scan": scan}
    if purge:
        _pip(["uninstall", "-y", *purge])
    ok, log = _pip(["install", "--no-cache-dir", *install])
    import shutil, os
    cc = os.path.expanduser("~/.cupy/kernel_cache")
    if os.path.isdir(cc):
        shutil.rmtree(cc, ignore_errors=True)
    return {"dry_run": False, "ok": ok, "log": log[-300:], **plan, "scan_after": scan_cuda_stack()}


def _present(probe):
    try:
        importlib.import_module(probe); return True
    except Exception:  # noqa: BLE001
        return False


def check(agents=None):
    """Probe which gated agents are READY vs MISSING (imports only; no install). Returns {agent: {ready, pip}}."""
    names = agents or list(REQUIRES)
    out = {}
    for a in names:
        req = REQUIRES.get(a)
        if not req:
            out[a] = {"ready": None, "note": "not a lib-gated agent (no known deps)"}; continue
        out[a] = {"ready": _present(req["probe"]), "pip": req["pip"], "abi": req["abi"]}
    return out


def _pip(args, timeout=1800):
    try:
        r = subprocess.run([sys.executable, "-m", "pip", *args], capture_output=True, text=True, timeout=timeout)
        return r.returncode == 0, (r.stdout + r.stderr)[-800:]
    except Exception as e:  # noqa: BLE001
        return False, f"{type(e).__name__}: {e}"


def _torch_cuda_ok():
    try:
        import torch; return bool(torch.cuda.is_available())
    except Exception:  # noqa: BLE001
        return None


def verify_gpu(run_matmul=True):
    """Standalone 5090 GPU-STABILITY guard — run this after ANY install (not just this agent's) to catch a
    torch/CUDA clobber like the one a stray `pip install` can cause (e.g. a dep pulling torch+cu130, which is
    binary-incompatible with the CUDA 12.8 driver → cuda.is_available() False). Checks: torch imports, build is
    +cu128 (NOT cu130), cuda available, capability is sm_120 (RTX 5090), and a real GPU matmul runs.
    Returns {ok, torch, build_cu128, cuda, capability, matmul_ok, reason}."""
    r = {"ok": False, "torch": None, "build_cu128": None, "cuda": None, "capability": None,
         "matmul_ok": None, "reason": ""}
    try:
        import torch
    except Exception as e:  # noqa: BLE001
        r["reason"] = f"torch import failed: {e}"; return r
    r["torch"] = torch.__version__
    r["build_cu128"] = "cu128" in torch.__version__
    r["cuda"] = bool(torch.cuda.is_available())
    if not r["cuda"]:
        r["reason"] = ("cuda unavailable — likely a wrong torch build (need +cu128 for the 5090/sm_120; "
                       "cu130 is incompatible with the 12.8 driver). Run repair_torch().")
        return r
    try:
        r["capability"] = tuple(torch.cuda.get_device_capability(0))
    except Exception:  # noqa: BLE001
        pass
    if run_matmul:
        try:
            x = torch.randn(512, 512, device="cuda"); float((x @ x).sum().item()); r["matmul_ok"] = True
        except Exception as e:  # noqa: BLE001
            r["matmul_ok"] = False; r["reason"] = f"GPU matmul failed: {e}"; return r
    r["ok"] = r["cuda"] and (r["build_cu128"] is not False) and (r["matmul_ok"] is not False)
    r["reason"] = "5090 GPU stable (cu128, cuda available, matmul OK)" if r["ok"] else "GPU degraded"
    return r


def repair_torch(dry_run=True):
    """Restore torch/vision/audio to the +cu128 build (the 5090's stable ABI). dry_run only prints the command."""
    cmd = [sys.executable, "-m", "pip", "install", "--force-reinstall",
           "torch", "torchvision", "torchaudio", "--index-url", "https://download.pytorch.org/whl/cu128"]
    if dry_run:
        return {"dry_run": True, "cmd": " ".join(cmd)}
    ok, log = _pip(["torch", "torchvision", "torchaudio", "--force-reinstall",
                    "--index-url", "https://download.pytorch.org/whl/cu128"], timeout=3600)
    return {"dry_run": False, "ok": ok and bool(_torch_cuda_ok()), "log": log[-300:]}


def install(agents, dry_run=True, upgrade=False):
    """Install the missing gated agents' deps SAFELY. dry_run (default) only plans. ABI-sensitive installs pin
    numpy and re-verify torch CUDA after, repairing torch (+cu128) if a dep broke it. Returns a per-agent report."""
    import numpy as np  # noqa: F401 (ensures numpy importable before we touch the env)
    rep = {}
    torch_before = _torch_cuda_ok()
    for a in agents:
        req = REQUIRES.get(a)
        if not req:
            rep[a] = {"status": "skip", "note": "no known deps"}; continue
        if _present(req["probe"]):
            rep[a] = {"status": "already-ready"}; continue
        specs = req["pip"]
        if dry_run:
            rep[a] = {"status": "would-install", "pip": specs, "abi": req["abi"],
                      "cmd": f"pip install {'--upgrade ' if upgrade else ''}{' '.join(specs)}"
                             + (f'  (with constraint {NUMPY_PIN})' if req["abi"] else "")}
            continue
        args = ["install"] + (["--upgrade"] if upgrade else []) + specs
        if req["abi"]:                              # keep numpy anchored so the heavy dep can't move the ABI
            args += ["--constraint", "/dev/stdin"] if False else []
            ok, log = _pip(["install", NUMPY_PIN, *specs])   # pin numpy alongside so pip can't downgrade it
        else:
            ok, log = _pip(args)
        ready = _present(req["probe"])
        entry = {"status": "installed" if (ok and ready) else "failed", "pip": specs, "log_tail": log[-300:], "ready": ready}
        if req["abi"]:                              # a heavy dep may have pulled a wrong torch → verify + repair
            tc = _torch_cuda_ok()
            entry["torch_cuda_after"] = tc
            if torch_before and tc is False:
                rok, rlog = _pip(TORCH_REPAIR + ["--force-reinstall", "--no-deps"])
                entry["torch_repaired"] = rok and _torch_cuda_ok(); entry["repair_log"] = rlog[-200:]
        rep[a] = entry
    return rep


class SetupEnv(BaseAgent):
    name = "setup-env"
    thread = "S"
    kind = "verdict"

    def run(self, q, worker):
        spec = self.spec(q)
        # gpu-guard mode: verify (and optionally repair) the 5090 cu128 stability after any install
        if spec.get("mode") == "gpu-guard" or spec.get("gpu_guard"):
            v = verify_gpu()
            if not v["ok"] and spec.get("repair"):
                v["repair"] = repair_torch(dry_run=not spec.get("apply_repair"))
                if spec.get("apply_repair"):
                    v = {**verify_gpu(), "repair": v["repair"]}
            msg = (f"[{worker}] **SETUP-ENV gpu-guard**: torch={v['torch']} cu128={v['build_cu128']} "
                   f"cuda={v['cuda']} cap={v['capability']} matmul={v['matmul_ok']} → "
                   f"{'✅ STABLE' if v['ok'] else '❌ DEGRADED: ' + v['reason']}"
                   + (f"\nrepair: {v.get('repair')}" if v.get('repair') else ""))
            self.log(summary=f"gpu-guard: {'stable' if v['ok'] else 'DEGRADED'} ({v['reason']})",
                     kind="verdict", recommendation="run repair_torch() / setup-env mode=gpu-guard repair=true "
                                                     "apply_repair=true if cuda broke after an install")
            self.post(worker, "leader", msg, routine=False, kind="verdict")
            return self.done(v, msg, to="leader")
        agents = spec.get("agents") or list(REQUIRES)
        if isinstance(agents, str):
            agents = [agents]
        do_install = bool(spec.get("install", False))
        dry = not do_install
        chk = check(agents)
        missing = [a for a, v in chk.items() if v.get("ready") is False]
        rep = install(missing, dry_run=dry, upgrade=bool(spec.get("upgrade", False))) if missing else {}
        self.save_state({"check": chk, "install": rep, "dry_run": dry})
        ready = [a for a, v in chk.items() if v.get("ready") is True]
        rows = "\n".join(
            f"| {a} | {'✅ ready' if chk[a].get('ready') else '❌ missing' if chk[a].get('ready') is False else '—'} | "
            f"{', '.join(chk[a].get('pip', [])) or '—'} | {rep.get(a, {}).get('status', '')} |"
            for a in agents)
        mode = "INSTALLED" if do_install else "DRY-RUN (pass install=true to apply)"
        msg = (f"[{worker}] **SETUP-ENV** ({mode}) · lib-gated agent dependencies\n"
               f"| agent | state | pip | action |\n|:-|:-|:-|:-|\n{rows}\n"
               f"→ ready: {len(ready)}/{len(agents)}; missing: {missing or 'none'}")
        self.log(summary=f"setup-env: {len(ready)}/{len(agents)} ready, missing={missing}, dry_run={dry}",
                 detail="safe installer (numpy 2.4.6 pinned, torch cu128 re-verified for ABI-heavy deps)",
                 kind="verdict", recommendation="run with install=true to resolve the missing gated agents; ABI-safe")
        self.post(worker, "leader", msg, routine=False, kind="verdict")
        return self.done({"check": chk, "install": rep, "ready": ready, "missing": missing}, msg, to="leader")


_AGENT = SetupEnv()


def run(q, worker):
    return _AGENT.run(q, worker)
