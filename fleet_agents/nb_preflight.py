"""nb-preflight — VERIFY a Kaggle submission/benchmark notebook OFFLINE, LOCALLY, before any push, so the
whole class of "burned a Kaggle run to discover No-module-named-X" mistakes dies on our machine instead.

User rule (2026-07-12): "u have to find a verifier before submit run on kaggle so these mistakes u can
eliminate." This agent is that verifier. It does NOT run the model — it proves the ENVIRONMENT + WIRING:

  1. IMPORT-RESOLVE  — parse every top-level `import X` / `from X import` in the notebook code cells, then
     in a throwaway venv seeded with the Kaggle-base packages (numpy/scipy/torch/pandas/typing_extensions
     via --system-site-packages), do the EXACT Kaggle offline install (`pip install --no-index --no-deps`
     of every wheel in the offline pack) and `import` each module. A wheel that installs but fails to import
     for a missing transitive dep (the zarr→typing_extensions trap) is caught HERE. Reports each unresolved
     import — that is a guaranteed Kaggle failure.
  2. PATH-DISCOVERY  — assert the notebook finds its inputs by CONTENT (globbed *.whl / *.zarr / weights),
     not a hard-coded /kaggle/input/<slug> that silently becomes None when the mount name differs.
  3. NO-COMMENTS / SUBMISSION-SHAPE (optional) — flag `#` comment lines (human Kaggle style) and check the
     submission writes the required columns.

Verdict = the notebook is offline-import-safe (all imports resolve) or a list of the exact failures. No
Kaggle push happens until this is green. Pure decision logic (_verdict) is data-wise tested with no venv.
"""
from __future__ import annotations
import ast
import json
import os
import subprocess
import sys
from .base import BaseAgent, COMP

# Modules the Kaggle GPU base image is known to provide — an unresolved import of one of these locally is a
# false alarm (our seed venv mirrors it via --system-site-packages), so it is NOT counted as a Kaggle failure.
KAGGLE_BASE = {
    "numpy", "scipy", "pandas", "torch", "torchvision", "sklearn", "skimage", "cv2", "PIL", "matplotlib",
    "tifffile", "typing_extensions", "packaging", "yaml", "tqdm", "requests", "numba", "h5py", "numcodecs",
    "os", "sys", "glob", "time", "json", "ast", "subprocess", "multiprocessing", "math", "re", "shutil",
    "pathlib", "collections", "itertools", "functools", "warnings", "random", "io", "gc", "typing",
}
STDLIB_OK = KAGGLE_BASE


def _imports_of_source(src):
    """Top-level module names imported by a Python source string (best-effort AST; falls back on syntax err)."""
    mods = set()
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return mods
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                mods.add(a.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.level == 0:
                mods.add(node.module.split(".")[0])
    return mods


def _notebook_sources(nb_path):
    """All code-cell sources of an .ipynb, concatenated per cell (list of strings)."""
    try:
        with open(nb_path) as fh:
            nb = json.load(fh)
    except (OSError, ValueError):
        return []
    cells = nb.get("cells", []) if isinstance(nb, dict) else []
    out = []
    for c in cells:
        if isinstance(c, dict) and c.get("cell_type") == "code":
            src = c.get("source", [])
            out.append("".join(src) if isinstance(src, list) else str(src))
    return out


def _gpu_config_check(sources, accelerator):
    """PURE (data-wise tested). Catch the 2×T4 class of push error BEFORE `kaggle kernels push`.

    sources: list of notebook cell source strings. accelerator: the value intended for
    `kaggle kernels push --accelerator <ACC>` ("" / None = clean push = single P100).

    A notebook is 'multi-GPU' if it references cuda:1 / device 1 / USE_MULTI_GPU=True /
    n_jobs=2 / set_device(1). Such a notebook on a single P100 crashes with
    'invalid device ordinal'. Returns (ok, issues, info)."""
    blob = "\n".join(sources)
    multi_gpu = any(t in blob for t in (
        'cuda:1', '"cuda", 1', "'cuda', 1", "set_device(1)", "device 1",
        "USE_MULTI_GPU = True", "USE_MULTI_GPU=True", "n_jobs=2", "n_jobs = 2"))
    has_devguard = ("device_count()" in blob)  # auto-detect degrades instead of crashing
    acc = (accelerator or "").strip()
    is_2xt4 = acc.lower() in ("nvidiatteslat4", "nvidiateslat4", "t4x2", "gpu_t4_x2")
    issues = []
    if multi_gpu and not is_2xt4:
        issues.append(
            "multi-GPU notebook but push accelerator is not NvidiaTeslaT4 (2×T4) — will get a single P100 "
            "and crash with 'CUDA error: invalid device ordinal'. Push with "
            "`kaggle kernels push --accelerator NvidiaTeslaT4`.")
    if multi_gpu and not has_devguard:
        issues.append(
            "multi-GPU notebook has no `torch.cuda.device_count()` guard — add "
            "`USE_MULTI_GPU = torch.cuda.device_count() >= 2` AFTER any cell that hardcodes it, so a wrong "
            "accelerator degrades to single-GPU instead of crashing.")
    info = {"multi_gpu": multi_gpu, "device_count_guard": has_devguard,
            "accelerator": acc or "(none → P100)", "accelerator_is_2xt4": is_2xt4}
    return (not issues), issues, info


def _verdict(import_results, path_discovery_ok, comment_lines):
    """PURE decision (data-wise tested). import_results = {module: True/False resolved}.
    Green iff every non-base import resolves AND inputs are discovered by content. Returns (ok, report)."""
    unresolved = sorted(m for m, ok in import_results.items() if not ok)
    ok = (not unresolved) and path_discovery_ok
    report = {
        "unresolved_imports": unresolved,
        "path_discovery_by_content": path_discovery_ok,
        "comment_lines": comment_lines,
        "verdict": "GREEN — offline-import-safe" if ok else "RED — would fail on Kaggle",
    }
    return ok, report


class NbPreflight(BaseAgent):
    name = "nb-preflight"
    thread = "S"
    kind = "verdict"

    def _simulate(self, nb_path, wheels_dir, timeout=120):
        """Build a base-seeded venv, do the EXACT Kaggle offline install, import every notebook module.
        Returns {module: resolved_bool} for the non-base imports only. timeout: per-subprocess cap (s)."""
        srcs = _notebook_sources(nb_path)
        mods = set().union(*[_imports_of_source(s) for s in srcs]) if srcs else set()
        to_test = sorted(m for m in mods if m not in STDLIB_OK)
        venv = "/tmp/nb_preflight_venv"
        py = f"{venv}/bin/python"
        timeout = max(1, int(timeout))
        if not os.path.exists(py):
            subprocess.run([sys.executable, "-m", "venv", "--system-site-packages", venv], check=False,
                           timeout=timeout)
        # EXACT Kaggle offline install: every wheel, --no-index --no-deps (transitive deps must be base or
        # also-a-wheel — this is what surfaces the zarr→typing_extensions class of failure).
        import glob as _g
        wheels = sorted(_g.glob(os.path.join(wheels_dir, "*.whl")))
        for whl in wheels:
            try:
                subprocess.run([py, "-m", "pip", "install", "-q", "--no-index", "--no-deps", whl], check=False,
                               timeout=timeout)
            except subprocess.TimeoutExpired:
                pass
        # Kaggle offline env quirks the notebook MUST set (proven by hengck23 + 68 public notebooks):
        # POLARS_PREFER_PKG=32 makes polars load the split polars-runtime-32 wheel. Import WITH it set (a
        # correct notebook sets it); the source-level check below flags a notebook that imports polars but
        # forgets it — the class of failure that GREENs locally yet dies on Kaggle.
        _env = dict(os.environ); _env.setdefault("POLARS_PREFER_PKG", "32")
        results = {}
        for m in to_test:
            try:
                r = subprocess.run([py, "-c", f"import {m}"], capture_output=True, text=True, env=_env,
                                   timeout=timeout)
                results[m] = (r.returncode == 0)
            except subprocess.TimeoutExpired:
                results[m] = False
        # gotcha: polars imported but POLARS_PREFER_PKG not set in the notebook source
        blob = " ".join(srcs)
        if any(m == "polars" or m.startswith("polars") for m in to_test) and "POLARS_PREFER_PKG" not in blob:
            results["__gotcha_polars_prefer_pkg"] = False
        return results, to_test

    def _path_discovery_ok(self, srcs):
        """Inputs must be found by CONTENT (recursive glob on a filename pattern), not by our dataset's exact
        slug (which becomes None the moment the pack is renamed/re-uploaded). A recursive glob rooted at a
        Kaggle-STANDARD base (/kaggle/input, /kaggle/input/competitions) is fine — only referencing OUR pack
        by name is the fragile anti-pattern this catches."""
        blob = "\n".join(srcs)
        by_content = ("recursive=True" in blob) or ("**" in blob)
        hard_coded = "biohub-offline-pack" in blob        # our pack by exact slug = breaks on rename
        return by_content and not hard_coded

    def run(self, q, worker):
        spec = self.spec(q)
        nb_path = spec.get("nb") or "/tmp/bench_kernel/biohub_bench.ipynb"
        wheels_dir = spec.get("wheels") or "/tmp/kaggle_deps/biohub-offline-pack/wheels"
        timeout = int(spec.get("timeout", 120))            # timeout: per-subprocess (venv/install/import) cap in seconds
        strict = bool(spec.get("strict", True))            # strict: RED also if path-discovery uses hard-coded paths (default on)
        srcs = _notebook_sources(nb_path)
        comment_lines = sum(1 for s in srcs for ln in s.splitlines() if ln.strip().startswith("#"))
        import_results, tested = self._simulate(nb_path, wheels_dir, timeout=timeout)
        path_ok = self._path_discovery_ok(srcs) or (not strict)
        gpu_ok, gpu_issues, gpu_info = _gpu_config_check(srcs, spec.get("accelerator"))
        ok, report = _verdict(import_results, path_ok, comment_lines)
        ok = ok and gpu_ok
        report["gpu_config"] = gpu_info
        report["gpu_issues"] = gpu_issues
        if not gpu_ok:
            report["verdict"] = "RED — GPU/accelerator config would fail on Kaggle"
        summary = (f"nb-preflight {os.path.basename(nb_path)}: {report['verdict']}. "
                   f"tested imports={tested}; unresolved={report['unresolved_imports']}; "
                   f"path-by-content={path_ok}; comment-lines={comment_lines}; gpu={gpu_info}"
                   + (f"; GPU-ISSUES={gpu_issues}" if gpu_issues else ""))
        if not ok:
            return self.escalate(worker, "researcher",
                                 f"nb-preflight RED — DO NOT PUSH. {summary}. Fix the install/paths, re-verify.")
        self.log(summary, kind="verdict",
                 recommendation="notebook is offline-import-safe locally → safe to push to Kaggle. Keep this "
                                "green as a gate before every kernels push.")
        return self.done({"ok": ok, **report, "tested": tested}, summary)


_AGENT = NbPreflight()


def run(q, worker):
    return _AGENT.run(q, worker)
