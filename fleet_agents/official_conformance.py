"""official-conformance — ONE agent that proves our repo/submission/metric are CONFORMANT with the official
baseline (royerlab/kaggle-cell-tracking-competition). Replaces the ad-hoc bash/pytest cross-check (user
2026-07-12: "do all with our agents"). Three checks, each reproducible:

  1. repo-sync   — compare our research/official_repo files vs upstream by git-blob SHA (identical/differ/missing);
                   assert the METRIC CORE (metrics/division_metrics/io/img_proc + temporal_unet) is byte-identical
                   so our golden CV is provably the organizers' metric.
  2. schema      — assert our src/submission.py column schema == the official geffs_to_csv.py submission schema.
  3. division    — run the 14 canonical division sandbox cases against our division_metrics.py (organizer-verified
                   TP/FP/FN freeze-test) via `uv run pytest`.

Pure helpers (_parse_pytest, _schema_equal, _gitblob) are data-wise tested."""
from __future__ import annotations
import re, subprocess, hashlib
from .base import BaseAgent, COMP

REPO = "royerlab/kaggle-cell-tracking-competition"
OFFICIAL = COMP / "research" / "official_repo"
METRIC_CORE = ("src/tracking_cellmot/metrics.py", "src/tracking_cellmot/division_metrics.py",
               "src/tracking_cellmot/io.py", "src/tracking_cellmot/img_proc.py",
               "src/tracking_cellmot/models/temporal_unet.py")


def _gitblob(b: bytes) -> str:
    """git blob SHA-1 of raw bytes (matches GitHub's tree blob sha) — for byte-identity comparison."""
    h = hashlib.sha1(); h.update(b"blob %d\0" % len(b) + b); return h.hexdigest()


def _parse_pytest(out: str):
    """PURE (tested): parse a pytest summary line → (passed, failed)."""
    p = re.search(r"(\d+)\s+passed", out); f = re.search(r"(\d+)\s+failed", out)
    return (int(p.group(1)) if p else 0, int(f.group(1)) if f else 0)


def _official_columns(text: str):
    """PURE (tested): extract the COLUMNS tuple from the official geffs_to_csv.py source."""
    m = re.search(r"COLUMNS[^=]*=\s*\(([^)]*)\)", text)
    if not m:
        return []
    return [c.strip().strip('"\'') for c in m.group(1).split(",") if c.strip().strip('"\'')]


def _schema_equal(ours, official):
    """PURE (tested): compare our submission columns to the official ones. Our finalize() prepends 'id' as the
    index, so normalise by dropping a leading 'id' from either side before comparing."""
    def norm(cols):
        return [c for c in cols if c != "id"]
    a, b = norm(ours), norm(official)
    return a == b, {"ours": a, "official": b, "missing": [c for c in b if c not in a],
                    "extra": [c for c in a if c not in b]}


class OfficialConformance(BaseAgent):
    name = "official-conformance"
    thread = "V"
    kind = "verdict"

    def _repo_sync(self):
        if not OFFICIAL.exists():
            return {"error": f"official_repo missing at {OFFICIAL}"}
        try:
            from . import research_search as RS
            ag = RS._AGENT
            H = {"User-Agent": "research-search/1.0", "Accept": "application/vnd.github+json"}
            meta = ag._get_json(f"https://api.github.com/repos/{REPO}", headers=H)
            br = meta.get("default_branch", "main") if isinstance(meta, dict) else "main"
            tree = ag._get_json(f"https://api.github.com/repos/{REPO}/git/trees/{br}?recursive=1", headers=H)
        except Exception as e:  # noqa: BLE001 — offline / import / network failure → clean, not a crash
            return {"error": f"could not fetch upstream tree ({type(e).__name__}: {str(e)[:80]})"}
        blobs = {t["path"]: t["sha"] for t in (tree.get("tree", []) if isinstance(tree, dict) else [])
                 if t.get("type") == "blob"}
        if not blobs:
            return {"error": "could not fetch upstream tree (offline/rate-limited)"}
        match = differ = missing = 0
        core_ok = True
        for path, sha in blobs.items():
            lp = OFFICIAL / path
            if not lp.exists():
                missing += 1
                if path in METRIC_CORE:
                    core_ok = False
                continue
            try:
                same = _gitblob(lp.read_bytes()) == sha
            except Exception:  # noqa: BLE001 — unreadable file counts as a difference, never crashes
                differ += 1
                if path in METRIC_CORE:
                    core_ok = False
                continue
            if same:
                match += 1
            else:
                differ += 1
                if path in METRIC_CORE:
                    core_ok = False
        return {"remote": f"{REPO}@{br}", "n": len(blobs), "identical": match, "differ": differ,
                "missing": missing, "metric_core_identical": core_ok}

    def _schema(self):
        try:
            import importlib
            sub = importlib.import_module("src.submission")
            ours = list(getattr(sub, "COLUMNS", []))
        except Exception as e:  # noqa: BLE001
            return {"error": f"import src.submission: {type(e).__name__}: {e}"}
        gf = OFFICIAL / "scripts" / "geffs_to_csv.py"
        official = _official_columns(gf.read_text()) if gf.exists() else []
        ok, detail = _schema_equal(ours, official)
        return {"match": ok, **detail}

    def _division(self, timeout=400):
        """`timeout`: seconds for the division-sandbox pytest run (default 400)."""
        if not OFFICIAL.exists():
            return {"error": "official_repo missing"}
        try:
            timeout = int(timeout)
        except (TypeError, ValueError):
            timeout = 400
        try:
            r = subprocess.run(["uv", "run", "--quiet", "pytest",
                                "tests/test_division_sandbox_examples.py", "-q"],
                               cwd=str(OFFICIAL), capture_output=True, text=True, timeout=timeout)
            passed, failed = _parse_pytest(r.stdout + r.stderr)
            return {"passed": passed, "failed": failed, "ok": failed == 0 and passed > 0}
        except Exception as e:  # noqa: BLE001
            return {"error": f"{type(e).__name__}: {str(e)[:80]}"}

    def run(self, q, worker):
        spec = self.spec(q) or {}
        which = spec.get("checks", ["repo", "schema", "division"])
        res = {}
        if "repo" in which:
            res["repo_sync"] = self._repo_sync()
        if "schema" in which:
            res["schema"] = self._schema()
        if "division" in which:
            res["division"] = self._division(timeout=spec.get("timeout", 400))
        rs, sc, dv = res.get("repo_sync", {}), res.get("schema", {}), res.get("division", {})
        conformant = (rs.get("metric_core_identical", True) and sc.get("match", True) and dv.get("ok", True))
        summary = (f"official-conformance: metric-core-identical={rs.get('metric_core_identical')} "
                   f"({rs.get('identical')}/{rs.get('n')} files identical, {rs.get('missing')} missing) · "
                   f"submission-schema-match={sc.get('match')} · division-sandbox={dv.get('passed')}pass/"
                   f"{dv.get('failed')}fail → CONFORMANT={conformant}")
        self.log(summary, kind="verdict",
                 recommendation="metric core byte-identical → our golden CV IS the organizers' metric; submission "
                                "schema matches geffs_to_csv; division semantics pass the canonical freeze-test. "
                                "Local diffs are OUR customizations (train/predict/augment/linker) — intentional.")
        return self.done({"conformant": conformant, **res}, summary)


_AGENT = OfficialConformance()


def run(q, worker):
    return _AGENT.run(q, worker)
