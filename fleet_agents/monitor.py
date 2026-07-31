"""train-monitor — LIVE watchdog over a running training (GPU/CPU + log freshness).

The 32-min silent hang (dataloader deadlock at epoch 0, GPU idle, log frozen) proved we cannot
trust a training just because :7799 says 'running'. This agent samples, each fleet cycle:
  * GPU utilisation + memory (nvidia-smi)
  * the job's log staleness (seconds since the train log last advanced)
  * the training process %CPU
and declares a HANG when the log is frozen AND the GPU is idle for too long → kills the process
(the train service can't cancel a *running* job), reconciles the job to 'failed', and escalates to
the leader. Healthy samples post routine progress (hidden from the leader unless 'show all').

Generic: reads the running job + its log_path straight from the :7799 board, so it works for any
competition's training as long as jobs go through the shared train service.
"""
from __future__ import annotations

import datetime
import json
import re
import subprocess
import time
import urllib.request
from pathlib import Path

COMP = Path(__file__).resolve().parent.parent
TRAIN_SERVICE = "http://127.0.0.1:7799"
JOBS_PATH = COMP / "tools" / "researchpapers" / ".research-mvp-data" / "train-service" / "jobs.json"
STATE_PATH = COMP / "tools" / "researchpapers" / ".research-mvp-data" / "runtime" / ".monitor_state.json"

MAX_RUN_MIN = 30   # fast-loop cap: a job over this long is blocking the queue → skip it (unless notes say 'no cap')
STALL_S = 240      # log frozen this long ...
GPU_IDLE = 5       # ... AND GPU under this % util  → HANG
SAMPLE_S = 6       # GPU sampling window


def _elapsed(job) -> tuple[float | None, str]:
    """Seconds + human string since the job started (so the user sees TIME SPENT live)."""
    ts = job.get("started_at") or job.get("created_at")
    if not ts:
        return None, "?"
    try:
        started = datetime.datetime.fromisoformat(ts)
        now = datetime.datetime.now(started.tzinfo)
        s = (now - started).total_seconds()
        h, m, sec = int(s // 3600), int((s % 3600) // 60), int(s % 60)
        return s, (f"{h}h {m}m" if h else f"{m}m {sec}s")
    except Exception:  # noqa: BLE001
        return None, "?"


_ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


def _clean(s: str) -> str:
    """Strip ANSI escapes + tqdm bar glyphs so the chat line is readable (no raw \\x1b[A or |███| bars)."""
    s = _ANSI.sub("", s or "").replace("\r", "")
    s = re.sub(r"[|│][█▉▊▋▌▍▎▏ #>=-]*[|│]?", "", s)  # drop the bar itself, keep the numbers
    return re.sub(r"\s{2,}", " ", s).strip()


def _phase_progress(text: str) -> tuple[str, str]:
    """(phase, clean human progress) from the trainer's log tail — epoch + iter + ETA, no raw bars/ANSI."""
    if not text:
        return "starting", ""
    tail = _ANSI.sub("", text[-6000:]).replace("\r", "\n")
    epoch = ""
    for m in re.finditer(r"Training:\s*(\d+)%[^\d]*(\d+)/(\d+)", tail):
        epoch = f"epoch {m.group(2)}/{m.group(3)}"
    # innermost iteration bar: "iters: 4%| | 6/150 [00:10<03:35, 1.49s/it]"
    it = None
    for it in re.finditer(r"(\d+)/(\d+)\s*\[[\d:]+<([\d:]+),\s*([\d.]+)(s/it|it/s)\]", tail):
        pass
    if it:
        n, tot, eta, rate, unit = it.groups()
        piece = f"iter {n}/{tot} · ETA {eta} · {rate}{unit}"
        return "training", (f"{epoch} · {piece}" if epoch else piece)
    if epoch:
        return "training", epoch
    if "Starting training" in tail:
        return "training", "epoch 0"
    mc = ""
    for m in re.finditer(r"(train|test):\s*(\d+)%", tail):
        mc = f"caching {m.group(1)} {m.group(2)}%"
    return ("caching", mc) if mc else ("running", "")


def _running_job():
    try:
        with urllib.request.urlopen(f"{TRAIN_SERVICE}/api/board", timeout=4) as r:
            for j in json.loads(r.read()).get("queue", {}).get("recent", []):
                if j.get("status") == "running":
                    return j
    except Exception:  # noqa: BLE001
        pass
    return None


def _gpu():
    """(util%, mem_MiB) — max over a few quick single-shot samples; (None, None) if nvidia-smi absent."""
    utils, mems = [], []
    for _ in range(3):
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5).stdout.strip()
            row = [r for r in out.splitlines() if "," in r]
            if row:
                u, m = row[0].split(",")
                utils.append(int(u)); mems.append(int(m))
        except Exception:  # noqa: BLE001
            pass
        time.sleep(1)
    if not utils:
        return None, None
    return max(utils), max(mems)   # max over samples: bursty compute still shows up


def _log_age(job) -> float | None:
    lp = job.get("log_path")
    if lp and Path(lp).exists():
        return time.time() - Path(lp).stat().st_mtime
    return None


def _last_line(job) -> str:
    lp = job.get("log_path")
    if not (lp and Path(lp).exists()):
        return ""
    try:
        return [ln for ln in Path(lp).read_text(errors="replace").splitlines() if ln.strip()][-1][:120]
    except Exception:  # noqa: BLE001
        return ""


MLFLOW_API = "http://127.0.0.1:5000/api/2.0/mlflow"
MLFLOW_UI = "http://gpu:5000"   # user-facing (Tailscale) — the live, auto-updating run view


def _method_from_job(job):
    for a in job.get("script_args", []):
        if str(a).endswith((".yml", ".yaml")):
            try:
                import yaml
                c = yaml.safe_load((COMP / a).read_text()) or {}
                return (c.get("train") or {}).get("method") or c.get("name")
            except Exception:  # noqa: BLE001
                return None
    return None


def _mlflow_url(method):
    """Deep-link to THIS run's live MLflow page (metrics update per-epoch while the run is going)."""
    if not method:
        return None
    try:
        exp = json.loads(urllib.request.urlopen(
            f"{MLFLOW_API}/experiments/get-by-name?experiment_name=kaggle-biohub-loeo", timeout=4).read())
        eid = exp["experiment"]["experiment_id"]
        body = json.dumps({"experiment_ids": [eid], "filter": f"tags.`mlflow.runName` = '{method}'",
                           "max_results": 1, "order_by": ["attributes.start_time DESC"]}).encode()
        req = urllib.request.Request(f"{MLFLOW_API}/runs/search", data=body,
                                     headers={"Content-Type": "application/json"})
        runs = json.loads(urllib.request.urlopen(req, timeout=4).read()).get("runs", [])
        if runs:
            return f"{MLFLOW_UI}/#/experiments/{eid}/runs/{runs[0]['info']['run_id']}"
    except Exception:  # noqa: BLE001
        pass
    return None


def _train_pids():
    """Python PIDs running the config-driven trainer (comm=python so we never match a shell)."""
    try:
        out = subprocess.run(["ps", "-eo", "pid,comm,args"], capture_output=True, text=True, timeout=6).stdout
    except Exception:  # noqa: BLE001
        return []
    pids = []
    for ln in out.splitlines():
        p = ln.split(None, 2)
        if len(p) == 3 and "python" in p[1] and ("train_from_config" in p[2] or "train_unet_transformer" in p[2]):
            pids.append(int(p[0]))
    return pids


def _kill(pids):
    for sig in ("-TERM", "-KILL"):
        for p in pids:
            try:
                subprocess.run(["kill", sig, str(p)], capture_output=True, timeout=10)
            except subprocess.TimeoutExpired:   # a wedged fork must not stall the watchdog itself
                pass
        time.sleep(2)


def _reconcile_failed(job_id: str, why: str):
    """The train service can't cancel a running job — mark the hung one failed so the queue frees."""
    try:
        d = json.loads(JOBS_PATH.read_text())
        now = __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat()
        for j in d.get("jobs", []):
            if isinstance(j, dict) and j.get("id") == job_id:
                j["status"], j["finished_at"], j["exit_code"] = "failed", now, 137
                j["error"] = (j.get("error") or "") + f" [train-monitor: {why}]"
                j.setdefault("events", []).append({"timestamp": now, "message": f"killed by train-monitor: {why}"})
        d["updated_at"] = now
        JOBS_PATH.write_text(json.dumps(d, indent=2))
    except Exception:  # noqa: BLE001
        pass


def watch(q, worker):
    """One health sample of the running training. HANG → kill+escalate; healthy → routine progress."""
    job = _running_job()
    if not job:
        return ("done", {"idle": True}, "all", f"[{worker}] no training running — nothing to watch.")
    util, mem = _gpu()
    age = _log_age(job)
    last = _last_line(job)
    jid, title = job.get("id"), (job.get("title") or "")[:40]
    el_s, _el_h = _elapsed(job)
    # RUNTIME CAP: a fast-loop job (screen train ≤10ep, ≤12-FOV score) should finish quick; if it runs way
    # over MAX_RUN_MIN it is blocking the queue (e.g. a 71-FOV score) → treat like a hang and skip to the next.
    over = el_s is not None and el_s > MAX_RUN_MIN * 60 and "no cap" not in (job.get("notes") or "")
    hung = ((age is not None and age > STALL_S) and (util is not None and util < GPU_IDLE)) or over
    if hung:
        pids = _train_pids()
        _kill(pids)
        why = (f"over runtime cap {MAX_RUN_MIN}m ({int(el_s / 60)}m) — blocking the fast loop"
               if over else f"log frozen {int(age)}s + GPU {util}% idle at: {last!r}")
        _reconcile_failed(jid, why)
        # tell the journal this experiment died
        try:
            from . import ledger
            for e in ledger.entries():
                if jid in (e.get("observation") or "") or title.split()[0] in (e.get("change") or ""):
                    ledger.set_scores(e["exp"], cv=("skipped" if over else "hang"), observation=f"killed by monitor: {why}")
                    break
        except Exception:  # noqa: BLE001
            pass
        tag = "⏭️ SKIPPED (runtime cap)" if over else "🛑 HANG DETECTED"
        return ("escalated", {"job": jid, "over_cap": over, "gpu": util, "killed": pids}, "leader",
                f"[{worker}] {tag} on {jid} ({title}): {why}. Killed {len(pids)} pid(s), GPU freed → next experiment. "
                + ("" if over else "Likely a num_workers dataloader deadlock — retry with num_workers=0 or a smoke gate."))
    # HEALTHY → post LIVE progress (visible) with elapsed time + phase, deduped so identical samples don't spam
    elapsed_s, elapsed_h = _elapsed(job)
    lp = job.get("log_path")
    text = Path(lp).read_text(errors="replace") if (lp and Path(lp).exists()) else ""
    phase, prog = _phase_progress(text)
    method = _method_from_job(job)
    mlflow_url = _mlflow_url(method)
    gpu_pct = f"{util}%" if util is not None else "—"
    gpu_gb = f"{mem / 1024:.1f} GB" if mem is not None else "—"
    data = {"job": jid, "title": title, "method": method, "phase": phase, "progress": prog,
            "elapsed_s": elapsed_s, "elapsed": elapsed_h, "gpu": util, "mem": mem,
            "mlflow_url": mlflow_url, "log_age": age, "healthy": True}
    # clean, professional ONE-LINE card (no raw tqdm bars / ANSI) with the live MLflow link
    head = f"🔬 {title or method or jid}"
    body = f"{phase}" + (f" · {prog}" if prog else "")
    tail = f"⏱ {elapsed_h} · 🖥 GPU {gpu_pct} · {gpu_gb}"
    link = f" · 📊 live: {mlflow_url}" if mlflow_url else ""
    msg = f"[{worker}] {head} — {body} · {tail}{link}"
    sig = f"{jid}|{phase}|{prog}"  # only re-post when the phase/progress actually advances
    try:
        st = json.loads(STATE_PATH.read_text()) if STATE_PATH.exists() else {}
    except Exception:  # noqa: BLE001
        st = {}
    if st.get("sig") == sig:
        # unchanged since last cycle → keep it routine (still live in `data`, just not a new visible line)
        return ("done", {**data, "unchanged": True}, "all", f"[{worker}] training healthy: {jid} — {phase} {prog}")
    st["sig"] = sig
    try:
        STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        STATE_PATH.write_text(json.dumps(st))
    except Exception:  # noqa: BLE001
        pass
    # status 'done' so board.reopen('train-monitor') re-samples next cycle; msg doesn't match the routine
    # regex → posted VISIBLE. The rich `data` payload lets the board render a live progress card + MLflow link.
    return ("done", data, "all", msg)
