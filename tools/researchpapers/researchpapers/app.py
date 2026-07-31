from __future__ import annotations

import asyncio
import json
import logging
import subprocess
import threading
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from researchpapers.models import (
    ConsoleInputRequest,
    ArtifactCreate,
    MessageCreate,
    ProjectCreate,
    ReviewRequest,
    TaskCompleteRequest,
    now_iso,
)
from researchpapers.logging_utils import configure_logging
from researchpapers import comp_env as _ce
from researchpapers.runtime_cli import (
    DEFAULT_CONFIG_PATH,
    ensure_runtime_layout,
    inbox_files,
    load_config,
    normalize_tmux_key,
    queue_message,
    read_json,
    refresh_status,
    runtime_dirs,
    sorted_thread_messages,
    summarize_tasks,
)
from researchpapers.store import ProjectStore

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
store = ProjectStore()
logger = configure_logging(BASE_DIR.parent / ".research-mvp-data" / "logs")


class RuntimeBoardMessageCreate(BaseModel):
    sender: str = "human"
    recipient: str
    content: str


class FleetExperimentCreate(BaseModel):
    config: str                      # the config/*.yml the researcher authored
    description: str = ""            # WHAT WE DID (leader-supplied) — becomes the journal DESCRIPTION
    kind: str = "aug-ablation"      # aug-ablation | arch-probe → experiments.run_config
    thread: str = "C"


class FleetExperimentYaml(BaseModel):
    """MANUAL mode — paste a full config YML (when the leader is on a Claude limit, or you prefer manual).
    Saved under config/_manual/ and enqueued so the fleet runs + journals it exactly like a leader POST."""
    yaml: str
    description: str = ""
    kind: str = "aug-ablation"


class RuntimeTrainingCallbackCreate(BaseModel):
    job_id: str
    status: str
    title: str = ""
    script_path: str = ""
    technical_focus: list[str] = Field(default_factory=list)
    script_args: list[str] = Field(default_factory=list)
    workdir: str = ""
    runtime_task_id: str = ""
    notify_agent: str = "trainer"
    log_path: str = ""
    notes: str = ""


def _runtime_cfg():
    cfg = load_config(DEFAULT_CONFIG_PATH)
    ensure_runtime_layout(cfg)
    return cfg


def _runtime_agent_snapshot(cfg) -> list[dict]:
    state = refresh_status(cfg)
    agents_state = state.get("agents", {}) if isinstance(state, dict) else {}
    snapshots: list[dict] = []
    for agent in cfg.agents:
        info = agents_state.get(agent, {}) if isinstance(agents_state, dict) else {}
        inbox = inbox_files(cfg, agent)
        pending = 0
        last_message = {}
        for path in inbox:
            payload = read_json(path, {})
            if not isinstance(payload, dict):
                continue
            last_message = {
                "id": path.name,
                "from": payload.get("from", ""),
                "to": payload.get("to", ""),
                "timestamp": payload.get("timestamp", ""),
                "task_id": payload.get("task_id", ""),
                "delivered_at": payload.get("delivered_at", ""),
                "content": payload.get("content", payload.get("message", "")),
            }
            if not payload.get("delivered_at"):
                pending += 1
        snapshots.append(
            {
                "name": agent,
                "status": info.get("status", "unknown"),
                "target": info.get("target", ""),
                "last_event": info.get("last_event", ""),
                "pending_inbox": pending,
                "last_message": last_message,
            }
        )
    return snapshots


def _runtime_board_state(slug: str | None = None, limit: int = 120) -> dict:
    # The tmux leader/researcher runtime + shared thread live in the HOME comp (biohub) only. For any
    # OTHER selected competition there is no local runtime yet → show an empty, comp-scoped board
    # (0 messages, 0 agents) rather than biohub's, so the numbers reflect ONLY the selected comp.
    if slug and slug != _home_slug():
        # No tmux/fleet.db runtime for this comp — but its experiments live in Postgres
        # (kaggle_<slug>.experiment_journal / experiment_decisions). Render THOSE as the activity so
        # the page isn't blank. Home (biohub) never enters this branch → its live board is unchanged.
        journal_activity: list[dict] = []
        try:
            rows = _journal_entries(slug)
            # all_journal is ts-ascending → newest-first for the panel
            for r in reversed(rows):
                if not isinstance(r, dict):
                    continue
                journal_activity.append({
                    "exp": r.get("exp") or "",
                    "cv": r.get("cv"),
                    "public": r.get("public"),
                    "private": r.get("private"),
                    "description": r.get("description") or r.get("desc") or "",
                    "ts": r.get("ts") or "",
                })
        except Exception as exc:  # noqa: BLE001
            logger.warning("journal_activity unavailable for %s: %s", slug, exc)
        decisions_feed: list[dict] = []
        try:
            drows = _decision_rows(slug)
            for r in reversed(drows):
                if not isinstance(r, dict):
                    continue
                decisions_feed.append({
                    "agent": r.get("agent") or "",
                    "summary": r.get("summary") or r.get("finding") or "",
                    "recommendation": r.get("recommendation") or "",
                    "ts": r.get("ts") or "",
                })
                if len(decisions_feed) >= 40:
                    break
        except Exception as exc:  # noqa: BLE001
            logger.warning("decisions_feed unavailable for %s: %s", slug, exc)
        # journal-derived KPIs (best private, latest exp) for the no-daemon card row
        def _best(key):
            vals = [j[key] for j in journal_activity if isinstance(j.get(key), (int, float))]
            return max(vals) if vals else None
        kpis = {
            "n_experiments": len(journal_activity),
            "best_private": _best("private"),
            "best_public": _best("public"),
            "best_cv": _best("cv"),
            "latest_exp": journal_activity[0]["exp"] if journal_activity else "",
            "n_decisions": len(decisions_feed),
        }
        # Synthesize one live "agent card" per distinct python agent from the activity feed, so the
        # /runtime Agents panel + messages render for a comp WITHOUT a tmux daemon (decisions_feed is
        # newest-first). status=online if it posted within 15 min, else idle. This is what makes the
        # python-fleet activity visible on the board (the front-end renders `agents` + `messages`).
        import datetime as _dt2
        def _age_min(ts):
            try:
                t = _dt2.datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
                if t.tzinfo:
                    t = t.replace(tzinfo=None)
                return (_dt2.datetime.now() - t).total_seconds() / 60.0
            except Exception:  # noqa: BLE001
                return 1e9
        synth_agents: list[dict] = []
        synth_messages: list[dict] = []
        _seen = set()
        for d in decisions_feed:
            a = d.get("agent") or ""
            summ = d.get("summary") or ""
            ts = d.get("ts") or ""
            synth_messages.append({"id": f"act-{len(synth_messages)}", "from": a, "to": "board",
                                   "timestamp": ts, "task_id": "", "delivered_at": ts,
                                   "content": summ + (f" → {d['recommendation']}" if d.get("recommendation") else "")})
            if a and a not in _seen:
                _seen.add(a)
                synth_agents.append({
                    "name": a,
                    "status": "online" if _age_min(ts) <= 15 else "idle",
                    "target": f"postgres:{slug}:{a}",
                    "last_event": summ[:160],
                    "pending_inbox": 0,
                    "last_message": {"id": f"act-{a}", "from": a, "to": "board",
                                     "timestamp": ts, "task_id": "", "delivered_at": ts, "content": summ},
                })
        # HONEST: the "Agents" panel is the Claude tmux runtime (leader/researcher) — it exists only for
        # the home comp. For rogii there is none, so `agents: []` (do NOT fake python-agent capabilities
        # into this Claude slot). The python fleet-agents render in the SEPARATE Python-agents panel
        # (/api/runtime/fleet ← fleet_board). `messages` carries the activity thread so it isn't blank.
        note = f"No Claude runtime (leader/researcher) for {slug} — python-agent activity + journal are in Postgres."
        return {"config_path": "", "session_name": slug, "workdir": "", "runtime_root": "",
                "thread_path": "", "workflow_state": {},
                "task_summary": {"counts": {}, "tasks": [], "current_task_marker": "-"},
                "agents": [], "messages": synth_messages[:limit],
                "has_local_runtime": False,
                "journal_activity": journal_activity,
                "decisions_feed": decisions_feed,
                "journal_kpis": kpis,
                "note": note}
    cfg = _runtime_cfg()
    paths = runtime_dirs(cfg)
    thread_rows = sorted_thread_messages(cfg)[-limit:]
    # the agent snapshot may shell out to tmux; if tmux is missing, DON'T take the whole page down —
    # messages come from thread.jsonl (no tmux) and must always render.
    try:
        agents = _runtime_agent_snapshot(cfg)
    except Exception as exc:  # noqa: BLE001
        logger.warning("agent snapshot unavailable (tmux?): %s", exc)
        agents = []
    return {
        "config_path": str(cfg.config_path),
        "session_name": cfg.session_name,
        "workdir": str(cfg.workdir),
        "runtime_root": str(cfg.runtime_root),
        "thread_path": str(paths["thread"]),
        "workflow_state": {},
        "task_summary": summarize_tasks(cfg),
        "agents": agents,
        "messages": thread_rows,
        "has_local_runtime": True,
    }


def _send_to_tmux_target(project_id: str, target: str, content: str, *, buffer_prefix: str) -> None:
    cfg = _runtime_cfg()
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, prefix=f"{buffer_prefix}-") as handle:
            handle.write(content)
            tmp_path = handle.name
        load = subprocess.run(
            ["tmux", "load-buffer", "-b", f"{buffer_prefix}-{project_id}", tmp_path],
            capture_output=True,
            text=True,
        )
        if load.returncode != 0:
            logger.warning("Failed to load tmux buffer project=%s target=%s", project_id, target)
            raise HTTPException(status_code=500, detail=load.stderr.strip() or "Failed to load tmux buffer")

        paste = subprocess.run(
            ["tmux", "paste-buffer", "-b", f"{buffer_prefix}-{project_id}", "-t", target],
            capture_output=True,
            text=True,
        )
        if paste.returncode != 0:
            logger.warning("Failed to paste tmux buffer project=%s target=%s", project_id, target)
            raise HTTPException(status_code=500, detail=paste.stderr.strip() or "Failed to paste tmux buffer")

        if cfg.submit_delay_ms > 0:
            time.sleep(cfg.submit_delay_ms / 1000)
        for key in cfg.submit_keys:
            subprocess.run(
                ["tmux", "send-keys", "-t", target, normalize_tmux_key(key)],
                capture_output=True,
                text=True,
            )
            if cfg.submit_delay_ms > 0:
                time.sleep(cfg.submit_delay_ms / 1000)
    finally:
        subprocess.run(["tmux", "delete-buffer", "-b", f"{buffer_prefix}-{project_id}"], capture_output=True, text=True)
        if tmp_path:
            Path(tmp_path).unlink(missing_ok=True)


class LeaderLoop:
    def __init__(self, project_store: ProjectStore):
        self.project_store = project_store
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self.project_store.tick()
            except Exception:
                logger.exception("Leader loop tick failed")
            self._stop.wait(3)


leader_loop = LeaderLoop(store)


@asynccontextmanager
async def lifespan(_: FastAPI):
    leader_loop.start()
    yield
    leader_loop.stop()


app = FastAPI(title="Autonomous ML Research Board", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")



def _adopted_repo_slugs(slug: str) -> list:
    """Adopted-repo folders for a competition, read from `docs/repos/<repo>/manifest.json` (written by
    paper-learn's repo mode). Used only to LINK to the knowledge hub's own repo page — the runboard does
    not re-render it, so there is one renderer, not two."""
    try:
        from pathlib import Path as _P
        root = _P("/home/seshu/kaggle/2026") / slug / "docs" / "repos"
        return sorted(d.name for d in root.iterdir() if (d / "manifest.json").exists()) if root.is_dir() else []
    except Exception:                                     # noqa: BLE001 — a nav link is never worth a 500
        return []


@app.get("/", response_class=HTMLResponse)
async def index() -> RedirectResponse:
    return RedirectResponse(url="/runtime", status_code=307)


@app.get("/projects/{project_id}", response_class=HTMLResponse)
async def project_thread(project_id: str) -> str:
    logger.info("Thread page requested for project=%s", project_id)
    return (STATIC_DIR / "thread.html").read_text(encoding="utf-8")


@app.get("/runtime", response_class=HTMLResponse)
async def runtime_board() -> str:
    return (STATIC_DIR / "runtime_board.html").read_text(encoding="utf-8")


@app.get("/api/state")
async def get_state() -> dict:
    logger.info("State requested")
    return store.state()


@app.get("/api/projects/{project_id}")
async def get_project_detail(project_id: str) -> dict:
    try:
        logger.info("Project detail requested for %s", project_id)
        return {"project": store.project_detail(project_id)}
    except KeyError:
        raise HTTPException(status_code=404, detail="Project not found")


@app.get("/api/events")
async def stream_events() -> StreamingResponse:
    async def event_stream():
        while True:
            payload = json.dumps(store.state(), ensure_ascii=False)
            yield f"data: {payload}\n\n"
            await asyncio.sleep(2)

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.get("/api/runtime/board")
async def get_runtime_board(request: Request, limit: int = 120) -> dict:
    try:
        return _runtime_board_state(_active_comp(request), limit=limit)
    except Exception as exc:
        logger.exception("Runtime board state request failed")
        raise HTTPException(status_code=500, detail=str(exc))


def _tokendb_spend() -> dict:
    """Real Claude $ + tokens the researchpapers AGENTS have spent (from tokendb, corrected Opus-4.8 pricing)."""
    sql = (
        "SELECT "
        "coalesce(round(sum(cost_usd)::numeric,2),0), coalesce(sum(total_tokens),0), count(*), "
        "coalesce(round(sum(cost_usd) FILTER (WHERE ts > now()-interval '60 minutes')::numeric,2),0), "
        "coalesce(sum(total_tokens) FILTER (WHERE ts > now()-interval '60 minutes'),0), "
        "coalesce(round(sum(cost_usd) FILTER (WHERE ts::date = now()::date)::numeric,2),0) "
        "FROM token_usage WHERE project IN ('researchpapers','2026');"
    )
    r = subprocess.run(
        ["psql", "-h", "localhost", "-U", "postgres", "-d", "tokendb", "-tAF", "\t", "-c", sql],
        env={"PGPASSWORD": "seshu", "PATH": "/usr/bin:/bin"}, capture_output=True, text=True, timeout=6,
    )
    parts = (r.stdout.strip().split("\t") if r.stdout.strip() else [])
    if len(parts) < 6:
        raise RuntimeError(r.stderr.strip()[-200:] or "no tokendb rows")
    f = lambda x: float(x or 0)
    return {
        "total_cost": f(parts[0]), "total_tokens": int(f(parts[1])), "messages": int(f(parts[2])),
        "last60_cost": f(parts[3]), "last60_tokens": int(f(parts[4])), "today_cost": f(parts[5]),
    }


@app.get("/api/runtime/spend")
async def get_runtime_spend(request: Request) -> dict:
    """How much Claude the researchpapers agents have really spent. This is a SESSION-GLOBAL number
    (one shared agent pool / tokendb, not per-competition) → flagged scope='global' so the board can
    label it honestly when a specific competition is selected."""
    try:
        s = _tokendb_spend()
        slug = _active_comp(request)
        if slug and slug != _home_slug():
            # For a selected competition show ONLY the LAST HOUR (the current live work) as the headline —
            # not the machine-lifetime total. Per-comp attribution isn't tracked (all logs share one project),
            # so this is honestly labeled as the last-60-min machine spend during this comp's session.
            return {"ok": True, "scope": "last60", "window": "last 60 min",
                    "total_cost": s["last60_cost"], "total_tokens": s["last60_tokens"],
                    "messages": s["messages"], "last60_cost": s["last60_cost"],
                    "last60_tokens": s["last60_tokens"], "today_cost": s["today_cost"]}
        return {"ok": True, "scope": "global", **s}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": str(exc)}


def _train_jobs() -> dict:
    """Live training-job status from the SAME source as the :7766 trend page (train-service :7799)."""
    import urllib.request
    with urllib.request.urlopen("http://127.0.0.1:7799/api/board", timeout=4) as r:
        b = json.loads(r.read())
    c, q = b.get("counts", {}), b.get("queue", {})
    recent = [
        {"id": j.get("id"), "title": (j.get("title") or "")[:80], "status": j.get("status"),
         "started_at": j.get("started_at"), "finished_at": j.get("finished_at")}
        for j in (q.get("recent") or [])[:6]
    ]
    return {
        "running": int(q.get("running_count", 0)), "queued": int(q.get("queued_count", 0)),
        "succeeded": int(c.get("succeeded", 0)), "failed": int(c.get("failed", 0)),
        "running_ids": q.get("running_ids", []), "recent": recent,
    }


@app.get("/api/runtime/trainjobs")
async def get_runtime_trainjobs(request: Request) -> dict:
    """Training-job status (running/queued/succeeded/failed). The GPU queue (:7799) is a SINGLE shared
    scheduler (one job at a time across the whole box). For the HOME comp we show the global numbers; for
    any OTHER selected competition we show ONLY that comp's jobs (matched by slug token in the title), so a
    new comp with no queued training honestly reads 0 instead of borrowing biohub's history."""
    try:
        g = _train_jobs()
        slug = _active_comp(request)
        if slug and slug != _home_slug():
            tok = slug.split("-")[0].lower()
            mine = [j for j in g.get("recent", []) if tok in (j.get("title") or "").lower()]
            return {"ok": True, "scope": "comp", "slug": slug,
                    "running": 0, "queued": 0,
                    "succeeded": sum(1 for j in mine if j.get("status") == "succeeded"),
                    "failed": sum(1 for j in mine if j.get("status") == "failed"),
                    "running_ids": [], "recent": mine,
                    "note": f"Shared GPU queue is machine-global; {len(mine)} recent jobs tagged for {slug}.",
                    "global": {k: g[k] for k in ("running", "queued", "succeeded", "failed")}}
        return {"ok": True, "scope": "global", **g}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": str(exc)}


def _home_slug() -> str:
    """The competition this board physically lives in (biohub) — the ONLY comp with a local SQLite
    fleet board + tmux runtime. Other comps read their fleet board from Postgres (kaggle_<slug>)."""
    from pathlib import Path as _P
    return _P(__file__).resolve().parents[3].name


def _fleet_rows(slug: str | None = None):
    """Fleet-board rows for a competition — SQLite fleet.db for the home comp (biohub), else the
    Postgres fleet_board (fleet_agents.db.all_board) so EVERY comp shows only ITS OWN work (zeros
    until it runs). Returns list of {kind,status,claimed_by,question,updated} or None if unavailable."""
    slug = slug or _home_slug()
    if slug == _home_slug():
        import sqlite3
        from pathlib import Path as _P
        db = _ce.fleet_db() or (_P(__file__).resolve().parent / "fleet" / "fleet.db")
        if not db.exists():
            return None
        try:
            c = sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=4)
            rows = [{"kind": k, "status": s, "claimed_by": w or "", "question": q or "", "updated": u or ""}
                    for (k, s, w, q, u) in c.execute(
                        "SELECT kind,status,claimed_by,question,updated FROM questions")]
            c.close()
            return rows
        except Exception:  # noqa: BLE001
            return None
    db = _fa_db()
    if db is None:
        return []
    try:
        return [{"kind": r.get("kind"), "status": r.get("status"), "claimed_by": r.get("claimed_by") or "",
                 "question": r.get("question") or "", "updated": r.get("updated") or ""}
                for r in db.all_board(slug, limit=100000)]
    except Exception:  # noqa: BLE001
        return []


def _fleet_now_running(slug: str | None = None) -> str:
    """The Python fleet's currently-CLAIMED experiment work for a competition, as a green RUNNING-NOW
    banner. Prefers the score/search agents that actually move the CV (fullconfig-search, combo-search…)."""
    import html as _html
    rows = _fleet_rows(slug)
    if not rows:
        return ""
    claimed_rows = [r for r in rows if r.get("status") == "claimed"]
    claimed_rows.sort(key=lambda r: r.get("updated") or "", reverse=True)
    if not claimed_rows:
        return ""
    PRIORITY = ["fullconfig-search", "combo-search", "config-ablate", "verify-cv", "block-synth",
                "combine-winners", "ablate-best", "pipeline-run", "reproduce-score", "score",
                "div-model", "deep-sister", "arch-probe", "aug-ablation"]
    claimed = {r["kind"] for r in claimed_rows}
    lead = next((k for k in PRIORITY if k in claimed), claimed_rows[0]["kind"])
    lr = next((r for r in claimed_rows if r["kind"] == lead), claimed_rows[0])
    others = [k for k in PRIORITY if k in claimed and k != lead]
    also = ", ".join(others[:6]) or ", ".join(sorted(claimed - {lead})[:6])
    det = _html.escape(f"worker {lr.get('claimed_by')} · {lr.get('question')}")[:200]
    return (f"<div class='now run'><b>▶ FLEET RUNNING NOW</b> — "
            f"<span class='mono'>{_html.escape(lead)}</span> &nbsp;·&nbsp; "
            f"searching golden-12 to beat the public bar &nbsp;·&nbsp; "
            f"<span class='muted'>also active: {_html.escape(also) or 'none'}</span>"
            f"<details><summary>details</summary>{det}</details></div>")


def _fleet_state(slug: str | None = None) -> dict:
    """Roster + activity of the DETERMINISTIC Python agents for a competition (read-only). Home comp =
    SQLite fleet.db; other comps = Postgres fleet_board, so a fresh comp (birdclef) shows all zeros."""
    rows = _fleet_rows(slug)
    if rows is None:
        return {"ok": False, "error": "fleet not started yet"}
    kinds: dict = {}
    for r in rows:
        kind, status = r.get("kind"), r.get("status")
        if not kind:
            continue
        kinds.setdefault(kind, {"open": 0, "claimed": 0, "done": 0, "escalated": 0, "failed": 0})
        kinds[kind][status] = kinds[kind].get(status, 0) + 1
    workers = sorted({r["claimed_by"] for r in rows if r.get("claimed_by")})
    recent = sorted([r for r in rows if r.get("claimed_by")],
                    key=lambda r: r.get("updated") or "", reverse=True)[:8]
    recent = [{"worker": r["claimed_by"], "kind": r["kind"], "question": (r.get("question") or "")[:70],
               "status": r["status"], "updated": r.get("updated")} for r in recent]
    return {"ok": True, "kinds": kinds, "workers": workers, "n_workers": len(workers), "recent": recent}


@app.get("/insights", response_class=HTMLResponse)
def fleet_insights(request: Request) -> str:
    """The deterministic fleet's INSIGHTS.md — the super-agent handoff (complete work + next direction)."""
    import html as _html
    from pathlib import Path as _P
    slug = _active_comp(request)                                   # active competition drives this view
    _sel = _comp_selector_html(slug)
    md = _comp_docs_dir(slug) / "INSIGHTS.md"
    body = md.read_text(encoding="utf-8") if md.exists() else f"# Insights — {slug}\n\n(not generated yet — the fleet writes this each cycle as experiments run)"
    import re as _re

    def _inline(s):  # bold + code on already-escaped text
        s = _re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", s)
        return _re.sub(r"`([^`]+)`", r"<code>\1</code>", s)

    def _cells(line):
        return [c.strip() for c in line.strip().strip("|").split("|")]

    lines = body.splitlines()
    html, i = [], 0
    while i < len(lines):
        ln = lines[i]
        e = _html.escape(ln)
        # markdown TABLE: header row + |---| separator + body rows
        if ln.strip().startswith("|") and i + 1 < len(lines) and _re.match(r"^\s*\|?[\s:|-]+\|?\s*$", lines[i + 1]):
            head = _cells(ln)
            i += 2
            rows = ""
            while i < len(lines) and lines[i].strip().startswith("|"):
                rows += "<tr>" + "".join(f"<td>{_inline(_html.escape(c))}</td>" for c in _cells(lines[i])) + "</tr>"
                i += 1
            html.append("<table><tr>" + "".join(f"<th>{_inline(_html.escape(c))}</th>" for c in head) + "</tr>" + rows + "</table>")
            continue
        if ln.startswith("# "):
            html.append(f"<h1>{_inline(_html.escape(ln[2:]))}</h1>")
        elif ln.startswith("## "):
            html.append(f"<h2>{_inline(_html.escape(ln[3:]))}</h2>")
        elif ln.strip().startswith("- "):
            html.append(f"<div class='li'>• {_inline(_html.escape(ln.strip()[2:]))}</div>")
        elif ln.strip() == "":
            html.append("<div class='sp'></div>")
        else:
            html.append(f"<p>{_inline(e)}</p>")
        i += 1
    return (f"<!doctype html><meta charset=utf-8><meta http-equiv=refresh content=30><title>Fleet Insights</title>"
            f"<style>body{{font:14px -apple-system,Segoe UI,sans-serif;background:#f4f7fb;color:#152238;padding:26px;max-width:960px;margin:auto}}"
            f"h1{{margin:0 0 6px}}h2{{margin:24px 0 8px;font-size:16px;color:#0369a1}}p{{margin:2px 0}}.li{{margin:2px 0 2px 8px}}.sp{{height:8px}}"
            f"code{{background:#eef2f8;padding:1px 5px;border-radius:5px;font-size:12.5px}}a{{color:#0369a1}}"
            f"table{{border-collapse:collapse;width:100%;background:#fff;border-radius:10px;overflow:hidden;box-shadow:0 1px 3px rgba(0,0,0,.06);margin:8px 0}}"
            f"th,td{{text-align:left;padding:7px 11px;border-bottom:1px solid #eef2f8;font-size:13px}}th{{background:#f1f5fb;color:#5b6b86}}"
            f"td:nth-child(2),td:nth-child(3){{font-family:ui-monospace,monospace;color:#b45309}}"
            f"@media(prefers-color-scheme:dark){{body{{background:#0b1220;color:#e6edf8}}h2{{color:#a5b4fc}}"
            f"code{{background:#1b2745;color:#c4b5fd}}table{{background:#111a2e;box-shadow:0 1px 3px rgba(0,0,0,.4)}}"
            f"th,td{{border-bottom-color:#22304d}}th{{background:#1b2745;color:#8ba0c4}}a{{color:#8b9bff}}"
            f"td:nth-child(2),td:nth-child(3){{color:#fbbf24}}}}</style>"
            f"<p><a href='/runtime'>← runtime</a> · <a href='/journal?comp={_html.escape(slug)}'>journal</a> · <a href='/prompts'>prompts</a></p>"
            f"{_sel}{''.join(html)}")


@app.get("/prompts", response_class=HTMLResponse)
def agent_prompts(request: Request) -> str:
    """Show the ACTUAL system prompts (identity files) the leader & researcher run with — inspectable.
    These belong to the HOME comp's Claude runtime; for any other selected comp (no runtime) we say so
    honestly instead of implying biohub's leader/researcher are running for it."""
    import html as _html
    _comp = _active_comp(request)
    _banner = ""
    if _comp and _comp != _home_slug():
        _banner = (f"<div style='background:#fffaf0;border:1px solid #f3e3b8;border-left:4px solid #e0a93a;"
                   f"border-radius:10px;padding:12px 16px;margin:0 0 18px;color:#7a5b12'>ℹ️ <b>{_html.escape(_comp)}</b> "
                   f"has no Claude leader/researcher runtime — it runs deterministic <b>python fleet-agents</b> "
                   f"(see the Python-agents panel on <a href='/runtime?comp={_html.escape(_comp)}'>runtime</a>). "
                   f"The identities below are the HOME ({_html.escape(_home_slug())}) runtime's.</div>")
    idir = BASE_DIR / "identities"
    order = [("leader", "🧭 Leader"), ("researcher", "🔬 Researcher")]
    blocks = _banner
    for slug, label in order:
        f = idir / f"{slug}.md"
        body = f.read_text(encoding="utf-8") if f.exists() else "(identity file not found)"
        # highlight the STRICT OUTPUT TEMPLATE section so it stands out
        esc = _html.escape(body)
        blocks += (f"<section><h2>{label} <small>{_html.escape(str(f))}</small></h2>"
                   f"<pre>{esc}</pre></section>")
    return (f"<!doctype html><meta charset=utf-8><title>Agent Prompts</title><style>"
            f"body{{font:14px -apple-system,Segoe UI,sans-serif;background:#f4f7fb;color:#152238;padding:26px;max-width:1000px;margin:auto}}"
            f"h1{{margin:0 0 4px}}h2{{margin:22px 0 6px;font-size:16px}}h2 small{{color:#94a3b8;font-weight:400;font-family:ui-monospace,monospace}}"
            f"p{{color:#5b6b86}}a{{color:#0369a1}}"
            f"pre{{background:#fff;border:1px solid #e6edf6;border-radius:10px;padding:14px 16px;white-space:pre-wrap;"
            f"word-break:break-word;font:12.5px/1.5 ui-monospace,SFMono-Regular,monospace;box-shadow:0 1px 3px rgba(0,0,0,.05)}}"
            f"@media(prefers-color-scheme:dark){{body{{background:#0b1220;color:#e6edf8}}p{{color:#8ba0c4}}a{{color:#8b9bff}}"
            f"h2 small{{color:#5b6b86}}pre{{background:#111a2e;border-color:#22304d;color:#e6edf8;box-shadow:0 1px 3px rgba(0,0,0,.4)}}}}</style>"
            f"<h1>🗒️ Agent system prompts</h1><p>The exact identity/instructions each Claude agent runs with "
            f"(edit these files to change behavior on next launch). <a href='/runtime'>← back to runtime</a> · "
            f"<a href='/journal'>journal →</a></p>{blocks}")


@app.get("/api/runtime/fleet")
async def get_runtime_fleet(request: Request) -> dict:
    """Deterministic Python-agent roster + activity for the ACTIVE competition (comp-scoped)."""
    try:
        return _fleet_state(_active_comp(request))
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": str(exc)}


def _md_to_html(body: str) -> str:
    """Minimal, CSP-safe markdown → HTML (tables, #/## headers, - bullets, **bold**, `code`).
    Shared by the experiments panel; mirrors the /insights renderer but emits h3/h4 for embedding."""
    import html as _h
    import re as _re

    def _inline(s):
        s = _re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", s)
        return _re.sub(r"`([^`]+)`", r"<code>\1</code>", s)

    def _cells(line):
        return [c.strip() for c in line.strip().strip("|").split("|")]

    lines = body.splitlines()
    out, i = [], 0
    while i < len(lines):
        ln = lines[i]
        e = _h.escape(ln)
        if ln.strip().startswith("|") and i + 1 < len(lines) and _re.match(r"^\s*\|?[\s:|-]+\|?\s*$", lines[i + 1]):
            head = _cells(ln)
            i += 2
            rows = ""
            while i < len(lines) and lines[i].strip().startswith("|"):
                rows += "<tr>" + "".join(f"<td>{_inline(_h.escape(c))}</td>" for c in _cells(lines[i])) + "</tr>"
                i += 1
            out.append("<div class='tw'><table>" + "<tr>" + "".join(f"<th>{_inline(_h.escape(c))}</th>" for c in head) + "</tr>" + rows + "</table></div>")
            continue
        if ln.startswith("# "):
            out.append(f"<h3>{_inline(_h.escape(ln[2:]))}</h3>")
        elif ln.startswith("## "):
            out.append(f"<h4>{_inline(_h.escape(ln[3:]))}</h4>")
        elif ln.strip().startswith("- "):
            out.append(f"<div class='li'>• {_inline(_h.escape(ln.strip()[2:]))}</div>")
        elif ln.strip() == "":
            out.append("<div class='sp'></div>")
        else:
            out.append(f"<p>{_inline(e)}</p>")
        i += 1
    return "".join(out)


def _experiments_state(slug: str | None = None) -> dict:
    """Live fleet-agent EXPERIMENT tracking: durable findings tables (linker baseline, division
    verifiers) + currently-running experiment processes + tails of the live scratchpad logs.
    All read-only, best-effort; never raises. `slug` selects the competition root (defaults to this comp)."""
    from pathlib import Path as _P
    import glob as _glob
    import subprocess as _sp
    import time as _t
    _default = _P(__file__).resolve().parents[3]
    comp = _default if (not slug or slug == _default.name) else (_P("/home/seshu/kaggle/2026") / slug)
    FINDINGS = [
        ("🧬 Linker baseline — per-embryo × mode (E10)", "experiments/divisions/e10_findings.md"),
        ("🔬 Division temporal verifier (E9)", "experiments/divisions/e9_findings.md"),
        ("🔬 Division appearance verifier (E8)", "experiments/divisions/e8_findings.md"),
        ("🧫 Division on real masks (E7)", "experiments/divisions/e7_findings.md"),
        ("📐 Division ILP-recoverable? verdict (E1)", "experiments/divisions/e1_findings.md"),
        ("🧩 Cellpose-SAM → Trackastra pipeline (E10)", "experiments/pipeline/e10_findings.md"),
        ("📏 Division distance feasibility (E31)", "experiments/divisions/e31_findings.md"),
        ("📏 Division feasibility (E36)", "experiments/divisions/e36_findings.md"),
    ]
    findings = []
    for label, rel in FINDINGS:
        p = comp / rel
        if not p.exists():
            continue
        try:
            findings.append({
                "label": label, "rel": rel,
                "mtime": _t.strftime("%Y-%m-%d %H:%M", _t.localtime(p.stat().st_mtime)),
                "mtime_ts": p.stat().st_mtime,
                "md": p.read_text(encoding="utf-8", errors="replace"),
            })
        except OSError:
            continue
    # NON-HOME comp: the hardcoded biohub division/linker files don't exist → build findings from THIS
    # comp's own docs (INSIGHTS.md + research/*.md) + its Postgres decisions, so /experiments shows the
    # selected competition — never biohub's linker/division placeholder.
    if slug and slug != _default.name and not findings:
        import time as _t2
        for rel in ["docs/INSIGHTS.md"] + [f"research/{n.name}" for n in sorted((comp / "research").glob("*.md"))[:6]]:
            p = comp / rel
            if p.exists():
                try:
                    findings.append({"label": p.stem.replace("_", " ").title(), "rel": rel,
                                     "mtime": _t2.strftime("%Y-%m-%d %H:%M", _t2.localtime(p.stat().st_mtime)),
                                     "mtime_ts": p.stat().st_mtime,
                                     "md": p.read_text(encoding="utf-8", errors="replace")[:20000]})
                except OSError:
                    continue
        try:
            drows = _decision_rows(slug)
            if drows:
                md = "\n".join(f"- **[{r.get('agent','')}]** {r.get('summary','')}"
                               + (f" → _{r.get('recommendation')}_" if r.get('recommendation') else "")
                               for r in reversed(drows))
                findings.insert(0, {"label": "🧭 Findings & decisions (live)", "rel": "postgres:experiment_decisions",
                                    "mtime": "live", "mtime_ts": 9e9, "md": md})
        except Exception:  # noqa: BLE001
            pass
    # currently-running experiment processes (division/linker/finetune work)
    running = []
    try:
        ps = _sp.run(["ps", "-eo", "pid,pcpu,etime,args"], capture_output=True, text=True, timeout=5).stdout.splitlines()
        for ln in ps:
            low = ln.lower()
            if "ps -eo" in low or "python" not in low:
                continue
            if any(k in low for k in ("experiments/divisions/e", "linker_baseline", "tk_train",
                                       "temporal_verifier", "div_verifier", "finetune")):
                parts = ln.split(None, 3)
                if len(parts) == 4:
                    running.append({"pid": parts[0], "cpu": parts[1], "etime": parts[2], "cmd": parts[3][:170]})
    except Exception:  # noqa: BLE001
        pass
    # live scratchpad logs — discover recent logs across claude sessions, keep newest per name, tail
    PATS = ("e10_run", "verifier", "tk_train", "finetune", "ilp", "linker", "e8_", "e9_")
    best: dict = {}
    for g in _glob.glob("/tmp/claude-*/-home-seshu-kaggle-2026/*/scratchpad/*.log"):
        base = _P(g).name
        if not any(k in base.lower() for k in PATS):
            continue
        try:
            st = _P(g).stat()
        except OSError:
            continue
        if _t.time() - st.st_mtime > 6 * 3600:                    # only logs touched in the last 6h
            continue
        if base not in best or st.st_mtime > best[base][1]:
            best[base] = (g, st.st_mtime)
    LOGMAP = [
        ("e10_run", "E10 · Linker baseline (per-embryo × mode)"),
        ("e10", "E10 · Linker baseline"),
        ("verifier2", "E9 · Division temporal verifier"),
        ("temporal", "E9 · Division temporal verifier"),
        ("verifier", "E8 · Division appearance verifier"),
        ("tk_train", "Trackastra linker fine-tune"),
        ("finetune", "Linker fine-tune"),
        ("ilp", "E7 · ILP linking confirm"),
        ("linker", "Linker experiment"),
    ]

    def _explabel(base):
        b = base.lower()
        for key, lab in LOGMAP:
            if key in b:
                return lab
        return base

    def _clean_tail(g):
        """Tail the log and collapse tqdm carriage-return spam to the final state of each line."""
        try:
            raw = _sp.run(["tail", "-n", "60", g], capture_output=True, timeout=5).stdout
        except Exception:  # noqa: BLE001
            return ""
        text = raw.decode("utf-8", "replace")
        lines = [ln.split("\r")[-1].rstrip() for ln in text.split("\n")]
        return "\n".join(lines).strip()[-7000:]

    logs = []
    for base, (g, ts) in best.items():
        logs.append({"name": base, "exp": _explabel(base),
                     "mtime": _t.strftime("%H:%M:%S", _t.localtime(ts)), "mtime_ts": ts,
                     "tail": _clean_tail(g)})
    logs.sort(key=lambda d: -d["mtime_ts"])
    return {"ok": True, "findings": findings, "running": running, "logs": logs[:6]}


@app.get("/api/runtime/experiments")
async def get_runtime_experiments(request: Request, full: int = 0) -> dict:
    """Live experiment tracking JSON. Default = light summary for the board side-card;
    `?full=1` also returns the running processes + tailed log bodies so the /experiments
    terminal panels can update live without a full-page reload."""
    try:
        slug = _active_comp(request)
        st = _experiments_state(slug)
        out = {"ok": True, "comp": slug, "n_journal": len(_journal_entries(slug)), "n_findings": len(st["findings"]),
               "findings": [{"label": f["label"], "rel": f["rel"], "mtime": f["mtime"]} for f in st["findings"]],
               "running": st["running"], "n_logs": len(st["logs"])}
        if full:
            out["logs"] = [{"name": l["name"], "exp": l.get("exp") or l["name"], "id": _re_sub_id(l["name"]),
                            "mtime": l["mtime"], "tail": l["tail"]} for l in st["logs"]]
        return out
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": str(exc)}


def _re_sub_id(name: str) -> str:
    import re as _re
    return "log-" + _re.sub(r"[^a-zA-Z0-9]+", "-", name).strip("-").lower()


@app.get("/experiments", response_class=HTMLResponse)
def experiments_page(request: Request) -> str:
    """Live fleet-agent EXPERIMENT board: per-embryo linker baseline, division-verifier numbers,
    fine-tune progress, and live experiment log tails. Theme-aware, auto-refresh."""
    import html as _h
    slug = _active_comp(request)                                   # active competition drives this view
    _sel = _comp_selector_html(slug)
    _njournal = len(_journal_entries(slug))
    st = _experiments_state(slug)
    run = st["running"]
    if run:
        chips = "".join(
            f"<div class='run-chip'><b>▶ pid {_h.escape(r['pid'])}</b> · cpu {_h.escape(r['cpu'])}% · "
            f"{_h.escape(r['etime'])} <span class='rc-cmd'>{_h.escape(r['cmd'])}</span></div>" for r in run)
        banner = f"<div class='banner live'><div class='bh'>▶ {len(run)} experiment{'s' if len(run)!=1 else ''} running now</div>{chips}</div>"
    else:
        banner = "<div class='banner idle'><div class='bh'>■ No experiment process running right now</div><div class='sub'>Durable findings below stay live; a new run appears here within seconds.</div></div>"
    banner = f"<div id='bannerBox'>{banner}</div>"

    cards = ""
    if not st["findings"]:
        cards = "<div class='card'><div class='ch'>No findings written yet</div><p>Experiment findings, research notes, and agent decisions for this competition render here as they are written (Postgres <code>experiment_decisions</code> + <code>docs/</code>/<code>research/</code>).</p></div>"
    for f in st["findings"]:
        cards += (f"<section class='card'><div class='ch'>{_h.escape(f['label'])}"
                  f"<span class='ctag'>{_h.escape(f['rel'])} · {_h.escape(f['mtime'])}</span></div>"
                  f"<div class='md'>{_md_to_html(f['md'])}</div></section>")

    def _term(lg):
        lid = _re_sub_id(lg["name"])
        return (f"<section class='term' data-log='{_h.escape(lg['name'])}'>"
                f"<div class='term-bar'><span class='dot r'></span><span class='dot y'></span><span class='dot g'></span>"
                f"<span class='term-title'><b>{_h.escape(lg.get('exp') or lg['name'])}</b> · {_h.escape(lg['name'])}</span>"
                f"<span class='term-live' id='{lid}-live'>● live · {_h.escape(lg['mtime'])}</span></div>"
                f"<pre class='term-body' id='{lid}'>{_h.escape((lg['tail'] or '(empty)').strip()[-6000:])}</pre></section>")
    logs = "".join(_term(lg) for lg in st["logs"])
    logs_section = (f"<h2 class='sec'>Live experiment logs <small>(tail · streaming)</small></h2>"
                    f"<div id='terms'>{logs}</div>") if logs else "<div id='terms'></div>"

    return (
        "<!doctype html><html lang=en><head><meta charset=utf-8>"
        "<meta name=viewport content='width=device-width,initial-scale=1'>"
        "<meta http-equiv=refresh content=60><title>Experiment Tracking</title><style>"
        ":root{--bg:#f4f7fb;--panel:#ffffff;--ink:#152238;--muted:#5b6b86;--line:#e6edf6;"
        "--brand:#4f46e5;--accent:#eef2ff;--mono:'SF Mono',ui-monospace,SFMono-Regular,Menlo,monospace;"
        "--num:#6d28d9;--ok:#16a34a;--warn:#b45309;--shadow:0 1px 3px rgba(15,23,42,.07)}"
        ":root[data-theme=dark]{--bg:#0b1220;--panel:#111a2e;--ink:#e6edf8;--muted:#8ba0c4;--line:#22304d;"
        "--brand:#8b9bff;--accent:#1b2745;--num:#c4b5fd;--ok:#4ade80;--warn:#fbbf24;--shadow:0 1px 3px rgba(0,0,0,.4)}"
        "@media(prefers-color-scheme:dark){:root:not([data-theme=light]){--bg:#0b1220;--panel:#111a2e;--ink:#e6edf8;"
        "--muted:#8ba0c4;--line:#22304d;--brand:#8b9bff;--accent:#1b2745;--num:#c4b5fd;--ok:#4ade80;--warn:#fbbf24;--shadow:0 1px 3px rgba(0,0,0,.4)}}"
        "*{box-sizing:border-box}body{margin:0;font:14px/1.55 -apple-system,'Segoe UI',Roboto,sans-serif;"
        "background:var(--bg);color:var(--ink);-webkit-font-smoothing:antialiased}"
        "header{position:sticky;top:0;z-index:9;background:color-mix(in srgb,var(--panel) 88%,transparent);"
        "backdrop-filter:blur(10px);border-bottom:1px solid var(--line);padding:14px 24px;display:flex;"
        "align-items:center;gap:14px;flex-wrap:wrap}"
        "header h1{margin:0;font-size:19px;letter-spacing:-.02em}header .nav{display:flex;gap:6px;flex-wrap:wrap;margin-left:auto}"
        "header .nav a{font-size:12.5px;color:var(--muted);border:1px solid var(--line);border-radius:20px;"
        "padding:5px 12px;text-decoration:none;background:var(--panel)}header .nav a:hover{color:var(--brand);border-color:var(--brand)}"
        "#tt{cursor:pointer;font-size:15px;line-height:1;background:var(--panel);border:1px solid var(--line);"
        "border-radius:20px;padding:5px 11px;color:var(--ink)}"
        ".wrap{max-width:1080px;margin:0 auto;padding:20px 24px 80px}"
        ".banner{border-radius:14px;padding:14px 16px;margin:6px 0 20px;box-shadow:var(--shadow);border:1px solid var(--line)}"
        ".banner .bh{font-weight:700;font-size:14px}.banner .sub{color:var(--muted);font-size:12.5px;margin-top:3px}"
        ".banner.live{background:color-mix(in srgb,var(--ok) 12%,var(--panel));border-color:var(--ok)}.banner.live .bh{color:var(--ok)}"
        ".banner.idle{background:var(--panel)}.banner.idle .bh{color:var(--muted)}"
        ".run-chip{font-family:var(--mono);font-size:11.5px;margin-top:7px;color:var(--ink)}"
        ".run-chip .rc-cmd{color:var(--muted);display:block;margin-top:1px;word-break:break-all}"
        "h2.sec{font-size:14px;text-transform:uppercase;letter-spacing:.05em;color:var(--muted);margin:34px 0 12px}"
        "h2.sec small{text-transform:none;letter-spacing:0;font-weight:400}"
        ".card{background:var(--panel);border:1px solid var(--line);border-radius:14px;box-shadow:var(--shadow);"
        "padding:16px 18px;margin:0 0 16px}"
        ".card .ch{font-weight:700;font-size:14.5px;display:flex;align-items:baseline;gap:10px;flex-wrap:wrap;margin-bottom:8px}"
        ".card .ctag{font-family:var(--mono);font-size:11px;color:var(--muted);font-weight:400}"
        ".md h3{font-size:15px;margin:4px 0 6px}.md h4{font-size:13px;margin:16px 0 6px;color:var(--brand);"
        "text-transform:uppercase;letter-spacing:.03em}"
        ".md p{margin:3px 0;color:var(--ink)}.md .li{margin:2px 0 2px 6px;color:var(--ink)}.md .sp{height:7px}"
        ".md code{background:var(--accent);padding:1px 6px;border-radius:6px;font-family:var(--mono);font-size:12px;color:var(--num)}"
        ".md .tw{overflow-x:auto;margin:8px 0}"
        ".md table{border-collapse:collapse;width:100%;font-size:12.5px;min-width:520px}"
        ".md th,.md td{text-align:left;padding:6px 10px;border-bottom:1px solid var(--line);white-space:nowrap}"
        ".md th{background:var(--accent);color:var(--muted);font-size:11px;text-transform:uppercase;letter-spacing:.03em}"
        ".md td:nth-child(n+3){font-family:var(--mono);color:var(--num);font-variant-numeric:tabular-nums}"
        ".md tr:hover td{background:color-mix(in srgb,var(--accent) 55%,transparent)}"
        ".term{border-radius:12px;overflow:hidden;margin:0 0 16px;box-shadow:0 8px 24px rgba(0,0,0,.28);border:1px solid #10151f}"
        ".term-bar{display:flex;align-items:center;gap:7px;background:#181e2a;padding:8px 12px;border-bottom:1px solid #0a0e15}"
        ".term-bar .dot{width:11px;height:11px;border-radius:50%;display:inline-block}"
        ".term-bar .dot.r{background:#ff5f56}.term-bar .dot.y{background:#ffbd2e}.term-bar .dot.g{background:#27c93f}"
        ".term-bar .term-title{color:#6b7688;font-family:var(--mono);font-size:11px;margin-left:6px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}"
        ".term-bar .term-title b{color:#e8eefc;font-weight:600}"
        ".term-bar .term-live{margin-left:auto;color:#27c93f;font-size:10.5px;font-family:var(--mono);letter-spacing:.03em}"
        ".term-bar .term-live.stale{color:#8b98b0}.term-bar .term-live.stale::first-letter{color:#8b98b0}"
        ".term-body{background:#0c1118;color:#cfe3d0;font-family:var(--mono);font-size:11.5px;line-height:1.55;"
        "margin:0;padding:14px 16px;overflow:auto;max-height:360px;white-space:pre;scroll-behavior:smooth}"
        ".term-body::selection{background:#264f2e}"
        "@keyframes blink{50%{opacity:.25}}.term-live{animation:blink 2s infinite}"
        "</style></head><body>"
        f"<header><h1>🧪 Experiment Tracking — {_h.escape(slug)}</h1>"
        f"<span style='font-size:12px;color:var(--muted)'>{_njournal} journaled experiments (Postgres) · live fleet-agent runs</span>"
        f"<nav class='nav'><a href='/runtime'>← Runtime</a><a href='/journal?comp={_h.escape(slug)}'>📓 Journal</a>"
        f"<a href='/insights?comp={_h.escape(slug)}'>🧠 Insights</a><a href='/prompts'>🗒 Prompts</a>"
        # adopted GitHub repos live on the knowledge hub (:7777); link every one we have adopted so the
        # runboard and the hub stay one surface. Discovered from docs/repos/, so a new repo needs no edit.
        + "".join(f"<a href='http://gpu:7777/repo/{_h.escape(slug)}/{_h.escape(_rp)}' "
                  f"target='_blank' rel='noopener'>⚙ {_h.escape(_rp)}</a>"
                  for _rp in _adopted_repo_slugs(slug))
        + "<span id=tt title='Toggle theme'>◐</span></nav></header>"
        f"<div class='wrap'>{_sel}{banner}<h2 class='sec'>Findings <small>(durable · newest tables win)</small></h2>{cards}{logs_section}</div>"
        "<script>(function(){var k='rp-theme',r=document.documentElement,b=document.getElementById('tt');"
        "var s=localStorage.getItem(k);if(s)r.setAttribute('data-theme',s);"
        "b.onclick=function(){var d=r.getAttribute('data-theme')==='dark'?'light':'dark';"
        "r.setAttribute('data-theme',d);localStorage.setItem(k,d);};})();</script>"
        "<script>(function(){"
        "function esc(s){return (s==null?'':''+s).replace(/[&<>]/g,function(c){return {'&':'&amp;','<':'&lt;','>':'&gt;'}[c];});}"
        "function termHTML(l){return \"<section class='term' data-log='\"+esc(l.name)+\"'><div class='term-bar'>\"+"
        "\"<span class='dot r'></span><span class='dot y'></span><span class='dot g'></span>\"+"
        "\"<span class='term-title'><b>\"+esc(l.exp||l.name)+\"</b> \\u00B7 \"+esc(l.name)+\"</span><span class='term-live' id='\"+l.id+\"-live'>\\u25CF live \\u00B7 \"+esc(l.mtime)+\"</span></div>\"+"
        "\"<pre class='term-body' id='\"+l.id+\"'>\"+esc((l.tail||'(empty)').trim())+\"</pre></section>\";}"
        "async function poll(){try{"
        "var res=await fetch('/api/runtime/experiments?full=1',{cache:'no-store'});var d=await res.json();if(!d.ok)return;"
        "var run=d.running||[];var bb=document.getElementById('bannerBox');"
        "if(run.length){bb.innerHTML=\"<div class='banner live'><div class='bh'>\\u25B6 \"+run.length+\" experiment\"+(run.length!=1?'s':'')+\" running now</div>\"+"
        "run.map(function(r){return \"<div class='run-chip'><b>\\u25B6 pid \"+esc(r.pid)+\"</b> \\u00B7 cpu \"+esc(r.cpu)+\"% \\u00B7 \"+esc(r.etime)+\" <span class='rc-cmd'>\"+esc(r.cmd)+\"</span></div>\";}).join('')+\"</div>\";}"
        "else{bb.innerHTML=\"<div class='banner idle'><div class='bh'>\\u25A0 No experiment process running right now</div><div class='sub'>Durable findings stay live; a new run appears here within seconds.</div></div>\";}"
        "var terms=document.getElementById('terms');var logs=d.logs||[];"
        "logs.forEach(function(l){var pre=document.getElementById(l.id);"
        "if(!pre){terms.insertAdjacentHTML('afterbegin',termHTML(l));pre=document.getElementById(l.id);var atB=true;}"
        "else{var atB=(pre.scrollTop+pre.clientHeight>=pre.scrollHeight-30);}"
        "var t=(l.tail||'(empty)').trim();if(pre.textContent!==t){pre.textContent=t;if(atB)pre.scrollTop=pre.scrollHeight;}"
        "var lv=document.getElementById(l.id+'-live');if(lv)lv.textContent='\\u25CF live \\u00B7 '+l.mtime;});"
        "}catch(e){}}"
        "document.querySelectorAll('.term-body').forEach(function(p){p.scrollTop=p.scrollHeight;});"
        "poll();setInterval(poll,4000);})();</script>"
        "</body></html>")


def _launch_agent_bg(agent: str) -> None:
    """Bring a Claude agent online in the background (launch_agent blocks on codex-ready)."""
    try:
        from researchpapers import runtime_cli as _rc
        _rc.launch_agent(_runtime_cfg(), agent)
        logger.info("Agent %s launched from web UI", agent)
    except Exception as exc:  # noqa: BLE001
        logger.warning("launch_agent(%s) failed: %s", agent, exc)


@app.post("/api/runtime/agents/{agent}/start")
async def start_runtime_agent(agent: str) -> dict:
    """Power ON a Claude agent (leader / researcher) — launches its tmux window and bootstraps it.
    Non-blocking: returns 'starting'; the board's status poll flips it to 'online' when ready."""
    import shutil
    cfg = _runtime_cfg()
    if agent not in cfg.agents:
        raise HTTPException(status_code=400, detail=f"Unknown agent: {agent}")
    if not shutil.which("tmux"):
        return {"ok": False, "agent": agent, "error": "tmux is not available on this host — agents can't be launched here"}
    threading.Thread(target=_launch_agent_bg, args=(agent,), daemon=True).start()
    return {"ok": True, "agent": agent, "status": "starting"}


@app.post("/api/runtime/agents/{agent}/stop")
async def stop_runtime_agent(agent: str) -> dict:
    """Power OFF a Claude agent — kills its tmux window and marks it offline."""
    import shutil
    from researchpapers import runtime_cli as _rc
    cfg = _runtime_cfg()
    if agent not in cfg.agents:
        raise HTTPException(status_code=400, detail=f"Unknown agent: {agent}")
    if not shutil.which("tmux"):
        return {"ok": False, "agent": agent, "error": "tmux is not available on this host"}
    r = subprocess.run(["tmux", "kill-window", "-t", f"{cfg.session_name}:{agent}"],
                       capture_output=True, text=True)
    try:
        _rc.mark_agent_status(cfg, agent, "offline", "powered off from web UI")
    except Exception:  # noqa: BLE001
        pass
    if r.returncode != 0 and "can't find" not in (r.stderr or "").lower() and "no such" not in (r.stderr or "").lower():
        return {"ok": False, "agent": agent, "error": r.stderr.strip() or "kill-window failed"}
    return {"ok": True, "agent": agent, "status": "offline"}


def _fleet_agents_import():
    """Import the competition's fleet_agents package (adds comp root to sys.path). Returns the module or None."""
    import sys
    from pathlib import Path as _P
    comp = _P(__file__).resolve().parents[3]
    if str(comp) not in sys.path:
        sys.path.insert(0, str(comp))
    try:
        import fleet_agents as _fa  # noqa: PLC0415
        return _fa
    except Exception as exc:  # noqa: BLE001
        logger.warning("fleet_agents import failed: %s", exc)
        return None


def _fleet_daemon_running() -> bool:
    """Is the deterministic fleet worker pool (python -m researchpapers.fleet) alive?"""
    try:
        ps = subprocess.run(["ps", "-eo", "args"], capture_output=True, text=True, timeout=5).stdout
        return ("researchpapers.fleet" in ps) and ("--workers" in ps)
    except Exception:  # noqa: BLE001
        return False


_COMP_DENYLIST = {"kaggle_ai", "unsloth_compiled_cache", "amio"}


def _active_competitions() -> list:
    """Active Kaggle PROJECTS (sibling comp dirs of this comp root) → their Kaggle competition modality, so the
    board's selector is by PROJECT (how the user picks on :7777) in the SAME Kaggle taxonomy the agents are tagged
    with. Known comps use comp-onboard's table; others resolve to '' (= all agents). Read-only, never raises."""
    from pathlib import Path as _P
    out: list = []
    try:
        try:
            from fleet_agents.comp_onboard import KNOWN_COMPS as known  # noqa: PLC0415
        except Exception:  # noqa: BLE001
            known = {}
        try:
            from fleet_agents.agent_routing import MODALITIES as _KMODS  # noqa: PLC0415
        except Exception:  # noqa: BLE001
            _KMODS = []
        try:
            # Kaggle-tag-grounded modality (cached offline map) AUGMENTS KNOWN_COMPS so birdclef/deep-past/
            # nemotron get their REAL modality in the dropdown instead of '(all)'. Read-only, never raises.
            from fleet_agents.kaggle_modality import load_map as _kml  # noqa: PLC0415
            _kmap = _kml()
        except Exception:  # noqa: BLE001
            _kmap = {}
        root = _P(__file__).resolve().parents[4]            # /home/seshu/kaggle/2026 (holds all comp projects)
        for d in sorted(root.iterdir()):
            if not d.is_dir() or d.name.startswith(".") or d.name in _COMP_DENYLIST:
                continue
            slug = d.name
            is_comp = (slug in known or (d / "input").exists() or (d / "recipe").exists()
                       or (d / ".env").exists() or (d / "fleet_agents").exists())
            if not is_comp:
                continue
            # KNOWN_COMPS is the high-confidence override; the Kaggle-tag cached map fills the gaps.
            comp_mod = (known.get(slug) or {}).get("modality", "")   # Kaggle taxonomy (comp-onboard)
            if not comp_mod:
                comp_mod = (_kmap.get(slug) or {}).get("modality", "") or ""
            routed = comp_mod if (comp_mod in _KMODS) else ""        # '' = unknown → show all agents
            out.append({"slug": slug, "modality": routed, "comp_modality": comp_mod or "unknown",
                        "known": slug in known})
    except Exception:  # noqa: BLE001
        return out
    return out


def _fleet_agents_state(slug: str | None = None) -> dict:
    """Roster of the Python fleet agents (fleet_agents.HANDLERS) grouped BY DOMAIN, each tagged with its
    domain + the competition modalities that use it (via fleet_agents.agent_routing), plus live board
    status (running/queued/done) for the ACTIVE competition and whether the worker daemon is up. The
    per-kind status counts come from the comp's fleet board (SQLite for home, Postgres otherwise), so a
    fresh competition (e.g. birdclef) shows zero active work. Read-only, never raises."""
    from pathlib import Path as _P
    fa = _fleet_agents_import()
    if fa is None:
        return {"ok": False, "error": "fleet_agents package could not be imported"}
    raw = getattr(fa, "_RAW_HANDLERS", None) or getattr(fa, "HANDLERS", {})
    # DOMAIN + MODALITY routing (single source of truth: fleet_agents.agent_routing)
    tagmap: dict = {}
    domain_order: list = []
    modalities: list = []
    try:
        from fleet_agents import agent_routing as _ar  # noqa: PLC0415
        tagmap = _ar.tag_map()                          # {agent: {domain, modalities}}
        domain_order = list(_ar.domains().keys())       # ordered domain groups
        modalities = list(getattr(_ar, "MODALITIES", []))  # the KAGGLE competition taxonomy
        if not modalities:
            modalities = sorted({m for v in tagmap.values() for m in v.get("modalities", [])})
    except Exception as exc:  # noqa: BLE001
        logger.warning("agent_routing unavailable, falling back to module grouping: %s", exc)

    def _fallback_group(fn):
        mod = getattr(fn, "__module__", "") or ""
        return (mod.split(".")[-1] if mod else "other") or "other"

    # live board status per kind — COMP-SCOPED (SQLite for home, Postgres fleet_board otherwise)
    status: dict = {}
    for r in (_fleet_rows(slug) or []):
        k, st = r.get("kind"), r.get("status")
        if k and st:
            status.setdefault(k, {})[st] = status.get(k, {}).get(st, 0) + 1
    agents = []
    domain_counts: dict = {}
    for k in sorted(raw):
        s = status.get(k, {})
        tag = tagmap.get(k, {})
        domain = tag.get("domain") or _fallback_group(raw[k])
        mods = list(tag.get("modalities", []))
        domain_counts[domain] = domain_counts.get(domain, 0) + 1
        agents.append({"name": k, "group": domain, "domain": domain, "modalities": mods,
                       "running": int(s.get("claimed", 0)) > 0, "queued": int(s.get("open", 0)),
                       "claimed": int(s.get("claimed", 0)), "done": int(s.get("done", 0))})
    domains_out = [{"name": d, "count": domain_counts[d]} for d in domain_order if d in domain_counts]
    for d in sorted(domain_counts):                     # any domain not in the routing order (fallbacks)
        if d not in domain_order:
            domains_out.append({"name": d, "count": domain_counts[d]})
    return {"ok": True, "fleet_running": _fleet_daemon_running(), "n": len(agents),
            "agents": agents, "domains": domains_out, "modalities": modalities,
            "competitions": _active_competitions()}


@app.get("/api/runtime/fleet_agents")
async def get_runtime_fleet_agents(request: Request) -> dict:
    """List the Python fleet agents + live status (comp-scoped) for the board's fleet-agent manager panel."""
    try:
        return _fleet_agents_state(_active_comp(request))
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": str(exc)}


@app.post("/api/runtime/fleet_agents/{kind}/start")
async def start_fleet_agent(kind: str) -> dict:
    """Power ON a Python fleet agent = dispatch a task of its KIND onto the fleet board; a worker
    from the pool claims + runs it. Enqueue is safe; if the daemon is down it stays queued (with a note)."""
    fa = _fleet_agents_import()
    if fa is None:
        return {"ok": False, "kind": kind, "error": "fleet_agents package could not be imported"}
    if kind not in fa.HANDLERS:
        raise HTTPException(status_code=400, detail=f"Unknown fleet agent: {kind}")
    try:
        from researchpapers.fleet import board as fb
        fb.add("C", kind, f"{kind}: launched from runtime board", {"launched_by": "web-ui"})
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "kind": kind, "error": f"enqueue failed: {exc}"}
    running = _fleet_daemon_running()
    note = ("dispatched — a worker will claim and run it" if running
            else "queued, but the fleet worker daemon is NOT running (start run_fleet.sh) so it won't execute yet")
    return {"ok": True, "kind": kind, "status": "queued", "fleet_running": running, "note": note}


@app.post("/api/runtime/fleet_agents/{kind}/stop")
async def stop_fleet_agent(kind: str) -> dict:
    """Power OFF a Python fleet agent = cancel its pending/claimed board tasks so no worker (re)runs it.
    A task already mid-run in a shared worker finishes its current pass (in-process pool — no hard kill)."""
    import sqlite3
    from pathlib import Path as _P
    fa = _fleet_agents_import()
    if fa is not None and kind not in fa.HANDLERS:
        raise HTTPException(status_code=400, detail=f"Unknown fleet agent: {kind}")
    db = _ce.fleet_db() or (_P(__file__).resolve().parent / "fleet" / "fleet.db")
    if not db.exists():
        return {"ok": False, "kind": kind, "error": "fleet board (fleet.db) not found"}
    try:
        c = sqlite3.connect(str(db), timeout=6)
        cur = c.execute("UPDATE questions SET status='cancelled',claimed_by='',updated=? "
                        "WHERE kind=? AND status IN ('open','claimed')", (now_iso(), kind))
        n = cur.rowcount
        c.commit()
        c.close()
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "kind": kind, "error": str(exc)}
    return {"ok": True, "kind": kind, "status": "cancelled", "cancelled": int(n),
            "note": "pending/claimed board tasks cancelled"}


@app.post("/api/fleet/experiment")
async def enqueue_fleet_experiment(payload: FleetExperimentCreate) -> dict:
    """The LEADER starts an experiment here → it lands on the fleet board → the experiments agent picks it
    up, runs it (screen → predict → score), and writes the journal row AUTOMATICALLY. This is how the
    leader launches a task without anyone hand-writing the journal."""
    from researchpapers.fleet import board as fb
    if not payload.config.strip():
        raise HTTPException(status_code=400, detail="config is required (the yml the researcher authored)")
    q = f"{payload.kind}: {payload.description or payload.config} ({payload.config})"
    fb.add(payload.thread or "C", payload.kind, q,
           {"config": payload.config, "description": payload.description, "approved": True})
    return {"queued": True, "kind": payload.kind, "config": payload.config, "question": q}


@app.post("/api/fleet/experiment_yaml")
async def enqueue_fleet_experiment_yaml(payload: FleetExperimentYaml) -> dict:
    """MANUAL fallback — paste a full YML; it's saved under config/_manual/ and enqueued for the fleet."""
    import yaml as _yaml
    from researchpapers.fleet import board as fb
    from pathlib import Path as _P
    try:
        cfg = _yaml.safe_load(payload.yaml) or {}
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"invalid YAML: {exc}")
    comp = _ce.comp_root() or _P(__file__).resolve().parents[3]    # competition root (RP_COMP_ROOT override)
    name = (cfg.get("train", {}) or {}).get("method") or cfg.get("name") or "manual_exp"
    name = "".join(ch for ch in str(name) if ch.isalnum() or ch in "_-")[:60] or "manual_exp"
    out = comp / "config" / "_manual" / f"{name}.yml"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(payload.yaml)
    rel = str(out.relative_to(comp))
    q = f"{payload.kind}: {payload.description or name} ({rel})"
    fb.add("C", payload.kind, q, {"config": rel, "description": payload.description, "approved": True})
    return {"queued": True, "saved": rel, "kind": payload.kind, "question": q}


def _pg_cv_by_run() -> dict:
    """Pull {run_name: best golden_cv} from the MLflow Postgres DB (mlflowdb) — the journal's scores
    also come from Postgres, not only the jsonl. Best-effort via psql CLI; returns {} on any failure."""
    import subprocess, os
    sql = ("SELECT t.value, MAX(m.value) FROM latest_metrics m "
           "JOIN tags t ON t.run_uuid=m.run_uuid AND t.key='mlflow.runName' "
           "WHERE m.key='golden_cv' GROUP BY t.value;")
    try:
        env = {"PGPASSWORD": "seshu", "PATH": os.environ.get("PATH", "/usr/bin:/bin")}
        r = subprocess.run(["psql", "-U", "postgres", "-h", "localhost", "-d", "mlflowdb", "-tAF", "\t", "-c", sql],
                           capture_output=True, text=True, timeout=5, env=env)
        out = {}
        for line in r.stdout.strip().splitlines():
            if "\t" in line:
                name, val = line.split("\t", 1)
                try:
                    out[name.strip()] = float(val)
                except ValueError:
                    pass
        return out
    except Exception:  # noqa: BLE001
        return {}


# ─────────────────────────── COMPETITION SELECTOR (one comp → all views) ───────────────────────────
# A single active competition drives EVERY data view. Resolved per request: ?comp=<slug> (which the JS
# selector also stores in the rp_comp cookie) → cookie → this board's own comp (biohub). All per-comp
# data is read from Postgres kaggle_<slug> (experiment_journal / decisions / research_index / lb_*),
# with the JSONL/markdown files as a dual-written fallback so biohub keeps rendering unchanged.
def _active_comp(request) -> str:
    from pathlib import Path as _P
    default = _P(__file__).resolve().parents[3].name
    try:
        q = request.query_params.get("comp")
        if q:
            return q
        c = request.cookies.get("rp_comp")
        if c:
            return c
    except Exception:  # noqa: BLE001
        pass
    return default


def _fa_db():
    """The fleet_agents.db module (per-competition Postgres helper), or None. Never raises."""
    try:
        _fleet_agents_import()                              # ensures comp root on sys.path
        import fleet_agents.db as _db  # noqa: PLC0415
        return _db
    except Exception:  # noqa: BLE001
        return None


def _comp_docs_dir(slug):
    from pathlib import Path as _P
    default = _P(__file__).resolve().parents[3]
    return (default / "docs") if slug == default.name else (_P("/home/seshu/kaggle/2026") / slug / "docs")


def _journal_entries(slug):
    """Experiment rows for a competition — Postgres first (source of truth), JSONL file fallback."""
    db = _fa_db()
    if db is not None:
        try:
            rows = db.all_journal(slug)
            if rows:
                return [r for r in rows if isinstance(r, dict) and r.get("exp")]
        except Exception:  # noqa: BLE001
            pass
    jl = _comp_docs_dir(slug) / "experiment_ledger.jsonl"
    out = []
    if jl.exists():
        import json as _j
        for ln in jl.read_text().splitlines():
            try:
                r = _j.loads(ln)
                if r.get("exp"):
                    out.append(r)
            except Exception:  # noqa: BLE001
                pass
    return out


def _decision_rows(slug):
    """Agent decision/finding rows for a competition — Postgres first, JSONL file fallback."""
    db = _fa_db()
    if db is not None:
        try:
            rows = db.all_decisions(slug)
            if rows:
                return rows
        except Exception:  # noqa: BLE001
            pass
    dj = _comp_docs_dir(slug) / "experiment_decisions.jsonl"
    out = []
    if dj.exists():
        import json as _j
        for ln in dj.read_text().splitlines():
            try:
                out.append(_j.loads(ln))
            except Exception:  # noqa: BLE001
                pass
    return out


def _comp_selector_html(slug):
    """A compact competition switcher (sets ?comp + rp_comp cookie, then reloads) shown atop per-comp views."""
    import html as _h
    comps = []
    try:
        comps = [c.get("slug") for c in _active_competitions() if c.get("slug")]
    except Exception:  # noqa: BLE001
        pass
    if slug not in comps:
        comps = [slug] + [c for c in comps if c != slug]
    opts = "".join(f"<option value='{_h.escape(c)}'{' selected' if c == slug else ''}>{_h.escape(c)}</option>"
                   for c in comps)
    return ("<div style='margin:0 0 14px;font:13px -apple-system,Segoe UI,sans-serif'>🎯 <b>Competition:</b> "
            "<select onchange=\"document.cookie='rp_comp='+this.value+';path=/;max-age=31536000';"
            "var u=new URL(location.href);u.searchParams.set('comp',this.value);location.href=u.href\" "
            f"style='padding:5px 9px;border-radius:8px;font:inherit'>{opts}</select> "
            "<span style='color:#8595ad'>— journal · experiments · insights · research · LB all follow this selection</span></div>")


@app.get("/journal", response_class=HTMLResponse)
def experiment_journal(request: Request) -> str:
    """Render the experiment journal — SYNC handler so the blocking :7799 status call runs in a
    threadpool (not the event loop), which keeps the NOW-RUNNING banner reliable under concurrent load."""
    import html as _html
    import re as _re
    from pathlib import Path as _P
    slug = _active_comp(request)                                   # active competition drives this view
    _sel = _comp_selector_html(slug)
    def clamp(text, n=90):
        """Long text → a native <details> 'more' expander (readable, no JS, survives auto-refresh)."""
        t = _html.escape(text)
        if len(text) <= n:
            return t
        return f"<details><summary>{_html.escape(text[:n]).rstrip()}…</summary>{t}</details>"

    # ── NOW RUNNING banner — the ONE job actually on the GPU right now (live from :7799), no confusion ──
    import urllib.request as _u
    banner, run_cfg, run_method = "", "", ""
    try:
        q = __import__("json").loads(_u.urlopen("http://127.0.0.1:7799/api/board", timeout=8).read()).get("queue", {})
        running = [j for j in q.get("recent", []) if j.get("status") == "running"]
        queued = [j for j in q.get("recent", []) if j.get("status") == "queued"]
        qnames = ", ".join((j.get("script_args") or ["?"])[0].split("/")[-1].replace(".yml", "") for j in queued[:6])
        if running:
            j = running[0]
            run_cfg = (j.get("script_args") or [""])[0]              # e.g. config/aug_ablation/31_contrast.yml
            method = (j.get("script_args") or ["?"])[0].split("/")[-1].replace(".yml", "")
            run_method = method
            title = _html.escape((j.get("title") or "")[:80])
            started = str(j.get("started_at", ""))[:19].replace("T", " ")
            det = _html.escape(f"job {j.get('id')} · started {started} · args {j.get('script_args')}")
            banner = (f"<div class='now run'><b>▶ RUNNING NOW</b> — <span class='mono'>{_html.escape(method)}</span> "
                      f"&nbsp;·&nbsp; {title} &nbsp;·&nbsp; <span class='muted'>waiting: {_html.escape(qnames) or 'none'}</span>"
                      f"<details><summary>details</summary>{det}</details></div>")
        else:
            # No GPU job → the fleet may still be actively working (post-proc search / scoring agents).
            # Highlight what the Python fleet is claiming RIGHT NOW so the banner is never falsely "idle".
            fleet_html = _fleet_now_running(slug)
            if fleet_html:
                banner = fleet_html
            else:
                banner = (f"<div class='now idle'><b>■ IDLE</b> — nothing on the GPU or in the fleet right now"
                          f"{(' · waiting: ' + _html.escape(qnames)) if qnames else ' · queue empty'}</div>")
    except Exception:  # noqa: BLE001
        banner = _fleet_now_running(slug) or "<div class='now idle'>■ GPU queue status unavailable (:7799)</div>"

    # ── MAIN JOURNAL — GROUPED BY DATASET (trn_set), each group sorted by CV↓ ──
    # CV across different datasets isn't comparable (mini vs golden12 vs full), so we section by dataset
    # and rank within each. Source of truth = docs/experiment_ledger.jsonl, CV enriched from Postgres (mlflowdb).
    import json as _json
    entries = _journal_entries(slug)                              # Postgres-first (per-comp), file fallback
    if not entries:
        return (f"<!doctype html><meta charset=utf-8><title>Experiment Journal</title>"
                f"<body style='font:15px -apple-system,Segoe UI,sans-serif;padding:26px'>{_sel}"
                f"<h1>📓 Experiment Journal — {_html.escape(slug)}</h1>"
                f"<p style='color:#5b6b86'>No experiments logged yet for <b>{_html.escape(slug)}</b> — "
                f"the journal fills as the fleet runs experiments (writes land in Postgres "
                f"<code>kaggle_{_html.escape(slug).replace('-', '_')}</code> when training runs with "
                f"<code>RP_COMP={_html.escape(slug)}</code>).</p></body>")
    led = {r["change"]: r for r in entries if r.get("change")}   # kept for the decisions table below
    pg_cv = _pg_cv_by_run()                                        # {run_name: golden_cv} from Postgres

    # ── PUBLIC LB BAR — the current bar to beat, pinned at the TOP (LB-only public rows sort to the
    # bottom of a CV-ranked group, so surface them here explicitly). Shows: top public LB + best public
    # NOTEBOOK we can reproduce + our best local CV. Answers "where's the 0.900 public notebook?".
    def _lbf(r):
        v = r.get("lb")
        return float(v) if isinstance(v, (int, float)) and v < 1.0 else None
    pub_lb = [(r, _lbf(r)) for r in entries if r.get("trn_set") == "public" and _lbf(r) is not None]
    pubbar = ""
    if pub_lb:
        top_r, top_lb = max(pub_lb, key=lambda t: t[1])
        nb = [(r, v) for r, v in pub_lb if "public notebook" in (r.get("desc") or "").lower()
              or "kernels pull" in (r.get("script") or "").lower()]
        nb_r, nb_lb = max(nb, key=lambda t: t[1]) if nb else (top_r, top_lb)
        our = [float(r.get("cv")) for r in entries if r.get("trn_set") in ("golden12", "full")
               and r.get("trn_set") != "public" and isinstance(r.get("cv"), (int, float)) and r.get("cv") <= 1.0]
        our_best = max(our) if our else None
        pubbar = (f"<div class='now' style='background:#eef2ff;border-left:4px solid #6366f1'>"
                  f"<b>🎯 PUBLIC BAR TO BEAT</b> — LB top <b>{top_lb:.3f}</b> "
                  f"({_html.escape(clamp(str(top_r.get('desc') or ''), 55))}) &nbsp;·&nbsp; "
                  f"best public NOTEBOOK <b>{nb_lb:.3f}</b> "
                  f"({_html.escape(clamp(str(nb_r.get('desc') or ''), 55))})"
                  + (f" &nbsp;·&nbsp; <span class='muted'>our best local CV {our_best:.4f}</span>" if our_best else "")
                  + "</div>")

    def _cvf(r):
        v = pg_cv.get(r.get("change"))                             # prefer the Postgres-logged CV when present
        if v is None:
            v = r.get("cv")
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    # dataset trust order (most-trusted first); unknown sets sort after these, alphabetically
    ORDER = {"full": 0, "golden12": 1, "loeo": 2, "loeo_mini": 3, "golden4": 4, "ft187": 5,
             "screen_matched": 6, "splits_stagebridge_6bba": 7, "mini": 8, "public": 9}
    # public-notebook experiments are scored on golden-12 → SAME measurement; merge into one golden-12
    # group. This is a biohub-specific scoring fact → only for the home comp; other comps keep "public".
    _ALIAS = {"public": "golden12"} if slug == _home_slug() else {}
    groups: dict = {}
    for r in entries:
        ds = r.get("trn_set") or "(unspecified)"
        groups.setdefault(_ALIAS.get(ds, ds), []).append(r)
    main_html = ""
    for ds in sorted(groups, key=lambda d: (ORDER.get(d, 50), d)):
        grp = sorted(groups[ds], key=lambda r: (_cvf(r) is not None, _cvf(r) if _cvf(r) is not None else -1.0),
                     reverse=True)
        # best-in-group = highest LEGITIMATE CV (≤ 1.0). A golden-CV > 1.0 is the metric's under-count
        # bonus artifact (severe under-detection), NOT a real win — it's flagged, never awarded 🏆.
        best = next((_cvf(r) for r in grp if _cvf(r) is not None and _cvf(r) <= 1.0), None)
        body = ""
        best_shown = False                                        # award 🏆 to the FIRST best only (ties are common)
        for rank, r in enumerate(grp, 1):
            cvf = _cvf(r)
            suspect = cvf is not None and cvf > 1.0                # >1 = under-count artifact
            from_pg = pg_cv.get(r.get("change")) is not None
            cv_disp = f"{cvf:.4f}" if cvf is not None else _html.escape(str(r.get("cv") or "—"))
            lb = r.get("lb")
            # LB is always < 1.0 for this comp; a ≥1.0 value is a bad extraction (e.g. the 1.625 voxel scale) → hide it.
            # When a submission carries explicit Kaggle public/private scores (sub-journal), show BOTH as "public / private".
            def _score_disp(v):
                return f"{v:.4f}" if isinstance(v, (int, float)) and not isinstance(v, bool) and v < 1.0 else None
            _pub = _score_disp(r.get("public")) or _score_disp(lb)
            _prv = _score_disp(r.get("private"))
            if _pub and _prv:
                lb_disp = f"{_pub} / {_prv}"
            elif _pub:
                lb_disp = _pub
            elif lb and not isinstance(lb, (int, float)):
                lb_disp = _html.escape(str(lb))
            else:
                lb_disp = "—"
            script = _html.escape(str(r.get("script") or ""))
            is_run = bool(run_cfg and r.get("script") and run_cfg in str(r.get("script")))
            is_best = cvf is not None and not suspect and cvf == best and not best_shown
            if is_best:
                best_shown = True
            pub = (r.get("trn_set") == "public")
            _rawdesc = str(r.get("desc") or r.get("change") or "")
            _m2 = _re.search(r"2-CV\[([^\]]*)\]", _rawdesc)               # explicit "2-CV[44b6=.. 6bba=..]"
            if _m2:
                twocv = _html.escape(_m2.group(1))
            else:                                                        # fallback: any "44b6 .. 6bba .." phrasing
                _m3 = _re.search(r"44b6[=:\s]+([0-9.]+).*?6bba[=:\s]+([0-9.]+)", _rawdesc)
                twocv = f"44b6={_m3.group(1)} 6bba={_m3.group(2)}" if _m3 else "—"
            desc = clamp(_re.sub(r"2-CV\[[^\]]*\]\s*(mean=[0-9.]+)?\s*—?\s*", "", _rawdesc), 80)
            kept = "✓" if r.get("kept") else ""
            badge = ("<span class='rbadge'>▶ RUNNING</span> " if is_run
                     else ("<span class='warn' title='golden-CV &gt; 1.0 = under-count bonus artifact, not a real win'>⚠︎ &gt;1</span> " if suspect
                           else ("🏆 " if is_best else "")))
            pubtag = " <span class='pubtag'>public</span>" if pub else ""
            cls = "run-row" if is_run else ("suspect-row" if suspect else ("best-row" if is_best else ""))
            gh = _html.escape(str(r.get("git_hash") or "—"))     # commit hash → maps this experiment to its code
            body += (f"<tr class='{cls}'><td class='rk'>{rank}</td><td class='exp'>{_html.escape(str(r.get('exp','?')))}{pubtag}</td>"
                     f"<td class='cv'>{cv_disp}{'<sup class=pg title=from-Postgres>pg</sup>' if from_pg else ''}</td>"
                     f"<td class='cv2 mono'>{twocv}</td>"
                     f"<td class='lb'>{lb_disp}</td><td class='kept'>{kept}</td>"
                     f"<td>{badge}{desc}</td><td class='mono scr'>{script}</td>"
                     f"<td class='git mono' title='code commit'>{gh}</td></tr>")
        best_disp = f"{best:.4f}" if best is not None else "—"
        main_html += (f"<h2 class='ds'>📂 {_html.escape('golden-12' if ds=='golden12' else ds)} <small>{len(grp)} runs · best CV {best_disp}</small></h2>"
                      f"<table class='exp'><thead><tr><th>#</th><th>EXP</th><th>CV ↓</th><th>2-CV (44b6/6bba)</th><th>LB (pub/priv)</th><th>kept</th>"
                      f"<th>Description</th><th>Script</th><th>git</th></tr></thead><tbody>{body}</tbody></table>")

    def _fmt(v):
        if v is None or v == "":
            return "pending"
        return f"{v:.4f}" if isinstance(v, (int, float)) else str(v)

    # AGENT CONTRIBUTIONS — same columns as the experiment journal: EXP | CV | LB | DESCRIPTION | SCRIPT | TRN_SET
    items = _decision_rows(slug)                                  # Postgres-first (per-comp), file fallback
    dec_rows = ""
    if items:
        for e in reversed(items):
            agent = str(e.get("agent", ""))
            kind = _html.escape(str(e.get("kind", "finding")))
            run = str(e.get("run") or "")
            lr = led.get(run, {})
            exp = _html.escape(lr.get("exp") or (run[:16] if run else "—"))
            cv = _html.escape(_fmt(lr.get("cv")))
            lb = _html.escape(_fmt(lr.get("lb")))
            trn = _html.escape(lr.get("trn_set") or "—")   # the run's REAL set, or — (never a phase name)
            summ = str(e.get("summary") or e.get("finding") or "")
            detail = str(e.get("detail") or "")
            rec = str(e.get("recommendation") or "")
            desc = clamp(summ, 90) + (f"<div class='det'>{_html.escape(detail)}</div>" if detail else "")
            desc += f"<div class='rec'>→ {clamp(rec, 80)}</div>" if rec and rec != "None" else ""
            script = f"<span class='agent'>{_html.escape(agent)}</span><span class='badge b-{kind}'>{kind}</span>"
            is_run = bool(run_method and run_method == run)   # highlight the analysis row of the live run
            cls = " class='run-row'" if is_run else ""
            expc = ("▶ " + exp) if is_run else exp
            gh = _html.escape(str(e.get("git_hash") or "—"))     # code commit for this agent contribution
            dec_rows += (f"<tr{cls}><td>{expc}</td><td>{cv}</td><td>{lb}</td>"
                         f"<td>{desc}</td><td>{script}</td><td>{trn}</td>"
                         f"<td class='git mono'>{gh}</td></tr>")
    dec_html = (f"<h2>🧭 Agent analysis &amp; decisions <small>(all agents contribute · newest first · click “…” for detail)</small></h2>"
                f"<table><tr><th>EXP</th><th>CV</th><th>LB</th><th>DESCRIPTION</th><th>SCRIPT</th><th>TRN_SET</th><th>git</th></tr>"
                f"{dec_rows}</table>") if dec_rows else ""
    return (f"<!doctype html><meta charset=utf-8><meta http-equiv=refresh content=20>"
            f"<title>Experiment Journal</title><style>"
            f"body{{font:14px -apple-system,Segoe UI,sans-serif;background:#f4f7fb;color:#152238;padding:26px}}"
            f"h1{{margin:0 0 4px}}h2{{margin:26px 0 8px;font-size:16px}}h2 small{{color:#8595ad;font-weight:400}}"
            f"p{{color:#5b6b86;margin:0 0 16px}}"
            f"table{{border-collapse:collapse;width:100%;background:#fff;border-radius:10px;overflow:hidden;box-shadow:0 1px 3px rgba(0,0,0,.06);margin-bottom:8px}}"
            f"th,td{{text-align:left;padding:8px 11px;border-bottom:1px solid #eef2f8;font-size:13px;vertical-align:top}}"
            f"th{{background:#f1f5fb;color:#5b6b86}}td:first-child{{font-family:ui-monospace,monospace;font-weight:700}}"
            f"td:nth-child(2),td:nth-child(3){{font-family:ui-monospace,monospace;color:#b45309}}"
            f"details summary{{cursor:pointer;color:#334155}}details[open] summary{{color:#0369a1;margin-bottom:4px}}"
            f".rec{{color:#0369a1;font-weight:600;margin-top:4px}}.det{{color:#64748b;margin-top:3px;font-size:12px}}"
            f".agent{{font-weight:600}}.badge{{margin-left:6px;padding:1px 6px;border-radius:8px;font-size:11px;color:#fff}}"
            f".b-finding{{background:#0ea5e9}}.b-decision{{background:#8b5cf6}}.b-verdict{{background:#f59e0b}}.b-data{{background:#10b981}}"
            f".b-pre{{background:#8b5cf6}}.b-running{{background:#f59e0b}}.b-post{{background:#16a34a}}"
            f".now{{padding:11px 14px;border-radius:10px;margin:0 0 16px;font-size:14px}}"
            f".now.run{{background:#052e16;color:#bbf7d0;border:1px solid #16a34a}}"
            f".now.idle{{background:#f1f5f9;color:#64748b;border:1px solid #e2e8f0}}"
            f".now .mono{{font-family:ui-monospace,monospace;font-weight:700;color:#fff}}"
            f".now .muted{{color:#94a3b8}}.now details{{margin-top:5px}}.now summary{{color:#86efac}}"
            f".git{{color:#64748b;font-size:11px;font-family:ui-monospace,monospace;white-space:nowrap}}"
            f".run-row td{{background:#ecfdf5 !important;box-shadow:inset 3px 0 #16a34a}}"
            f".run-row td:first-child{{color:#16a34a}}"
            f".pub-row td{{background:#f5f3ff}}.pub-row td:first-child{{color:#7c3aed}}"
            f".pubtag{{margin-left:6px;padding:1px 6px;border-radius:8px;font-size:10px;background:#7c3aed;color:#fff;font-family:inherit}}"
            # grouped-by-dataset journal styling
            f"h2.ds{{margin:26px 0 6px;font-size:15px;color:#0f172a;border-left:4px solid #4f46e5;padding-left:9px}}"
            f"h2.ds small{{color:#94a3b8;font-weight:400;font-size:12px;margin-left:6px}}"
            f"table.exp{{table-layout:auto}}table.exp thead th{{position:sticky;top:0;background:#eef2ff;color:#3730a3;font-size:11px;text-transform:uppercase;letter-spacing:.03em}}"
            f"table.exp td.rk{{width:30px;text-align:center;color:#94a3b8;font-variant-numeric:tabular-nums}}"
            f"table.exp td.exp{{width:70px;font-weight:600;color:#0369a1}}"
            f"table.exp td.cv{{width:78px;text-align:right;font-family:ui-monospace,monospace;font-weight:700;color:#6d28d9;font-variant-numeric:tabular-nums}}"
            f"table.exp td.lb{{width:64px;text-align:right;font-family:ui-monospace,monospace;color:#b45309}}"
            f"table.exp td.kept{{width:34px;text-align:center;color:#16a34a}}"
            f"table.exp td.scr{{color:#64748b;font-size:11px;max-width:260px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}}"
            f"sup.pg{{color:#2563eb;font-size:8px;margin-left:2px;cursor:help}}"
            f"tr.best-row td{{background:#fffbeb}}tr.best-row td.cv{{color:#b45309}}"
            f"tr.suspect-row td{{background:#fff7ed}}tr.suspect-row td.cv{{color:#c2410c;text-decoration:line-through wavy #f59e0b}}"
            f".warn{{background:#f59e0b;color:#7c2d12;border-radius:9px;padding:1px 6px;font-size:9.5px;font-weight:700;white-space:nowrap}}"
            f".rbadge{{background:#dc2626;color:#fff;border-radius:9px;padding:1px 7px;font-size:9.5px;font-weight:700;letter-spacing:.02em}}"
            f"@media(prefers-color-scheme:dark){{body{{background:#0b1220;color:#e6edf8}}"
            f"h2 small{{color:#8ba0c4}}p{{color:#8ba0c4}}"
            f"table{{background:#111a2e;box-shadow:0 1px 3px rgba(0,0,0,.4)}}"
            f"th,td{{border-bottom-color:#22304d}}th{{background:#1b2745;color:#8ba0c4}}"
            f"td:nth-child(2),td:nth-child(3){{color:#fbbf24}}"
            f"table.exp thead th{{background:#1b2745;color:#a5b4fc}}"
            f"h2.ds{{color:#e6edf8;border-left-color:#8b9bff}}"
            f"tr.best-row td{{background:#2a2410}}tr.suspect-row td{{background:#2a1e10}}"
            f".now.idle{{background:#111a2e;color:#8ba0c4;border-color:#22304d}}"
            f"details summary{{color:#8ba0c4}}.det{{color:#8ba0c4}}}}</style>"
            f"<h1>📓 Experiment Journal — {_html.escape(slug)}</h1>{_sel}"
            f"<p>Grouped by <b>dataset</b>, ranked by <b>CV</b> (best first) · 🏆 best-in-group · "
            f"▶ running highlighted · <sup class=pg>pg</sup> = score from Postgres · failures kept · Auto-refresh 20s.</p>"
            f"{banner}{pubbar}{main_html}{dec_html}")


@app.post("/api/runtime/messages")
async def post_runtime_message(payload: RuntimeBoardMessageCreate) -> dict:
    cfg = _runtime_cfg()
    sender = payload.sender.strip() or "human"
    recipient = payload.recipient.strip()
    content = payload.content.strip()
    if sender != "human" and sender not in cfg.agents:
        raise HTTPException(status_code=400, detail=f"Unknown sender: {sender}")
    if recipient != "all" and recipient not in cfg.agents:
        raise HTTPException(status_code=400, detail=f"Unknown recipient: {recipient}")
    if not content:
        raise HTTPException(status_code=400, detail="Message cannot be empty")
    message_id, task_id = queue_message(
        cfg,
        sender,
        recipient,
        content,
        create_task_record=(sender == "human" and recipient != "all"),
        task_marker=now_iso() if sender == "human" else "",
    )
    logger.info("Runtime board queued message %s -> %s id=%s", sender, recipient, message_id)
    return {"ok": True, "message_id": message_id, "task_id": task_id, "board": _runtime_board_state(limit=120)}


@app.post("/api/runtime/training-callback")
async def post_runtime_training_callback(payload: RuntimeTrainingCallbackCreate) -> dict:
    cfg = _runtime_cfg()
    recipient = payload.notify_agent.strip() or "trainer"
    if recipient not in cfg.agents:
        raise HTTPException(status_code=400, detail=f"Unknown notify_agent: {recipient}")
    lines = [
        f"Training job `{payload.job_id}` finished with status `{payload.status}`.",
        f"Title: {payload.title.strip() or '-'}",
        f"Script: {payload.script_path.strip() or '-'}",
    ]
    if payload.technical_focus:
        lines.append(f"Technical focus: {json.dumps(payload.technical_focus, ensure_ascii=False)}")
    if payload.script_args:
        lines.append(f"Script args: {json.dumps(payload.script_args, ensure_ascii=False)}")
    if payload.workdir.strip():
        lines.append(f"Workdir: {payload.workdir.strip()}")
    if payload.log_path.strip():
        lines.append(f"Log: {payload.log_path.strip()}")
    if payload.notes.strip():
        lines.append(f"Notes: {payload.notes.strip()}")
    # The DETERMINISTIC fleet auto-handles the result (predict→score→journal→metrics-table→INSIGHTS.md),
    # so DON'T ask a Claude agent to inspect + hand-write a result note — that was redundant Claude spend.
    lines.append(
        "FYI only — the fleet auto-scores this run, backfills the journal, and refreshes docs/INSIGHTS.md. "
        "No action needed unless a genuine DECISION or Kaggle submission is required."
    )
    content = "\n".join(lines)
    message_id, task_id = queue_message(
        cfg,
        "system",
        recipient,
        content,
        create_task_record=False,
        linked_task_id=payload.runtime_task_id.strip(),
        task_marker="",
    )
    logger.info(
        "Training callback queued for %s from job=%s status=%s message=%s",
        recipient,
        payload.job_id,
        payload.status,
        message_id,
    )
    return {"ok": True, "message_id": message_id, "task_id": task_id, "board": _runtime_board_state(limit=120)}


@app.post("/api/projects")
async def create_project(payload: ProjectCreate) -> dict:
    project = store.create_project(payload)
    return {"project": project.model_dump(mode="json")}


@app.post("/api/projects/{project_id}/submit")
async def submit_project(project_id: str) -> dict:
    try:
        project = store.submit_project(project_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Project not found")
    return {"project": project.model_dump(mode="json")}


@app.post("/api/projects/{project_id}/leader/start")
async def start_leader_chat(project_id: str) -> dict:
    try:
        project = store.start_leader_chat(project_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Project not found")
    return {"project": project.model_dump(mode="json")}


@app.post("/api/projects/{project_id}/leader/restart")
async def restart_leader_chat(project_id: str) -> dict:
    try:
        store.clear_leader_runtime(project_id, "Leader chat was manually restarted from the web UI.")
        project = store.start_leader_chat(project_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Project not found")
    return {"project": project.model_dump(mode="json")}


@app.get("/api/projects/{project_id}/leader-console")
async def get_leader_console(project_id: str) -> dict:
    try:
        project = store.project_detail(project_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Project not found")

    target = project.get("leader_tmux_target") or ""
    if not target:
        return {"available": False, "output": "", "target": "", "status": "Leader tmux target not available"}

    result = subprocess.run(
        ["tmux", "capture-pane", "-pt", target, "-S", "-120"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        logger.warning("Failed to capture tmux pane for project=%s target=%s", project_id, target)
        reason = result.stderr.strip() or "Failed to capture tmux pane"
        if "no server running" in reason.lower() or "failed to connect" in reason.lower() or "operation not permitted" in reason.lower():
            store.clear_leader_runtime(
                project_id,
                "Leader tmux session is no longer available. Send a new message to restart Codex.",
            )
        return {
            "available": False,
            "output": reason,
            "target": "",
            "status": "Leader tmux target unavailable",
        }

    logger.info("Leader console snapshot requested project=%s target=%s", project_id, target)
    return {
        "available": True,
        "output": result.stdout,
        "target": target,
        "status": "ok",
    }


@app.post("/api/projects/{project_id}/leader-console/send")
async def send_leader_console_input(project_id: str, payload: ConsoleInputRequest) -> dict:
    try:
        project = store.project_detail(project_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Project not found")

    target = project.get("leader_tmux_target") or ""
    if not target:
        logger.info("Auto-starting leader chat before console input project=%s", project_id)
        try:
            project = store.start_leader_chat(project_id)
        except KeyError:
            raise HTTPException(status_code=404, detail="Project not found")
        target = project.leader_tmux_target
    if not target:
        raise HTTPException(status_code=400, detail="Leader tmux target not available")

    content = payload.content.rstrip("\n")
    if not content:
        raise HTTPException(status_code=400, detail="Input cannot be empty")

    try:
        _send_to_tmux_target(project_id, target, content, buffer_prefix="console")
    except HTTPException as exc:
        logger.warning("Console input failed project=%s target=%s detail=%s", project_id, target, exc.detail)
        store.clear_leader_runtime(
            project_id,
            "Leader tmux session went away while sending console input. Restarting leader chat.",
        )
        project = store.start_leader_chat(project_id)
        target = project.leader_tmux_target
        if not target:
            raise exc
        _send_to_tmux_target(project_id, target, content, buffer_prefix="console")
    store.add_message(project_id, "human", f"[console] {content}", recipient=project.get("leader_agent_name") or "leader")
    logger.info("Sent manual console input project=%s target=%s", project_id, target)
    return {"ok": True}


@app.post("/api/projects/{project_id}/messages")
async def add_message(project_id: str, payload: MessageCreate) -> dict:
    try:
        project = store.get_project(project_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Project not found")
    if payload.sender == "human" and payload.recipient == "leader" and not project.leader_tmux_target:
        logger.info("Auto-starting leader chat before thread message project=%s", project_id)
        project = store.start_leader_chat(project_id)
    store.add_message(project_id, payload.sender, payload.content, recipient=payload.recipient)
    if payload.sender == "human" and payload.recipient == "leader":
        target = project.leader_tmux_target
        if target:
            tmux_message = (
                "Human posted a new thread comment. Reply in the shared thread, not only in the terminal.\n\n"
                f"{payload.content}"
            )
            try:
                _send_to_tmux_target(project_id, target, tmux_message, buffer_prefix="thread")
            except HTTPException as exc:
                logger.warning("Thread forward failed project=%s target=%s detail=%s", project_id, target, exc.detail)
                store.clear_leader_runtime(
                    project_id,
                    "Leader tmux session went away. Restarting leader chat and retrying the message.",
                )
                project = store.start_leader_chat(project_id)
                target = project.leader_tmux_target
                if not target:
                    raise exc
                _send_to_tmux_target(project_id, target, tmux_message, buffer_prefix="thread")
            logger.info("Forwarded human thread message to leader tmux project=%s target=%s", project_id, target)
        else:
            logger.info("Leader tmux unavailable for human thread message project=%s", project_id)
    return {"ok": True}


@app.post("/api/projects/{project_id}/docs")
async def add_doc(project_id: str, payload: ArtifactCreate) -> dict:
    try:
        project = store.add_artifact(project_id, payload)
    except KeyError:
        raise HTTPException(status_code=404, detail="Project not found")
    return {"project": project.model_dump(mode="json")}


@app.post("/api/projects/{project_id}/agents/{role}/heartbeat")
async def heartbeat(project_id: str, role: str, event: str = "") -> dict:
    try:
        project = store.heartbeat(project_id, role, event=event)
    except KeyError:
        raise HTTPException(status_code=404, detail="Project not found")
    return {"project": project.model_dump(mode="json")}


@app.post("/api/projects/{project_id}/tasks/{task_id}/complete")
async def complete_task(project_id: str, task_id: str, payload: TaskCompleteRequest, role: str) -> dict:
    try:
        project = store.complete_task(project_id, task_id, role, payload.summary, payload.artifacts)
    except KeyError:
        raise HTTPException(status_code=404, detail="Project not found")
    return {"project": project.model_dump(mode="json")}


@app.post("/api/projects/{project_id}/review/approve")
async def approve_review(project_id: str, payload: ReviewRequest) -> dict:
    try:
        project = store.approve_review(project_id, payload.note)
    except KeyError:
        raise HTTPException(status_code=404, detail="Project not found")
    return {"project": project.model_dump(mode="json")}


@app.post("/api/projects/{project_id}/review/requeue")
async def requeue_review(project_id: str, payload: ReviewRequest) -> dict:
    try:
        project = store.send_back_to_todo(project_id, payload.note)
    except KeyError:
        raise HTTPException(status_code=404, detail="Project not found")
    return {"project": project.model_dump(mode="json")}
