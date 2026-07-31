import glob
import json
import mimetypes
import os
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parent
PORT = 7766
MLFLOW = "http://127.0.0.1:5000"
TRAIN_SERVICE = "http://127.0.0.1:7799"   # the queue runs ONE job at a time
EXPERIMENT = "kaggle-biohub-cell-tracking"


def train_queue():
    """Live queue/running/succeeded/failed counts from the train-service (single-job scheduler)."""
    try:
        with urllib.request.urlopen(f"{TRAIN_SERVICE}/api/board", timeout=4) as r:
            b = json.loads(r.read())
        c, q = b.get("counts", {}), b.get("queue", {})
        return {
            "queued": int(q.get("queued_count", 0)),
            "running": int(q.get("running_count", 0)),
            "succeeded": int(c.get("succeeded", 0)),
            "failed": int(c.get("failed", 0)),
        }, None
    except Exception as e:  # noqa: BLE001
        return None, str(e)

# 0..1 "higher is better" metrics plotted as lines on the shared-axis chart (each its own colour)
CHART_KEYS = ["official_score", "golden_cv", "adj_edge_jaccard", "micro_adjJ", "mean_node_recall",
              "division_jaccard", "score", "recall", "best_score", "acc"]
# colour per metric (stable)
PALETTE = {
    "official_score": "#2563eb", "golden_cv": "#7c3aed", "adj_edge_jaccard": "#0891b2",
    "micro_adjJ": "#0d9488", "mean_node_recall": "#16a34a", "division_jaccard": "#db2777",
    "score": "#ea580c", "recall": "#65a30d", "best_score": "#9333ea", "acc": "#0284c7",
}


def _mlflow(path, payload=None):
    url = f"{MLFLOW}{path}"
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data,
                                 headers={"Content-Type": "application/json"},
                                 method="POST" if data else "GET")
    with urllib.request.urlopen(req, timeout=5) as r:
        return json.loads(r.read())


def mlflow_runs():
    try:
        exp = _mlflow(f"/api/2.0/mlflow/experiments/get-by-name?experiment_name={EXPERIMENT}")
        eid = exp["experiment"]["experiment_id"]
        res = _mlflow("/api/2.0/mlflow/runs/search",
                      {"experiment_ids": [eid], "max_results": 100,
                       "order_by": ["attributes.start_time ASC"]})
        rows = []
        for r in res.get("runs", []):
            i = r["info"]; data = r.get("data", {})
            m = {}
            for x in data.get("metrics", []):
                try:
                    m[x["key"]] = float(x["value"])
                except (TypeError, ValueError):
                    pass
            p = {x["key"]: x["value"] for x in data.get("params", [])}
            rows.append({
                "name": i.get("run_name", i["run_id"][:8]),
                "status": i.get("status", "?"),
                "start": int(i.get("start_time", 0)),
                "end": int(i.get("end_time", 0) or 0),
                "config": p.get("config_file", "-"),
                "downsample": p.get("downsample", "-"),
                "metrics": m,
            })
        return rows, None
    except Exception as e:  # noqa: BLE001
        return [], str(e)


def fmt_dur(ms):
    if not ms or ms < 0:
        return "—"
    s = ms / 1000.0
    if s < 60:
        return f"{s:.0f}s"
    if s < 3600:
        return f"{int(s//60)}m {int(s % 60)}s"
    return f"{int(s//3600)}h {int((s % 3600)//60)}m"


def fmt_ts(ms):
    """epoch-ms -> local 'YYYY-MM-DD HH:MM:SS' (detailed start/end datetime per run)."""
    if not ms or ms <= 0:
        return "—"
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ms / 1000.0))


# display order (important first), then anything else alphabetically. losses grouped last.
METRIC_ORDER = ["official_score", "golden_cv", "adj_edge_jaccard", "micro_adjJ", "mean_node_recall",
                "division_jaccard", "score", "recall", "best_score", "acc",
                "adjJ_6bba", "adjJ_44b6", "mean_count_ratio", "div_tp_total", "n",
                "det_loss", "edge_loss", "test_loss"]
LOSS_KEYS = {"det_loss", "edge_loss", "test_loss"}
COUNT_KEYS = {"n", "div_tp_total"}


def _fmt_val(k, v):
    if k in COUNT_KEYS:
        return f"{v:.0f}"
    return f"{v:.4f}"


# MLflow-style: one chart per metric, grouped into sections
SECTIONS = [
    ("Validation — Golden-CV (vs public 0.890)",
     ["official_score", "golden_cv", "adj_edge_jaccard", "micro_adjJ", "mean_node_recall",
      "adjJ_6bba", "adjJ_44b6"]),
    ("Divisions — the +0.1 lever", ["division_jaccard", "div_tp_total"]),
    ("Training curves", ["score", "recall", "acc", "best_score"]),
    ("Losses (lower = better)", ["det_loss", "edge_loss", "test_loss"]),
    ("Counts / calibration", ["n", "mean_count_ratio"]),
]


def _metric_chart(rows, key):
    """One readable chart for a single metric across runs — zoomed y-axis, value gridlines,
    reference lines for the score metrics, big latest value in the header."""
    pts = [(i, rows[i]["metrics"][key]) for i in range(len(rows)) if key in rows[i]["metrics"]]
    if not pts:
        return ""
    refs = [(0.890, "0.890", "#e11d48"), (0.8708, "base", "#94a3b8")] if key in ("official_score", "golden_cv") else []
    col = PALETTE.get(key, "#f97316" if key in LOSS_KEYS else "#0ea5e9")
    allv = [v for _, v in pts] + [r[0] for r in refs]
    lo, hi = min(allv), max(allv)
    span = (hi - lo) or (abs(hi) or 1.0)
    lo -= span * 0.15
    hi += span * 0.15
    span = hi - lo
    W, H, padL, padR, padT, padB = 900, 240, 54, 56, 14, 40
    plotw, ploth = W - padL - padR, H - padT - padB
    n = len(rows)

    def X(i):
        return padL + plotw * (i / max(1, n - 1))

    def Y(v):
        return padT + ploth * (1 - (v - lo) / span)

    grid = ""
    for t in range(5):
        v = lo + span * t / 4
        y = Y(v)
        grid += (f'<line x1="{padL}" y1="{y:.1f}" x2="{padL+plotw}" y2="{y:.1f}" stroke="#eef2f8"/>'
                 f'<text x="{padL-7}" y="{y+3:.1f}" font-size="9.5" text-anchor="end" fill="#9aa7bd">{v:.3f}</text>')
    for val, label, rc in refs:
        if lo <= val <= hi:
            y = Y(val)
            grid += (f'<line x1="{padL}" y1="{y:.1f}" x2="{padL+plotw}" y2="{y:.1f}" stroke="{rc}" '
                     f'stroke-width="1.3" stroke-dasharray="5 3"/>'
                     f'<text x="{padL+plotw+4}" y="{y+3:.1f}" font-size="9" fill="{rc}">{label}</text>')
    xlabels = "".join(f'<text x="{X(i):.1f}" y="{H-padB+15:.1f}" font-size="8.5" text-anchor="middle" '
                      f'fill="#5b6b86">{rows[i]["name"][:14]}</text>' for i in range(n))
    line = (f'<polyline points="{" ".join(f"{X(i):.1f},{Y(v):.1f}" for i,v in pts)}" '
            f'fill="none" stroke="{col}" stroke-width="2.4"/>') if len(pts) > 1 else ""
    dots = "".join(f'<circle cx="{X(i):.1f}" cy="{Y(v):.1f}" r="3.4" fill="{col}"/>' for i, v in pts)
    latest = pts[-1][1]
    return (f'<div class=card><div class=cardh><span class=dot style="background:{col}"></span>'
            f'<span class=mk>{key}</span><span class=lv>{_fmt_val(key, latest)}</span></div>'
            f'<svg viewBox="0 0 {W} {H}" width="100%">{grid}{line}{dots}{xlabels}</svg></div>')


def _sections(rows):
    present = set()
    for r in rows:
        present |= set(r["metrics"])
    # charts show ML metrics only — drop MLflow system/* telemetry (gpu watts, disk MB, network…)
    present = {k for k in present if not k.startswith("system/")}
    def block(title, keys):
        charts = "".join(_metric_chart(rows, k) for k in keys if k in present)
        if not charts:
            return ""
        nkeys = sum(1 for k in keys if k in present)
        return (f'<details class=sec open><summary>{title} <span class=cnt>{nkeys}</span></summary>'
                f'<div class=grid>{charts}</div></details>')

    out, used = "", set()
    for title, keys in SECTIONS:
        out += block(title, keys)
        used |= set(keys)
    out += block("Other", sorted(present - used))
    return out or '<p class=dim>No metrics yet.</p>'


def trend_images():
    seen = {}
    for pat in ("docs/**/*trend*.png", "docs/**/*trend*.jpg", "output/**/*trend*.png"):
        for p in glob.glob(str(ROOT / pat), recursive=True):
            seen[os.path.relpath(p, ROOT)] = os.path.getmtime(p)
    return sorted(seen, key=lambda r: -seen[r])


def _journal_html():
    """Render the grandmaster experiment journal (docs/experiment_ledger.md) as an HTML table."""
    import html as _h
    md = ROOT.parent.parent / "docs" / "experiment_ledger.md"
    if not md.exists():
        return "<p class=note>No experiments logged yet — the journal fills as the fleet runs experiments.</p>"
    rows = ""
    for ln in md.read_text().splitlines():
        ln = ln.strip()
        if not ln.startswith("|") or set(ln) <= set("|:- "):
            continue
        cells = [c.strip().strip("`") for c in ln.strip("|").split("|")]
        tag = "th" if cells and cells[0] == "EXP" else "td"
        rows += "<tr>" + "".join(f"<{tag}>{_h.escape(c)}</{tag}>" for c in cells) + "</tr>"
    return f"<table class=journal>{rows}</table>" if rows else "<p class=note>journal empty</p>"


BIOHUB_SLUG = "biohub-cell-tracking-during-development"


def _comp_journal_rows(slug):
    """Experiment-journal rows (cv/lb over time) for ANY competition, from its Postgres kaggle_<slug>."""
    import sys as _sys
    for p in ("/home/seshu/kaggle/2026/biohub-cell-tracking-during-development",
              "/home/seshu/kaggle/2026/biohub-cell-tracking-during-development/tools/researchpapers"):
        if p not in _sys.path:
            _sys.path.insert(0, p)
    try:
        from fleet_agents import db
        return db.all_journal(slug) or []
    except Exception:
        return []


def render_comp_trend(slug):
    """Comp-scoped metric-trend page (cv/lb per experiment) read from Postgres experiment_journal.
    MLflow run history is biohub-specific, so for other competitions this is the honest metric view."""
    import html as _h
    rows = _comp_journal_rows(slug)

    def _f(v):
        try:
            return f"{float(v):.4f}"
        except (TypeError, ValueError):
            return _h.escape(str(v)) if v not in (None, "") else "—"
    body = ""
    for r in rows:
        cv, lb = r.get("cv"), r.get("lb")
        body += (f"<tr><td class=rn>{_h.escape(str(r.get('exp') or ''))}</td>"
                 f"<td class=cf>{_h.escape(str(r.get('change') or r.get('trn_set') or ''))}</td>"
                 f"<td class=cf>{_h.escape(str(r.get('trn_set') or ''))}</td>"
                 f"<td class=vv><b>{_f(cv)}</b></td><td class=vv><b class=lb>{_f(lb)}</b></td>"
                 f"<td>{_h.escape(str(r.get('desc') or '')[:160])}</td></tr>")
    table = (f"<table class=journal><tr><th>EXP</th><th>CHANGE</th><th>TRN_SET</th><th>CV</th>"
             f"<th>LB</th><th>DESCRIPTION</th></tr>{body}</table>" if body else
             "<p class=note>No experiments logged yet for this competition — the journal fills as the "
             "fleet runs experiments (writes land in Postgres when training runs with RP_COMP=&lt;slug&gt;).</p>")
    return (f"""<!doctype html><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1">
<title>Metric trend — {_h.escape(slug)}</title>
<style>body{{font:14px -apple-system,Segoe UI,sans-serif;background:#f4f7fb;color:#152238;margin:0;padding:24px;max-width:1080px;margin:auto}}
h1{{font-size:20px;margin:0 0 4px}}.sub{{color:#5b6b86;font-size:13px;margin-bottom:16px}}
table{{border-collapse:collapse;width:100%;background:#fff;border-radius:10px;overflow:hidden;box-shadow:0 1px 3px rgba(0,0,0,.06)}}
th,td{{text-align:left;padding:7px 11px;border-bottom:1px solid #eef2f8;font-size:13px}}th{{background:#f1f5fb;color:#5b6b86}}
.rn{{font-family:ui-monospace,monospace}}.vv{{font-variant-numeric:tabular-nums}}.lb{{color:#b45309}}
a{{color:#0369a1}}.note{{color:#5b6b86}}
@media(prefers-color-scheme:dark){{body{{background:#0b1220;color:#e6edf8}}table{{background:#111a2e}}th{{background:#1b2745;color:#8ba0c4}}th,td{{border-bottom-color:#22304d}}a{{color:#8b9bff}}}}</style>
<h1>📈 Metric trend — {_h.escape(slug)}</h1>
<div class=sub>CV / LB per experiment, from Postgres <code>{_h.escape(slug).replace('-', '_')}</code> ·
<a href="//{{h}}:7788/journal?comp={_h.escape(slug)}">journal</a> ·
<a href="//{{h}}:7788/experiments?comp={_h.escape(slug)}">experiments</a> ·
<a href="//{{h}}:7777/?comp={_h.escape(slug)}">hub</a></div>
{table}
<script>document.querySelectorAll('a[href*="{{h}}"]').forEach(function(a){{a.href=a.getAttribute('href').split('{{h}}').join(location.hostname);}});</script>
""").encode("utf-8")


def render_index():
    rows, err = mlflow_runs()
    now_ms = time.time() * 1000
    total_ms = 0.0
    trows = []
    # table shows LATEST run first (rows arrive oldest-first — that order is kept for the charts' x-axis)
    for r in sorted(rows, key=lambda x: x["start"], reverse=True):
        m = r["metrics"]
        dur = (r["end"] or now_ms) - r["start"] if r["start"] else 0
        total_ms += max(0, dur)
        valid = next((m[k] for k in ("official_score", "golden_cv") if k in m), None)
        lb = m.get("lb_score")
        # ALL metrics dumped (validation, training, whatever the run has)
        allm = " ".join(f'<span class=mtag>{k}={float(v):.4f}</span>' for k, v in sorted(m.items()))
        badge = "run" if r["status"] == "RUNNING" else ("ok" if r["status"] == "FINISHED" else "x")
        val_cell = f'<b>{valid:.4f}</b>' if valid is not None else '<span class=dim>—</span>'
        lb_cell = f'<b class=lb>{lb:.4f}</b>' if lb is not None else '<span class=dim>—</span>'
        trows.append(
            f'<tr><td class=rn>{r["name"]}</td>'
            f'<td><span class="st {badge}">{r["status"]}</span></td>'
            f'<td class=cf>{r["config"]}</td><td class=cf>{r["downsample"]}</td>'
            f'<td class=ts>{fmt_ts(r["start"])}</td>'
            f'<td class=ts>{"running" if r["status"] == "RUNNING" else fmt_ts(r["end"])}</td>'
            f'<td class=dur>{fmt_dur(dur)}</td>'
            f'<td class=vv>{val_cell}</td><td class=vv>{lb_cell}</td>'
            f'<td>{allm or "<span class=dim>— training… —</span>"}</td></tr>')
    # live training-queue status (single-job scheduler on :7799) — queued / running / done / failed
    tq, tqerr = train_queue()
    if tq:
        queue_pills = (
            f'<span class="qp run">▶ running {tq["running"]}</span>'
            f'<span class="qp queue">⏳ queued {tq["queued"]}</span>'
            f'<span class="qp ok">✓ succeeded {tq["succeeded"]}</span>'
            f'<span class="qp fail">✗ failed {tq["failed"]}</span>'
            f'<span class=qnote>· scheduler runs one at a time</span>')
    else:
        queue_pills = f'<span class=qnote>train-service (:7799) unreachable: {tqerr}</span>'
    summary = (f'{queue_pills}<br>'
               f'<span class=pill>{len(rows)} MLflow runs</span> '
               f'<span class=pill>total compute {fmt_dur(total_ms)}</span>')
    # TOP: the runs table (run/status/config/ds/time/validation/LB/all metrics). Charts go at the BOTTOM.
    table_html = (f'<p class=err>MLflow query failed: {err}</p>' if err else
                  (f'<p class=sumline>{summary}</p>'
                   f'<table><thead><tr><th>run</th><th>status</th><th>config</th><th>ds</th>'
                   f'<th>start</th><th>end</th><th>time</th><th>validation</th><th>LB</th><th>all metrics</th></tr></thead>'
                   f'<tbody>{"".join(trows)}</tbody></table>'
                   if rows else '<p class=dim>No runs yet.</p>'))
    # BOTTOM: per-metric SVG charts — ONE metric per chart, each with its own y-scale (never mixed)
    charts_html = ('' if err else _sections(rows)) if rows else ''
    # after-the-fact version PNGs
    pngs = "".join(f'<figure><figcaption>{rp}</figcaption>'
                   f'<a href="/img/{rp}" target=_blank><img src="/img/{rp}"></a></figure>'
                   for rp in trend_images())
    png_section = pngs or ('<p class=dim>No cross-version PNG yet — the trainer writes '
                           '<code>docs/baseline_vX_to_vY_top3_trend.png</code> after a version completes.</p>')
    return f"""<!doctype html><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<meta http-equiv=refresh content=15>
<title>biohub — training trend</title>
<style>
 body{{font:15px/1.5 -apple-system,Segoe UI,Roboto,sans-serif;background:#f4f7fb;color:#152238;margin:0;padding:26px}}
 h1{{margin:0 0 2px}} h2{{margin:26px 0 10px;font-size:16px}} .sub{{color:#5b6b86;margin:0 0 18px}}
 .note{{font-size:12px;font-weight:400;color:#8493a8}}
 table{{border-collapse:collapse;width:100%;background:#fff;border-radius:10px;overflow:hidden;box-shadow:0 1px 3px rgba(0,0,0,.05)}}
 th,td{{text-align:left;padding:9px 12px;border-bottom:1px solid #eef2f8;font-size:13px;vertical-align:top}}
 th{{background:#f1f5fb;color:#5b6b86;font-weight:600}}
 .rn{{font-family:ui-monospace,monospace;font-weight:600}} .cf{{font-family:ui-monospace,monospace;color:#475}}
 .mtag{{display:inline-block;background:#eef4ff;color:#1d4ed8;border-radius:6px;padding:1px 7px;margin:1px 3px 1px 0;font-family:ui-monospace,monospace;font-size:12px}}
 .st{{border-radius:20px;padding:1px 9px;font-size:11px;font-weight:600;color:#fff}}
 .st.run{{background:#f59e0b}} .st.ok{{background:#16a34a}} .st.x{{background:#94a3b8}}
 .dim{{color:#94a3b8}} .err{{color:#b91c1c}} .chart{{background:#fff;border-radius:10px;padding:14px;margin-bottom:14px;box-shadow:0 1px 3px rgba(0,0,0,.05)}}
 .dur{{font-variant-numeric:tabular-nums;color:#475}} .vv b{{font-variant-numeric:tabular-nums}} .lb{{color:#b45309}}
 .ts{{font-family:ui-monospace,monospace;font-size:12px;color:#475;white-space:nowrap;font-variant-numeric:tabular-nums}}
 .pill{{display:inline-block;background:#e8eefc;color:#1e40af;border-radius:20px;padding:2px 11px;margin-right:6px;font-size:12px;font-weight:600}} .sumline{{margin:0 0 12px}}
 .qp{{display:inline-block;border-radius:20px;padding:3px 12px;margin:0 6px 6px 0;font-size:13px;font-weight:700;color:#fff}}
 .qp.run{{background:#f59e0b}} .qp.queue{{background:#6366f1}} .qp.ok{{background:#16a34a}} .qp.fail{{background:#dc2626}}
 .qnote{{color:#8493a8;font-size:12px;font-weight:400}}
 details.sec{{margin:18px 0 4px}}
 details.sec>summary{{font-size:14px;font-weight:700;color:#334;margin:0 0 10px;padding:8px 4px;border-bottom:1px solid #e2e8f2;cursor:pointer;list-style:none;user-select:none}}
 details.sec>summary::-webkit-details-marker{{display:none}}
 details.sec>summary::before{{content:"▾ ";color:#94a3b8;font-size:12px}}
 details.sec:not([open])>summary::before{{content:"▸ "}}
 details.sec:not([open])>summary{{color:#5b6b86}}
 .cnt{{display:inline-block;background:#eef2f9;color:#5b6b86;border-radius:20px;padding:0 8px;font-size:11px;font-weight:600;margin-left:4px}}
 .grid{{display:flex;flex-direction:column;gap:14px}}
 .card{{background:#fff;border:1px solid #e8edf5;border-radius:12px;padding:12px 16px;box-shadow:0 1px 2px rgba(0,0,0,.04);max-width:960px}}
 .cardh{{display:flex;align-items:center;gap:7px;margin-bottom:4px}}
 .dot{{width:9px;height:9px;border-radius:50%;flex:none}}
 .mk{{font-family:ui-monospace,monospace;font-size:12.5px;color:#475;font-weight:600}}
 .lv{{margin-left:auto;font-size:17px;font-weight:700;font-variant-numeric:tabular-nums;color:#1e293b}}
 figure{{background:#fff;border:1px solid #e2e8f2;border-radius:12px;padding:14px;margin:0 0 18px}}
 figcaption{{font-family:ui-monospace,monospace;font-size:12px;font-weight:600;margin-bottom:8px}} img{{max-width:100%;border-radius:8px;display:block}}
 code{{background:#eef2f9;padding:1px 6px;border-radius:5px;font-size:12px}}
 table.journal td:first-child,table.journal th:first-child{{font-family:ui-monospace,monospace;font-weight:700;white-space:nowrap}}
 table.journal td:nth-child(2),table.journal td:nth-child(3){{font-family:ui-monospace,monospace;color:#b45309;font-weight:600}}
 table.journal td{{font-size:12.5px}}
</style>
<h1>biohub — training trend</h1>
<p class=sub>Live from MLflow ({EXPERIMENT}) · auto-refresh 15s</p>
<h2>📓 Experiment journal <span class=note>— grandmaster-style (one change per row · CV &amp; LB separate · failures kept; icecube-journal / rna-stanford)</span></h2>
{_journal_html()}
<h2>Cross-version trend images</h2>
{png_section}
<h2>Runs</h2>
{table_html}
<h2>Trend charts <span class=note>— one metric per chart, each with its own y-scale (values are never mixed across scales)</span></h2>
{charts_html}
""".encode("utf-8")


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        from urllib.parse import parse_qs
        qs = parse_qs(self.path.split("?", 1)[1]) if "?" in self.path else {}
        comp = (qs.get("comp", [""])[0] or "").strip()
        path = unquote(self.path.split("?", 1)[0])
        if path in ("/", "/index.html"):
            # ?comp=<other-slug> → comp-scoped metric trend from Postgres; biohub/default → full MLflow view
            data = render_comp_trend(comp) if (comp and comp != BIOHUB_SLUG) else render_index()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        if path.startswith("/img/"):
            full = (ROOT / path[len("/img/"):]).resolve()
            if str(full).startswith(str(ROOT)) and full.is_file():
                self.send_response(200)
                self.send_header("Content-Type", mimetypes.guess_type(str(full))[0] or "application/octet-stream")
                data = full.read_bytes()
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                return
        self.send_response(404)
        self.end_headers()
        self.wfile.write(b"not found")


if __name__ == "__main__":
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
