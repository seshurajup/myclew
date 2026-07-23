"""metric-probe — REUSABLE (any competition) adversarial METRIC-VULNERABILITY prober. Given a competition's
official scorer (`score_fn(pred)->float`) and a baseline prediction, it systematically applies a library of
STRUCTURAL perturbations that change the graph/prediction WITHOUT improving real correctness, and reports each
one with the exact score delta it produced. Any |delta| above a tolerance is flagged as an exploitable
degeneracy, with the inferred BUG CLASS.

Why it exists (grandmaster discipline — NOT for silently submitting exploits):
  (a) understand WHY a public LB is unreliable (which degeneracy the top scores are riding),
  (b) guard our OWN CV against a degenerate metric before we trust it,
  (c) report the metric bug to the organizers.

Comp-agnostic core:
  probe(score_fn, baseline_pred, perturbations, tol) -> ranked report
      score_fn(pred)->float ; perturbations = [named callables pred->pred'].
  Nothing hardcodes 'biohub' or any path — the scorer + representation are INJECTED.

Graph-perturbation library (generic over a node/edge representation, but the biohub-relevant set):
  offvolume_fork_inject(k) · hub_unify_fpfree() · garbage_edges(n) · node_sparsify(frac)
  + the composed division_credit_hack(k) (hub-unify + off-volume forks) that reproduces the proven biohub
  division exploit. They operate on a plain submission-style dict graph (DictGraph, zero deps) OR a tracksdata
  graph (TracksDataAdapter, lazy import) — the same perturbation code drives both.

Inferred bug classes:
  "unmatched-prediction-not-penalized" — adding prediction structure that matches no GT is free (no FP term),
  "global-reachability-credit"         — a credit granted for global weak-reachability (a hub unifies all),
  "under-prediction-bonus"             — a size-ratio adjustment that REWARDS predicting fewer nodes.

A BaseAgent with its own data-wise test (a toy score_fn with a KNOWN degeneracy → the prober must detect it and
quantify the delta). The biohub end-to-end proof (research/metric_probe_check.py) wires the REAL official scorer.
"""
from __future__ import annotations
import random
from .base import BaseAgent

# far-away off-volume coordinate: outside any real acquisition volume, so injected nodes match NO GT node
_FAR = -10000
TOL = 1e-6


# ───────────────────────────── generic graph adapters ─────────────────────────────
class DictGraph:
    """Zero-dependency submission-style node/edge graph. nodes: {id: attrs}; edges: [(src,tgt)]. This is the
    generic representation the perturbation library targets; the unit test uses it with no heavy deps."""

    def __init__(self, nodes=None, edges=None):
        self.nodes = {int(k): dict(v) for k, v in (nodes or {}).items()}
        self.edges = [(int(s), int(t)) for s, t in (edges or [])]
        self._next = (max(self.nodes) + 1) if self.nodes else 0
        self.meta = {}                                  # scratch (e.g. the last hub id) — carried by copy()

    def copy(self):
        g = DictGraph({k: dict(v) for k, v in self.nodes.items()}, list(self.edges))
        g._next = self._next; g.meta = dict(self.meta); return g

    def add_node(self, attrs=None):
        nid = self._next; self._next += 1; self.nodes[nid] = dict(attrs or {}); return nid

    def add_edge(self, s, t):
        self.edges.append((int(s), int(t)))

    def node_ids(self):
        return list(self.nodes)

    def out_degree(self, nid):
        return sum(1 for s, _ in self.edges if s == nid)

    def in_degree(self, nid):
        return sum(1 for _, t in self.edges if t == nid)

    def leaves(self):
        return [n for n in self.nodes if self.out_degree(n) == 0]

    def roots(self):
        return [n for n in self.nodes if self.in_degree(n) == 0]

    def remove_nodes(self, ids):
        ids = set(int(i) for i in ids)
        for i in ids:
            self.nodes.pop(i, None)
        self.edges = [(s, t) for s, t in self.edges if s not in ids and t not in ids]

    def num_nodes(self):
        return len(self.nodes)

    def num_edges(self):
        return len(self.edges)


class TracksDataAdapter:
    """Thin adapter over a tracksdata graph exposing the same interface the perturbation library needs. Lazy —
    tracksdata is imported only when this class is instantiated, so the generic core/unit-test need no deps."""

    def __init__(self, g):
        self.g = g
        self.meta = {}

    def copy(self):
        a = TracksDataAdapter(self.g.copy()); a.meta = dict(self.meta); return a

    def add_node(self, attrs=None):
        return int(self.g.add_node(dict(attrs or {})))

    def add_edge(self, s, t):
        self.g.add_edge(int(s), int(t), {})

    def node_ids(self):
        return [int(n) for n in self.g.node_ids()]

    def _deg(self, direction):
        ids = self.node_ids()
        fn = self.g.out_degree if direction == "out" else self.g.in_degree
        return dict(zip(ids, fn(ids)))

    def out_degree(self, nid):
        return int(self._deg("out").get(int(nid), 0))

    def in_degree(self, nid):
        return int(self._deg("in").get(int(nid), 0))

    def leaves(self):
        d = self._deg("out"); return [n for n, v in d.items() if v == 0]

    def roots(self):
        d = self._deg("in"); return [n for n, v in d.items() if v == 0]

    def remove_nodes(self, ids):
        self.g.bulk_remove_nodes([int(i) for i in ids])

    def num_nodes(self):
        return int(self.g.num_nodes())

    def num_edges(self):
        return int(self.g.num_edges())


def as_graph(pred):
    """Coerce a prediction into a perturbable adapter. Passes DictGraph/TracksDataAdapter through; wraps a raw
    tracksdata graph (has add_node+node_ids+copy) in TracksDataAdapter; builds a DictGraph from a
    {'nodes':..,'edges':..} dict."""
    if isinstance(pred, (DictGraph, TracksDataAdapter)):
        return pred
    if isinstance(pred, dict) and "nodes" in pred:
        return DictGraph(pred.get("nodes"), pred.get("edges"))
    if all(hasattr(pred, m) for m in ("add_node", "add_edge", "node_ids", "copy")):
        return TracksDataAdapter(pred)
    raise TypeError(f"metric-probe: cannot interpret prediction of type {type(pred).__name__} as a graph")


# ───────────────────────────── perturbation library (generic; named + bug-tagged) ─────────────────────────────
def _tag(fn, name, bug, direction="any"):
    """direction = which score move is an EXPLOIT. 'any' (additive off-volume injections add nothing that matches
    GT ⇒ any nonzero move is degenerate) or 'positive' (node-drop: a score DROP is legitimate loss of correctness,
    only a RISE — the under-prediction bonus — is the degeneracy)."""
    fn.perturb_name = name; fn._bug_class = bug; fn._exploit_direction = direction; return fn


def _far_attrs(t, dz=0):
    return {"t": t, "z": _FAR + dz, "y": _FAR, "x": _FAR}


def hub_unify_fpfree(far_t=-1000):
    """Add ONE off-volume 'hub' node and attach every real LEAF (out-degree 0) to it via a leaf→hub edge. On a
    metric that credits GLOBAL weak-reachability + does not penalize unmatched prediction structure, this fuses
    all tracks into one weakly-connected component at ZERO cost (leaf→hub edge is unmatched ⇒ not counted FP).
    Records the hub id in g.meta['hub'] so a following fork-inject can anchor to it."""
    def f(g):
        g = g.copy()
        hub = g.add_node(_far_attrs(far_t))
        for lf in g.leaves():
            if lf != hub:
                g.add_edge(lf, hub)
        g.meta["hub"] = hub
        return g
    return _tag(f, "hub_unify_fpfree", "global-reachability-credit")


def offvolume_fork_inject(k=5, anchor=None, far_t=-999):
    """Inject k off-volume fake 'forks' (a divider node -> {child, continuation}), CHAINED so they are mutually
    reachable, anchored to `anchor` (or g.meta['hub'] if present, else a root). Every injected node is off-volume
    ⇒ matches no GT ⇒ excluded from division-FP; if the anchor is globally reachable, each fake divider becomes
    weakly-reachable from the matched nodes → fake division credit."""
    def f(g):
        g = g.copy()
        prev = anchor if anchor is not None else g.meta.get("hub")
        if prev is None:
            rts = g.roots(); prev = rts[0] if rts else (g.node_ids()[0] if g.node_ids() else g.add_node(_far_attrs(far_t - 1)))
        for i in range(int(k)):
            t = far_t + 2 * i
            d = g.add_node(_far_attrs(t))
            c = g.add_node(_far_attrs(t + 1))
            k2 = g.add_node(_far_attrs(t + 1, dz=1))
            g.add_edge(prev, d); g.add_edge(d, c); g.add_edge(d, k2)
            prev = k2
        return g
    return _tag(f, f"offvolume_fork_inject(k={k})", "global-reachability-credit")


def garbage_edges(n=4, far_t=-2000):
    """Add n edges between fresh OFF-VOLUME nodes (endpoints match no GT node). On a metric that does not count
    an edge whose endpoints match no GT as a false positive, these are FREE — a probe for the
    'unmatched-prediction-not-penalized' degeneracy (the score should NOT move; if a naive metric counts them as
    TP or the node-ratio shifts, it will)."""
    def f(g):
        g = g.copy()
        for i in range(int(n)):
            a = g.add_node(_far_attrs(far_t - 2 * i))
            b = g.add_node(_far_attrs(far_t - 2 * i + 1))
            g.add_edge(a, b)
        return g
    return _tag(f, f"garbage_edges(n={n})", "unmatched-prediction-not-penalized")


def node_sparsify(frac=0.5, seed=0):
    """Remove a fraction of nodes (and their incident edges). On a metric with a size-ratio BONUS
    (adj = J·(1 - α·(N_pred - N_est)/N_est)), dropping nodes makes the ratio more negative → the adjusted score
    can go UP even though fewer things are predicted — the 'under-prediction-bonus' degeneracy."""
    def f(g):
        g = g.copy()
        ids = g.node_ids()
        rng = random.Random(seed)
        drop = rng.sample(ids, int(frac * len(ids))) if ids else []
        g.remove_nodes(drop)
        return g
    return _tag(f, f"node_sparsify(frac={frac})", "under-prediction-bonus", direction="positive")


def division_credit_hack(k=5, far_t=-1000):
    """The COMPOSED, proven biohub division exploit: FP-free hub-unify THEN k off-volume forks anchored to the
    hub. Drives division_jaccard 0→1 on a missed-every-division prediction (measured +~0.098 on the official
    metric) while leaving real correctness unchanged. Exposed as ONE perturbation so probe() reports its full
    delta; the atomic hub_unify_fpfree / offvolume_fork_inject remain available for isolating each mechanism."""
    hub = hub_unify_fpfree(far_t=far_t)
    forks = offvolume_fork_inject(k=k, far_t=far_t + 1)

    def f(g):
        return forks(hub(g))
    return _tag(f, f"division_credit_hack(k={k})", "global-reachability-credit")


# the default probe set for a GRAPH/tracking metric (the biohub-relevant modality). Reusable: pass your own list.
def graph_perturbation_suite(k=5, garbage=4, sparsify=0.5):
    return [hub_unify_fpfree(), offvolume_fork_inject(k=k), garbage_edges(n=garbage),
            node_sparsify(frac=sparsify), division_credit_hack(k=k)]


# ───────────────────────────── the generic CORE prober ─────────────────────────────
_BUG_KEYWORDS = (("fork", "global-reachability-credit"), ("hub", "global-reachability-credit"),
                 ("garbage", "unmatched-prediction-not-penalized"), ("sparsify", "under-prediction-bonus"),
                 ("division_credit", "global-reachability-credit"))


def _bug_class(fn, name):
    bc = getattr(fn, "_bug_class", None)
    if bc:
        return bc
    low = (name or "").lower()
    for kw, cls in _BUG_KEYWORDS:
        if kw in low:
            return cls
    return "unknown-degeneracy"


def _name_of(fn, i):
    return getattr(fn, "perturb_name", None) or getattr(fn, "__name__", None) or f"perturbation_{i}"


def probe(score_fn, baseline_pred, perturbations, tol=TOL, wrap=True):
    """GENERIC core. score_fn(pred)->float ; perturbations = [callable pred->pred'] (named via .perturb_name).
    Applies each perturbation to a COPY of the baseline, scores it, and ranks by |delta|. Flags |delta|>tol as an
    exploitable degeneracy and infers the bug class. Never raises on a single perturbation's failure — records the
    error and continues. Returns {base_score, results[sorted], n_exploits, verdict}.

    `wrap`: coerce predictions to a perturbable adapter (as_graph). Set False if perturbations act on raw preds."""
    base_g = as_graph(baseline_pred) if wrap else baseline_pred
    base_score = float(score_fn(base_g))
    results = []
    for i, fn in enumerate(perturbations):
        name = _name_of(fn, i)
        try:
            src = (as_graph(baseline_pred) if wrap else baseline_pred)
            src = src.copy() if hasattr(src, "copy") else src
            perturbed = fn(src)
            s = float(score_fn(perturbed))
            delta = s - base_score
            direction = getattr(fn, "_exploit_direction", "any")
            # an EXPLOIT is a move the perturbation should NOT cause: 'any'-direction perturbations inject only
            # off-volume/unmatched structure (no real gain) so any nonzero move is degenerate; 'positive'-direction
            # perturbations remove real prediction (a DROP is legitimate) so only a score RISE is the degeneracy.
            is_exploit = abs(delta) > tol and (direction == "any" or (direction == "positive" and delta > tol))
            results.append({"name": name, "base_score": round(base_score, 6),
                            "perturbed_score": round(s, 6), "delta": round(delta, 6),
                            "exploit": is_exploit,
                            "bug_class": _bug_class(fn, name) if is_exploit else None})
        except Exception as e:  # noqa: BLE001 — a perturbation that can't apply is not an exploit; record + move on
            results.append({"name": name, "base_score": round(base_score, 6), "perturbed_score": None,
                            "delta": None, "exploit": False, "error": f"{type(e).__name__}: {str(e)[:120]}"})
    results.sort(key=lambda r: (abs(r["delta"]) if r.get("delta") is not None else -1.0), reverse=True)
    exploits = [r for r in results if r["exploit"]]
    if exploits:
        top = exploits[0]
        verdict = (f"DEGENERATE — {len(exploits)} exploit(s); largest `{top['name']}` moves the score "
                   f"{top['delta']:+.4f} ({top['bug_class']}) with NO gain in real correctness")
    else:
        verdict = "robust — no probed perturbation moved the score beyond tolerance"
    return {"base_score": round(base_score, 6), "results": results, "n_exploits": len(exploits),
            "bug_classes": sorted({r["bug_class"] for r in exploits if r.get("bug_class")}), "verdict": verdict}


def report_md(rep, title="METRIC-PROBE"):
    """Compact degeneracy report (markdown) — which perturbation moved the metric, by how much, and the bug class."""
    rows = []
    for r in rep["results"]:
        d = "—" if r.get("delta") is None else f"{r['delta']:+.4f}"
        flag = "⚠️ EXPLOIT" if r["exploit"] else ("err" if r.get("error") else "ok")
        rows.append(f"| `{r['name']}` | {r.get('perturbed_score', '—')} | {d} | {flag} | {r.get('bug_class') or ''} |")
    body = "\n".join(rows)
    return (f"**{title}** — base score `{rep['base_score']}` · {rep['verdict']}\n"
            f"| perturbation | perturbed | Δscore | flag | bug class |\n|:-|--:|--:|:-|:-|\n{body}")


# ───────────────────────────── biohub OFFICIAL-scorer bridge (comp-parameterized, lazy) ─────────────────────────────
def biohub_score_fn(gt, scale, n_total, max_distance=7.0, term="score"):
    """Build a score_fn(pred_adapter)->float over the OFFICIAL biohub metric (evaluate → per_sample_metrics →
    summarise). `term` selects what to return: 'score' (adj_edge + 0.1·div), 'division_jaccard', 'edge_jaccard',
    'adj_edge_jaccard'. Copies the predicted graph before evaluate (evaluate mutates in place). tracksdata/metrics
    are imported lazily here so the generic core never depends on them."""
    from tracking_cellmot.metrics import evaluate, per_sample_metrics, summarise, node_recall  # lazy

    def score(pred):
        g = pred.g if isinstance(pred, TracksDataAdapter) else pred
        g = g.copy()                                                    # evaluate() writes matching attrs in place
        er = evaluate(g, gt, scale=scale, max_distance=max_distance)
        row = per_sample_metrics(er=er, n_total=n_total, node_recall=node_recall(g, gt))
        s = summarise([row])
        if term == "division_jaccard":
            v = s["division_jaccard"]
        elif term == "edge_jaccard":
            v = row["edge_jaccard"]
        elif term == "adj_edge_jaccard":
            v = row["adj_edge_jaccard"]
        else:
            v = s["score"]
        return 0.0 if v != v else float(v)                              # NaN (no divisions) → 0.0
    return score


# ───────────────────────────── self-contained synthetic demo (no deps) ─────────────────────────────
def _toy_reachability_metric(n_real=6):
    """A toy metric with the SAME two degeneracies as the real biohub bug, in pure python. Correctness =
    fraction of the n_real 'real' nodes present (detection); PLUS a +0.1 credit iff all real nodes lie in ONE
    weakly-connected component (global-reachability-credit); unmatched (non-real) nodes/edges are NOT penalized."""
    real_ids = set(range(n_real))

    def _one_component(g):
        parent = {n: n for n in g.node_ids()}

        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]; a = parent[a]
            return a
        for s, t in g.edges if isinstance(g, DictGraph) else []:
            if s in parent and t in parent:
                parent[find(s)] = find(t)
        present_real = [n for n in g.node_ids() if n in real_ids]
        if not present_real:
            return False
        roots = {find(n) for n in present_real}
        return len(roots) == 1

    def score(g):
        present = sum(1 for n in g.node_ids() if n in real_ids)
        detect = present / n_real
        credit = 0.1 if _one_component(g) else 0.0
        return detect + credit
    return score, real_ids


def _toy_baseline(n_real=6):
    """6 disconnected real nodes (perfect detection, ZERO connectivity) — the missed-every-division analogue."""
    return DictGraph({i: {"t": i, "z": 0, "y": 0, "x": 0, "real": True} for i in range(n_real)}, [])


def synthetic_demo():
    """Run the prober on the built-in toy degenerate metric — proves the agent detects + quantifies the exploit
    with no external deps. Returns the probe report."""
    score, _ = _toy_reachability_metric()
    base = _toy_baseline()
    perts = [hub_unify_fpfree(), garbage_edges(n=3), division_credit_hack(k=2)]
    return probe(score, base, perts)


# ───────────────────────────── agent ─────────────────────────────
class MetricProbe(BaseAgent):
    name = "metric-probe"
    thread = "V"          # validation / security thread
    kind = "verdict"

    def run(self, q, worker):
        spec = self.spec(q)
        # spec may inject a live scorer + baseline + perturbations (rare across the board); else run the
        # self-contained synthetic degeneracy demo so the agent always returns an honest, reproducible verdict.
        score_fn = spec.get("score_fn"); baseline = spec.get("baseline")
        if callable(score_fn) and baseline is not None:
            perts = spec.get("perturbations") or graph_perturbation_suite(
                k=int(spec.get("k", 5)), garbage=int(spec.get("garbage", 4)),
                sparsify=float(spec.get("sparsify", 0.5)))
            rep = probe(score_fn, baseline, perts, tol=float(spec.get("tol", TOL)),
                        wrap=bool(spec.get("wrap", True)))
            src = "injected scorer"
        else:
            rep = synthetic_demo(); src = "synthetic toy-metric demo (no scorer injected)"
        self.save_state({"metric_probe": rep, "source": src})
        msg = (f"[{worker}] **METRIC-PROBE** ({src}) — adversarial metric-vulnerability scan\n"
               f"{report_md(rep)}\n"
               f"→ **{rep['verdict']}**"
               + (f"\nbug classes: {', '.join(rep['bug_classes'])}. Use to explain LB unreliability / guard CV / "
                  f"report the bug — NOT to silently submit exploits." if rep["n_exploits"] else ""))
        self.post(worker, "leader", msg, routine=False, kind="verdict")
        self.log(summary=f"metric-probe: {rep['n_exploits']} exploit(s); {rep['verdict'][:120]}",
                 detail=f"source={src}; bug_classes={rep['bug_classes']}",
                 kind="verdict",
                 recommendation="degenerate metric ⇒ distrust that LB axis + guard CV; report bug; never submit the exploit")
        return self.done({"n_exploits": rep["n_exploits"], "bug_classes": rep["bug_classes"],
                          "base_score": rep["base_score"], "results": rep["results"],
                          "verdict": rep["verdict"]}, msg, to="leader")


_AGENT = MetricProbe()


def run(q, worker):
    return _AGENT.run(q, worker)
