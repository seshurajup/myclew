"""forecast_sports_pack — more recurring PURE agents from the one-by-one pass (forecasting / sports-prob /
best-of-N / segmentation-decode). All numpy/sklearn, offline-verifiable, CompConfig-agnostic:

  • ts-decompose-forecaster     — multiplicative decomposition into interpretable ratio factors (calendar/
                                  group) + reconstruction (playground-s5e1 sticker-sales lever).
  • forecast-trend-extrapolator — choose the future-horizon global trend multiplier (const/linear/ReLU) for
                                  years beyond the training range (the s5e1 decisive out-of-range lever).
  • rating-systems              — competitive ratings from a win/loss+margin game graph: Elo (MOV+carry),
                                  Colley matrix, SRS (march-mania).
  • outcome-sharpen             — Brier/log-loss tail sharpening + expert overrides (sports prob comps).
  • best-of-n-diversity-allocator — for max-over-N metrics, pick N DIVERSE candidates (not averaged) to
                                  maximize expected max (rna best-of-5, TM-score).
  • temporal-segment-decoder    — frame-probabilities → scored action segments (per-group threshold +
                                  min-duration filter) for detection/segmentation metrics (MABe).
"""
from __future__ import annotations
import numpy as np
from .base import BaseAgent


# ---------------------------------------------------------------- ts-decompose-forecaster
def decompose_multiplicative(y, group_ids):
    """Decompose a positive series into a global level × per-group ratio factors. group_ids = dict{name: array
    of group index per sample}. Returns (level, factors dict, reconstruction)."""
    y = np.nan_to_num(np.asarray(y, float), nan=0.0, posinf=0.0, neginf=0.0)
    level = np.exp(np.mean(np.log(np.clip(y, 1e-9, None)))) if len(y) else 1.0
    logr = np.log(np.clip(y, 1e-9, None)) - np.log(level)
    factors = {}; recon = np.full_like(y, level)
    for name, g in group_ids.items():
        g = np.asarray(g); f = np.array([np.exp(logr[g == k].mean()) if (g == k).any() else 1.0 for k in range(g.max() + 1)])
        factors[name] = f; recon = recon * f[g]
    return float(level), factors, recon


# ---------------------------------------------------------------- forecast-trend-extrapolator
def choose_trend(years, values, future_years, forms=("const", "linear", "relu")):
    """Fit candidate future-trend multipliers on the observed years and return per-form multipliers for the
    future horizon + the in-sample best form. values = yearly aggregate ratios (e.g. YoY growth)."""
    years = np.nan_to_num(np.asarray(years, float)); values = np.nan_to_num(np.asarray(values, float))
    out = {}
    # const: mean YoY multiplier
    m = float(np.mean(values)) if len(values) else 1.0; out["const"] = {y: m for y in future_years}
    # linear: fit slope (needs >=2 distinct points; else flat at the mean)
    if len(years) >= 2 and np.ptp(years) > 0:
        a, b = np.polyfit(years, values, 1)
    else:
        a, b = 0.0, m
    out["linear"] = {y: float(a * y + b) for y in future_years}
    # relu: linear but floored at last observed (no decay below)
    last = float(values[-1]); out["relu"] = {y: max(last, float(a * y + b)) for y in future_years}
    # in-sample fit quality picks the safest default
    resid = {"const": np.mean((values - m) ** 2), "linear": np.mean((values - (a * years + b)) ** 2)}
    best = min(resid, key=resid.get)
    return {f: out[f] for f in forms if f in out}, best


# ---------------------------------------------------------------- rating-systems
def elo_ratings(games, k=20, base=1500.0, mov=True):
    """games = list of (team_i, team_j, margin_i_minus_j). Returns dict team->Elo. MOV scales K by margin."""
    teams = sorted({t for g in games for t in g[:2]}); R = {t: base for t in teams}
    for i, j, m in games:
        exp_i = 1 / (1 + 10 ** ((R[j] - R[i]) / 400))
        s_i = 1.0 if m > 0 else (0.5 if m == 0 else 0.0)
        kk = k * (np.log1p(abs(m)) if mov else 1.0)
        R[i] += kk * (s_i - exp_i); R[j] += kk * ((1 - s_i) - (1 - exp_i))
    return R


def colley_ratings(games):
    """Colley matrix ratings (bias-free from win/loss only). games = (i,j,margin)."""
    teams = sorted({t for g in games for t in g[:2]}); idx = {t: n for n, t in enumerate(teams)}; n = len(teams)
    C = np.eye(n) * 2; b = np.ones(n)
    wins = {t: 0 for t in teams}; loss = {t: 0 for t in teams}
    for i, j, m in games:
        a, c = idx[i], idx[j]; C[a, a] += 1; C[c, c] += 1; C[a, c] -= 1; C[c, a] -= 1
        if m > 0: wins[i] += 1; loss[j] += 1
        elif m < 0: wins[j] += 1; loss[i] += 1
    for t in teams:
        b[idx[t]] = 1 + 0.5 * (wins[t] - loss[t])
    try:
        r = np.linalg.solve(C, b)
    except np.linalg.LinAlgError:                       # singular Colley matrix → least-squares
        r = np.linalg.lstsq(C, b, rcond=None)[0]
    return {t: float(r[idx[t]]) for t in teams}


# ---------------------------------------------------------------- outcome-sharpen
def sharpen(probs, hi=0.97, lo=0.03, overrides=None):
    """Push confident predictions to the tails (Brier-EV gamble) + apply expert overrides {idx: prob}."""
    p = np.asarray(probs, float).copy()
    p = np.where(p >= hi, np.maximum(p, 0.995), p); p = np.where(p <= lo, np.minimum(p, 0.005), p)
    for i, v in (overrides or {}).items():
        p[int(i)] = v
    return np.clip(p, 0, 1)


# ---------------------------------------------------------------- best-of-n-diversity-allocator
def allocate_best_of_n(candidates, n, quality=None):
    """Pick N diverse candidates to maximize expected MAX (not average). candidates = (m, d) array of m
    candidate feature-vectors; greedy: start from highest quality, add the most-distant each step."""
    C = np.asarray(candidates, float); m = len(C)
    q = np.asarray(quality, float) if quality is not None else np.zeros(m)
    chosen = [int(np.argmax(q))]
    while len(chosen) < min(n, m):
        d = np.array([min(np.linalg.norm(C[i] - C[c]) for c in chosen) for i in range(m)])
        d[chosen] = -1
        chosen.append(int(np.argmax(d + 0.01 * q)))
    return chosen


# ---------------------------------------------------------------- temporal-segment-decoder
def decode_segments(frame_prob, threshold=0.5, min_len=3, merge_gap=0):
    """frame_prob = (T,) probability of the action per frame. Returns list of (start, end, score) segments
    above threshold with a minimum duration (flicker filter).
    merge_gap: bridge two above-threshold runs separated by a sub-threshold gap of <= this many frames."""
    p = np.nan_to_num(np.asarray(frame_prob, float), nan=0.0, posinf=1.0, neginf=0.0)
    above = p >= threshold
    if merge_gap and int(merge_gap) > 0:               # close short gaps before segmenting
        above = above.copy(); mg = int(merge_gap); t = 0
        while t < len(above):
            if not above[t]:
                s = t
                while t < len(above) and not above[t]:
                    t += 1
                if 0 < s and t < len(above) and (t - s) <= mg:
                    above[s:t] = True
            else:
                t += 1
    segs = []
    t = 0
    while t < len(p):
        if above[t]:
            s = t
            while t < len(p) and above[t]:
                t += 1
            if t - s >= min_len:
                segs.append((s, t - 1, float(p[s:t].mean())))
        else:
            t += 1
    return segs


# ---------------------------------------------------------------- agents
class _B(BaseAgent):
    thread = "M"; kind = "finding"


class TsDecompose(_B):
    name = "ts-decompose-forecaster"
    def run(self, q, worker):
        s = self.spec(q)
        missing = [k for k in ("y", "groups") if k not in s]
        if missing: return self.escalate(worker, "leader", f"ts-decompose-forecaster needs spec keys {missing} — none provided")
        level, factors, recon = decompose_multiplicative(s["y"], s["groups"])
        msg = f"ts-decompose-forecaster: level={level:.3f}, {len(factors)} ratio factors; reconstruction ready"
        self.log(msg, kind="finding", recommendation="forecast by multiplying factors; extrapolate trend separately")
        return self.done({"level": level, "factor_names": list(factors), "_recon": recon.tolist()}, msg)


class TrendExtrapolator(_B):
    name = "forecast-trend-extrapolator"
    def run(self, q, worker):
        s = self.spec(q)
        missing = [k for k in ("years", "values", "future_years") if k not in s]
        if missing: return self.escalate(worker, "leader", f"forecast-trend-extrapolator needs spec keys {missing} — none provided")
        mult, best = choose_trend(s["years"], s["values"], s["future_years"])
        msg = f"forecast-trend-extrapolator: in-sample best form='{best}'; multipliers for {s['future_years']} ready (submit const+linear as 2 finals)"
        self.log(msg, kind="finding", recommendation="hedge: submit the two safest trend hypotheses")
        return self.done({"best_form": best, "multipliers": {f: {int(k): v for k, v in d.items()} for f, d in mult.items()}}, msg)


class RatingSystems(_B):
    name = "rating-systems"
    def run(self, q, worker):
        s = self.spec(q)
        missing = [k for k in ("games",) if k not in s]
        if missing: return self.escalate(worker, "leader", f"rating-systems needs spec keys {missing} — none provided")
        games = [tuple(g) for g in s["games"]]
        elo = elo_ratings(games, mov=bool(s.get("mov", True))); colley = colley_ratings(games)
        msg = f"rating-systems: Elo + Colley ratings for {len(elo)} teams"
        self.log(msg, kind="finding", recommendation="use rating DIFFS as features; add SRS/Massey for consensus")
        return self.done({"elo": elo, "colley": colley}, msg)


class OutcomeSharpen(_B):
    name = "outcome-sharpen"
    def run(self, q, worker):
        s = self.spec(q)
        missing = [k for k in ("probs",) if k not in s]
        if missing: return self.escalate(worker, "leader", f"outcome-sharpen needs spec keys {missing} — none provided")
        out = sharpen(s["probs"], float(s.get("hi", 0.97)), float(s.get("lo", 0.03)), s.get("overrides"))
        msg = "outcome-sharpen: tail-sharpened probabilities (Brier/log-loss EV gamble) + overrides applied"
        self.log(msg, kind="finding", recommendation="size the sharpened set to the loss-EV tradeoff; verify on CV")
        return self.done({"_probs": out.tolist()}, msg)


class BestOfNAllocator(_B):
    name = "best-of-n-diversity-allocator"
    def run(self, q, worker):
        s = self.spec(q)
        missing = [k for k in ("candidates", "n") if k not in s]
        if missing: return self.escalate(worker, "leader", f"best-of-n-diversity-allocator needs spec keys {missing} — none provided")
        chosen = allocate_best_of_n(s["candidates"], int(s["n"]), s.get("quality"))
        msg = f"best-of-n-diversity-allocator: selected {len(chosen)} DIVERSE candidates for the max-over-N metric"
        self.log(msg, kind="finding", recommendation="submit distinct candidates; never average for best-of-N metrics")
        return self.done({"chosen": chosen}, msg)


class TemporalSegmentDecoder(_B):
    name = "temporal-segment-decoder"
    def run(self, q, worker):
        s = self.spec(q)
        missing = [k for k in ("frame_prob",) if k not in s]
        if missing: return self.escalate(worker, "leader", f"temporal-segment-decoder needs spec keys {missing} — none provided")
        segs = decode_segments(s["frame_prob"], float(s.get("threshold", 0.5)),
                                                 int(s.get("min_len", 3)), merge_gap=int(s.get("merge_gap", 0)))
        msg = f"temporal-segment-decoder: {len(segs)} action segments (thr={s.get('threshold',0.5)}, min_len={s.get('min_len',3)})"
        self.log(msg, kind="finding", recommendation="tune per-group threshold on OOF; filter short flickers")
        return self.done({"segments": segs, "n_segments": len(segs)}, msg)


_TS = TsDecompose(); _TR = TrendExtrapolator(); _RS = RatingSystems(); _OS = OutcomeSharpen()
_BN = BestOfNAllocator(); _SD = TemporalSegmentDecoder()


def run_ts(q, worker): return _TS.run(q, worker)
def run_trend(q, worker): return _TR.run(q, worker)
def run_rating(q, worker): return _RS.run(q, worker)
def run_sharpen(q, worker): return _OS.run(q, worker)
def run_bestofn(q, worker): return _BN.run(q, worker)
def run_segdecode(q, worker): return _SD.run(q, worker)
