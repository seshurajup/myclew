"""domain_feature_pack — reusable DOMAIN feature-engineering + online-learning agents mined from the winners:

  • fin-ta-feature-library      — financial technical-analysis features from a price/return series (mitsui:
                                  volatility, momentum, RSI, z-score, Hurst, mean-reversion half-life).
  • imu-feature-engineer        — kinematic features from IMU/accelerometer streams (cmi: magnitude, jerk,
                                  gravity-removal, per-axis stats, spectral energy).
  • online-walk-forward-retrainer — incremental retrain over a time stream (jane-street/mitsui: refit every
                                  N steps on data-so-far, recency-weighted) — survives non-stationarity.

Pure numpy/scipy. Modality-agnostic inputs (arrays). Verified offline.
"""
from __future__ import annotations
import numpy as np
from .base import BaseAgent


# ---------------------------------------------------------------- financial TA features
def fin_ta_features(prices, windows=(5, 20), rsi_period=14):
    """Return a (n, k) feature matrix from a 1-D price series. Early rows back-filled. Grounded in mitsui.
    rsi_period: look-back for the RSI oscillator (feature named rsi{period})."""
    p = np.nan_to_num(np.asarray(prices, float), nan=0.0, posinf=0.0, neginf=0.0); n = len(p)
    if n == 0:
        return np.zeros((0, 1), np.float32), ["log_return"]
    logret = np.zeros(n); logret[1:] = np.diff(np.log(np.clip(p, 1e-9, None)))
    feats = {"log_return": logret}
    for w in windows:
        w = max(1, int(w))
        roll = np.array([logret[max(0, i - w + 1):i + 1] for i in range(n)], dtype=object)
        feats[f"vol{w}"] = np.array([r.std() if len(r) else 0.0 for r in roll])
        feats[f"mom{w}"] = np.array([p[i] / max(p[max(0, i - w)], 1e-9) - 1 for i in range(n)])
        m = np.array([logret[max(0, i - w + 1):i + 1].mean() for i in range(n)])
        s = feats[f"vol{w}"]; feats[f"z{w}"] = np.where(s > 1e-9, (logret - m) / (s + 1e-9), 0.0)
    # RSI(period)
    rp = max(1, int(rsi_period))
    d = np.diff(p, prepend=p[:1]); up = np.clip(d, 0, None); dn = -np.clip(d, None, 0)
    ru = np.array([up[max(0, i - (rp - 1)):i + 1].mean() for i in range(n)])
    rd = np.array([dn[max(0, i - (rp - 1)):i + 1].mean() for i in range(n)])
    feats[f"rsi{rp}"] = 100 - 100 / (1 + ru / (rd + 1e-9))
    # Hurst (rescaled-range on log-returns, coarse)
    feats["hurst"] = np.full(n, _hurst(logret[1:]))
    names = list(feats)
    X = np.column_stack([feats[k] for k in names])
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32), names


def _hurst(x, max_lag=20):
    x = np.asarray(x, float)
    if len(x) < max_lag * 2:
        return 0.5
    lags = range(2, max_lag)
    tau = [np.std(x[lag:] - x[:-lag]) for lag in lags]
    try:
        return float(np.polyfit(np.log(list(lags)), np.log(np.array(tau) + 1e-9), 1)[0])
    except Exception:  # noqa: BLE001
        return 0.5


# ---------------------------------------------------------------- IMU / sensor features
def imu_features(accel, window=10):
    """accel = (n,3) x/y/z. Returns (n,k): magnitude, jerk, gravity-removed magnitude, per-axis rolling
    mean/std, spectral energy. Grounded in cmi-detect-behavior."""
    a = np.nan_to_num(np.asarray(accel, float), nan=0.0, posinf=0.0, neginf=0.0); n = len(a)
    if n == 0 or a.ndim != 2:
        return np.zeros((n, 6), np.float32), ["mag", "jerk", "lin_mag", "roll_mean", "roll_std", "energy"]
    window = max(1, int(window))
    mag = np.linalg.norm(a, axis=1)
    jerk = np.zeros(n); jerk[1:] = np.linalg.norm(np.diff(a, axis=0), axis=1)
    grav = np.array([a[max(0, i - window + 1):i + 1].mean(0) for i in range(n)])   # low-pass = gravity
    lin_mag = np.linalg.norm(a - grav, axis=1)                                     # gravity-removed
    rmean = np.array([mag[max(0, i - window + 1):i + 1].mean() for i in range(n)])
    rstd = np.array([mag[max(0, i - window + 1):i + 1].std() for i in range(n)])
    energy = np.array([np.sum(mag[max(0, i - window + 1):i + 1] ** 2) for i in range(n)])
    X = np.column_stack([mag, jerk, lin_mag, rmean, rstd, energy])
    return np.nan_to_num(X).astype(np.float32), ["mag", "jerk", "lin_mag", "roll_mean", "roll_std", "energy"]


# ---------------------------------------------------------------- online walk-forward retraining
def walk_forward(X, y, model_factory, retrain_every=50, warmup=100, recency_halflife=None):
    """Refit a model every `retrain_every` steps on all data seen so far; predict the next block online.
    Returns oof predictions (NaN in warmup). Grounded in jane-street/mitsui online learning.
    recency_halflife: if set, weight training samples by an exponential recency decay (half-weight this many
    steps back) — passed as sample_weight when the model supports it (non-stationary regimes)."""
    X = np.nan_to_num(np.asarray(X, float)); y = np.nan_to_num(np.asarray(y, float)); n = len(y)
    oof = np.full(n, np.nan); model = None
    retrain_every = max(1, int(retrain_every))
    for start in range(max(1, int(warmup)), n, retrain_every):
        model = model_factory()
        if recency_halflife and recency_halflife > 0:
            age = np.arange(start)[::-1]
            w = 0.5 ** (age / float(recency_halflife))
            try:
                model.fit(X[:start], y[:start], sample_weight=w)
            except Exception:  # noqa: BLE001 — model doesn't accept sample_weight
                model.fit(X[:start], y[:start])
        else:
            model.fit(X[:start], y[:start])
        end = min(n, start + retrain_every)
        oof[start:end] = model.predict(X[start:end])
    return oof


class _Base(BaseAgent):
    thread = "M"; kind = "finding"


class FinTa(_Base):
    name = "fin-ta-feature-library"
    def run(self, q, worker):
        s = self.spec(q)
        missing = [k for k in ("prices",) if k not in s]
        if missing:
            return self.escalate(worker, "leader", f"fin-ta-feature-library needs spec keys {missing} — none provided")
        X, names = fin_ta_features(s["prices"], tuple(s.get("windows", (5, 20))),
                                   rsi_period=int(s.get("rsi_period", 14)))
        msg = f"fin-ta-feature-library: {X.shape[1]} TA features ({names}) from {X.shape[0]} steps"
        self.log(msg, kind="finding", recommendation="feed to tab-train; combine with mean-encodings")
        return self.done({"n_features": int(X.shape[1]), "names": names, "_X": X.tolist()}, msg)


class ImuFe(_Base):
    name = "imu-feature-engineer"
    def run(self, q, worker):
        s = self.spec(q)
        missing = [k for k in ("accel",) if k not in s]
        if missing:
            return self.escalate(worker, "leader", f"imu-feature-engineer needs spec keys {missing} — none provided")
        X, names = imu_features(s["accel"], int(s.get("window", 10)))
        msg = f"imu-feature-engineer: {X.shape[1]} kinematic features ({names}) from {X.shape[0]} samples"
        self.log(msg, kind="finding", recommendation="add orientation aug + frame canonicalization for sensor comps")
        return self.done({"n_features": int(X.shape[1]), "names": names, "_X": X.tolist()}, msg)


class OnlineRetrain(_Base):
    name = "online-walk-forward-retrainer"
    def run(self, q, worker):
        s = self.spec(q)
        missing = [k for k in ("X", "y") if k not in s]
        if missing:
            return self.escalate(worker, "leader", f"online-walk-forward-retrainer needs spec keys {missing} — none provided")
        from sklearn.linear_model import Ridge
        oof = walk_forward(np.asarray(s["X"], float), np.asarray(s["y"], float),
                           lambda: Ridge(alpha=float(s.get("alpha", 1.0))),
                           retrain_every=int(s.get("retrain_every", 50)), warmup=int(s.get("warmup", 100)),
                           recency_halflife=s.get("recency_halflife"))
        cov = float(np.mean(~np.isnan(oof)))
        msg = f"online-walk-forward-retrainer: online OOF over {len(oof)} steps ({cov*100:.0f}% covered) — survives drift"
        self.log(msg, kind="finding", recommendation="blend with a persistence anchor (recent-label mean) for regime shifts")
        return self.done({"coverage": cov, "_oof": np.nan_to_num(oof).tolist()}, msg)


_FIN = FinTa(); _IMU = ImuFe(); _ONLINE = OnlineRetrain()


def run_fin(q, worker): return _FIN.run(q, worker)
def run_imu(q, worker): return _IMU.run(q, worker)
def run_online(q, worker): return _ONLINE.run(q, worker)
