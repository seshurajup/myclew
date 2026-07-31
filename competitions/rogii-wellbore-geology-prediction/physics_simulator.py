"""physics_simulator.py — generate UNLIMITED physically-valid synthetic wells from the PROVEN generative model
(eda.py): TVT is an Integrated Random Walk (proof 1: q_s,q_v), observed through a real typewell's material GR
(proof 3), with Student-t measurement noise (proof 6). Each synthetic well has a KNOWN TVT trajectory, so it is
independent training data (unlike the 3.7M correlated real rows). Used to pretrain the vision CNN. Pure numpy
generation (CPU, cheap); images built on GPU downstream."""
import glob, os, numpy as np, pandas as pd

_TW = None
def _typewells():
    global _TW
    if _TW is None:
        _TW = []
        for tp in sorted(glob.glob("input/train/*__typewell.csv")):
            tw = pd.read_csv(tp).sort_values("TVT")
            T = tw.TVT.to_numpy(float); G = tw.GR.to_numpy(float); g = tw.Geology.fillna("").astype(str).str.strip().to_numpy()
            m = np.isfinite(T) & np.isfinite(G)
            if m.sum() > 200: _TW.append((T[m], G[m], g[m]))
    return _TW

def simulate(rng, n_min=800, n_max=6000):
    tws = _typewells(); T, G, gid = tws[rng.integers(len(tws))]         # borrow a real material profile
    n = int(rng.integers(n_min, n_max)); dmd = 1.0
    # IRW dip trajectory (proof 1): v is a random walk, s integrates v
    q_s = np.exp(rng.uniform(np.log(0.01), np.log(0.3))) ** 2            # per-well curvature (proof 5 spread)
    q_v = np.exp(rng.uniform(np.log(0.005), np.log(0.05))) ** 2
    # anchor TVT inside a random formation band so the well "steers" within a formation (proof: 87% stay)
    labs = [x for x in np.unique(gid) if x]; home = labs[rng.integers(len(labs))]
    band = (T[gid == home].min(), T[gid == home].max())
    lo_, hi_ = band[0] + 5, band[1] - 5
    if hi_ <= lo_: lo_, hi_ = band[0], band[1]
    tvt0 = rng.uniform(lo_, hi_) if hi_ > lo_ else float(band[0])
    v = np.zeros(n); s = np.zeros(n); s[0] = tvt0; v[0] = rng.normal(0, 0.02)
    for i in range(1, n):
        v[i] = v[i-1] + rng.normal(0, np.sqrt(q_v * dmd))
        s[i] = s[i-1] + v[i-1] * dmd + rng.normal(0, np.sqrt(q_s * dmd))
    s = np.clip(s, T.min(), T.max())
    # measurement: GR = typewell GR at the trajectory TVT + heavy-tailed noise (proof 6)
    gr_clean = np.interp(s, T, G)
    sig = rng.uniform(8, 18); nu = 4.0
    noise = sig * rng.standard_t(nu, size=n)
    GRw = gr_clean + noise
    MD = 11000 + np.arange(n) * dmd
    Z = np.linspace(rng.uniform(-9500, -9000), rng.uniform(-9500, -9000) - rng.uniform(20, 80), n)  # near-horizontal
    ps = int(rng.uniform(0.25, 0.45) * n)                                # random prediction-start
    return dict(MD=MD, Z=Z, GR=GRw, TVT=s, ps=ps, twT=T, twG=G, twGid=gid, home=home, band=band)

def to_frame(sim):
    """Return (hw_df, tw_df) mimicking the real CSV schema so downstream image code is reused unchanged."""
    n = len(sim["MD"]); tvti = sim["TVT"].copy(); tvti[sim["ps"]:] = np.nan
    hw = pd.DataFrame(dict(MD=sim["MD"], X=np.cumsum(np.ones(n)), Y=np.zeros(n), Z=sim["Z"],
                           GR=sim["GR"], TVT=sim["TVT"], TVT_input=tvti))
    tw = pd.DataFrame(dict(TVT=sim["twT"], GR=sim["twG"], Geology=sim["twGid"]))
    return hw, tw

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    hw, tw = to_frame(simulate(rng))
    print("sample synthetic well:", hw.shape, "hidden rows:", hw.TVT_input.isna().sum(),
          "| TVT range", round(hw.TVT.min()), round(hw.TVT.max()), "| home", None)
    # sanity: does a synthetic well look like a real one? (IRW 2nd-diff autocorr negative)
    s = hw.TVT.to_numpy(); d2 = np.diff(s, 2); r1 = np.corrcoef(d2[:-1], d2[1:])[0,1]
    print(f"synthetic IRW check: rho1(2nd diff)={r1:.3f} (should be <0, matching eda proof 1)")
