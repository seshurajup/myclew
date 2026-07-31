"""Data-wise: plant wells whose datum s=TVT+Z is a smooth plane over (X,Y); the structural agent must
reconstruct TVT for a held-out well to within a small tolerance (the identity TVT=surface(X,Y)-Z+b)."""
import importlib.util, sys, types, numpy as np, pandas as pd
from pathlib import Path
FA = str(Path(__file__).resolve().parent.parent / "fleet_agents")
if "fleet_agents" not in sys.modules:
    pkg = types.ModuleType("fleet_agents"); pkg.__path__ = [FA]; sys.modules["fleet_agents"] = pkg
spec = importlib.util.spec_from_file_location("fleet_agents.geology_structural", FA + "/geology_structural.py")
gs = importlib.util.module_from_spec(spec); spec.loader.exec_module(gs)
def _well(rng, x0, y0):
    n = 300; X = x0 + np.cumsum(rng.uniform(0.5,1.5,n)); Y = y0 + rng.normal(0,0.3,n).cumsum()
    s = 11000 + 0.02*X - 0.01*Y                      # smooth datum plane
    Z = np.linspace(-9000,-9050,n); TVT = s - Z
    return pd.DataFrame(dict(X=X,Y=Y,Z=Z,TVT=TVT,TVT_input=np.where(np.arange(n)<120,TVT,np.nan)))
def test_recovers_surface():
    rng = np.random.default_rng(0)
    train = {f"w{i}": _well(rng, rng.uniform(0,50), rng.uniform(0,50)) for i in range(40)}
    surf = gs.build_surface_from(train)
    test = _well(rng, 20, 20)                          # inside the cloud
    dt, pr, used = gs.predict_well(test, surf, None)
    ev = test.TVT_input.isna().to_numpy(); tvtps = test.TVT_input.dropna().iloc[-1]
    true = test.TVT.to_numpy()[ev] - tvtps
    rmse = np.sqrt(np.mean((dt-true)**2))
    print(f"structural recovery RMSE={rmse:.3f}, prefix={pr:.3f}")
    assert rmse < 2.0 and pr < 2.0
if __name__ == "__main__":
    test_recovers_surface(); print("PASS")
