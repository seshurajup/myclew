"""Data-wise: plant wells on a line; the retrieval must return the SPATIALLY nearest well first."""
import importlib.util, sys, types, numpy as np, pandas as pd, tempfile, os
from pathlib import Path
FA = str(Path(__file__).resolve().parent.parent / "fleet_agents")
if "fleet_agents" not in sys.modules:
    pkg = types.ModuleType("fleet_agents"); pkg.__path__ = [FA]; sys.modules["fleet_agents"] = pkg
spec = importlib.util.spec_from_file_location("fleet_agents.well_retrieval", FA + "/well_retrieval.py")
wr = importlib.util.module_from_spec(spec); spec.loader.exec_module(wr)
def test_nearest_first():
    d = tempfile.mkdtemp()
    for i,x0 in enumerate([0,100,200,300]):
        n=100; X=x0+np.arange(n)*0.5; Y=np.zeros(n)
        pd.DataFrame(dict(MD=np.arange(n),X=X,Y=Y,Z=-9000-np.zeros(n),GR=np.zeros(n),
                          TVT=11000+np.zeros(n),TVT_input=11000+np.zeros(n))).to_csv(f"{d}/w{i}__horizontal_well.csv",index=False)
    idx = wr.build(d)
    q = pd.DataFrame(dict(X=105+np.arange(50)*0.5, Y=np.zeros(50)))   # nearest to well at x0=100 (w1)
    nn,_ = idx.query(q, k=2)
    assert nn[0]=="w1", nn
if __name__=="__main__":
    test_nearest_first(); print("PASS")
