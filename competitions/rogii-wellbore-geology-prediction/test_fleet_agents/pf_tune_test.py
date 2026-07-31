"""Data-wise: _combine reproduces the softmax(ll/scale) weighting exactly on a planted 2-seed case."""
import importlib.util,sys,types,numpy as np
from pathlib import Path
if 'fleet_agents' not in sys.modules:
    pkg=types.ModuleType('fleet_agents');pkg.__path__=[str(Path(__file__).resolve().parent.parent/'fleet_agents')];sys.modules['fleet_agents']=pkg
spec=importlib.util.spec_from_file_location('fleet_agents.pf_tune', Path(__file__).resolve().parent.parent/'fleet_agents'/'pf_tune.py')
pf=importlib.util.module_from_spec(spec); spec.loader.exec_module(pf)
def test_combine_matches_softmax():
    ll=np.array([2.0,0.0]); res=np.array([[10.,10.,10.],[20.,20.,20.]])  # 2 seeds x 3 rows
    scale=5.0
    w=np.exp((ll-ll.max())/scale); w/=w.sum()
    expect=(w[:,None]*res).sum(0)
    got=pf._combine(ll,res,scale)
    assert np.allclose(got,expect), (got,expect)
    # selection mask zeros a seed
    got2=pf._combine(ll,res,scale,sel=np.array([1.0,0.0]))
    assert np.allclose(got2,res[0]), got2
if __name__=="__main__":
    test_combine_matches_softmax(); print("PASS")
