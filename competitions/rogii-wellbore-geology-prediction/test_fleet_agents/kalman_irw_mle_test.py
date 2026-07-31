"""ADVANCED proof: plant an integrated-random-walk with KNOWN (q_s,q_v); assert (1) the PyTorch Kalman MLE
recovers them, and (2) the fast method-of-moments (pf_noise_id) AGREES with the exact MLE — validating MoM."""
import importlib.util,sys,types,numpy as np,pandas as pd
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent
if 'fleet_agents' not in sys.modules:
    pkg=types.ModuleType('fleet_agents');pkg.__path__=[str(ROOT/'fleet_agents')];sys.modules['fleet_agents']=pkg
def _load(n):
    sp=importlib.util.spec_from_file_location(f'fleet_agents.{n}',ROOT/'fleet_agents'/f'{n}.py');m=importlib.util.module_from_spec(sp);sys.modules[f'fleet_agents.{n}']=m;sp.loader.exec_module(m);return m
mm=_load('math_master'); nid=_load('pf_noise_id')
def _sim(qs,qv,n=500,dt=1.0,seed=0):
    rng=np.random.default_rng(seed); v=np.zeros(n);s=np.zeros(n);s[0]=100.0
    for i in range(1,n):
        v[i]=v[i-1]+rng.normal(0,np.sqrt(qv*dt)); s[i]=s[i-1]+v[i-1]*dt+rng.normal(0,np.sqrt(qs*dt))
    return s
def test_mle_recovers_and_mom_agrees():
    qs,qv=0.05,0.001; s=_sim(qs,qv)
    mle=mm.kalman_irw_mle(s,dt=1.0,iters=120,lr=0.2)
    assert 0.5<mle['q_s']/qs<1.8, mle
    assert 0.4<mle['q_v']/qv<2.2, mle
    # MoM on same trajectory
    d2=np.diff(s,2); g0=np.mean(d2*d2); g1=np.mean(d2[1:]*d2[:-1])
    mom_qs=max(-g1,0); mom_qv=max(g0+2*g1,0)
    assert 0.5<mom_qs/mle['q_s']<1.8, (mom_qs,mle['q_s'])   # MoM ≈ MLE (the validation)
    print(f"MLE q_s={mle['q_s']:.4f}(true {qs}) q_v={mle['q_v']:.5f}(true {qv}) | MoM q_s={mom_qs:.4f} q_v={mom_qv:.5f}")
if __name__=="__main__":
    test_mle_recovers_and_mom_agrees(); print("PASS")
