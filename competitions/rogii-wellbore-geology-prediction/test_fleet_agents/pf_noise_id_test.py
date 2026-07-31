"""Data-wise: plant a state-space trajectory with KNOWN q_s,q_v,r on a synthetic heel and assert recovery."""
import importlib.util,numpy as np,pandas as pd
from pathlib import Path
spec=importlib.util.spec_from_file_location('nid', Path(__file__).resolve().parent.parent/'fleet_agents'/'pf_noise_id.py')
nid=importlib.util.module_from_spec(spec); spec.loader.exec_module(nid)
def test_recovers_planted_noise():
    rng=np.random.default_rng(0); n=4000; dmd=1.0
    qv=(0.02)**2; qs=(0.05)**2                     # planted process variances (per unit depth)
    v=np.zeros(n); s=np.zeros(n); s[0]=11000.0; v[0]=0.0
    for i in range(1,n):
        v[i]=v[i-1]+rng.normal(0,np.sqrt(qv*dmd))
        s[i]=s[i-1]+v[i-1]*dmd+rng.normal(0,np.sqrt(qs*dmd))
    Z=np.linspace(0,-50,n); TVT=s-Z; MD=11000+np.arange(n)*dmd
    # simple typewell so h(TVT)=TVT (r estimation not the focus of recovery assert)
    tw=pd.DataFrame({'TVT':np.linspace(TVT.min()-5,TVT.max()+5,3000)}); tw['GR']=tw['TVT']; 
    hw=pd.DataFrame({'MD':MD,'Z':Z,'TVT':TVT,'TVT_input':TVT,'GR':TVT})
    out=nid.identify(hw,tw)
    # recover q_v and q_s within a factor of ~1.6 (MoM on 4000 pts)
    assert 0.6 < (out['vn']**2)/qv < 1.7, (out['vn']**2, qv)
    assert 0.6 < (out['pn']**2)/qs < 1.7, (out['pn']**2, qs)
if __name__=="__main__":
    test_recovers_planted_noise(); print("PASS")
