import sys, types, time
p=types.ModuleType('fleet_agents'); p.__path__=['fleet_agents']; sys.modules['fleet_agents']=p
from fleet_agents import geology_trackB as TB
import numpy as np, pandas as pd
t=time.time()
TB.trackB_oof('input/train','config/_auto/trackB_oof.csv',training=True,n_seeds=64,n_particles=500)
TB.trackB_oof('input/test','config/_auto/trackB_test.csv',training=False,n_seeds=64,n_particles=500)
d=pd.read_csv('config/_auto/trackB_oof.csv')
rmse=np.sqrt(((d.trackB_dtvt-d.true_dtvt)**2).mean())
print(f'DONE {time.time()-t:.0f}s | Track-B full OOF pooled RMSE={rmse:.3f} vs const 15.91 vs TrackA 15.46', flush=True)
