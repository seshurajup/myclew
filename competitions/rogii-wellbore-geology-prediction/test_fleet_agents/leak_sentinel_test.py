"""Data-wise: sentinel calls a real-signal model significant, a noise model NOT, and catches an ID drift leak."""
import importlib.util,numpy as np
from pathlib import Path
spec=importlib.util.spec_from_file_location('ls', Path(__file__).resolve().parent.parent/'fleet_agents'/'leak_sentinel.py')
ls=importlib.util.module_from_spec(spec); spec.loader.exec_module(ls)
def test_sentinel():
    s=ls.LeakSentinel()
    assert s.permutation_test(lambda shuffle_target,ablate,seed: 15.9 if shuffle_target else 6.0,6.0,K=5)[1] is True
    assert s.permutation_test(lambda shuffle_target,ablate,seed: 15.9,15.9,K=5)[1] is False
    assert s.group_disjoint([1,2,3],[4,5])[1] is True
    assert s.group_disjoint([1,2,3],[3,4])[1] is False
    assert s.verify() is True
if __name__=="__main__":
    test_sentinel(); print("PASS")
