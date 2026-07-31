"""Data-wise test: the probe PASSES an honest model, FAILS a known-leak model on the null-target test."""
import importlib.util
from pathlib import Path
spec=importlib.util.spec_from_file_location('sp', Path(__file__).resolve().parent.parent/'fleet_agents'/'stability_probe.py')
sp=importlib.util.module_from_spec(spec); spec.loader.exec_module(sp)
def test_null_catches_leak():
    probe=sp.StabilityProbe()
    honest=lambda shuffle_target,ablate,seed: 15.9 if shuffle_target else 6.0
    leaky =lambda shuffle_target,ablate,seed: 6.0
    assert probe.null_test(honest,15.9)[1] is True
    assert probe.null_test(leaky ,15.9)[1] is False
    assert probe.verify() is True
if __name__=="__main__":
    test_null_catches_leak(); print("PASS")
