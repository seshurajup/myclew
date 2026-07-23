"""xai_test — DATA-WISE quality verifier for the XAI agent (lesson from the sandboxed-AutoML notebook's
verify_execution.py: don't check "did it run", check "did it produce the CORRECT output on known data").

Each test plants a KNOWN signal (a bright blob at a fixed voxel, or a feature that drives the label) and
asserts the method RECOVERS it — localises the blob, or ranks the true driver feature top. A method that
merely runs but points at the wrong place FAILS here. Run:

    research/cellmot_venv/bin/python test_fleet_agents/xai_test.py      # standalone (prints PASS/FAIL table)
    pytest test_fleet_agents/xai_test.py                               # or via pytest

Extend this file with one <agent>_test.py per fleet agent so every agent has a ground-truth quality gate.
"""
import os
import sys
COMP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, COMP)
sys.path.insert(0, os.path.join(COMP, "tools", "researchpapers"))

import numpy as np
import torch
from torch import nn
from fleet_agents import xai

# ground truth: the blob sits here; a method must peak inside this box to pass
BLOB = {"z": (2, 5), "y": (6, 9), "x": (6, 9)}


def _localises(peak):
    return (BLOB["z"][0] <= peak[0] <= BLOB["z"][1] and BLOB["y"][0] <= peak[1] <= BLOB["y"][1]
            and BLOB["x"][0] <= peak[2] <= BLOB["x"][1])


def _cam_peak(cam, vol):
    import torch.nn.functional as F
    if tuple(cam.shape) != tuple(vol.shape[2:]):
        cam = F.interpolate(cam[None, None].float(), size=vol.shape[2:], mode="trilinear", align_corners=False)[0, 0]
    arr = cam.detach().cpu().numpy()
    return np.unravel_index(int(arr.argmax()), arr.shape)


# ─────────────────────────── CNN family (must LOCALISE the blob) ───────────────────────────
def test_cnn_methods_localise_the_signal():
    model, vol, layer = xai._tiny_cnn(torch, nn); model.eval()
    methods = {
        "grad_cam": lambda: xai._grad_cam(torch, model, vol, layer),
        "grad_cam_pp": lambda: xai._grad_cam(torch, model, vol, layer, plusplus=True),
        "score_cam": lambda: xai._score_cam(torch, model, vol, layer),
        "smoothgrad": lambda: xai._smoothgrad(torch, model, vol),
        "occlusion": lambda: xai._occlusion(torch, model, vol),
        "rise": lambda: xai._rise(torch, model, vol),
        "layer_cam": lambda: xai._layer_cam(torch, model, vol, layer),
        "xgrad_cam": lambda: xai._xgrad_cam(torch, model, vol, layer),
        "ablation_cam": lambda: xai._ablation_cam(torch, model, vol, layer),
        "eigen_cam": lambda: xai._eigen_cam(torch, model, vol, layer),
        "g_came_detector": lambda: xai._g_came(torch, model, vol, layer),
        "lrp": lambda: xai._lrp(torch, model, vol, layer),
    }
    results = {}
    for name, fn in methods.items():
        peak = _cam_peak(fn(), vol)
        results[name] = _localises(peak)
        assert results[name], f"{name} peaked at {peak}, not on the blob {BLOB}"
    return results


# ─────────────────────────── mechanistic family (must pass its causal check) ───────────────────────────
def test_mechanistic_methods_pass_checks():
    model, vol, layer = xai._tiny_cnn(torch, nn); model.eval()
    checks = {
        "sparse_autoencoder": xai._sae(torch, nn, model, vol, layer),
        "activation_patching": xai._activation_patching(torch, model, vol, layer),
        "tcav_concept": xai._tcav(torch, nn, model, vol, layer),
        "protopnet": xai._protopnet(torch, model, vol, layer),
    }
    for name, r in checks.items():
        assert r.get("ok"), f"{name} failed its data check: {r}"
    return {k: v.get("ok") for k, v in checks.items()}


# ─────────────────────────── feature family (must rank the TRUE driver top) ───────────────────────────
def test_feature_attribution_finds_the_driver():
    """Plant a label driven by feature index 2; every attribution method must rank feature 2 highest."""
    torch.manual_seed(0); rng = np.random.RandomState(0)
    N, F, driver = 3000, 5, 2
    X = rng.randn(N, F).astype("float32")
    Y = (X[:, driver] + 0.25 * rng.randn(N) > 1.0).astype("float32")
    # tiny model that learns Y from X
    net = nn.Sequential(nn.Linear(F, 32), nn.GELU(), nn.Linear(32, 1))
    opt = torch.optim.Adam(net.parameters(), lr=1e-2)
    xt = torch.tensor(X); yt = torch.tensor(Y).unsqueeze(1)
    bce = nn.BCEWithLogitsLoss()
    for _ in range(300):
        opt.zero_grad(); bce(net(xt), yt).backward(); opt.step()
    net.eval()
    feat = [f"f{i}" for i in range(F)]
    res = xai._feature_methods(np, torch, net, X, Y, feat, "all")
    passed = {}
    for method, attrib in res.items():
        top = max(attrib, key=lambda k: abs(attrib[k]))
        passed[method] = (top == f"f{driver}")
        assert passed[method], f"{method} ranked {top} top, not the true driver f{driver}: {attrib}"
    return passed


def _run():
    print("=== XAI DATA-WISE VERIFIER (ground-truth checks) ===")
    total_pass = total = 0
    for label, fn in [("CNN localise", test_cnn_methods_localise_the_signal),
                      ("mechanistic", test_mechanistic_methods_pass_checks),
                      ("feature driver", test_feature_attribution_finds_the_driver)]:
        try:
            r = fn()
            for m, ok in r.items():
                total += 1; total_pass += int(bool(ok))
                print(f"  {'✅' if ok else '❌'} [{label}] {m}")
        except AssertionError as e:
            total += 1
            print(f"  ❌ [{label}] FAILED: {e}")
    print(f"\n=== {total_pass}/{total} methods pass DATA-WISE verification ===")
    return total_pass == total


if __name__ == "__main__":
    sys.exit(0 if _run() else 1)
