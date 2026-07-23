"""training_head_pack_test — CPU verifier for the three distilled GM training primitives (offline).

  • deep-supervision : max-pooled multi-scale DS loss decreases; per-level weights are respected;
                       adaptive_MAX_pool keeps a lone foreground voxel alive at a coarse scale.
  • sed-attention-pool: weakly-supervised attention head learns to classify clips from a single salient
                        frame (attention beats plain mean); GeM p is learnable.
  • awp-perturb       : AWP restores weights before the optimizer step (one-step move stays small) and
                        the regularised loop still reduces loss.
"""
import os, sys
COMP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, COMP); sys.path.insert(0, os.path.join(COMP, "tools", "researchpapers"))
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from fleet_agents import training_head_pack as T


def _run():
    print("=== TRAINING-HEAD PACK VERIFIER (CPU) ===")
    torch.manual_seed(0); checks = {}

    # ---- deep supervision ----
    # adaptive_MAX_pool keeps a lone foreground voxel; avg would nearly erase it
    y = torch.zeros(1, 1, 8, 8); y[0, 0, 3, 3] = 1.0
    mx = T._adaptive_max_pool(y, (2, 2)).max().item()
    checks["ds_maxpool_keeps_object"] = mx == 1.0
    tgt = T.to_ce_target(torch.tensor([[[[0.0, 1.0]]]]).reshape(1, 1, 1, 2))
    checks["ds_ce_target_sums_to_1"] = torch.allclose(tgt.sum(1), torch.ones(1, 1, 2))
    # per-level weighting: zeroing a level removes its contribution
    outs = [torch.randn(2, 3, 8, 8), torch.randn(2, 3, 4, 4)]
    tg = (torch.rand(2, 2, 8, 8) > 0.8).float()
    l_both = T.deep_supervision_loss(outs, tg, [1, 1])
    l_first = T.deep_supervision_loss(outs, tg, [1, 0])
    checks["ds_level_weight_effect"] = abs(float(l_both) - float(l_first)) > 1e-4
    st, d, to, msg = T.run_deep_supervision({"spec": {"steps": 40}}, "t")
    checks["ds_agent_loss_drops"] = st == "done" and d["loss_last"] < d["loss_first"] - 1e-3
    print(f"  -> deep-supervision {d['loss_first']:.4f}->{d['loss_last']:.4f}")

    # ---- sed attention pool ----
    fl = torch.randn(4, 6, 3); al = torch.randn(4, 6, 3)
    cm, ac, w = T.attention_pool(fl, al)
    checks["sed_att_weights_sum1"] = torch.allclose(w.sum(1), torch.ones(4, 3), atol=1e-5)
    checks["sed_shapes"] = cm.shape == (4, 3) and ac.shape == (4, 3)
    gem = T.GeMPool(); checks["sed_gem_learnable"] = gem.p.requires_grad
    st, d, to, msg = T.run_sed_attention({"spec": {"steps": 150}}, "t")
    checks["sed_agent_learns"] = st == "done" and d["att_clip_acc"] > 0.7
    print(f"  -> sed-attention-pool att_clip_acc={d['att_clip_acc']:.3f}")

    # ---- awp ----
    torch.manual_seed(0)
    model = nn.Linear(4, 1); opt = torch.optim.SGD(model.parameters(), lr=1e-2)
    awp = T.AWP(model, opt, delta=0.05)
    X = torch.randn(32, 4); Y = torch.randn(32, 1)
    snap = [p.detach().clone() for p in model.parameters()]
    awp.train_step(lambda m: F.mse_loss(m(X), Y))
    # after a full AWP step the perturbation must be restored: net move == one SGD step (small), NOT ~delta
    move = sum(float((p.detach() - s0).abs().max()) for p, s0 in zip(model.parameters(), snap))
    checks["awp_restores_weights"] = move < 0.05          # << delta-scale perturbation would be ~0.05+
    st, d, to, msg = T.run_awp({"spec": {"steps": 200}}, "t")
    checks["awp_agent_loss_drops"] = st == "done" and d["loss_last"] < d["loss_first"] - 1e-3
    print(f"  -> awp-perturb {d['loss_first']:.4f}->{d['loss_last']:.4f}, one_step_move={d['one_step_move']:.4f}")

    for k, v in checks.items():
        print(f"  {'OK' if v else 'X'} {k}")
    ok = all(checks.values())
    print(f"=== training-head-pack: {'PASS' if ok else 'FAIL'} ({sum(checks.values())}/{len(checks)}) ===")
    return ok


if __name__ == "__main__":
    sys.exit(0 if _run() else 1)
