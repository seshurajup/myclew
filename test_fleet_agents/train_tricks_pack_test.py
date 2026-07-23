"""train_tricks_pack_test — DATA-WISE verifier for the REUSABLE train-tricks pack. Checks each winner
technique on a tiny synthetic batch: EMA shadow tracks the live weights; mixup/cutmix return valid lam∈[0,1]
with correct shapes; focal loss ≤ CE on easy examples and up-weights hard ones; label-smoothing targets sum
to 1 and match plain CE at smoothing=0; SAM takes two steps and actually moves the weights; ArcFace forward
has the right shape and the margin lowers the true-class logit vs plain cosine. Also runs the agent run()."""
import os, sys
COMP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, COMP); sys.path.insert(0, os.path.join(COMP, "tools", "researchpapers"))
import torch
import torch.nn.functional as F
from torch import nn
from fleet_agents import train_tricks_pack as TT


def _run():
    print("=== TRAIN-TRICKS DATA-WISE VERIFIER ===")
    torch.manual_seed(0)
    checks = {}

    # (1) EMA — shadow tracks the live weights (moves toward them after update, stays between old and new)
    model = nn.Linear(8, 4)
    w0 = model.weight.detach().clone()
    ema = TT.ModelEMA(model, decay=0.5)
    with torch.no_grad():
        model.weight.add_(torch.ones_like(model.weight))     # bump live weights by +1
    ema.update(model)
    ew = dict(ema.ema.named_parameters())["weight"].detach()
    # decay=0.5, warmup ramp → ema moved off w0 toward w0+1, strictly between
    moved = (ew - w0).abs().mean().item()
    between = torch.all(ew >= w0 - 1e-6) and torch.all(ew <= w0 + 1.0 + 1e-6)
    checks["ema_tracks_weights"] = moved > 1e-3 and bool(between)
    # copy_to writes EMA weights into a fresh model
    m2 = nn.Linear(8, 4)
    ema.copy_to(m2)
    checks["ema_copy_to"] = torch.allclose(m2.weight.detach(), ew)

    # (2) mixup — valid lam and correct shapes
    x = torch.randn(6, 8); y = torch.randint(0, 4, (6,))
    mx, ya, yb, lam = TT.mixup_data(x, y, alpha=0.4, seed=1)
    checks["mixup_lam"] = (0.0 <= lam <= 1.0) and mx.shape == x.shape and ya.shape == y.shape and yb.shape == y.shape

    # (3) cutmix — valid lam, image shape preserved
    img = torch.randn(6, 3, 16, 16)
    cx, ca, cb, lam2 = TT.cutmix_data(img, y, alpha=1.0, seed=2)
    checks["cutmix_lam"] = (0.0 <= lam2 <= 1.0) and cx.shape == img.shape

    # (4) mixup_criterion reduces to plain loss at lam=1
    crit = nn.CrossEntropyLoss()
    logits = torch.randn(6, 4)
    mc = TT.mixup_criterion(crit, logits, y, y[torch.randperm(6)], 1.0)
    checks["mixup_criterion"] = torch.allclose(mc, crit(logits, y))

    # (5) focal loss — down-weights EASY examples (focal < CE) and focuses HARD ones (focal_hard/ce_hard bigger)
    #   easy: confident-correct binary examples; hard: confident-wrong
    easy_logit = torch.tensor([4.0, 4.0, -4.0, -4.0]); easy_tgt = torch.tensor([1., 1., 0., 0.])
    hard_logit = torch.tensor([-4.0, -4.0, 4.0, 4.0]); hard_tgt = torch.tensor([1., 1., 0., 0.])
    ce_easy = F.binary_cross_entropy_with_logits(easy_logit, easy_tgt)
    fl_easy = TT.binary_focal_loss(easy_logit, easy_tgt, alpha=-1, gamma=2.0)
    ce_hard = F.binary_cross_entropy_with_logits(hard_logit, hard_tgt)
    fl_hard = TT.binary_focal_loss(hard_logit, hard_tgt, alpha=-1, gamma=2.0)
    # focal shrinks easy-example loss far more than hard-example loss
    easy_shrink = (fl_easy / ce_easy).item()
    hard_shrink = (fl_hard / ce_hard).item()
    checks["focal_downweights_easy"] = fl_easy < ce_easy and easy_shrink < hard_shrink

    # multiclass focal finite + ≤ CE on an easy (confident-correct) batch
    easy_ml = torch.tensor([[6.0, 0, 0, 0], [0, 6.0, 0, 0]]); easy_y = torch.tensor([0, 1])
    fl_ml = TT.FocalLoss(gamma=2.0)(easy_ml, easy_y)
    ce_ml = F.cross_entropy(easy_ml, easy_y)
    checks["focal_multiclass"] = torch.isfinite(fl_ml) and fl_ml <= ce_ml + 1e-6

    # (6) label smoothing — smoothed targets sum to 1; loss == CE at smoothing=0
    st = TT.smooth_one_hot(y, 4, smoothing=0.1)
    checks["smooth_target_sums_to_1"] = torch.allclose(st.sum(dim=1), torch.ones(6), atol=1e-5)
    ls0 = TT.label_smoothing_cross_entropy(logits, y, smoothing=0.0)
    checks["ls_equals_ce_at_zero"] = torch.allclose(ls0, F.cross_entropy(logits, y), atol=1e-5)
    ls1 = TT.label_smoothing_cross_entropy(logits, y, smoothing=0.1)
    checks["ls_finite"] = bool(torch.isfinite(ls1))

    # (7) SAM — two steps run without error AND actually move the weights
    m3 = nn.Linear(8, 4)
    before = m3.weight.detach().clone()
    opt = TT.make_sam(torch.optim.SGD, m3.parameters(), lr=0.1, momentum=0.9, rho=0.05)
    F.cross_entropy(m3(x), y).backward(); opt.first_step(zero_grad=True)
    F.cross_entropy(m3(x), y).backward(); opt.second_step(zero_grad=True)
    checks["sam_two_steps_move"] = not torch.allclose(before, m3.weight.detach())

    # (8) ArcFace — forward shape correct; margin lowers the true-class logit vs plain scaled cosine
    head = TT.build_arcface(8, 4, s=30.0, m=0.5, k=3)
    emb = torch.randn(6, 8)
    train_logits = head(emb, y)          # with margin
    infer_logits = head(emb, None)       # plain scaled cosine
    checks["arcface_shape"] = tuple(train_logits.shape) == (6, 4)
    true_margin = train_logits.gather(1, y.view(-1, 1))
    true_plain = infer_logits.gather(1, y.view(-1, 1))
    checks["arcface_margin_penalizes_true"] = torch.all(true_margin <= true_plain + 1e-4)

    # (9) SWA — averaged model builds and update_bn is a safe no-op when there is no BN
    swa = TT.swa_average_model(nn.Linear(8, 4))
    swa.update_parameters(nn.Linear(8, 4))
    TT.swa_update_bn([(torch.randn(4, 8),)], swa)
    checks["swa_builds"] = True

    # (10) agent run() emits a done verdict with per-trick checks
    st_, d, to, msg = TT.TrainTricksPack().run({"question": "tricks", "spec": {"device": "cpu"}}, "test")
    checks["agent_runs"] = st_ == "done" and isinstance(d.get("checks"), dict) and all(d["checks"].values())

    for k, v in checks.items():
        print(f"  {'OK' if v else 'X'} {k}")
    ok = all(checks.values())
    print(f"\n=== train-tricks: {'PASS' if ok else 'FAIL'} ({sum(bool(v) for v in checks.values())}/{len(checks)}) ===")
    return ok


if __name__ == "__main__":
    try:
        sys.exit(0 if _run() else 1)
    except AssertionError as e:
        print("ASSERTION", e); sys.exit(1)
