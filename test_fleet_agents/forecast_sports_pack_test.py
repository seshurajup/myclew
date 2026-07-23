"""forecast_sports_pack_test — verifier for forecast/sports/best-of-N/segment agents (offline, synthetic)."""
import os, sys
COMP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, COMP); sys.path.insert(0, os.path.join(COMP, "tools", "researchpapers"))
import numpy as np
from fleet_agents import forecast_sports_pack as F


def _run():
    print("=== FORECAST/SPORTS PACK VERIFIER ===")
    rng = np.random.RandomState(0); checks = {}

    # ts-decompose: y = level * dow_factor * store_factor → reconstruction ~ y
    n = 700; dow = rng.randint(0, 7, n); store = rng.randint(0, 3, n)
    dow_f = np.array([0.8, 0.9, 1.0, 1.1, 1.2, 1.5, 0.7]); store_f = np.array([1.0, 1.3, 0.7])
    y = 100 * dow_f[dow] * store_f[store] * np.exp(rng.normal(0, 0.02, n))
    level, factors, recon = F.decompose_multiplicative(y, {"dow": dow, "store": store})
    checks["decompose_reconstructs"] = np.mean(np.abs(recon - y) / y) < 0.05
    checks["decompose_factors"] = "dow" in factors and len(factors["dow"]) == 7

    # trend extrapolator: values trending up → linear multiplier for future > const
    years = np.array([2015, 2016, 2017, 2018, 2019.0]); vals = np.array([1.02, 1.04, 1.06, 1.08, 1.10])
    mult, best = F.choose_trend(years, vals, [2020, 2021])
    checks["trend_linear_grows"] = mult["linear"][2021] > mult["linear"][2020] > 1.10
    checks["trend_best_linear"] = best == "linear"

    # rating-systems: a dominant team gets the top rating
    games = []
    for _ in range(200):
        i, j = rng.choice(4, 2, replace=False)
        margin = (10 if i == 0 else (-10 if j == 0 else rng.randint(-5, 6)))  # team 0 always wins big
        games.append((int(i), int(j), int(margin)))
    elo = F.elo_ratings(games); colley = F.colley_ratings(games)
    checks["elo_dominant_top"] = max(elo, key=elo.get) == 0
    checks["colley_dominant_top"] = max(colley, key=colley.get) == 0

    # outcome-sharpen: confident preds pushed to tails, mid untouched, override applied
    p = np.array([0.98, 0.5, 0.02, 0.7])
    sp = F.sharpen(p, overrides={1: 0.99})
    checks["sharpen_tails"] = sp[0] >= 0.995 and sp[2] <= 0.005 and sp[3] == 0.7 and sp[1] == 0.99

    # best-of-n: 5 candidates, 2 tight clusters + 1 outlier → pick spread-out ones
    cands = np.array([[0, 0], [0.1, 0], [5, 5], [5.1, 5], [10, 0]])
    chosen = F.allocate_best_of_n(cands, 3, quality=np.array([1, 0.9, 1, 0.9, 0.5]))
    # should pick one from each distinct region (indices spanning the 3 clusters)
    regions = {0: {0, 1}, 1: {2, 3}, 2: {4}}
    covered = sum(1 for r in regions.values() if r & set(chosen))
    checks["bestofn_diverse"] = covered == 3

    # temporal-segment-decoder: two runs above threshold, one too short
    fp = np.array([0.1, 0.9, 0.9, 0.9, 0.1, 0.9, 0.1, 0.8, 0.8, 0.8, 0.8])
    segs = F.decode_segments(fp, threshold=0.5, min_len=3)
    checks["segdecode"] = len(segs) == 2 and segs[0][:2] == (1, 3) and segs[1][0] == 7

    # agent contracts
    st, d, to, msg = F.run_rating({"spec": {"games": [list(g) for g in games]}}, "t")
    checks["rating_agent"] = st == "done" and "elo" in d
    st, d, to, msg = F.run_segdecode({"spec": {"frame_prob": fp.tolist(), "threshold": 0.5, "min_len": 3}}, "t")
    checks["seg_agent"] = st == "done" and d["n_segments"] == 2

    for k, v in checks.items():
        print(f"  {'OK' if v else 'X'} {k}")
    ok = all(checks.values())
    print(f"=== forecast-sports-pack: {'PASS' if ok else 'FAIL'} ({sum(checks.values())}/{len(checks)}) ===")
    return ok


if __name__ == "__main__":
    sys.exit(0 if _run() else 1)
