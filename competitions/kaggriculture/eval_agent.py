"""Multi-seed agent evaluation — because single-episode scores are noise.

Weeds spawn stochastically and the shared market makes outcomes opponent-dependent, so one episode cannot
rank two agents. This runs N seeds per matchup in parallel and reports mean, std and WIN RATE — win rate
being what the ladder actually scores (Bradley-Terry on win/loss; coin margin is ignored).
"""
import sys, statistics
from multiprocessing import Pool


def _one(args):
    a, b, seed = args
    from kaggle_environments import make
    env = make("kaggriculture", configuration={"episodeSteps": 720}, debug=False)
    env.reset(2)
    env.run([a, b])
    f = env.steps[-1]
    return float(f[0]["reward"] or 0), float(f[1]["reward"] or 0)


def evaluate(a, b, n=16, workers=8):
    with Pool(workers) as p:
        res = p.map(_one, [(a, b, s) for s in range(n)])
    us = [r[0] for r in res]
    them = [r[1] for r in res]
    wins = sum(1 for u, t in zip(us, them) if u > t)
    ties = sum(1 for u, t in zip(us, them) if u == t)
    return {"n": n, "mean": statistics.mean(us), "std": statistics.pstdev(us),
            "opp_mean": statistics.mean(them), "wins": wins, "ties": ties,
            "win_rate": wins / n}


if __name__ == "__main__":
    a, b = sys.argv[1], sys.argv[2]
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 16
    r = evaluate(a, b, n)
    print(f"{a} vs {b}   n={r['n']}")
    print(f"  ours  {r['mean']:8.0f} +/- {r['std']:.0f}")
    print(f"  opp   {r['opp_mean']:8.0f}")
    print(f"  WIN RATE {r['win_rate']:.1%}  ({r['wins']}W {r['ties']}T {r['n']-r['wins']-r['ties']}L)")
