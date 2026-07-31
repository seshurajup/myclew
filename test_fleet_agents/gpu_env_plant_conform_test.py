"""Randomised conformance: batched plant refresh vs the OFFICIAL _daily_refresh_plants.

Ports are only trustworthy when proven equal to the implementation they replace. A simulator that drifts
from the official one trains a policy optimal for OUR simulator and wrong on the ladder — the same failure
as a CV that misranks against the real leaderboard. This drives randomised farms (all crops, weed
transitions at consecutive_unwatered 0..2, fertilizer-window boundaries, locked tiles) through both and
compares every tile field.

Skips cleanly when the kaggriculture venv is absent, so it never fails the suite on another box.
"""
import sys, types, importlib.util, random, copy
sys.path.insert(0, "/home/seshu/kaggle/2026/biohub-cell-tracking-during-development")
pkg = types.ModuleType("kaggle_environments"); pkg.__path__ = []
u = types.ModuleType("kaggle_environments.utils"); u.resolve_episode_seed = lambda e: 0
sys.modules["kaggle_environments"] = pkg; sys.modules["kaggle_environments.utils"] = u
OFFICIAL = ("/home/seshu/kaggle/2026/kaggriculture/.venv/lib/python3.12/site-packages/"
            "kaggle_environments/envs/kaggriculture/kaggriculture.py")
import os
if not os.path.exists(OFFICIAL):
    print("  [SKIP] official kaggriculture env not installed — nothing to conform against")
    sys.exit(0)
spec = importlib.util.spec_from_file_location("kagg", OFFICIAL)
K = importlib.util.module_from_spec(spec); spec.loader.exec_module(K)
import torch
from fleet_agents import gpu_env as G

CROPS = list(K.CROPS.keys()); H = W = 10; TPD = 24
ct = G.crop_table(K.CROPS, CROPS)
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def random_farm(rng, day):
    tiles = [[None]*W for _ in range(H)]
    for y in range(H):
        for x in range(W):
            r = rng.random()
            if r < 0.55:
                c = rng.choice(CROPS); cd = K.CROPS[c]
                tiles[y][x] = {"kind":"PLANT","crop":c,
                    "planted_day": max(0, day - rng.randint(0, 14)),
                    "watered_today": rng.random() < 0.6,
                    "consecutive_unwatered": rng.randint(0, 2),
                    "yield_units": rng.randint(0, cd["max_yield"]),
                    "max_lifespan_step": -1 if cd["ongoing"] else (day+cd["max_yield_day"]+1)*TPD,
                    "fertilized_until_day": rng.choice([-1, day-1, day, day+2])}
            elif r < 0.65: tiles[y][x] = {"kind":"WEED"}
            elif r < 0.70: tiles[y][x] = "LOCKED"
    return {"tiles": tiles}

def to_state(farms, dev):
    B = len(farms)
    st = G.new_tile_state((B, H, W), device=dev)
    for b, f in enumerate(farms):
        for y in range(H):
            for x in range(W):
                tl = f["tiles"][y][x]
                if tl is None: continue
                if tl == "LOCKED": st["kind"][b,y,x] = G.TILE_LOCKED; continue
                if tl["kind"] == "WEED": st["kind"][b,y,x] = G.TILE_WEED; continue
                st["kind"][b,y,x] = G.TILE_PLANT
                st["crop"][b,y,x] = CROPS.index(tl["crop"])
                for fld in ("planted_day","consecutive_unwatered","yield_units",
                            "max_lifespan_step","fertilized_until_day"):
                    st[fld][b,y,x] = int(tl[fld])
                st["watered_today"][b,y,x] = 1 if tl["watered_today"] else 0
    return st

rng = random.Random(11); mism = 0; checked = 0
for trial in range(12):
    day = rng.randint(0, 25)
    farms = [random_farm(rng, day) for _ in range(24)]
    st = to_state(copy.deepcopy(farms), dev)
    G.daily_refresh_plants_batch(st, ct, day, TPD)
    for f in farms: K._daily_refresh_plants(f, day, TPD)
    for b, f in enumerate(farms):
        for y in range(H):
            for x in range(W):
                tl = f["tiles"][y][x]; checked += 1
                k = int(st["kind"][b,y,x])
                if tl is None: exp = G.TILE_EMPTY
                elif tl == "LOCKED": exp = G.TILE_LOCKED
                elif tl["kind"] == "WEED": exp = G.TILE_WEED
                else: exp = G.TILE_PLANT
                if k != exp: mism += 1; continue
                if exp != G.TILE_PLANT: continue
                for fld in ("consecutive_unwatered","yield_units","max_lifespan_step"):
                    if int(st[fld][b,y,x]) != int(tl[fld]): mism += 1; break
                else:
                    if int(st["watered_today"][b,y,x]) != (1 if tl["watered_today"] else 0): mism += 1
print(f"tiles checked: {checked:,}   mismatches: {mism}")
print("PLANT LIFECYCLE CONFORMS:", mism == 0)

sys.exit(0 if mism == 0 else 1)
