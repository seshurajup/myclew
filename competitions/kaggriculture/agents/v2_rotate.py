"""v2 — one unit SERVICES MANY TILES.

v1 scored 4859 vs starter 3506 but left 20 of 25 tiles empty all season: each unit walked to one tile and
worked that tile forever (plant -> water -> harvest -> replant). With 24 turns per day a single unit can
service many tiles, so the fix is to give every unit a SET of tiles and send it to the most urgent one.

Priority per tile: HARVEST (money now) > PLANT (start the clock) > WATER (protect the yield) > DIG a weed.
Distance breaks ties, so a unit does the nearest useful thing rather than crossing the farm.

Two "obvious" upgrades were tried and REJECTED on 16-seed evidence, both of which single runs would have
sold as wins:
  * BUY_LAND once the home quadrant saturates (it does saturate: 24/25 tiles planted by day 20).
    8468 -> 3489, win rate 100% -> 50%. Extra tiles sit across a 10x10 board, so units spend turns walking
    instead of farming, while the 1k-4k purchase drains the cash that buys seed.
  * More hands (10 instead of 6): 8468 -> 4787, 100% -> 62.5%. The fibonacci hire tail outruns the extra
    labour's output.
Neither is re-tried without a mechanism that fixes the underlying cost (local land, or labour that pays for
itself).
"""

CROP = "CARROT"
SEED_COST = 20
MAX_YIELD_DAY = 3
MAX_HANDS = 6


def _tiles(farm):
    return [(x, y) for y, row in enumerate(farm["tiles"]) for x, t in enumerate(row) if t != "LOCKED"]


def _dist(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _need(farm, xy, day, seeds_left):
    """(priority, action) for a tile — lower priority number is more urgent. None if nothing to do."""
    x, y = xy
    t = farm["tiles"][y][x]
    if t is None:
        return (1, ["PLANT", CROP]) if seeds_left > 0 else None
    if isinstance(t, dict):
        if t.get("kind") == "WEED":
            return (3, ["DIG"])
        if t.get("kind") == "PLANT" and t.get("crop") == CROP:
            if day - t.get("planted_day", day) >= MAX_YIELD_DAY:
                return (0, ["HARVEST"])
            if not t.get("watered_today"):
                return (2, ["WATER"])
    return None


def _move(pos, target):
    x, y = pos
    tx, ty = target
    if x < tx: return ["EAST"]
    if x > tx: return ["WEST"]
    if y < ty: return ["SOUTH"]
    if y > ty: return ["NORTH"]
    return None


def agent(obs):
    farms = obs.get("farms", [])
    player = obs.get("player", 0)
    if not farms or player >= len(farms):
        return {"farmer": ["PASS"], "hands": [], "market": []}
    farm = farms[player]
    priv = obs.get("private", {}) or {}
    seeds = dict(priv.get("seeds", {}) or {})
    shed = priv.get("shed", {}) or {}
    day, hour, money = obs.get("day", 0), obs.get("hour", 0), farm["money"]

    hands = farm.get("hands") or []
    positions = [tuple(farm["farmer"])] + [tuple(h) for h in hands]
    n_units = len(positions)
    all_tiles = _tiles(farm)

    market = []
    if hour == 0:
        # HIRE costs fib(n_already_today): 1,1,2,3,5,8,13,21,34,55,89,144,... Measured sweep at 720 steps:
        # 6 hands ~9.6k, 12 hands 2.4k, 20 hands 2.6k — past ~8 the fibonacci tail bankrupts the farm
        # (hires 13-20 alone cost thousands per DAY). So budget explicitly against the actual cost instead
        # of a flat money guard, and stop when the next hand costs more than a carrot harvest is worth.
        already = farm.get("hires_today", 0)
        budget = money * 0.15
        a, b = 1, 1
        for _ in range(already):
            a, b = b, a + b
        for _ in range(max(0, MAX_HANDS - already)):
            if a > budget or a > 3 * SEED_COST:
                break
            budget -= a
            market.append(["HIRE"])
            a, b = b, a + b
    # keep enough seed for every empty tile a unit might reach today
    empties = sum(1 for (x, y) in all_tiles if farm["tiles"][y][x] is None)
    want = max(0, min(empties, 12) - seeds.get(CROP, 0))
    if want > 0 and money > SEED_COST * want + 100:
        market.append(["BUY_SEED", CROP, want])
    if shed.get(CROP, 0) > 0:
        market.append(["SELL", CROP, shed[CROP]])

    # Greedy assignment: each unit takes the most urgent unclaimed tile, nearest first. Claiming prevents
    # two units walking to the same tile and wasting a turn.
    seeds_left = seeds.get(CROP, 0)
    claimed = set()
    acts = []
    for pos in positions:
        best = None
        for xy in all_tiles:
            if xy in claimed:
                continue
            n = _need(farm, xy, day, seeds_left)
            if n is None:
                continue
            key = (n[0], _dist(pos, xy))
            if best is None or key < best[0]:
                best = (key, xy, n[1])
        if best is None:
            acts.append(["PASS"])
            continue
        _, xy, action = best
        claimed.add(xy)
        mv = _move(pos, xy)
        if mv:
            acts.append(mv)
        else:
            if action[0] == "PLANT":
                seeds_left -= 1
            acts.append(action)
    return {"farmer": acts[0], "hands": acts[1:], "market": market}
