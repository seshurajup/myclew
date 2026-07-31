"""v1 — multi-tile carrot loop with hired hands.

The built-in `starter` scores 3503 by farming ONE tile with ONE farmer: 24 of its 25 unlocked tiles sit
idle all season. Two levers it ignores:

  * HIRE costs fib(n) per day and RESETS daily — six hands cost 1+1+2+3+5+8 = 20 coins total. Labour is
    effectively free, and every unit acts independently every turn.
  * The NW quadrant is 25 tiles. Work them in parallel instead of one.

Crop choice is carrot on revenue per tile-day: carrot yields 4 units over 3 days at base 35 (~46.6/tile/day)
versus wheat 6 over 4 at base 25 (~37.5). Tomato is higher still but needs 8 days to first yield, so it is a
v2 question once the loop is proven.

Each unit owns a tile, walks to it, then runs PLANT -> WATER -> HARVEST. Deliberately simple: it exists to
establish a real baseline above `starter`, measured locally, before anything cleverer is attempted.
"""

CROP = "CARROT"
SEED_COST = 20
MAX_YIELD_DAY = 3
MAX_HANDS = 6


def _quadrant_tiles(farm):
    """Unlocked, workable tiles (skip the shed row so units are not stuck on it)."""
    out = []
    for y, row in enumerate(farm["tiles"]):
        for x, t in enumerate(row):
            if t == "LOCKED":
                continue
            out.append((x, y))
    return out


def _step_toward(pos, target):
    x, y = pos
    tx, ty = target
    if x < tx:
        return ["EAST"]
    if x > tx:
        return ["WEST"]
    if y < ty:
        return ["SOUTH"]
    if y > ty:
        return ["NORTH"]
    return None


def _unit_action(farm, pos, target, seeds, day):
    """Walk to the assigned tile, then work it."""
    mv = _step_toward(pos, target)
    if mv:
        return mv
    x, y = pos
    tile = farm["tiles"][y][x]
    if tile is None:
        return ["PLANT", CROP] if seeds.get(CROP, 0) > 0 else ["PASS"]
    if isinstance(tile, dict):
        if tile.get("kind") == "WEED":
            return ["DIG"]
        if tile.get("kind") == "PLANT":
            if tile.get("crop") != CROP:
                return ["PASS"]
            if day - tile.get("planted_day", day) >= MAX_YIELD_DAY:
                return ["HARVEST"]
            if not tile.get("watered_today"):
                return ["WATER"]
    return ["PASS"]


def agent(obs):
    farms = obs.get("farms", [])
    player = obs.get("player", 0)
    if not farms or player >= len(farms):
        return {"farmer": ["PASS"], "hands": [], "market": []}
    farm = farms[player]
    private = obs.get("private", {}) or {}
    seeds = private.get("seeds", {}) or {}
    shed = private.get("shed", {}) or {}
    day = obs.get("day", 0)
    hour = obs.get("hour", 0)
    money = farm["money"]

    hands = farm.get("hands") or []
    n_units = 1 + len(hands)
    tiles = _quadrant_tiles(farm)

    market = []
    # hire at the START of each day, while fib() is still cheap. A turn accepts up to
    # maxMarketOrdersPerTurn (10) orders, so issue the whole day's hires at once rather than one per turn —
    # one-per-turn capped us at 3 hands and left most tiles idle.
    if hour == 0:
        already = farm.get("hires_today", 0)
        for _ in range(max(0, MAX_HANDS - already)):
            if money > 300:
                market.append(["HIRE"])
    # keep a seed buffer sized to the workforce
    want = max(0, n_units + 2 - seeds.get(CROP, 0))
    if want > 0 and money > SEED_COST * want + 100:
        market.append(["BUY_SEED", CROP, want])
    if shed.get(CROP, 0) > 0:
        market.append(["SELL", CROP, shed[CROP]])

    positions = [tuple(farm["farmer"])] + [tuple(h) for h in hands]
    acts = []
    for i, pos in enumerate(positions):
        target = tiles[i % len(tiles)] if tiles else pos
        acts.append(_unit_action(farm, pos, target, seeds, day))

    return {"farmer": acts[0], "hands": acts[1:], "market": market}
