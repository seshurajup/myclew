"""gpu_env — batched, CUDA-resident simulation for agent-ladder competitions, with a CONFORMANCE gate.

Why: `kaggle-environments` runs one episode at a time in Python. Measured on kaggriculture: 0.9s/episode,
11.1 eps/s across 8 cores = ~40k episodes/hour. Fine for tuning a heuristic, thin for RL from scratch, which
wants millions. A batched tensor port runs thousands of games simultaneously on one GPU (the Gymnax/Brax/PGX
pattern), turning hours into minutes.

The DANGER is the whole point of this module. A reimplemented environment that differs subtly from the
official one trains a policy that is optimal for OUR simulator and wrong on the ladder — the identical
failure mode as a CV that misranks against the real leaderboard. So nothing here is trusted until it is
proven equal to the official implementation, step by step, on the same inputs.

    conform(gpu_fn, ref_fn, cases)   ->  {"equal": bool, "max_abs_err": float, "first_mismatch": ...}

Reusable across competitions: a new agent comp implements `BatchedEnv` and ships its conformance cases; the
framework, the gate and the throughput harness are unchanged.
"""
from __future__ import annotations

import time


def torch_device(prefer_gpu=True):
    """The device to simulate on. CPU fallback keeps every test runnable on a box without CUDA."""
    import torch
    if prefer_gpu and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class BatchedEnv:
    """Base class: B independent games advanced in lockstep as tensors.

    Subclasses implement `reset(B)` and `step(actions)`, keeping ALL state in tensors with a leading batch
    dimension so a single kernel advances every game. `n_envs` games cost roughly one game's wall time.
    """

    def __init__(self, n_envs=1024, device=None, dtype=None):
        import torch
        self.n_envs = int(n_envs)
        self.device = device or torch_device()
        self.dtype = dtype or torch.float32

    def reset(self, seed=None):
        raise NotImplementedError

    def step(self, actions):
        raise NotImplementedError

    def throughput(self, steps=720, warmup=2):
        """Measured env-steps/second and episodes/hour — the number that justifies (or kills) the port."""
        import torch
        self.reset()
        for _ in range(warmup):
            self.step(None)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(steps):
            self.step(None)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        dt = time.time() - t0
        eps_h = (self.n_envs * steps / max(dt, 1e-9)) / steps * 3600
        return {"device": str(self.device), "n_envs": self.n_envs, "steps": steps,
                "seconds": round(dt, 3),
                "env_steps_per_sec": round(self.n_envs * steps / max(dt, 1e-9), 1),
                "episodes_per_hour": round(eps_h)}


def conform(gpu_fn, ref_fn, cases, atol=1e-6, rtol=1e-5):
    """Prove a batched implementation EQUALS the official scalar one on `cases`.

    `gpu_fn(batch)` returns a tensor/array of results for the whole batch; `ref_fn(case)` returns the
    official scalar for one case. Reports the FIRST mismatch, not just a count — when a port diverges you
    need the input that broke it, not the fact that something did.
    """
    import torch
    got = gpu_fn(cases)
    got = got.detach().cpu() if hasattr(got, "detach") else torch.as_tensor(got)
    want = torch.as_tensor([float(ref_fn(c)) for c in cases], dtype=torch.float64)
    got = got.to(torch.float64).reshape(-1)
    if got.shape != want.shape:
        return {"equal": False, "why": f"shape {tuple(got.shape)} vs {tuple(want.shape)}",
                "n": len(cases)}
    diff = (got - want).abs()
    tol = atol + rtol * want.abs()
    bad = (diff > tol).nonzero().flatten()
    first = None
    if bad.numel():
        i = int(bad[0])
        first = {"index": i, "case": cases[i], "gpu": float(got[i]), "official": float(want[i]),
                 "abs_err": float(diff[i])}
    return {"equal": bad.numel() == 0, "n": len(cases), "n_mismatch": int(bad.numel()),
            "max_abs_err": float(diff.max()) if diff.numel() else 0.0, "first_mismatch": first}


# ─────────────────────────────────────────────────────────────────────────────────────────────────────
# KAGGRICULTURE — the market model, ported exactly.
#
# The market is where this game is decided, and it is pure arithmetic, so it ports cleanly and is fully
# verifiable against the official `market_price`. Ported first for that reason: it is the piece a strategy
# search actually needs (when to sell, what to dump, what to corner) and the piece we can PROVE correct.
# ─────────────────────────────────────────────────────────────────────────────────────────────────────

SHAPES = ("linear", "sq", "sqrt", "log", "log10")


def shape_batch(func, x):
    """Vectorised `_shape`: f(0)=0 for every shape; `log` is ln(1+x), matching the official implementation."""
    import torch
    x = torch.clamp(x, min=0.0)
    if func == "linear":
        return x
    if func == "sq":
        return x * x
    if func == "sqrt":
        return torch.sqrt(x)
    if func == "log":
        return torch.log1p(x)
    if func == "log10":
        return torch.log1p(x) / torch.log(torch.tensor(10.0, dtype=x.dtype, device=x.device))
    return x


def market_price_batch(params, inventory, price_floor=1.0):
    """price(inv) = base ± amp·f(|inv − I0|), amp = target·base/f(T), floored — for a whole batch at once.

    `params`: dict with base/I0/T/below_func/below_target/above_func/above_target (one product).
    `inventory`: tensor [B] of market inventories.
    Returns tensor [B] of prices, rounded and floored exactly as the official code does.
    """
    import torch
    inv = torch.as_tensor(inventory, dtype=torch.float64)
    i0 = float(params["I0"])
    base = float(params["base"])
    t = float(params["T"])
    below = inv < i0
    dist = (inv - i0).abs()

    def amp(func, target):
        f_t = shape_batch(func, torch.tensor([t], dtype=torch.float64))
        return target * base / float(f_t.item() if f_t.item() != 0 else 1.0)

    p_below = base + amp(params["below_func"], params["below_target"]) * shape_batch(params["below_func"], dist)
    p_above = base - amp(params["above_func"], params["above_target"]) * shape_batch(params["above_func"], dist)
    price = torch.where(below, p_below, p_above)
    return torch.clamp(torch.round(price), min=float(price_floor))


# ─────────────────────────────────────────────────────────────────────────────────────────────────────
# EPISODE-LEVEL CONFORMANCE — the operational definition of "identical".
#
# Per-function checks (market prices, shape functions) prove pieces. They cannot prove a SIMULATION, where
# an error appears at step 3 and only becomes visible in the reward 700 steps later. This runs the SAME
# action sequence through the official environment and a port, comparing the full observable state after
# EVERY step, and reports the first step and field that diverged.
#
# Determinism note: the official env draws weeds and shop unlocks from `random.Random((seed*1_000_003)^day)`
# — reseeded per day, consumed farm-by-farm then y,x. Exact parity is therefore achievable by generating
# those draws on CPU and feeding the identical stream to the port, rather than trying to reproduce the
# Mersenne Twister on-device. Set `weed_chance=0` to compare the purely deterministic core first: an
# unexplained divergence there is a REAL logic bug, not RNG drift.
# ─────────────────────────────────────────────────────────────────────────────────────────────────────

def farm_fingerprint(farm, private=None):
    """A comparable, order-stable summary of one farm's observable state."""
    tiles = []
    for row in farm.get("tiles", []):
        for t in row:
            if t is None:
                tiles.append("_")
            elif t == "LOCKED":
                tiles.append("L")
            else:
                k = t.get("kind", "?")
                if k == "PLANT":
                    tiles.append(f"P:{t.get('crop')}:{t.get('yield_units')}:{t.get('watered_today')}:"
                                 f"{t.get('consecutive_unwatered')}")
                elif k == "WEED":
                    tiles.append("W")
                else:
                    tiles.append(f"{k}:{t.get('animal')}:{t.get('yield_units')}:{t.get('fed_today')}:"
                                 f"{t.get('pending_care_bonus')}")
    out = {"money": farm.get("money"), "farmer": tuple(farm.get("farmer") or ()),
           "hands": tuple(tuple(h) for h in farm.get("hands") or ()),
           "quadrants": tuple(sorted(farm.get("unlocked_quadrants") or ())),
           "hires_today": farm.get("hires_today"), "tiles": tuple(tiles)}
    if private is not None:
        out["shed"] = tuple(sorted((private.get("shed") or {}).items()))
        out["seeds"] = tuple(sorted((private.get("seeds") or {}).items()))
    return out


def episode_conform(official_env, port_step, actions_per_step, max_steps=720, compare_market=True):
    """Step the official env and a port in lockstep; return the FIRST divergence.

    `official_env` — a live kaggle_environments env, already made.
    `port_step(step_idx, actions) -> {"farms": [...], "market": {...}}` — the port's state after that step.
    `actions_per_step(step_idx) -> [action_p0, action_p1]` — the SAME actions given to both.

    Returns {"identical": bool, "steps_compared": int, "first_divergence": {...}}. A port is only
    trustworthy for training once this is identical over full episodes on many seeds.
    """
    diverged = None
    n = 0
    for i in range(int(max_steps)):
        acts = actions_per_step(i)
        official_env.step(acts)
        obs = official_env.state[0].observation
        port = port_step(i, acts)
        n = i + 1
        off_fp = [farm_fingerprint(f) for f in obs["farms"]]
        prt_fp = [farm_fingerprint(f) for f in port["farms"]]
        if off_fp != prt_fp:
            for pi, (a, b) in enumerate(zip(off_fp, prt_fp)):
                if a != b:
                    fields = [k for k in a if a[k] != b.get(k)]
                    diverged = {"step": i, "player": pi, "fields": fields,
                                "official": {k: a[k] for k in fields},
                                "port": {k: b.get(k) for k in fields}}
                    break
            break
        if compare_market:
            om = dict(obs["market"].get("inventory") or {})
            pm = dict((port.get("market") or {}).get("inventory") or {})
            if om != pm:
                diverged = {"step": i, "player": None, "fields": ["market.inventory"],
                            "official": om, "port": pm}
                break
        if official_env.done:
            break
    return {"identical": diverged is None, "steps_compared": n, "first_divergence": diverged}


def precompute_market(params_by_product, products, device=None, dtype=None):
    """Hoist every per-product CONSTANT to the host, once.

    Measured defect in `market_price_batch`: it recomputed `amp = target*base/f(T)` on EVERY call by running
    `shape_batch` over a 1-element tensor — launching CUDA kernels to do scalar arithmetic that never
    changes. Profiling showed 14 kernels/call with only 17.6us of real GPU work, and a per-call cost pinned
    at ~51us from B=128 to B=4096 (32x the work, same wall time) — i.e. launch-bound, not compute-bound.
    Constants belong on the host; the device should only ever see the batch.
    """
    import torch
    dtype = dtype or torch.float64
    def _f(func, x):
        import math
        x = max(0.0, float(x))
        return {"linear": x, "sq": x * x, "sqrt": math.sqrt(x),
                "log": math.log1p(x), "log10": math.log10(1.0 + x)}.get(func, x)
    base, i0, amp_b, amp_a = [], [], [], []
    below_f, above_f = [], []
    for p in products:
        pr = params_by_product[p]
        base.append(float(pr["base"])); i0.append(float(pr["I0"]))
        fb, fa = _f(pr["below_func"], pr["T"]) or 1.0, _f(pr["above_func"], pr["T"]) or 1.0
        amp_b.append(pr["below_target"] * float(pr["base"]) / fb)
        amp_a.append(pr["above_target"] * float(pr["base"]) / fa)
        below_f.append(pr["below_func"]); above_f.append(pr["above_func"])
    t = lambda v: torch.tensor(v, dtype=dtype, device=device)
    # GRAPH-SAFE masks. Selecting columns with a Python list (`dist[:, cols]`) allocates an index tensor on
    # every call, which CUDA Graph capture forbids outright ("operation not permitted when stream is
    # capturing") and which costs a gather/scatter even in eager mode. Precompute one [P] mask per distinct
    # shape function instead, so the hot path is pure elementwise arithmetic over static tensors.
    masks_b = {f: t([1.0 if bf == f else 0.0 for bf in below_f]) for f in set(below_f)}
    masks_a = {f: t([1.0 if af == f else 0.0 for af in above_f]) for f in set(above_f)}
    return {"base": t(base), "I0": t(i0), "amp_below": t(amp_b), "amp_above": t(amp_a),
            "below_func": below_f, "above_func": above_f, "products": list(products),
            "masks_below": masks_b, "masks_above": masks_a}


def market_price_all(pre, inventory, price_floor=1.0):
    """Prices for ALL products at once: [B, P] in, [B, P] out.

    Replaces P separate calls with ONE. Combined with hoisted constants this cuts the launch count that the
    profiler showed to be the real cost. Shapes are applied group-wise (products sharing a shape function
    are computed together), so the kernel count depends on the number of DISTINCT shapes (<=5), not on the
    number of products.
    """
    import torch
    inv = torch.as_tensor(inventory)
    dist = (inv - pre["I0"]).abs()
    below = inv < pre["I0"]
    # Evaluate each distinct shape over the WHOLE tensor and select with a precomputed mask. Slightly more
    # arithmetic than column-slicing, but no allocation and no indexing — so it is CUDA-Graph capturable and
    # avoids a gather/scatter per call. There are at most 5 shapes regardless of product count.
    fb = torch.zeros_like(dist)
    fa = torch.zeros_like(dist)
    for fname, m in pre["masks_below"].items():
        fb = fb + m * shape_batch(fname, dist)
    for fname, m in pre["masks_above"].items():
        fa = fa + m * shape_batch(fname, dist)
    price = torch.where(below, pre["base"] + pre["amp_below"] * fb,
                        pre["base"] - pre["amp_above"] * fa)
    return torch.clamp(torch.round(price), min=float(price_floor))


class GraphedRollout:
    """Replay a CHUNK of simulation steps from one CUDA graph — no Python in the inner loop.

    Design, from measurement rather than intuition (RTX 5090, 720-step market rollout):

      eager, 720 Python iterations   B=64 138.5 ms | B=512 87.6 ms
      chunk=24  (30 replays)         B=64  41.0 ms | B=512 48.7 ms
      chunk=720 (1 replay)           B=64  44.7 ms | B=512 50.6 ms

    Two conclusions worth keeping. Python IS eliminable — a whole episode captures into a single graph and
    replays with ONE call. But replay time is FLAT from chunk=24 onward, so past a day's worth of steps the
    remaining cost is genuine kernel execution, not Python; the full unroll is marginally slower and costs
    210 ms to capture versus 43 ms. So the default chunk is one in-game day: Python is already gone at that
    size, capture is cheap, and the boundary lines up with end-of-day processing.

    Requirements for capture (each one cost a failed attempt):
      * static shapes and no data-dependent control flow;
      * NO Python-list fancy indexing inside the captured region — it allocates mid-capture and CUDA rejects
        it outright ("operation not permitted when stream is capturing"). Use precomputed masks;
      * warm up on a side stream before capturing.
    """

    def __init__(self, step_fn, state, chunk=24, warmup=3):
        """`step_fn(t)` performs ONE step IN PLACE on tensors in `state`; `t` is static at capture time."""
        import torch
        self.chunk = int(chunk)
        self.state = state
        self._graph = None
        st = torch.cuda.Stream()
        st.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(st):
            for t in range(min(warmup, self.chunk)):
                step_fn(t)
        torch.cuda.current_stream().wait_stream(st)
        self._graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self._graph):
            for t in range(self.chunk):
                step_fn(t)

    def replay(self, n=1):
        """Run `n * chunk` simulation steps with exactly `n` Python calls."""
        for _ in range(int(n)):
            self._graph.replay()

    def run_steps(self, total_steps):
        """Run `total_steps`, reporting how many were covered by whole chunks."""
        n = int(total_steps) // self.chunk
        self.replay(n)
        return {"steps_run": n * self.chunk, "python_calls": n,
                "steps_remaining": int(total_steps) - n * self.chunk}


# ─────────────────────────────────────────────────────────────────────────────────────────────────────
# KAGGRICULTURE — tile state as tensors, and the daily plant refresh (the growth engine).
#
# Tiles are dicts of Python objects in the official env. Batched, every field becomes an int tensor of
# shape [B, players, H, W], so one kernel advances every tile of every farm of every game.
# ─────────────────────────────────────────────────────────────────────────────────────────────────────

TILE_EMPTY, TILE_LOCKED, TILE_PLANT, TILE_WEED, TILE_COOP, TILE_PASTURE = 0, 1, 2, 3, 4, 5


def crop_table(crops, order):
    """Per-crop constants as parallel lists, indexed by crop id — host-side, never recomputed on device."""
    return {"order": list(order),
            "ongoing": [1 if crops[c]["ongoing"] else 0 for c in order],
            "first_yield_day": [crops[c]["first_yield_day"] for c in order],
            "max_yield_day": [crops[c]["max_yield_day"] for c in order],
            "interval": [crops[c]["interval"] for c in order],
            "max_yield": [crops[c]["max_yield"] for c in order],
            "seed": [crops[c]["seed"] for c in order]}


def new_tile_state(shape, device=None):
    """Zeroed tile state: {field: tensor[*shape]}. `shape` is typically (B, players, H, W)."""
    import torch
    z = lambda: torch.zeros(shape, dtype=torch.int64, device=device)
    st = {k: z() for k in ("kind", "crop", "planted_day", "watered_today", "consecutive_unwatered",
                           "yield_units", "max_lifespan_step", "fertilized_until_day")}
    st["crop"].fill_(-1)
    st["max_lifespan_step"].fill_(-1)
    st["fertilized_until_day"].fill_(-1)
    return st


def daily_refresh_plants_batch(st, ct, current_day, turns_per_day, device=None):
    """Batched port of the official `_daily_refresh_plants`, IN PLACE.

    Order of operations follows the original exactly, because it is load-bearing:
      1. read `was_watered` BEFORE clearing it — the fertilizer bonus depends on it;
      2. update `consecutive_unwatered`, then clear `watered_today`;
      3. two consecutive unwatered days turn the tile into a WEED and it is skipped entirely;
      4. only ONGOING crops accrue yield here, on their exact production days.
    """
    import torch
    dev = device or st["kind"].device
    t = lambda v: torch.tensor(v, dtype=torch.int64, device=dev)
    next_day = current_day + 1

    is_plant = st["kind"] == TILE_PLANT
    was_watered = (st["watered_today"] == 1) & is_plant

    st["consecutive_unwatered"] = torch.where(
        is_plant,
        torch.where(was_watered, torch.zeros_like(st["consecutive_unwatered"]),
                    st["consecutive_unwatered"] + 1),
        st["consecutive_unwatered"])
    st["watered_today"] = torch.where(is_plant, torch.zeros_like(st["watered_today"]),
                                      st["watered_today"])

    # 2 consecutive unwatered days -> WEED (and the tile stops being a plant this refresh)
    to_weed = is_plant & (st["consecutive_unwatered"] >= 2)
    if bool(to_weed.any()):
        st["kind"] = torch.where(to_weed, torch.full_like(st["kind"], TILE_WEED), st["kind"])
        for f, v in (("crop", -1), ("planted_day", 0), ("watered_today", 0),
                     ("consecutive_unwatered", 0), ("yield_units", 0),
                     ("max_lifespan_step", -1), ("fertilized_until_day", -1)):
            st[f] = torch.where(to_weed, torch.full_like(st[f], v), st[f])

    alive = (st["kind"] == TILE_PLANT)
    crop = st["crop"].clamp(min=0)
    ongoing = t(ct["ongoing"])[crop] == 1
    first_y = t(ct["first_yield_day"])[crop]
    interval = t(ct["interval"])[crop].clamp(min=1)          # non-ongoing crops have interval 0
    max_y = t(ct["max_yield"])[crop]

    days_since_first = next_day - st["planted_day"] - first_y
    produces = alive & ongoing & (days_since_first >= 0) & (days_since_first % interval == 0)
    production_count = days_since_first // interval + 1
    produces = produces & (production_count <= max_y)

    fertilized = was_watered & (st["fertilized_until_day"] >= current_day)
    gain = torch.where(fertilized, torch.full_like(st["yield_units"], 2),
                       torch.ones_like(st["yield_units"]))
    st["yield_units"] = torch.where(produces,
                                    torch.minimum(max_y, st["yield_units"] + gain),
                                    st["yield_units"])

    final = produces & (production_count == max_y)
    st["max_lifespan_step"] = torch.where(
        final, torch.full_like(st["max_lifespan_step"], (next_day + 1) * turns_per_day),
        st["max_lifespan_step"])
    return st
