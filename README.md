# myclew

Mirror of the **fleet agent codebase** — reusable, config-driven Kaggle agents + their data-wise
verifiers — plus the per-competition experiment code built on top of them.

- `fleet_agents/` — the agents (BaseAgent subclasses): tabular/GBM, geology particle-filter (Track B),
  neural-sequence Muon trainer (Track C), low-bit QAT (NVFP4/MXFP4/int8), setup-env GPU guard
  (the verified **cu128 / CUDA-12.8 sm_120 stack** for the RTX 5090), and many more.
- `test_fleet_agents/` — one data-wise verifier per agent (offline, deterministic).
- `competitions/<slug>/` — experiment code + YAML configs we build per competition (no data).

Source of truth lives in the biohub competition workspace; this repo is a **stable mirror**, synced
and committed every 3 hours (only when changes are pending, and only if every `.py` byte-compiles).

Sync manually: `./sync_and_commit.sh`
