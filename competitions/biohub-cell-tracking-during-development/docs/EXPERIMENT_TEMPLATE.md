# Experiment note template — MACHINE-PARSEABLE (so Python gets exactly what it needs)

Write EVERY experiment as a block: a `### EXP` header followed by `key: value` lines. The Python
`notes-sync` fleet agent parses these into the experiment journal (`docs/experiment_ledger.md`)
deterministically — no LLM interpretation needed. This is how the researcher (Claude OR human) hands
structured results to the deterministic agents. Keep one change per block (grandmaster journal rule).

Write blocks into `docs/research_notes/<name>.md`. Fields (all lowercase keys):

```
### EXP
stage: 3
parent: EXP_00
change: <the ONE thing changed vs parent>
config: config/aug_ablation/rot90.yml
script: bash start_train.sh config/aug_ablation/rot90.yml
cv: 0.8421           # full-metric CV; or 'pending' / 'bad' / 'nan' / 'overfit'
lb: pending          # fill only when a Kaggle slot is spent
trn_set: loeo        # mini | loeo | golden12 | full
kept: false          # true ONLY if it beats CV AND transfers to the held-out embryo
observation: helps 44b6 not 6bba -> rejected (train-only gain)
```

Rules the fields encode (the grandmaster invariants):
- `parent` + `change` = "same as EXP_X but <change>" — one change per experiment.
- `cv` and `lb` are separate — the gap is the overfit signal.
- `kept: true` only if it beats CV **and** transfers to the held-out embryo (reject train-only gains).
- Failures are still logged (`cv: bad` / `nan` / `overfit`) — the full history, dead-ends included.
