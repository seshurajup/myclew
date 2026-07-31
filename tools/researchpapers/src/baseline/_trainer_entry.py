#!/usr/bin/env python
"""FD-safe entry shim for the official trainer.

Sets torch's DataLoader tensor-sharing strategy to 'file_system' BEFORE the trainer builds any
DataLoader, then runs the official `train_unet_transformer.py` in-process via runpy. This is the
root-cause fix for `RuntimeError: Too many open files` (FD/shared-memory exhaustion): the default
'file_descriptor' strategy consumes one FD per shared tensor, which accumulates across DataLoader
workers over a long (max_iters=1000+) run and blows past the soft FD limit. 'file_system' passes
tensors by filename instead — no FD accumulation. Injecting it here avoids editing the reference
trainer (research/official_repo/scripts/train_unet_transformer.py).

Usage (from src/baseline/train.py build_cmd):
    python -m src.baseline._trainer_entry <trainer.py> <trainer args...>
"""
import runpy
import sys

import torch.multiprocessing as _mp

_mp.set_sharing_strategy("file_system")

if len(sys.argv) < 2:
    sys.exit("usage: _trainer_entry.py <trainer.py> [trainer args...]")

_trainer = sys.argv[1]
# Present the trainer with the argv IT expects (its own path + flags), so its argparse is unchanged.
sys.argv = [_trainer] + sys.argv[2:]
runpy.run_path(_trainer, run_name="__main__")
