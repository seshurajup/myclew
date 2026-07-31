"""Load the competition's fleet-adapter package — keeps the fleet framework competition-agnostic.

The competition provides a package (default `fleet_agents`, at FLEET_COMPETITION_ROOT) exposing
NAME, SEED, HANDLERS. Swap that package per competition; this framework never changes.
"""
from __future__ import annotations

import importlib
import os
import sys


def load_competition():
    root = os.environ.get("FLEET_COMPETITION_ROOT")
    if root and root not in sys.path:
        sys.path.insert(0, root)  # so `import fleet_agents` (and its `import src.*`) resolve
    name = os.environ.get("FLEET_COMPETITION", "fleet_agents")
    return importlib.import_module(name)
