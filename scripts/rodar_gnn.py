#!/usr/bin/env python3
"""Executa a GNN. Ver docstring de src/models/gnn.py.

Uso:
  ~/.virtualenvs/dsa-gnn/bin/python scripts/rodar_gnn.py [--smoke]
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.gnn import run_all  # noqa: E402

if __name__ == "__main__":
    run_all(smoke="--smoke" in sys.argv)
