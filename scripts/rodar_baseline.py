#!/usr/bin/env python3
"""Executa o baseline ML. Ver docstring de src/models/baseline.py.

Uso:
  ~/.virtualenvs/datascience/bin/python scripts/rodar_baseline.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.baseline import run_all  # noqa: E402

if __name__ == "__main__":
    run_all()
