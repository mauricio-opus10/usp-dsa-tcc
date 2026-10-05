#!/usr/bin/env python3
"""Executa os testes de robustez. Ver docstring de src/models/robustez.py.

Uso:
  ~/.virtualenvs/datascience/bin/python scripts/rodar_robustez.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.robustez import run_all  # noqa: E402

if __name__ == "__main__":
    run_all()
