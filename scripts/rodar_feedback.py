#!/usr/bin/env python3
"""Executa os itens do feedback do orientador.
Ver docstring de src/models/feedback_orientador.py.

Uso:
  ~/.virtualenvs/datascience/bin/python scripts/rodar_feedback.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.feedback_orientador import run_all  # noqa: E402

if __name__ == "__main__":
    run_all()
