#!/usr/bin/env python3
"""
Figuras REAIS dos Resultados Preliminares – a partir dos dados COW.
Tudo traceável (Regra de Ouro). NÃO gera figuras de modelos/SHAP (não existem).

Saídas:
  reports/figuras_prelim/fig1_series_densidade_modularidade.png
  reports/figuras_prelim/fig2_composicao_sistema.png
  reports/figuras_prelim/fig3_suavizacao_disputas.png
  data/interim/prelim_series_densidade_modularidade.csv   (série traceável)

Uso: ~/.virtualenvs/datascience/bin/python scripts/gerar_figuras_prelim.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config as cfg
from src.network import annual

np.random.seed(cfg.RANDOM_SEED)

FIGDIR = cfg.REPORTS_DIR / "figuras_prelim"
FIGDIR.mkdir(parents=True, exist_ok=True)

LAYERS = ["alliance", "trade", "disputes", "diplomacy"]
LAYER_PT = {"alliance": "Alianças", "trade": "Comércio",
            "disputes": "Disputas (jan. 3a)", "diplomacy": "Diplomacia"}
COLORS = {"alliance": "#1f77b4", "trade": "#2ca02c",
          "disputes": "#d62728", "diplomacy": "#9467bd"}
EVENTS = {1914: "WWI", 1939: "WWII", 1947: "G. Fria", 1991: "Fim URSS"}

nmc_years = [y for y in range(cfg.YEAR_MIN, 2013)]


def build_layer(year, ns, layer):
    if layer == "alliance":
        return annual.alliance_graph(year, ns)
    if layer == "trade":
        return annual.trade_graph(year, ns)
    if layer == "disputes":
        return annual.dispute_graph(year, ns, window=3)
    if layer == "diplomacy":
        return annual.diplomacy_graph(year, ns)


# ---------------------------------------------------------------------------
# Série densidade + modularidade (traceável)
# ---------------------------------------------------------------------------
print("Computando série densidade+modularidade (1890–2012)...")
rows = []
for y in nmc_years:
    ns = annual.states_in_year(y)
    for layer in LAYERS:
        G = build_layer(y, ns, layer)
        if G is None:
            rows.append({"year": y, "layer": layer, "nodes": len(ns),
                         "edges": np.nan, "density": np.nan, "modularity": np.nan})
        else:
            rows.append({"year": y, "layer": layer, "nodes": G.number_of_nodes(),
                         "edges": G.number_of_edges(),
                         "density": nx.density(G), "modularity": annual.modularity(G)})
series = pd.DataFrame(rows)
series.to_csv(cfg.INTERIM_DIR / "prelim_series_densidade_modularidade.csv", index=False)
print(f"  série salva ({len(series)} linhas).")


def _annotate(ax):
    for yr, lbl in EVENTS.items():
        ax.axvline(yr, color="grey", ls="--", lw=0.8, alpha=0.6)
        ax.text(yr, ax.get_ylim()[1] * 0.97, lbl, rotation=90, va="top",
                ha="right", fontsize=7, color="grey")


# ---------------------------------------------------------------------------
# Figura 1 – séries de densidade e modularidade por camada
# ---------------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
for layer in LAYERS:
    s = series[series.layer == layer].dropna(subset=["density"])
    style = dict(color=COLORS[layer], label=LAYER_PT[layer])
    if layer == "diplomacy":
        ax1.plot(s.year, s.density, "o", ms=4, **style)
    else:
        ax1.plot(s.year, s.density, lw=1.4, **style)
ax1.set_ylabel("Densidade")
ax1.set_title("Evolução estrutural das redes interestatais (snapshots anuais, 1890–2012)")
ax1.legend(loc="center left", fontsize=8, ncol=2)
ax1.set_ylim(0, 1.05)
_annotate(ax1)

for layer in LAYERS:
    s = series[series.layer == layer].dropna(subset=["modularity"])
    style = dict(color=COLORS[layer], label=LAYER_PT[layer])
    if layer == "diplomacy":
        ax2.plot(s.year, s.modularity, "o", ms=4, **style)
    else:
        ax2.plot(s.year, s.modularity, lw=1.4, **style)
ax2.set_ylabel("Modularidade (Louvain)")
ax2.set_xlabel("Ano")
_annotate(ax2)
fig.tight_layout()
fig.savefig(FIGDIR / "fig1_series_densidade_modularidade.png", dpi=300)
plt.close(fig)
print("  fig1 ok")

# ---------------------------------------------------------------------------
# Figura 2 – composição do sistema (n_states + churn)
# ---------------------------------------------------------------------------
comp = pd.read_csv(cfg.INTERIM_DIR / "B2_composicao_sistema_por_ano.csv")
fig, ax = plt.subplots(figsize=(11, 4.5))
ax.bar(comp.year, comp.churn, color="#cccccc", label="Churn (entradas+saídas)")
ax.set_ylabel("Churn (nº de Estados)")
ax2 = ax.twinx()
ax2.plot(comp.year, comp.n_states, color="#1f77b4", lw=1.8, label="Nº de Estados (CINC≥1%)")
ax2.set_ylabel("Nº de Estados no sistema")
for yr in (1940, 1946, 1991):
    ax.axvline(yr, color="grey", ls="--", lw=0.8, alpha=0.6)
    ax.text(yr, ax.get_ylim()[1] * 0.95, str(yr), rotation=90, va="top",
            ha="right", fontsize=7, color="grey")
ax.set_xlabel("Ano")
ax.set_title("Composição do sistema internacional por ano (CINC ≥ 1%), 1890–2012")
lines = ax.get_legend_handles_labels()[0] + ax2.get_legend_handles_labels()[0]
labels = ax.get_legend_handles_labels()[1] + ax2.get_legend_handles_labels()[1]
ax.legend(lines, labels, loc="upper left", fontsize=8)
fig.tight_layout()
fig.savefig(FIGDIR / "fig2_composicao_sistema.png", dpi=300)
plt.close(fig)
print("  fig2 ok")

# ---------------------------------------------------------------------------
# Figura 3 – efeito da suavização nas disputas
# ---------------------------------------------------------------------------
b4 = pd.read_csv(cfg.INTERIM_DIR / "B4_disputas_suavizacao.csv")
fig, ax = plt.subplots(figsize=(11, 4.5))
for w, c in zip([1, 3, 5], ["#fdae6b", "#e6550d", "#a63603"]):
    s = b4[b4.window == w]
    ax.plot(s.year, s.density, lw=1.2, color=c, label=f"Janela trailing = {w} ano(s)")
ax.set_xlabel("Ano")
ax.set_ylabel("Densidade da camada de disputas")
ax.set_title("Efeito da janela de suavização na camada de disputas (1890–2012)")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(FIGDIR / "fig3_suavizacao_disputas.png", dpi=300)
plt.close(fig)
print("  fig3 ok")

print(f"\nFiguras salvas em {FIGDIR}")
