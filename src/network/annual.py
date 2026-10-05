"""
Builders de redes ANUAIS (snapshot por ano) com Opção A.

Reaproveita a convenção da versão corrigida do BIA
(v_final_correcoes/dados/build_network_corrigido.py): todos os Estados com
CINC >= threshold entram como nós de TODAS as camadas, mesmo isolados; métricas
sobre N completo. Comércio usa smoothtotrade. Seed=42 para Louvain.

Estes builders são a base reutilizável para o feature pipeline (próxima fase) e
para as figuras dos Resultados Preliminares.
"""

from functools import lru_cache
from pathlib import Path

import networkx as nx
import pandas as pd

try:
    import community as community_louvain  # python-louvain
except Exception:  # pragma: no cover
    community_louvain = None

from .. import config as cfg

RAW = cfg.cow_raw_dir()


# ---------------------------------------------------------------------------
# Carregamento (cacheado)
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _load():
    nmc = pd.read_csv(RAW / cfg.COW_FILES["nmc"])
    trade = pd.read_csv(RAW / cfg.COW_FILES["trade_dyadic"])
    ally = pd.read_csv(RAW / cfg.COW_FILES["alliances_yearly"])
    midb = pd.read_csv(RAW / cfg.COW_FILES["midb"])
    dip = pd.read_csv(RAW / cfg.COW_FILES["diplomatic"])
    return nmc, trade, ally, midb, dip


def diplomacy_years():
    _, _, _, _, dip = _load()
    return sorted(int(y) for y in dip["year"].unique())


def states_in_year(year: int, threshold: float = cfg.CINC_THRESHOLD):
    """Conjunto de Estados com CINC >= threshold no ano (composição do sistema)."""
    nmc, *_ = _load()
    sub = nmc[(nmc["year"] == year) & (nmc["cinc"] >= threshold)]
    return set(int(c) for c in sub["ccode"].unique())


# ---------------------------------------------------------------------------
# Builders por camada (Opção A: add_nodes_from(nodeset) sempre)
# ---------------------------------------------------------------------------
def alliance_graph(year, nodeset, weighted=True):
    _, _, ally, _, _ = _load()
    G = nx.Graph()
    G.add_nodes_from(nodeset)
    sub = ally[(ally["year"] == year)
               & (ally["ccode1"].isin(nodeset)) & (ally["ccode2"].isin(nodeset))]
    for r in sub.itertuples(index=False):
        w = (cfg.ALLIANCE_WEIGHTS["defense"] if r.defense else
             cfg.ALLIANCE_WEIGHTS["neutrality"] if r.neutrality else
             cfg.ALLIANCE_WEIGHTS["nonaggression"] if r.nonaggression else
             cfg.ALLIANCE_WEIGHTS["entente"] if r.entente else 0)
        if w:
            u, v = int(r.ccode1), int(r.ccode2)
            if G.has_edge(u, v):
                G[u][v]["weight"] = max(G[u][v]["weight"], w)
            else:
                G.add_edge(u, v, weight=w if weighted else 1)
    return G


def trade_graph(year, nodeset, weighted=True):
    _, trade, _, _, _ = _load()
    G = nx.Graph()
    G.add_nodes_from(nodeset)
    sub = trade[(trade["year"] == year) & (trade["smoothtotrade"] > 0)
                & (trade["ccode1"].isin(nodeset)) & (trade["ccode2"].isin(nodeset))]
    for r in sub.itertuples(index=False):
        w = float(r.smoothtotrade) if weighted else 1
        G.add_edge(int(r.ccode1), int(r.ccode2), weight=w)
    return G


def dispute_graph(year, nodeset, window=3, weighted=True):
    """Disputas ativas em [year-window+1, year]; arestas Side A vs Side B."""
    _, _, _, midb, _ = _load()
    G = nx.Graph()
    G.add_nodes_from(nodeset)
    lo = year - window + 1
    active = midb[(midb["styear"] <= year) & (midb["endyear"] >= lo)
                  & (midb["ccode"].isin(nodeset))]
    for _, part in active.groupby("dispnum"):
        a = part.loc[part["sidea"] == 1, "ccode"].unique()
        b = part.loc[part["sidea"] == 0, "ccode"].unique()
        for s1 in a:
            for s2 in b:
                if s1 != s2:
                    u, v = int(s1), int(s2)
                    if G.has_edge(u, v):
                        G[u][v]["weight"] += 1
                    else:
                        G.add_edge(u, v, weight=1)
    return G


def diplomacy_graph(year, nodeset):
    """Definida só em anos medidos (~quinquenal); senão None."""
    _, _, _, _, dip = _load()
    if year not in diplomacy_years():
        return None
    G = nx.Graph()
    G.add_nodes_from(nodeset)
    sub = dip[(dip["year"] == year) & (dip["DE"] == 1)
              & (dip["ccode1"].isin(nodeset)) & (dip["ccode2"].isin(nodeset))]
    for r in sub.itertuples(index=False):
        G.add_edge(int(r.ccode1), int(r.ccode2), weight=1)
    return G


# ---------------------------------------------------------------------------
# Métricas
# ---------------------------------------------------------------------------
def modularity(G):
    """Modularidade de Louvain (ponderada, seed=42). Nós isolados -> singletons.
    Retorna None se não houver arestas."""
    if community_louvain is None or G.number_of_edges() == 0:
        return None
    part = community_louvain.best_partition(G, weight="weight",
                                            random_state=cfg.RANDOM_SEED)
    return community_louvain.modularity(part, G, weight="weight")
