"""
Matriz anual de features estruturais + merge com alvos.

Para cada ano t em 1890–2012 (Opção A: nós = Estados com CINC>=1% em t):

(i)   Por camada (ali=alianças, com=comércio, disp=disputas suavizada
      trailing [t-2, t], dipl=diplomacia): densidade, modularidade Louvain
      (ponderada, seed=42), clustering médio, centralização de grau (Freeman),
      betweenness média e máxima (não-ponderada, normalizada).
(ii)  Cross-layer: Jaccard de arestas por par de camadas (6 pares) e
      edge-overlap médio (nº médio de camadas por díade conectada) – versão
      3 camadas (sem diplomacia, sempre disponível) e 4 camadas (com
      diplomacia, sujeita à imputação).
(iii) Composição: n_estados, churn_entradas/saidas/total vs t-1.
(iv)  Capacidade: concentração CINC via **HHI** (escolhido sobre Gini por ser
      o índice padrão de concentração com leitura direta de "quase-monopólio";
      shares renormalizados dentro do sistema CINC>=1%) e share do líder
      (renormalizado).
Adicional: deltas de 1 ano (d_*) para TODAS as métricas de (i). Para que os
deltas e o churn existam já em 1890, as métricas são computadas internamente
desde 1889 e a matriz final é recortada em 1890–2012.

Diplomacia (cadência ~quinquenal, termina em 2005):
  - anos medidos: métricas computadas no próprio ano (nodeset do ano);
  - anos NÃO medidos até 2005: forward-fill do último ano medido (inclusive
    anterior a 1890, se for o caso), com ``dipl_imputado=1``;
  - após 2005 (último ano medido): NaN + ``dipl_imputado=1`` – a decisão de
    uso fica para a modelagem.
  As colunas cross-layer que envolvem diplomacia (jaccard_*_dipl,
  overlap_medio_4c) seguem a MESMA regra (computadas no ano medido contra as
  camadas daquele ano, depois forward-fill).

ANTI-LEAKAGE: nenhuma feature usa informação de t+1 em diante – disputas usam
janela TRAILING [t-2, t]; churn e deltas comparam com t-1; forward-fill da
diplomacia só olha para trás. (`smoothtotrade` é a série suavizada da própria
fonte COW, decisão herdada do BIA.) Declarado em reports/FEATURES_TARGETS.md.

Uso:
  ~/.virtualenvs/datascience/bin/python -m src.features.build_features
Saídas:
  data/processed/features_1890_2012.csv          (só features)
  data/processed/features_targets_1890_2012.csv  (features + variantes de y)
"""

import numpy as np
import pandas as pd
import networkx as nx

from .. import config as cfg
from ..network import annual
from . import target as tgt

LAYERS = ("ali", "com", "disp", "dipl")
METRICS = ("densidade", "modularidade", "clustering",
           "centralizacao_grau", "betw_media", "betw_max")
PAIRS = (("ali", "com"), ("ali", "disp"), ("ali", "dipl"),
         ("com", "disp"), ("com", "dipl"), ("disp", "dipl"))
Y0, Y1 = 1890, 2012
_Y_START_INTERNO = Y0 - 1  # 1889: garante delta/churn definidos em 1890


# ---------------------------------------------------------------------------
# Métricas de uma camada
# ---------------------------------------------------------------------------
def degree_centralization(G) -> float:
    """Centralização de grau de Freeman: sum(dmax-di)/((n-1)(n-2))."""
    n = G.number_of_nodes()
    if n < 3:
        return np.nan
    degs = [d for _, d in G.degree()]
    dmax = max(degs)
    return sum(dmax - d for d in degs) / ((n - 1) * (n - 2))


def layer_metrics(G) -> dict:
    """As 6 métricas de (i) para um grafo Opção A (N completo)."""
    if G is None:
        return {m: np.nan for m in METRICS}
    bet = nx.betweenness_centrality(G, normalized=True)
    vals = list(bet.values())
    mod = annual.modularity(G)  # None se sem arestas
    return {
        "densidade": nx.density(G),
        "modularidade": np.nan if mod is None else mod,
        "clustering": nx.average_clustering(G),
        "centralizacao_grau": degree_centralization(G),
        "betw_media": float(np.mean(vals)) if vals else np.nan,
        "betw_max": float(np.max(vals)) if vals else np.nan,
    }


# ---------------------------------------------------------------------------
# Cross-layer
# ---------------------------------------------------------------------------
def _edges(G) -> set:
    return {frozenset((u, v)) for u, v in G.edges()} if G is not None else None


def jaccard(e1, e2) -> float:
    if e1 is None or e2 is None:
        return np.nan
    union = e1 | e2
    if not union:
        return np.nan  # ambas vazias: indefinido (não 0 silencioso)
    return len(e1 & e2) / len(union)


def edge_overlap_mean(edge_sets) -> float:
    """Nº médio de camadas por díade conectada em >=1 das camadas dadas."""
    if any(e is None for e in edge_sets):
        return np.nan
    all_dyads = set().union(*edge_sets)
    if not all_dyads:
        return np.nan
    return float(np.mean([sum(d in e for e in edge_sets) for d in all_dyads]))


# ---------------------------------------------------------------------------
# Vetor de um ano
# ---------------------------------------------------------------------------
def year_graphs(t: int, nodeset) -> dict:
    """Grafos das 4 camadas no ano t (dipl=None se ano não medido)."""
    return {
        "ali": annual.alliance_graph(t, nodeset),
        "com": annual.trade_graph(t, nodeset),
        "disp": annual.dispute_graph(t, nodeset, window=3),
        "dipl": annual.diplomacy_graph(t, nodeset),
    }


def year_vector(t: int, prev_nodeset=None) -> dict:
    """Todas as features de nível (sem deltas/ffill) do ano t."""
    nodeset = annual.states_in_year(t)
    graphs = year_graphs(t, nodeset)
    row = {"ano": t}

    # (i) por camada
    for lay in LAYERS:
        for m, v in layer_metrics(graphs[lay]).items():
            row[f"{lay}_{m}"] = v

    # (ii) cross-layer
    es = {lay: _edges(graphs[lay]) for lay in LAYERS}
    for a, b in PAIRS:
        row[f"jaccard_{a}_{b}"] = jaccard(es[a], es[b])
    row["overlap_medio_3c"] = edge_overlap_mean([es["ali"], es["com"], es["disp"]])
    row["overlap_medio_4c"] = edge_overlap_mean(
        [es["ali"], es["com"], es["disp"], es["dipl"]])

    # (iii) composição
    row["n_estados"] = len(nodeset)
    if prev_nodeset is None:
        row["churn_entradas"] = np.nan
        row["churn_saidas"] = np.nan
        row["churn_total"] = np.nan
    else:
        ent = len(nodeset - prev_nodeset)
        sai = len(prev_nodeset - nodeset)
        row["churn_entradas"] = ent
        row["churn_saidas"] = sai
        row["churn_total"] = ent + sai

    # (iv) capacidade (shares CINC renormalizados dentro do sistema)
    nmc, *_ = annual._load()
    sub = nmc[(nmc["year"] == t) & (nmc["ccode"].isin(nodeset))]
    shares = sub["cinc"] / sub["cinc"].sum()
    row["cinc_hhi"] = float((shares ** 2).sum())
    row["cinc_lider_share"] = float(shares.max())

    return row, nodeset


# ---------------------------------------------------------------------------
# Matriz completa
# ---------------------------------------------------------------------------
# Colunas dependentes de diplomacia (recebem ffill + indicador)
DIPL_COLS = ([f"dipl_{m}" for m in METRICS]
             + [f"jaccard_{a}_{b}" for a, b in PAIRS if "dipl" in (a, b)]
             + ["overlap_medio_4c"])


def build_feature_matrix() -> pd.DataFrame:
    dipl_measured = set(annual.diplomacy_years())
    last_dipl = max(y for y in dipl_measured)  # 2005 (medido)

    rows, prev = [], None
    for t in range(_Y_START_INTERNO, Y1 + 1):
        row, prev = year_vector(t, prev)
        rows.append(row)
    df = pd.DataFrame(rows).set_index("ano")

    # Diplomacia: ffill até o último ano medido; depois, NaN. Indicador 0/1.
    df["dipl_imputado"] = [int(t not in dipl_measured) for t in df.index]
    ffill_zone = df.index <= last_dipl
    df.loc[ffill_zone, DIPL_COLS] = df.loc[ffill_zone, DIPL_COLS].ffill()
    # (após last_dipl não há medição futura alguma: mantém NaN por decisão)

    # Deltas de 1 ano para TODAS as métricas por camada (pós-ffill p/ dipl)
    level_cols = [f"{lay}_{m}" for lay in LAYERS for m in METRICS]
    for c in level_cols:
        df[f"d_{c}"] = df[c].diff()

    df = df.loc[Y0:Y1].reset_index()
    return df


# ---------------------------------------------------------------------------
# main – features + alvos
# ---------------------------------------------------------------------------
def main():
    cfg.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print("Construindo matriz de features 1890–2012 (interno desde 1889)...")
    feats = build_feature_matrix()
    f_path = cfg.PROCESSED_DIR / "features_1890_2012.csv"
    feats.to_csv(f_path, index=False)
    print(f"  {f_path.name}: {feats.shape[0]} anos x {feats.shape[1]} colunas "
          f"(1 'ano' + {feats.shape[1] - 1} features)")

    onsets = {"proxy": tgt.proxy_onsets(), "war": tgt.wardata_onsets()}
    targets = tgt.build_targets(onsets)

    full = feats.merge(targets, on="ano", validate="1:1")
    ft_path = cfg.PROCESSED_DIR / "features_targets_1890_2012.csv"
    full.to_csv(ft_path, index=False)
    n_y = targets.shape[1] - 1
    print(f"  {ft_path.name}: {full.shape[0]} anos x {full.shape[1]} colunas "
          f"({feats.shape[1] - 1} features + {n_y} colunas de alvo)")

    # Resumo de NaN (traceabilidade – nada de imputação silenciosa)
    nan_cols = full.isna().sum()
    nan_cols = nan_cols[nan_cols > 0]
    print("\nColunas com NaN (esperado: diplomacia pós-2005 e afins):")
    for c, n in nan_cols.items():
        print(f"  {c}: {n}")
    return full


if __name__ == "__main__":
    main()
