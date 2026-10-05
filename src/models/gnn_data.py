"""
Dados como grafos para a GNN. SEM dependência de torch –
retorna estruturas numpy/python; a conversão a tensores fica em gnn.py.

Para cada ano t (1890–2012): um grafo por camada (ali, com, disp, dipl),
nós = Estados CINC >= 1% do ano (Opção A, isolados incluídos; builders
reutilizados de src/network/annual.py).

Pesos de aresta (pré-declarados em reports/GNN_RESULTADOS.md):
  ali  – compromisso 1–4 / 4 (constante);
  com  – log(1+smoothtotrade) BRUTO aqui; a divisão pelo máximo dos anos de
         TREINO do fold é feita em gnn.py (única normalização dependente de
         dados – nunca vê o teste);
  disp – contagem de disputas ativas na janela trailing [t-2, t] BRUTA aqui;
         normalização por máximo de treino também em gnn.py;
  dipl – binária; anos não medidos até 2005 usam as arestas do último ano
         MEDIDO <= t restritas ao nodeset de t (análogo em grafo ao ffill do
         baseline). Variante sem_dipl: camada ausente, janela toda.

Node features (todas naturalmente em [0,1]; nenhuma deriva do alvo; toda
informação é <= t):
  cinc_share (renormalizado no sistema), grau/(n-1) por camada,
  betweenness normalizada por camada, flag de entrada no sistema vs t-1.

Rótulos: y1_proxy e y1_war (>=2 Estados do sistema) + flags de censura,
reutilizados de src/features/target.py.
"""

import numpy as np
import networkx as nx

from .. import config as cfg
from ..features import target as tgt
from ..network import annual

Y0, Y1 = 1890, 2012
LAYERS_ALL = ("ali", "com", "disp", "dipl")
LAYERS_SEM_DIPL = ("ali", "com", "disp")


def _dipl_graph_ffill(t, nodeset, dipl_years):
    """Grafo de diplomacia com ffill estrutural (só anos <= 2005)."""
    measured = [y for y in dipl_years if y <= t]
    if not measured:
        return None
    src_year = max(measured)
    G_src = annual.diplomacy_graph(src_year, annual.states_in_year(src_year))
    G = nx.Graph()
    G.add_nodes_from(nodeset)
    for u, v in G_src.edges():
        if u in nodeset and v in nodeset:
            G.add_edge(u, v, weight=1)
    return G


def year_graphs(t, nodeset, layers):
    """Grafos networkx por camada; pesos brutos conforme docstring."""
    gs = {}
    if "ali" in layers:
        gs["ali"] = annual.alliance_graph(t, nodeset)
    if "com" in layers:
        gs["com"] = annual.trade_graph(t, nodeset)
    if "disp" in layers:
        gs["disp"] = annual.dispute_graph(t, nodeset, window=3)
    if "dipl" in layers:
        dipl_years = annual.diplomacy_years()
        g = annual.diplomacy_graph(t, nodeset)
        gs["dipl"] = g if g is not None else _dipl_graph_ffill(t, nodeset,
                                                               dipl_years)
    return gs


def _edge_arrays(G, index, transform=None):
    """(edge_index 2xE bidirecional, edge_weight E) com pesos transformados."""
    src, dst, w = [], [], []
    for u, v, data in G.edges(data=True):
        wt = float(data.get("weight", 1.0))
        if transform is not None:
            wt = transform(wt)
        i, j = index[u], index[v]
        src += [i, j]
        dst += [j, i]
        w += [wt, wt]
    return (np.array([src, dst], dtype=np.int64).reshape(2, -1),
            np.array(w, dtype=np.float32))


_TRANSFORMS = {
    "ali": lambda w: w / 4.0,          # compromisso 1–4 -> [0,25, 1]
    "com": lambda w: np.log1p(w),      # log(1+smoothtotrade); /max no fold
    "disp": lambda w: w,               # contagem; /max no fold
    "dipl": lambda w: 1.0,             # binária
}


def year_data(t, layers):
    """Dicionário numpy de um ano: nodeset, features, arestas por camada."""
    nodeset = annual.states_in_year(t)
    prev = annual.states_in_year(t - 1)
    nodes = sorted(nodeset)
    index = {c: i for i, c in enumerate(nodes)}
    n = len(nodes)
    gs = year_graphs(t, set(nodes), layers)

    # CINC share renormalizado no sistema
    nmc, *_ = annual._load()
    sub = nmc[(nmc["year"] == t) & (nmc["ccode"].isin(nodeset))]
    tot = float(sub["cinc"].sum())
    share = {int(r.ccode): float(r.cinc) / tot for r in sub.itertuples()}

    feats = [np.array([share.get(c, 0.0) for c in nodes], dtype=np.float32)]
    for lay in layers:
        G = gs[lay]
        deg = np.array([G.degree(c) / (n - 1) if n > 1 else 0.0
                        for c in nodes], dtype=np.float32)
        bet_d = nx.betweenness_centrality(G, normalized=True)
        bet = np.array([bet_d[c] for c in nodes], dtype=np.float32)
        feats += [deg, bet]
    entry = np.array([1.0 if c not in prev else 0.0 for c in nodes],
                     dtype=np.float32)
    feats.append(entry)
    x = np.stack(feats, axis=1)   # n x (1 + 2*n_camadas + 1)

    edges = {lay: _edge_arrays(gs[lay], index, _TRANSFORMS[lay])
             for lay in layers}
    return {"ano": t, "n": n, "x": x, "edges": edges}


def build_dataset(variant):
    """variant: 'com_dipl' (4 camadas, anos <= 2005) | 'sem_dipl' (3, todos).

    Retorna (dict ano->dados, DataFrame de alvos ms=2, camadas)."""
    layers = LAYERS_ALL if variant == "com_dipl" else LAYERS_SEM_DIPL
    y_max = 2005 if variant == "com_dipl" else Y1
    data = {t: year_data(t, layers) for t in range(Y0, y_max + 1)}
    onsets = {"proxy": tgt.proxy_onsets(min_system_states=2),
              "war": tgt.wardata_onsets(min_system_states=2)}
    targets = tgt.build_targets(onsets)
    return data, targets, layers


def usable_years(data, targets, familia, N=1):
    """Anos com grafo E rótulo não censurado (mesma regra do baseline)."""
    cens = dict(zip(targets["ano"], targets[f"y{N}_{familia}_cens"]))
    return sorted(t for t in data if cens.get(t, 1) == 0)


def sanity_check():
    """Nº de grafos, distribuição de N, verificação anti-leakage."""
    print("== Sanity check gnn_data ==")
    for variant in ("com_dipl", "sem_dipl"):
        data, targets, layers = build_dataset(variant)
        ns = np.array([d["n"] for d in data.values()])
        print(f"\n[{variant}] {len(data)} anos-grafo x {len(layers)} camadas "
              f"= {len(data) * len(layers)} grafos")
        print(f"  N por ano: min={ns.min()} p25={np.percentile(ns, 25):.0f} "
              f"mediana={np.median(ns):.0f} p75={np.percentile(ns, 75):.0f} "
              f"max={ns.max()} média={ns.mean():.1f}")
        d0 = next(iter(data.values()))
        print(f"  node features: {d0['x'].shape[1]} "
              f"(1 cinc + {2 * len(layers)} grau/betw + 1 entrada)")
        for fam in ("proxy", "war"):
            yrs = usable_years(data, targets, fam)
            ys = dict(zip(targets["ano"], targets[f"y1_{fam}"]))
            pos = sum(ys[t] for t in yrs)
            print(f"  y1_{fam}: {len(yrs)} anos utilizáveis "
                  f"({yrs[0]}–{yrs[-1]}), {pos} positivos "
                  f"(prev={pos / len(yrs):.3f})")
        # anti-leakage estrutural: features de t dependem só de <= t por
        # construção (builders recebem apenas t; disp usa [t-2,t]; entrada usa
        # t-1; dipl-ffill usa max(ano medido <= t)). Checagem executável:
        # nenhum NaN/Inf e feature range [0,1].
        bad = [t for t, d in data.items()
               if not np.isfinite(d["x"]).all()
               or d["x"].min() < 0 or d["x"].max() > 1.0 + 1e-6]
        print(f"  features fora de [0,1] ou não-finitas: {len(bad)} anos "
              f"{'OK' if not bad else bad[:5]}")
    print("\nRegra anti-leakage: builders só recebem o próprio ano t "
          "(disp janela trailing [t-2,t]; entrada usa t-1; dipl-ffill usa "
          "último ano medido <= t). Nenhum acesso a t+1 existe no código.")


if __name__ == "__main__":
    sanity_check()
