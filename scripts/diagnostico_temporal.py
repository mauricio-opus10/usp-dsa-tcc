#!/usr/bin/env python3
"""
DIAGNÓSTICO TEMPORAL EMPÍRICO – TCC DSA
===================================================
Mede, a partir dos dados COW REAIS, as evidências que vão informar as decisões
temporais acopladas (granularidade do snapshot, janela de suavização, horizonte
N do alvo, threshold de hostilidade) ANTES de escrever o feature pipeline.

Escopo: read-only sobre os brutos COW. Só escreve em data/interim/ (CSVs) e
imprime um resumo. NÃO constrói features finais, NÃO treina modelos, NÃO escolhe
target/janela/threshold finais – apenas mede e expõe.

Regra de Ouro: todo número é traceável (arquivo de origem + cálculo). Lacuna de
dado = TODO explícito, nunca estimativa silenciosa.

Convenções herdadas do BIA:
  - Opção A: Estados CINC >= threshold entram em todas as camadas, mesmo isolados;
             densidade/centralidades sobre N completo.
  - Comércio usa smoothtotrade (não flow1+flow2).
  - seed = 42.

Uso:
  ~/.virtualenvs/datascience/bin/python scripts/diagnostico_temporal.py
"""

import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config as cfg  # noqa: E402

np.random.seed(cfg.RANDOM_SEED)

RAW = cfg.cow_raw_dir()
OUT = cfg.INTERIM_DIR
OUT.mkdir(parents=True, exist_ok=True)
CINC = cfg.CINC_THRESHOLD
Y0, Y1 = cfg.YEAR_MIN, cfg.YEAR_MAX


def _p(name):
    return RAW / cfg.COW_FILES[name]


def log(msg=""):
    print(msg)
    LOG.append(str(msg))


LOG = []


# ===========================================================================
# Carregamento (uma vez)
# ===========================================================================
log("=" * 70)
log("DIAGNÓSTICO TEMPORAL – TCC DSA")
log(f"Fonte COW: {RAW}")
log(f"CINC threshold = {CINC} (1%) · seed = {cfg.RANDOM_SEED} · networkx {nx.__version__}")
log("=" * 70)

nmc = pd.read_csv(_p("nmc"))
trade = pd.read_csv(_p("trade_dyadic"))
ally = pd.read_csv(_p("alliances_yearly"))
midb = pd.read_csv(_p("midb"))
dip = pd.read_csv(_p("diplomatic"))

# nomes de Estado para referência
names = nmc[["ccode", "stateabb"]].drop_duplicates().set_index("ccode")["stateabb"].to_dict()


# ===========================================================================
# B.1 – Cobertura temporal real por dataset
# ===========================================================================
log("\n" + "#" * 70)
log("B.1 – COBERTURA TEMPORAL REAL POR DATASET")
log("#" * 70)

# alianças: usar a coluna 'year' do arquivo yearly (ano de atividade da díade)
ally_years = ally.loc[ally["year"] > 0, "year"]
dip_years = sorted(dip["year"].unique())
dip_gaps = [b - a for a, b in zip(dip_years[:-1], dip_years[1:])]

cov_rows = [
    {"camada": "alianças (CINC source: n/a)", "arquivo": cfg.COW_FILES["alliances_yearly"],
     "col_ano": "year", "ano_min": int(ally_years.min()), "ano_max": int(ally_years.max()),
     "cadencia": "anual", "obs": "year==0 = censura/sem ano; ignorado"},
    {"camada": "comércio", "arquivo": cfg.COW_FILES["trade_dyadic"],
     "col_ano": "year", "ano_min": int(trade.year.min()), "ano_max": int(trade.year.max()),
     "cadencia": "anual", "obs": "usar smoothtotrade>0"},
    {"camada": "disputas (MIDB)", "arquivo": cfg.COW_FILES["midb"],
     "col_ano": "styear/endyear", "ano_min": int(midb.styear.min()), "ano_max": int(midb.endyear.max()),
     "cadencia": "anual (intervalos st..end)", "obs": "ativa em t se styear<=t<=endyear"},
    {"camada": "CINC (NMC)", "arquivo": cfg.COW_FILES["nmc"],
     "col_ano": "year", "ano_min": int(nmc.year.min()), "ano_max": int(nmc.year.max()),
     "cadencia": "anual", "obs": "LIMITA composição do sistema (nós)"},
    {"camada": "diplomacia", "arquivo": cfg.COW_FILES["diplomatic"],
     "col_ano": "year", "ano_min": int(dip.year.min()), "ano_max": int(dip.year.max()),
     "cadencia": f"~quinquenal ({len(dip_years)} anos medidos)", "obs": "NÃO é anual"},
]
cov = pd.DataFrame(cov_rows)
cov.to_csv(OUT / "B1_cobertura_temporal.csv", index=False)
log(cov.to_string(index=False))
log(f"\nAnos medidos da diplomacia ({len(dip_years)}): {dip_years}")
log(f"Gaps entre medições diplomáticas (anos): {dip_gaps}")
log(f"  -> menor gap={min(dip_gaps)}, maior gap={max(dip_gaps)} (1940->1950)")
log("\nIMPLICAÇÃO B.1: a janela utilizável do multilayer COMPLETO com filtro CINC")
log("é 1890–2012 (NMC termina em 2012). Trade/MIDs vão a 2014, mas sem nós CINC.")
log("Diplomacia anual exige carry-forward/interpolação (ver TODO).")


# ===========================================================================
# B.2 – Composição do sistema por ano (CINC >= 1%)
# ===========================================================================
log("\n" + "#" * 70)
log("B.2 – COMPOSIÇÃO DO SISTEMA POR ANO (CINC >= 1%)")
log("#" * 70)

nmc_years = sorted(y for y in nmc["year"].unique() if Y0 <= y <= Y1)


def states_in_year(year):
    sub = nmc[(nmc["year"] == year) & (nmc["cinc"] >= CINC)]
    return set(int(c) for c in sub["ccode"].unique())


comp_rows = []
prev = None
for y in nmc_years:
    s = states_in_year(y)
    entrants = sorted(s - prev) if prev is not None else []
    exits = sorted(prev - s) if prev is not None else []
    comp_rows.append({
        "year": y, "n_states": len(s),
        "entrants": len(entrants), "exits": len(exits),
        "churn": len(entrants) + len(exits),
        "entrants_abb": ",".join(names.get(c, str(c)) for c in entrants),
        "exits_abb": ",".join(names.get(c, str(c)) for c in exits),
    })
    prev = s
comp = pd.DataFrame(comp_rows)
comp.to_csv(OUT / "B2_composicao_sistema_por_ano.csv", index=False)
log(f"Anos cobertos (NMC, 1890–2014 efetivo): {nmc_years[0]}–{nmc_years[-1]} "
    f"({len(nmc_years)} anos)")
log(f"n_states: min={comp.n_states.min()}, max={comp.n_states.max()}, "
    f"média={comp.n_states.mean():.1f}")
log(f"churn ano-a-ano: média={comp.churn[1:].mean():.2f}, máx={comp.churn.max()}")
log("\nAmostra (a cada ~10 anos):")
log(comp[comp.year % 10 == 0][["year", "n_states", "entrants", "exits", "churn"]]
    .to_string(index=False))


# ===========================================================================
# Builders por ano (snapshot estrito) – Opção A
# ===========================================================================
def alliance_graph(year, nodeset):
    G = nx.Graph()
    G.add_nodes_from(nodeset)
    sub = ally[(ally["year"] == year)
               & (ally["ccode1"].isin(nodeset)) & (ally["ccode2"].isin(nodeset))]
    for r in sub.itertuples(index=False):
        if (r.defense or r.neutrality or r.nonaggression or r.entente):
            G.add_edge(int(r.ccode1), int(r.ccode2))
    return G


def trade_graph(year, nodeset):
    G = nx.Graph()
    G.add_nodes_from(nodeset)
    sub = trade[(trade["year"] == year) & (trade["smoothtotrade"] > 0)
                & (trade["ccode1"].isin(nodeset)) & (trade["ccode2"].isin(nodeset))]
    for r in sub.itertuples(index=False):
        G.add_edge(int(r.ccode1), int(r.ccode2))
    return G


def dispute_graph(year, nodeset, window=1):
    """Disputas ativas em [year-window+1, year]; arestas Side A vs Side B."""
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
                    G.add_edge(int(s1), int(s2))
    return G


def diplomacy_graph(year, nodeset):
    """Só definida em anos medidos; senão retorna None (sem dado)."""
    if year not in dip_years:
        return None
    G = nx.Graph()
    G.add_nodes_from(nodeset)
    sub = dip[(dip["year"] == year) & (dip["DE"] == 1)
              & (dip["ccode1"].isin(nodeset)) & (dip["ccode2"].isin(nodeset))]
    for r in sub.itertuples(index=False):
        G.add_edge(int(r.ccode1), int(r.ccode2))
    return G


# ===========================================================================
# B.3 – Esparsidade das redes anuais, por camada (snapshot estrito)
# ===========================================================================
log("\n" + "#" * 70)
log("B.3 – ESPARSIDADE DAS REDES ANUAIS POR CAMADA (snapshot de ANO ESTRITO)")
log("#" * 70)

spar_rows = []
for y in nmc_years:
    ns = states_in_year(y)
    layers = {
        "alliance": alliance_graph(y, ns),
        "trade": trade_graph(y, ns),
        "disputes": dispute_graph(y, ns, window=1),
        "diplomacy": diplomacy_graph(y, ns),
    }
    for layer, G in layers.items():
        if G is None:
            spar_rows.append({"year": y, "layer": layer, "nodes": len(ns),
                              "edges": np.nan, "density": np.nan, "data": "sem_dado"})
        else:
            spar_rows.append({"year": y, "layer": layer, "nodes": G.number_of_nodes(),
                              "edges": G.number_of_edges(),
                              "density": round(nx.density(G), 4), "data": "ok"})
spar = pd.DataFrame(spar_rows)
spar.to_csv(OUT / "B3_esparsidade_redes_anuais.csv", index=False)


def _summ(layer):
    s = spar[(spar.layer == layer) & (spar.data == "ok")]
    if s.empty:
        return f"  {layer:10s}: SEM DADO em snapshot estrito"
    degen = int((s.edges <= 1).sum())
    return (f"  {layer:10s}: anos_ok={len(s):3d}  dens_média={s.density.mean():.3f}  "
            f"E_médio={s.edges.mean():5.1f}  anos_degenerados(E<=1)={degen}")


log("Resumo por camada (1890–2012):")
for layer in ["alliance", "trade", "disputes", "diplomacy"]:
    log(_summ(layer))

dip_ok = spar[(spar.layer == "diplomacy") & (spar.data == "ok")]
dip_nodata = spar[(spar.layer == "diplomacy") & (spar.data == "sem_dado")]
log(f"\nDiplomacia: {len(dip_ok)} anos COM dado, {len(dip_nodata)} anos SEM dado "
    f"(de {len(nmc_years)} anos) -> {len(dip_nodata)/len(nmc_years)*100:.0f}% sem dado.")

disp = spar[(spar.layer == "disputes") & (spar.data == "ok")]
log(f"Disputas (ano estrito): {int((disp.edges == 0).sum())} anos com 0 arestas; "
    f"{int((disp.edges <= 1).sum())} anos degenerados (E<=1) de {len(disp)}.")


# ===========================================================================
# B.4 – Efeito da janela de suavização nas disputas
# ===========================================================================
log("\n" + "#" * 70)
log("B.4 – JANELA DE SUAVIZAÇÃO NAS DISPUTAS (trailing {1,3,5} anos)")
log("#" * 70)

b4_rows = []
for w in (1, 3, 5):
    densities, nondegen = [], 0
    series = []
    for y in nmc_years:
        ns = states_in_year(y)
        G = dispute_graph(y, ns, window=w)
        d = nx.density(G)
        densities.append(d)
        e = G.number_of_edges()
        if e > 1:
            nondegen += 1
        series.append({"year": y, "window": w, "nodes": G.number_of_nodes(),
                       "edges": e, "density": round(d, 4)})
    b4_rows.extend(series)
    log(f"  janela={w}a: dens_média={np.mean(densities):.4f}  "
        f"anos_não_degenerados(E>1)={nondegen}/{len(nmc_years)}  "
        f"({nondegen/len(nmc_years)*100:.0f}%)")
b4 = pd.DataFrame(b4_rows)
b4.to_csv(OUT / "B4_disputas_suavizacao.csv", index=False)


# ===========================================================================
# B.5 – Inventário de severidade + balanceamento de classes (EXPLORATÓRIO)
# ===========================================================================
log("\n" + "#" * 70)
log("B.5 – SEVERIDADE DE CONFLITO + BALANCEAMENTO DE CLASSES (EXPLORATÓRIO)")
log("#" * 70)

# Inventário: existe arquivo de Inter-State Wars do COW?
war_candidates = list(RAW.rglob("*[Ww]ar*.csv"))
log(f"Inter-State War (COW) dataset presente? {'SIM' if war_candidates else 'NÃO'} "
    f"-> {[str(p.relative_to(RAW)) for p in war_candidates] if war_candidates else 'TODO'}")

# Distribuição de hostlev (MIDB, participantes) no recorte
host = midb[(midb["styear"] >= Y0) & (midb["styear"] <= Y1)]
host_dist = host["hostlev"].value_counts().sort_index()
log("\nDistribuição de hostlev (participações MIDB, onset 1890–2014):")
for lvl, n in host_dist.items():
    log(f"  hostlev={lvl} ({cfg.HOSTLEV_LABELS.get(lvl, '?')}): {n}")

fatal_dist = host["fatality"].value_counts().sort_index()
log("\nDistribuição de fatality (participações MIDB):")
for lvl, n in fatal_dist.items():
    log(f"  fatality={lvl}: {n}")


# Onset por disputa: max severidade entre participantes do SISTEMA, com >=2 nós-sistema
def onset_years(severity_col, threshold):
    """Anos de onset (styear) de disputas com max(severity)>=threshold e >=2
    participantes do sistema (CINC>=1% no styear)."""
    years = set()
    for dispnum, part in midb.groupby("dispnum"):
        sy = int(part["styear"].iloc[0])
        if not (Y0 <= sy <= Y1):
            continue
        ns = states_in_year(sy) if Y0 <= sy <= max(nmc_years) else set()
        sys_part = part[part["ccode"].isin(ns)]
        if sys_part["ccode"].nunique() < 2:
            continue
        sev = sys_part[severity_col].replace(-9, np.nan).max()
        if pd.notna(sev) and sev >= threshold:
            years.add(sy)
    return years


trial_defs = {
    "war (hostlev==5)": ("hostlev", 5),
    "use_of_force+ (hostlev>=4)": ("hostlev", 4),
    "fatal (fatality>=1)": ("fatality", 1),
}

b5_rows = []
feat_years = [y for y in nmc_years]  # anos-feature candidatos
log("\nContagem de classe positiva (alvo-trial = há onset em [t+1, t+N]):")
log(f"  anos-feature avaliados: {feat_years[0]}–{feat_years[-1]} ({len(feat_years)})")
for label, (col, thr) in trial_defs.items():
    oy = onset_years(col, thr)
    for N in (1, 3, 5):
        pos = sum(1 for t in feat_years if any((t + k) in oy for k in range(1, N + 1)))
        neg = len(feat_years) - pos
        b5_rows.append({"definicao": label, "horizonte_N": N,
                        "n_onset_years": len(oy), "positivos": pos, "negativos": neg,
                        "taxa_positiva": round(pos / len(feat_years), 3)})
        log(f"  {label:28s} N={N}: pos={pos:3d} neg={neg:3d} "
            f"(taxa_pos={pos/len(feat_years)*100:4.1f}%)  [onset_years={len(oy)}]")
b5 = pd.DataFrame(b5_rows)
b5.to_csv(OUT / "B5_balanceamento_classes_trial.csv", index=False)


# ===========================================================================
# Persistir log
# ===========================================================================
log("\n" + "=" * 70)
log("CSVs gerados em data/interim/:")
for f in sorted(OUT.glob("B*_*.csv")):
    log(f"  - {f.name}")
log("=" * 70)

with open(OUT / "diagnostico_log.txt", "w") as fh:
    fh.write("\n".join(LOG) + "\n")
print(f"\n[ok] log salvo em {OUT/'diagnostico_log.txt'}")
