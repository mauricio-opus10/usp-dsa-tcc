#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Análises EXPLORATÓRIAS pós-hoc do branch (27/09/2026) – NÃO pré-declaradas.

Nenhum modelo é re-treinado: tudo é calculado sobre as previsões fora da
amostra já persistidas em data/interim/predicoes_oos_baseline.csv (célula
proxy_N1_ge2, protocolo walk_forward), com a mesma função de bootstrap do
etapa de robustez (src.models.robustez.block_bootstrap_ci, seed=42, 2.000 reamostragens).

Saídas (nesta pasta):
  - pos_hoc_periodos.csv : AUC/PR-AUC por período (tudo, 1931–1945, 1946+),
    com IC 95% por block bootstrap apenas onde a série comporta (tudo, 1946+).
  - pos_hoc_blocos.csv   : sensibilidade do IC ao tamanho do bloco (3–20 anos).
  - pos_hoc_rotulo.csv   : autocorrelação do rótulo e probabilidades
    condicionais (persistência do rótulo).
  - pos_hoc_tendencia.csv: diagnóstico da tendência temporal – AUC/PR-AUC de
    uma ordenação dos anos de teste só pelo calendário (ano mais antigo =
    maior risco), sem atributos. A direção foi escolhida após os resultados:
    é diagnóstico, não baseline.
  - pos_hoc_gnn.csv      : inspeção das previsões do ensemble das redes
    neurais de grafos na mesma célula (faixa de probabilidades, correlação
    com o ano, anos classificados como positivos, AUC por período).
  - pos_hoc_robustez_periodos.csv : AUC/PR-AUC por período das previsões da
    ablação e do teste de circularidade (data/interim/predicoes_oos_robustez.csv).
  - pos_hoc_rotulo_duracao.csv : duração das disputas que geram o rótulo
    do proxy (lê os brutos COW; a severidade é o máximo de toda a disputa).

Uso: ~/.virtualenvs/datascience/bin/python tcc_branch_h1_operacionalizacao/analises/analise_pos_hoc.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))
from src import config as cfg  # noqa: E402
from src.models.robustez import block_bootstrap_ci  # noqa: E402

CELULA = "proxy_N1_ge2"
CORTE = 1946            # mesmo corte do split entre eras
BLOCOS = (3, 5, 10, 15, 20)
MODELOS = ("logreg", "rf", "xgb", "prevalencia", "persistencia_purgada")
PERIODOS = (("tudo", None, None), ("ate_1945", None, CORTE - 1),
            ("de_1946", CORTE, None))

preds = pd.read_csv(ROOT / "data" / "interim" / "predicoes_oos_baseline.csv")
preds = preds[(preds.celula == CELULA) & (preds.protocolo == "walk_forward")]


def serie(fs, modelo):
    g = preds[(preds.feature_set == fs) & (preds.modelo == modelo)]
    return g.sort_values("ano").reset_index(drop=True)


# --- 1. Desempenho por período ---------------------------------------------
rows = []
for fs in ("completo", "sem_dipl"):
    for modelo in MODELOS:
        g = serie(fs, modelo)
        for nome, a0, a1 in PERIODOS:
            m = pd.Series(True, index=g.index)
            if a0 is not None:
                m &= g.ano >= a0
            if a1 is not None:
                m &= g.ano <= a1
            s = g[m].reset_index(drop=True)
            row = {"celula": CELULA, "feature_set": fs, "modelo": modelo,
                   "periodo": nome, "ano_min": int(s.ano.min()),
                   "ano_max": int(s.ano.max()), "n_anos": len(s),
                   "positivos": int(s.y.sum()),
                   "prevalencia": s.y.mean(),
                   "prob_media": s.prob.mean(),
                   "auc_roc": roc_auc_score(s.y, s.prob),
                   "pr_auc": average_precision_score(s.y, s.prob)}
            # IC só onde a série comporta blocos de 10 anos (>= 60 anos)
            if nome != "ate_1945":
                ci = block_bootstrap_ci(s)
                for k in ("auc_ic95_lo", "auc_ic95_hi", "pr_auc_ic95_lo",
                          "pr_auc_ic95_hi", "tamanho_bloco", "n_boot",
                          "n_descartadas_classe_unica"):
                    row[k] = ci[k]
            rows.append(row)
per = pd.DataFrame(rows)
per.to_csv(HERE / "pos_hoc_periodos.csv", index=False)

# --- 2. Sensibilidade ao tamanho do bloco ----------------------------------
rows = []
for fs in ("completo", "sem_dipl"):
    g = serie(fs, "rf")
    for b in BLOCOS:
        ci = block_bootstrap_ci(g, block=b)
        rows.append({"celula": CELULA, "feature_set": fs, "modelo": "rf",
                     "n_anos": len(g), **{k: ci[k] for k in (
                         "tamanho_bloco", "n_boot",
                         "n_descartadas_classe_unica", "auc_ic95_lo",
                         "auc_ic95_hi", "pr_auc_ic95_lo", "pr_auc_ic95_hi")}})
blo = pd.DataFrame(rows)
blo.to_csv(HERE / "pos_hoc_blocos.csv", index=False)

# --- 3. Persistência do rótulo ----------------------------------------------
rows = []
for fs in ("completo", "sem_dipl"):
    y = serie(fs, "rf").y.values
    prev, cur = y[:-1], y[1:]
    row = {"celula": CELULA, "feature_set": fs, "n_anos": len(y),
           "positivos": int(y.sum()),
           "p_pos_dado_anterior_pos": cur[prev == 1].mean(),
           "p_pos_dado_anterior_neg": cur[prev == 0].mean()}
    for lag in (1, 2, 3):
        row[f"acf_lag{lag}"] = np.corrcoef(y[:-lag], y[lag:])[0, 1]
    rows.append(row)
rot = pd.DataFrame(rows)
rot.to_csv(HERE / "pos_hoc_rotulo.csv", index=False)

# --- 4. Tendência temporal pura (diagnóstico) -------------------------------
def _mask(g, a0, a1):
    m = pd.Series(True, index=g.index)
    if a0 is not None:
        m &= g.ano >= a0
    if a1 is not None:
        m &= g.ano <= a1
    return m


rows = []
for fs in ("completo", "sem_dipl"):
    g = serie(fs, "rf")
    for nome, a0, a1 in PERIODOS:
        s_ = g[_mask(g, a0, a1)]
        rows.append({"celula": CELULA, "feature_set": fs, "periodo": nome,
                     "n_anos": len(s_), "positivos": int(s_.y.sum()),
                     "auc_roc": roc_auc_score(s_.y, -s_.ano),
                     "pr_auc": average_precision_score(s_.y, -s_.ano)})
ten = pd.DataFrame(rows)
ten.to_csv(HERE / "pos_hoc_tendencia.csv", index=False)

# --- 5. Previsões das redes neurais de grafos -------------------------------
gnn = pd.read_csv(ROOT / "data" / "interim" / "predicoes_oos_gnn.csv")
gnn = gnn[gnn.celula == CELULA]
rows = []
for (var, arch), d in gnn.groupby(["variante", "arch"]):
    e = (d.groupby("ano").agg(prob=("prob", "mean"), y=("y", "first"))
         .reset_index())                       # ensemble = média das sementes
    pos = sorted(e[e.prob >= 0.5].ano.tolist())
    row = {"celula": CELULA, "variante": var, "arch": arch, "n_anos": len(e),
           "prob_min": e.prob.min(), "prob_max": e.prob.max(),
           "corr_prob_ano": np.corrcoef(e.prob, e.ano)[0, 1],
           "n_previstos_pos": len(pos), "previsto_ano_min": pos[0],
           "previsto_ano_max": pos[-1],
           "bloco_contiguo": pos == list(range(pos[0], pos[-1] + 1)),
           "verdadeiros_pos": int(e[(e.prob >= 0.5) & (e.y == 1)].shape[0])}
    for nome, a0, a1 in PERIODOS:
        s_ = e[_mask(e, a0, a1)]
        row[f"auc_{nome}"] = roc_auc_score(s_.y, s_.prob)
    rows.append(row)
gnd = pd.DataFrame(rows)
gnd.to_csv(HERE / "pos_hoc_gnn.csv", index=False)

# --- 6. Duração das disputas que geram o rótulo do proxy --------------------
from src.features.target import proxy_onsets  # noqa: E402

ons = proxy_onsets(min_system_states=2)        # mesma função do rótulo
midb = pd.read_csv(cfg.cow_raw_dir() / cfg.COW_FILES["midb"])
fim = midb.groupby("dispnum").endyear.max()
dur = fim.reindex(ons.conflito_id).values - ons.onset_year.values
durd = pd.DataFrame([{"celula": CELULA, "n_disputas": len(ons),
                      "terminam_no_ano_de_inicio": int((dur == 0).sum()),
                      "terminam_no_ano_seguinte": int((dur == 1).sum()),
                      "duram_dois_anos_ou_mais": int((dur >= 2).sum()),
                      "terminam_depois_do_inicio": int((dur >= 1).sum())}])
durd.to_csv(HERE / "pos_hoc_rotulo_duracao.csv", index=False)

# --- 7. Ablação e teste de circularidade por período -----------------------
rob = pd.read_csv(ROOT / "data" / "interim" / "predicoes_oos_robustez.csv")
rob = rob[(rob.celula == CELULA) & rob.modelo.isin(["rf", "xgb"])]
rows = []
for (fs, modelo), d in rob.groupby(["feature_set", "modelo"]):
    d = d.sort_values("ano").reset_index(drop=True)
    for nome, a0, a1 in PERIODOS:
        s_ = d[_mask(d, a0, a1)]
        rows.append({"celula": CELULA, "feature_set": fs, "modelo": modelo,
                     "periodo": nome, "n_anos": len(s_),
                     "positivos": int(s_.y.sum()),
                     "auc_roc": roc_auc_score(s_.y, s_.prob),
                     "pr_auc": average_precision_score(s_.y, s_.prob)})
robp = pd.DataFrame(rows)
robp.to_csv(HERE / "pos_hoc_robustez_periodos.csv", index=False)

pd.set_option("display.width", 250, "display.max_columns", 30)
print(f"[seed={cfg.RANDOM_SEED}] saídas em {HERE}")
print(per.round(3).to_string(index=False))
print(blo.round(3).to_string(index=False))
print(rot.round(3).to_string(index=False))
print(ten.round(3).to_string(index=False))
print(gnd.round(3).to_string(index=False))
print(durd.to_string(index=False))
print(robp.round(3).to_string(index=False))
