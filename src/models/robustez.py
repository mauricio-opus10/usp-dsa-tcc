"""
Robustez do sinal: três testes baratos sobre a única célula com
sinal do baseline (proxy_N1_ge2), ANTES de qualquer decisão de narrativa/GNN.

Protocolo idêntico ao baseline (reutiliza walk_forward/cell_frame/make_models
de baseline.py): walk-forward purgado (s + N < t, treino mín. 40 anos),
censura respeitada, seed=42, hiperparâmetros fixos, SEM tuning.

Teste A – ablação H1 na célula COM sinal: proxy_N1_ge2, RF e XGB, em
  nivel_delta (48) vs completo (63) vs sem_dipl (46). A ablação do baseline
  usou a célula primária (war_N1, sem sinal); esta é a evidência H1 que vale.

Teste B – circularidade: conjuntos sem NENHUMA feature da camada de disputas
  (nível disp_* + delta d_disp_*) nem cross-layer envolvendo disputas
  (jaccard_ali_disp, jaccard_com_disp, jaccard_disp_dipl, overlap_medio_3c,
  overlap_medio_4c):
    completo_sem_disputas  – completo (63) - 17 = 46 colunas;
    sem_dipl_sem_disputas  – sem_dipl (46) - 15 = 31 colunas
      (jaccard_disp_dipl e overlap_medio_4c já saem com a diplomacia).
  Interpretação PRÉ-REGISTRADA em reports/ROBUSTEZ_SINAL.md (escrita antes
  destes números).

Teste C – moving block bootstrap sobre os anos de teste out-of-sample:
  blocos contíguos de BLOCK=10 anos na série OOS ordenada (posicional; a série
  é quase-contígua em calendário), N_BOOT=2.000 reamostragens, seed=42,
  IC 95% percentílico de AUC e PR-AUC. Reamostragens com classe única são
  descartadas e contadas. Células: proxy_N1 RF completo/sem_dipl; war_N1
  melhor modelo do baseline (lido de resultados_baseline.csv, previsões
  reutilizadas de predicoes_oos_baseline.csv); runs do Teste B; baseline de
  prevalência das mesmas células.

Uso:  ~/.virtualenvs/datascience/bin/python scripts/rodar_robustez.py
Saídas: data/processed/resultados_robustez.csv (coluna rodada=robustez),
        data/processed/robustez_bootstrap_ic.csv,
        data/interim/predicoes_oos_robustez.csv.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from .. import config as cfg
from ..features.build_features import METRICS, PAIRS
from . import baseline as bl

SEED = cfg.RANDOM_SEED
BLOCK = 10
N_BOOT = 2000
RODADA = "robustez"

CELL = next(c for c in bl.CELLS if c[0] == "proxy_N1_ge2")
WAR_CELL = "war_N1_ge2"

# ---------------------------------------------------------------------------
# Conjuntos de features do Teste B
# ---------------------------------------------------------------------------
DISP_DEP_COLS = ([f"disp_{m}" for m in METRICS]
                 + [f"d_disp_{m}" for m in METRICS]
                 + [f"jaccard_{a}_{b}" for a, b in PAIRS if "disp" in (a, b)]
                 + ["overlap_medio_3c", "overlap_medio_4c"])            # 17

SEM_DISPUTAS_SETS = {
    "completo_sem_disputas": [c for c in bl.FEATURE_SETS["completo"]
                              if c not in DISP_DEP_COLS],
    "sem_dipl_sem_disputas": [c for c in bl.FEATURE_SETS["sem_dipl"]
                              if c not in DISP_DEP_COLS],
}
bl.FEATURE_SETS.update(SEM_DISPUTAS_SETS)   # p/ cell_frame resolver os nomes

TESTE_A_SETS = ("nivel_delta", "completo", "sem_dipl")
TESTE_B_SETS = tuple(SEM_DISPUTAS_SETS)
MODELS = ("rf", "xgb")


# ---------------------------------------------------------------------------
# Teste C – moving block bootstrap
# ---------------------------------------------------------------------------
def block_bootstrap_ci(preds: pd.DataFrame, block: int = BLOCK,
                       n_boot: int = N_BOOT, seed: int = SEED) -> dict:
    """IC 95% (percentílico) de AUC e PR-AUC por moving block bootstrap.

    Blocos posicionais de `block` anos consecutivos da série OOS ordenada;
    amostra ceil(n/block) blocos com reposição e trunca em n. Reamostragens
    com classe única não definem AUC – são descartadas e contadas.
    """
    g = preds.sort_values("ano")
    y = g["y"].to_numpy(dtype=float)
    p = g["prob"].to_numpy(dtype=float)
    n = len(y)
    eff_block = min(block, n)
    starts_all = np.arange(0, n - eff_block + 1)
    n_blocks = int(np.ceil(n / eff_block))
    rng = np.random.default_rng(seed)
    aucs, aps = [], []
    skipped = 0
    for _ in range(n_boot):
        starts = rng.choice(starts_all, size=n_blocks, replace=True)
        idx = np.concatenate([np.arange(s, s + eff_block) for s in starts])[:n]
        yy, pp = y[idx], p[idx]
        if yy.min() == yy.max():
            skipped += 1
            continue
        aucs.append(roc_auc_score(yy, pp))
        aps.append(average_precision_score(yy, pp))
    out = {
        "n_anos_teste": n,
        "ano_min": int(g["ano"].min()), "ano_max": int(g["ano"].max()),
        "positivos_teste": int(y.sum()),
        "auc_pontual": round(float(roc_auc_score(y, p)), 3),
        "pr_auc_pontual": round(float(average_precision_score(y, p)), 3),
        "n_reamostras_validas": len(aucs), "n_descartadas_classe_unica": skipped,
        "tamanho_bloco": eff_block, "n_boot": n_boot,
    }
    for nome, vals in (("auc", aucs), ("pr_auc", aps)):
        out[f"{nome}_ic95_lo"] = round(float(np.percentile(vals, 2.5)), 3)
        out[f"{nome}_ic95_hi"] = round(float(np.percentile(vals, 97.5)), 3)
        out[f"{nome}_boot_media"] = round(float(np.mean(vals)), 3)
    return out


# ---------------------------------------------------------------------------
# Orquestração
# ---------------------------------------------------------------------------
def run_all():
    for nome, cols in SEM_DISPUTAS_SETS.items():
        print(f"[{nome}] {len(cols)} colunas")

    feats, targets = bl.load_data()
    cell, fam, N, ms = CELL
    all_metrics, all_preds = [], []

    def _record(preds, **meta):
        met = bl.aggregate_metrics(preds)
        for k, v in meta.items():
            met[k] = v
        all_metrics.append(met)
        p = preds.copy()
        for k, v in meta.items():
            p[k] = v
        all_preds.append(p)

    # --- Testes A e B: walk-forward em proxy_N1_ge2, RF e XGB + ingênuos
    for teste, fsets in (("A_ablacao", TESTE_A_SETS), ("B_sem_disputas", TESTE_B_SETS)):
        for fs in fsets:
            X, y, y_full, _ = bl.cell_frame(feats, targets, fam, N, ms, fs)
            print(f"[{teste} | {fs} ({X.shape[1]} col.)] anos utilizáveis: "
                  f"{X.index.min()}–{X.index.max()} ({len(X)}), prev={y.mean():.3f}")
            wf = bl.walk_forward(X, y, N, model_names=MODELS)
            nv = bl.walk_forward_naive(X, y, y_full, N)
            _record(pd.concat([wf, nv]), rodada=RODADA, teste=teste,
                    celula=cell, familia=fam, N=N, min_sys=ms, feature_set=fs,
                    n_features=X.shape[1], protocolo="walk_forward")

    metrics = pd.concat(all_metrics, ignore_index=True)
    preds = pd.concat(all_preds, ignore_index=True)

    # --- Teste C: bootstrap das células-chave
    base_preds = pd.read_csv(cfg.INTERIM_DIR / "predicoes_oos_baseline.csv")
    base_metrics = pd.read_csv(cfg.PROCESSED_DIR / "resultados_baseline.csv")

    # melhor modelo de war_N1 no baseline (PR-AUC, walk-forward, qualquer fs)
    war = base_metrics[(base_metrics.celula == WAR_CELL)
                       & (base_metrics.protocolo == "walk_forward")
                       & (base_metrics.modelo.isin(["logreg", "rf", "xgb"]))]
    war_best = war.sort_values("pr_auc", ascending=False).iloc[0]
    print(f"\nMelhor modelo war_N1 (baseline, PR-AUC WF): "
          f"{war_best.modelo} | {war_best.feature_set} "
          f"(AUC {war_best.auc_roc}, PR-AUC {war_best.pr_auc})")

    def _sel_nova(fs, modelo):
        return preds[(preds.feature_set == fs) & (preds.modelo == modelo)]

    def _sel_base(celula, fs, modelo):
        return base_preds[(base_preds.celula == celula)
                          & (base_preds.protocolo == "walk_forward")
                          & (base_preds.feature_set == fs)
                          & (base_preds.modelo == modelo)]

    alvos = [
        # (celula, feature_set, modelo, frame de previsões OOS)
        (cell, "completo", "rf", _sel_nova("completo", "rf")),
        (cell, "sem_dipl", "rf", _sel_nova("sem_dipl", "rf")),
        (cell, "completo", "prevalencia", _sel_nova("completo", "prevalencia")),
        (cell, "sem_dipl", "prevalencia", _sel_nova("sem_dipl", "prevalencia")),
        (WAR_CELL, war_best.feature_set, war_best.modelo,
         _sel_base(WAR_CELL, war_best.feature_set, war_best.modelo)),
        (WAR_CELL, "completo", "prevalencia",
         _sel_base(WAR_CELL, "completo", "prevalencia")),
    ]
    for fs in TESTE_B_SETS:
        for m in MODELS:
            alvos.append((cell, fs, m, _sel_nova(fs, m)))
        alvos.append((cell, fs, "prevalencia", _sel_nova(fs, "prevalencia")))

    ic_rows = []
    for celula, fs, modelo, g in alvos:
        if g.empty:
            raise RuntimeError(f"Sem previsões OOS para {celula}/{fs}/{modelo}")
        row = {"rodada": RODADA, "celula": celula, "feature_set": fs,
               "modelo": modelo}
        row.update(block_bootstrap_ci(g))
        ic_rows.append(row)
        print(f"  IC95 [{celula} | {fs} | {modelo}] "
              f"AUC {row['auc_pontual']} ({row['auc_ic95_lo']}–{row['auc_ic95_hi']}) "
              f"PR {row['pr_auc_pontual']} ({row['pr_auc_ic95_lo']}–{row['pr_auc_ic95_hi']})")
    ics = pd.DataFrame(ic_rows)

    # --- Persistir
    meta_cols = ["rodada", "teste", "celula", "familia", "N", "min_sys",
                 "feature_set", "n_features", "protocolo"]
    metrics = metrics[meta_cols + [c for c in metrics.columns if c not in meta_cols]]
    metrics.to_csv(cfg.PROCESSED_DIR / "resultados_robustez.csv", index=False)
    preds.to_csv(cfg.INTERIM_DIR / "predicoes_oos_robustez.csv", index=False)
    ics.to_csv(cfg.PROCESSED_DIR / "robustez_bootstrap_ic.csv", index=False)
    print(f"\n{len(metrics)} linhas de métricas -> resultados_robustez.csv; "
          f"{len(preds)} previsões OOS -> predicoes_oos_robustez.csv; "
          f"{len(ics)} ICs -> robustez_bootstrap_ic.csv")

    modelos_show = list(MODELS) + ["prevalencia", "persistencia_purgada"]
    show = metrics[metrics.modelo.isin(modelos_show)][
        ["teste", "feature_set", "n_features", "modelo", "n_anos_teste",
         "prevalencia_teste", "auc_roc", "pr_auc", "prec_05", "rec_05"]]
    print("\n" + show.to_string(index=False))
    return metrics, ics


if __name__ == "__main__":
    run_all()
