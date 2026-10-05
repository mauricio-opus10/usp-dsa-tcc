"""
Baseline ML: LogReg / RandomForest / XGBoost sobre as variantes de
alvo, com validação walk-forward purgada + split entre eras, baselines
ingênuos, ablação H1 e SHAP.

Células de alvo (a escolha final aguarda sign-off do orientador):
  {war, proxy} x {N=1, N=3} na definição >=2 Estados do sistema no onset
  (a família de definição que reproduz o B.5; aqui na versão "limpa" de
  target.py – min(styear), severidade nível-disputa – prevalências 0,252/0,593
  vs 0,244/0,569 do trial, diferença documentada em FEATURES_TARGETS.md §2).
  + célula de sensibilidade: war N=1 com >=1 Estado.

Censura: anos com flag *_cens==1 ficam FORA de treino e teste (nunca são 0).

Conjuntos de features:
  completo  – 63 colunas; utilizável até 2005 (diplomacia = NaN em 2006–12);
  sem_dipl  – 46 colunas (drop das 16 dependentes de diplomacia + dipl_imputado);
              cobre a janela toda.
  nivel_delta – ablação H1: só nível+delta por camada (48), sem cross-layer,
              composição e CINC.

Protocolos:
  walk_forward – expanding window com PURGA: para prever t, treina em anos s
    com s + N < t (o rótulo y_N(s) usa [s+1, s+N] e precisa estar totalmente
    resolvido antes de t); treino mínimo = 40 anos.
  era_split – treino 1890–1945 (com a mesma purga: s <= 1945 - N), teste
    1946 em diante ("o modelo do mundo pré-1945 reconhece a Guerra Fria?").

Modelos (baseline honesto, SEM tuning – hiperparâmetros fixos documentados):
  logreg – StandardScaler + LogisticRegression(L2, C=1.0, balanced) em
           Pipeline (scaler fitado SÓ no treino de cada fold);
  rf     – RandomForest(500 árvores, max_depth=4, min_samples_leaf=5,
           balanced) – profundidade limitada contra overfit; árvores não
           precisam de scaler;
  xgb    – XGBoost(300 árvores, max_depth=3, lr=0.05, subsample=0.8,
           colsample=0.8, scale_pos_weight=neg/pos do treino do fold).
  seed=42 em tudo.

Baselines ingênuos: (a) prevalência do treino como score constante;
(b) persistência y(t)=y(t-1) COMO PEDIDO NO PROTOCOLO – com o caveat de que
y(t-1) só se resolve em t-1+N, então também reportamos a versão purgada
y(t)=y(t-1-N), que é a única de fato disponível em t.

Uso:  ~/.virtualenvs/datascience/bin/python scripts/rodar_baseline.py
Saídas: data/processed/resultados_baseline.csv,
        data/interim/predicoes_oos_baseline.csv,
        reports/figuras_modelos/shap_summary_*.png (+ shap_top15 CSV).
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, confusion_matrix,
                             precision_recall_curve, roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from .. import config as cfg
from ..features import target as tgt
from ..features.build_features import DIPL_COLS, LAYERS, METRICS, PAIRS

SEED = cfg.RANDOM_SEED
MIN_TRAIN = 40
ERA_SPLIT_TEST_START = 1946

# ---------------------------------------------------------------------------
# Conjuntos de colunas
# ---------------------------------------------------------------------------
LEVEL_COLS = [f"{lay}_{m}" for lay in LAYERS for m in METRICS]          # 24
DELTA_COLS = [f"d_{c}" for c in LEVEL_COLS]                             # 24
CROSS_COLS = ([f"jaccard_{a}_{b}" for a, b in PAIRS]
              + ["overlap_medio_3c", "overlap_medio_4c"])               # 8
COMP_COLS = ["n_estados", "churn_entradas", "churn_saidas", "churn_total"]
CINC_COLS = ["cinc_hhi", "cinc_lider_share"]
ALL_FEATURES = (LEVEL_COLS + DELTA_COLS + CROSS_COLS + COMP_COLS
                + CINC_COLS + ["dipl_imputado"])                        # 63

DIPL_DEP_COLS = DIPL_COLS + [f"d_dipl_{m}" for m in METRICS]            # 16

FEATURE_SETS = {
    "completo": ALL_FEATURES,
    "sem_dipl": [c for c in ALL_FEATURES
                 if c not in DIPL_DEP_COLS and c != "dipl_imputado"],   # 46
    "nivel_delta": LEVEL_COLS + DELTA_COLS,                             # 48
}

# Células: (nome, família, N, min_system_states)
CELLS = [
    ("war_N1_ge2", "war", 1, 2),        # PRIMÁRIA
    ("war_N3_ge2", "war", 3, 2),
    ("proxy_N1_ge2", "proxy", 1, 2),
    ("proxy_N3_ge2", "proxy", 3, 2),
    ("war_N1_ge1_sens", "war", 1, 1),   # sensibilidade >=1
]
PRIMARY_CELL = "war_N1_ge2"


# ---------------------------------------------------------------------------
# Dados
# ---------------------------------------------------------------------------
def load_data():
    """Features (CSV processado) + alvos por min_system_states (in-memory)."""
    feats = pd.read_csv(cfg.PROCESSED_DIR / "features_1890_2012.csv")
    targets = {}
    for ms in (1, 2):
        onsets = {"proxy": tgt.proxy_onsets(min_system_states=ms),
                  "war": tgt.wardata_onsets(min_system_states=ms)}
        targets[ms] = tgt.build_targets(onsets)
    return feats, targets


def cell_frame(feats, targets, familia, N, min_sys, feature_set):
    """DataFrame ano-indexado com X (sem NaN), y e máscara utilizável.

    Utilizável = features completas (sem NaN no conjunto escolhido) E rótulo
    não censurado. Anos censurados NUNCA entram (nem como 0).
    """
    cols = FEATURE_SETS[feature_set]
    df = feats.merge(targets[min_sys][["ano", f"y{N}_{familia}",
                                       f"y{N}_{familia}_cens"]], on="ano")
    df = df.set_index("ano").sort_index()
    usable = df[cols].notna().all(axis=1) & (df[f"y{N}_{familia}_cens"] == 0)
    X = df.loc[usable, cols]
    y = df.loc[usable, f"y{N}_{familia}"].astype(int)
    y_full = df[f"y{N}_{familia}"].astype(int)      # p/ persistência
    cens_full = df[f"y{N}_{familia}_cens"].astype(int)
    return X, y, y_full, cens_full


# ---------------------------------------------------------------------------
# Modelos
# ---------------------------------------------------------------------------
def make_models(y_train):
    pos = int(y_train.sum())
    neg = len(y_train) - pos
    spw = neg / pos if pos > 0 else 1.0
    return {
        "logreg": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(penalty="l2", C=1.0,
                                       class_weight="balanced",
                                       max_iter=5000, random_state=SEED)),
        ]),
        "rf": RandomForestClassifier(n_estimators=500, max_depth=4,
                                     min_samples_leaf=5,
                                     class_weight="balanced",
                                     random_state=SEED, n_jobs=-1),
        "xgb": XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05,
                             subsample=0.8, colsample_bytree=0.8,
                             scale_pos_weight=spw, random_state=SEED,
                             eval_metric="logloss", n_jobs=-1, verbosity=0),
    }


def best_f1_threshold(y_true, probs):
    """Threshold que maximiza F1 (sobre as previsões DE TREINO do fold)."""
    prec, rec, thr = precision_recall_curve(y_true, probs)
    with np.errstate(divide="ignore", invalid="ignore"):
        f1 = np.where((prec + rec) > 0, 2 * prec * rec / (prec + rec), 0.0)
    if len(thr) == 0:
        return 0.5
    return float(thr[int(np.argmax(f1[:-1]))])


# ---------------------------------------------------------------------------
# Protocolos
# ---------------------------------------------------------------------------
def walk_forward(X, y, N, model_names=("logreg", "rf", "xgb")):
    """Expanding window purgado. Retorna DataFrame longo de previsões OOS:
    (ano, modelo, y, prob, pred05, predthr, thr)."""
    years = list(X.index)
    rows = []
    for t in years:
        train_years = [s for s in years if s + N < t]
        if len(train_years) < MIN_TRAIN:
            continue
        Xtr, ytr = X.loc[train_years], y.loc[train_years]
        Xte = X.loc[[t]]
        if ytr.nunique() < 2:   # guarda: treino de classe única
            continue
        models = make_models(ytr)
        for name in model_names:
            m = models[name]
            m.fit(Xtr, ytr)
            p_tr = m.predict_proba(Xtr)[:, 1]
            thr = best_f1_threshold(ytr, p_tr)
            p = float(m.predict_proba(Xte)[:, 1][0])
            rows.append({"ano": t, "modelo": name, "y": int(y.loc[t]),
                         "prob": p, "pred05": int(p >= 0.5),
                         "predthr": int(p >= thr), "thr": thr})
    return pd.DataFrame(rows)


def walk_forward_naive(X, y, y_full, N):
    """Baselines ingênuos nos MESMOS anos de teste do walk-forward."""
    years = list(X.index)
    rows = []
    for t in years:
        train_years = [s for s in years if s + N < t]
        if len(train_years) < MIN_TRAIN:
            continue
        prev_rate = float(y.loc[train_years].mean())
        naive = {"prevalencia": prev_rate}
        if (t - 1) in y_full.index:
            naive["persistencia"] = float(y_full.loc[t - 1])
        if (t - 1 - N) in y_full.index:
            naive["persistencia_purgada"] = float(y_full.loc[t - 1 - N])
        for name, p in naive.items():
            rows.append({"ano": t, "modelo": name, "y": int(y.loc[t]),
                         "prob": p, "pred05": int(p >= 0.5),
                         "predthr": int(p >= 0.5), "thr": 0.5})
    return pd.DataFrame(rows)


def era_split(X, y, N, model_names=("logreg", "rf", "xgb")):
    """Treino 1890..(1946-N-1) [purga], teste 1946..fim utilizável."""
    tr_years = [s for s in X.index if s + N < ERA_SPLIT_TEST_START]
    te_years = [t for t in X.index if t >= ERA_SPLIT_TEST_START]
    Xtr, ytr = X.loc[tr_years], y.loc[tr_years]
    rows = []
    models = make_models(ytr)
    for name in model_names:
        m = models[name]
        m.fit(Xtr, ytr)
        p_tr = m.predict_proba(Xtr)[:, 1]
        thr = best_f1_threshold(ytr, p_tr)
        probs = m.predict_proba(X.loc[te_years])[:, 1]
        for t, p in zip(te_years, probs):
            rows.append({"ano": t, "modelo": name, "y": int(y.loc[t]),
                         "prob": float(p), "pred05": int(p >= 0.5),
                         "predthr": int(p >= thr), "thr": thr})
    # baseline de prevalência do treino
    prev_rate = float(ytr.mean())
    for t in te_years:
        rows.append({"ano": t, "modelo": "prevalencia", "y": int(y.loc[t]),
                     "prob": prev_rate, "pred05": int(prev_rate >= 0.5),
                     "predthr": int(prev_rate >= 0.5), "thr": 0.5})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Métricas agregadas
# ---------------------------------------------------------------------------
def _cm(y, pred):
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return int(tn), int(fp), int(fn), int(tp)


def _prf(y, pred):
    tn, fp, fn, tp = _cm(y, pred)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return prec, rec, f1, (tn, fp, fn, tp)


def aggregate_metrics(preds: pd.DataFrame) -> pd.DataFrame:
    """Métricas sobre o conjunto agregado de previsões out-of-sample."""
    out = []
    for name, g in preds.groupby("modelo"):
        y, prob = g["y"].values, g["prob"].values
        auc = roc_auc_score(y, prob) if len(np.unique(y)) == 2 else np.nan
        ap = average_precision_score(y, prob) if len(np.unique(y)) == 2 else np.nan
        p05, r05, f05, cm05 = _prf(y, g["pred05"].values)
        pth, rth, fth, cmth = _prf(y, g["predthr"].values)
        out.append({
            "modelo": name, "n_anos_teste": len(g), "positivos_teste": int(y.sum()),
            "prevalencia_teste": round(float(y.mean()), 3),
            "auc_roc": round(float(auc), 3), "pr_auc": round(float(ap), 3),
            "prec_05": round(p05, 3), "rec_05": round(r05, 3), "f1_05": round(f05, 3),
            "cm_05_tn_fp_fn_tp": "/".join(map(str, cm05)),
            "thr_medio": round(float(g["thr"].mean()), 3),
            "prec_thr": round(pth, 3), "rec_thr": round(rth, 3), "f1_thr": round(fth, 3),
            "cm_thr_tn_fp_fn_tp": "/".join(map(str, cmth)),
        })
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------
# SHAP sobre as previsões out-of-sample (walk-forward)
# ---------------------------------------------------------------------------
def shap_walk_forward(X, y, N, model_name):
    """Refaz o walk-forward do modelo de árvore escolhido acumulando os SHAP
    values de cada ano de teste (explicados pelo modelo do próprio fold –
    100% out-of-sample). Retorna (matriz SHAP, X dos anos explicados)."""
    import shap
    years = list(X.index)
    sv_rows, x_rows, idx = [], [], []
    for t in years:
        train_years = [s for s in years if s + N < t]
        if len(train_years) < MIN_TRAIN:
            continue
        ytr = y.loc[train_years]
        if ytr.nunique() < 2:
            continue
        m = make_models(ytr)[model_name]
        m.fit(X.loc[train_years], ytr)
        sv = shap.TreeExplainer(m).shap_values(X.loc[[t]])
        if isinstance(sv, list):            # RF (versões antigas): [cls0, cls1]
            sv = sv[1]
        if sv.ndim == 3:                    # RF (versões novas): (n, feat, cls)
            sv = sv[:, :, 1]
        sv_rows.append(sv[0])
        x_rows.append(X.loc[t])
        idx.append(t)
    sv_mat = pd.DataFrame(sv_rows, index=idx, columns=X.columns)
    x_mat = pd.DataFrame(x_rows, index=idx, columns=X.columns)
    return sv_mat, x_mat


def shap_outputs(sv_mat, x_mat, tag):
    """Ranking top-15 por média |SHAP| (CSV) + summary plot (PNG)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import shap
    fig_dir = cfg.REPORTS_DIR / "figuras_modelos"
    fig_dir.mkdir(parents=True, exist_ok=True)

    rank = (sv_mat.abs().mean().sort_values(ascending=False)
            .rename("mean_abs_shap").reset_index()
            .rename(columns={"index": "feature"}))
    rank.to_csv(cfg.PROCESSED_DIR / f"shap_ranking_{tag}.csv", index=False)

    shap.summary_plot(sv_mat.values, x_mat, max_display=15, show=False)
    plt.gcf().set_size_inches(9, 6)
    plt.tight_layout()
    plt.savefig(fig_dir / f"shap_summary_{tag}.png", dpi=150)
    plt.close("all")
    return rank


# ---------------------------------------------------------------------------
# Orquestração
# ---------------------------------------------------------------------------
def run_all():
    feats, targets = load_data()
    all_metrics, all_preds = [], []

    def _record(preds, **meta):
        met = aggregate_metrics(preds)
        for k, v in meta.items():
            met[k] = v
        all_metrics.append(met)
        p = preds.copy()
        for k, v in meta.items():
            p[k] = v
        all_preds.append(p)

    for cell, fam, N, ms in CELLS:
        for fs in ("completo", "sem_dipl"):
            X, y, y_full, _ = cell_frame(feats, targets, fam, N, ms, fs)
            print(f"[{cell} | {fs}] anos utilizáveis: {X.index.min()}–"
                  f"{X.index.max()} ({len(X)}), prev={y.mean():.3f}")
            wf = walk_forward(X, y, N)
            nv = walk_forward_naive(X, y, y_full, N)
            _record(pd.concat([wf, nv]), celula=cell, familia=fam, N=N,
                    min_sys=ms, feature_set=fs, protocolo="walk_forward")
            es = era_split(X, y, N)
            _record(es, celula=cell, familia=fam, N=N, min_sys=ms,
                    feature_set=fs, protocolo="era_split")

    metrics = pd.concat(all_metrics, ignore_index=True)

    # --- Ablação H1 na célula primária (melhor modelo por PR-AUC no WF completo)
    prim = metrics[(metrics.celula == PRIMARY_CELL)
                   & (metrics.feature_set == "completo")
                   & (metrics.protocolo == "walk_forward")
                   & (metrics.modelo.isin(["logreg", "rf", "xgb"]))]
    best_model = prim.sort_values("pr_auc", ascending=False)["modelo"].iloc[0]
    print(f"\nMelhor modelo da célula primária (PR-AUC, WF completo): {best_model}")

    cell, fam, N, ms = next(c for c in CELLS if c[0] == PRIMARY_CELL)
    Xa, ya, ya_full, _ = cell_frame(feats, targets, fam, N, ms, "nivel_delta")
    wf_abl = walk_forward(Xa, ya, N, model_names=(best_model,))
    wf_abl["modelo"] = best_model
    _record(wf_abl, celula=cell, familia=fam, N=N, min_sys=ms,
            feature_set="nivel_delta", protocolo="walk_forward")
    metrics = pd.concat(all_metrics, ignore_index=True)

    # --- SHAP no melhor modelo de ÁRVORE da célula primária (WF completo)
    tree = prim[prim.modelo.isin(["rf", "xgb"])].sort_values(
        "pr_auc", ascending=False)["modelo"].iloc[0]
    print(f"Melhor modelo de árvore (SHAP): {tree}")
    Xs, ys, _, _ = cell_frame(feats, targets, fam, N, ms, "completo")
    sv_mat, x_mat = shap_walk_forward(Xs, ys, N, tree)
    rank = shap_outputs(sv_mat, x_mat, tag=f"{PRIMARY_CELL}_{tree}")
    print("\nTop-15 SHAP (média |valor|):")
    print(rank.head(15).to_string(index=False))

    # --- Persistir
    meta_cols = ["celula", "familia", "N", "min_sys", "feature_set", "protocolo"]
    metrics = metrics[meta_cols + [c for c in metrics.columns if c not in meta_cols]]
    metrics.to_csv(cfg.PROCESSED_DIR / "resultados_baseline.csv", index=False)
    preds = pd.concat(all_preds, ignore_index=True)
    preds.to_csv(cfg.INTERIM_DIR / "predicoes_oos_baseline.csv", index=False)
    print(f"\n{len(metrics)} linhas de métricas -> resultados_baseline.csv; "
          f"{len(preds)} previsões OOS -> predicoes_oos_baseline.csv")
    return metrics, rank
