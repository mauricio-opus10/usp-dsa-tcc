"""
Itens do feedback do orientador. Mesmo protocolo do baseline e dos testes de robustez:
walk-forward purgado (s + N < t, treino mín. 40 anos), censura respeitada,
seed=42, SEM tuning. Células: proxy_N1_ge2 e war_N1_ge2; modelos: RF e o
melhor modelo por célula (proxy → o melhor É o RF, coincide; war → logreg em
nivel_delta, escolhido por PR-AUC no walk-forward do baseline).

A – Estratégias de desbalanceamento, comparadas NA MESMA validação:
  balanced      – class_weight='balanced' (protocolo atual);
  sem_peso      – sem ponderação alguma;
  undersampling – RandomUnderSampler (imblearn, seed=42) da classe majoritária
                  no treino de CADA fold; modelo sem class_weight; threshold
                  F1 escolhido nas previsões do treino ORIGINAL (não
                  reamostrado), como no protocolo do baseline;
  thr_calibrado – mesmo modelo de `balanced` (probabilidades idênticas), mas o
                  threshold vem de VALIDAÇÃO INTERNA temporal no treino do
                  fold: ajusta nos primeiros 80% dos anos de treino, escolhe o
                  threshold F1-máximo nos 20% finais (nunca toca o teste) e
                  reajusta no treino completo p/ prever t.
  Nota honesta (registrada antes de rodar): AUC e PR-AUC dependem só do
  ranking das probabilidades – são IDÊNTICOS entre `balanced` e
  `thr_calibrado` por construção, e só mudam entre estratégias que alteram o
  modelo (peso/undersampling). O que o threshold muda é precisão/revocação/F1.

B – Curvas Precision-Recall agregadas das previsões out-of-sample
  (reutilizadas de data/interim/predicoes_oos_baseline.csv, idênticas às do
  etapa de robustez por reprodutibilidade): proxy_N1 RF (completo e sem_dipl) e war_N1
  logreg (nivel_delta), com a prevalência como referência. 300 DPI, PT-BR.

C – Sensibilidade ao corte CINC: matriz de features E alvos reconstruídos sob
  CINC >= 0,5% e >= 1,5% (pipeline idêntico; o corte entra em
  annual.states_in_year, aplicado por monkeypatch em runtime tanto em
  build_features quanto em target – target.py importa a função por nome).
  RF em proxy_N1_ge2 (completo e sem_dipl), walk-forward. Matrizes salvas em
  data/interim/features_cinc_*.csv (traceabilidade). O corte 1% NÃO é
  re-rodado: reutiliza resultados da etapa de robustez (replicados do baseline).

Uso:  ~/.virtualenvs/datascience/bin/python scripts/rodar_feedback.py
Saídas: data/processed/resultados_desbalanceamento.csv,
        data/processed/resultados_cinc_sensibilidade.csv,
        data/interim/features_cinc_0p5.csv, features_cinc_1p5.csv,
        reports/figuras_modelos/pr_curve_proxy_N1_rf.png,
        reports/figuras_modelos/pr_curve_war_N1_logreg.png.
"""

import numpy as np
import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .. import config as cfg
from ..features import build_features as bf
from ..features import target as tgt
from ..network import annual
from . import baseline as bl

SEED = cfg.RANDOM_SEED
RODADA = "feedback_orientador"
INNER_VAL_FRAC = 0.2   # validação interna temporal p/ thr_calibrado

# (célula, família, N, min_sys, modelo, feature_set, papel)
A_CONFIGS = [
    ("proxy_N1_ge2", "proxy", 1, 2, "rf", "completo", "RF (= melhor da célula)"),
    ("proxy_N1_ge2", "proxy", 1, 2, "rf", "sem_dipl", "RF (= melhor da célula)"),
    ("war_N1_ge2", "war", 1, 2, "rf", "completo", "RF"),
    ("war_N1_ge2", "war", 1, 2, "logreg", "nivel_delta", "melhor da célula (P4, PR-AUC)"),
]
STRATEGIES = ("balanced", "sem_peso", "undersampling", "thr_calibrado")

CINC_GRID = (0.005, 0.015)   # além do 1% base (reutilizado da etapa de robustez)


# ---------------------------------------------------------------------------
# A – desbalanceamento
# ---------------------------------------------------------------------------
def _make_model(name, class_weight):
    """Mesmos hiperparâmetros do baseline, variando só a ponderação."""
    if name == "rf":
        return RandomForestClassifier(n_estimators=500, max_depth=4,
                                      min_samples_leaf=5,
                                      class_weight=class_weight,
                                      random_state=SEED, n_jobs=-1)
    if name == "logreg":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(penalty="l2", C=1.0,
                                       class_weight=class_weight,
                                       max_iter=5000, random_state=SEED)),
        ])
    raise ValueError(name)


def _inner_threshold(model_name, Xtr, ytr):
    """Threshold F1-máximo por validação interna TEMPORAL no treino do fold.

    Ajusta nos primeiros (1-INNER_VAL_FRAC) anos, escolhe o threshold nos
    INNER_VAL_FRAC finais. Fallback (contado): classe única em algum lado ->
    threshold das previsões do treino completo (protocolo do baseline).
    """
    n = len(Xtr)
    n_val = max(int(round(n * INNER_VAL_FRAC)), 5)
    Xin, yin = Xtr.iloc[:-n_val], ytr.iloc[:-n_val]
    Xval, yval = Xtr.iloc[-n_val:], ytr.iloc[-n_val:]
    if yin.nunique() < 2 or yval.nunique() < 2:
        return None
    m = _make_model(model_name, "balanced")
    m.fit(Xin, yin)
    return bl.best_f1_threshold(yval, m.predict_proba(Xval)[:, 1])


def walk_forward_strategies(X, y, N, model_name):
    """Walk-forward purgado com as 4 estratégias por fold. Retorna
    (previsões OOS longas com coluna `estrategia`, nº de fallbacks do
    thr_calibrado)."""
    years = list(X.index)
    rows, n_fallback = [], 0
    rus = RandomUnderSampler(random_state=SEED)
    for t in years:
        train_years = [s for s in years if s + N < t]
        if len(train_years) < bl.MIN_TRAIN:
            continue
        Xtr, ytr = X.loc[train_years], y.loc[train_years]
        if ytr.nunique() < 2:
            continue
        Xte = X.loc[[t]]

        fits = {}
        m_bal = _make_model(model_name, "balanced")
        m_bal.fit(Xtr, ytr)
        fits["balanced"] = (m_bal, bl.best_f1_threshold(
            ytr, m_bal.predict_proba(Xtr)[:, 1]))

        m_np = _make_model(model_name, None)
        m_np.fit(Xtr, ytr)
        fits["sem_peso"] = (m_np, bl.best_f1_threshold(
            ytr, m_np.predict_proba(Xtr)[:, 1]))

        Xus, yus = rus.fit_resample(Xtr, ytr)
        m_us = _make_model(model_name, None)
        m_us.fit(Xus, yus)
        fits["undersampling"] = (m_us, bl.best_f1_threshold(
            ytr, m_us.predict_proba(Xtr)[:, 1]))   # thr no treino ORIGINAL

        thr_cal = _inner_threshold(model_name, Xtr, ytr)
        if thr_cal is None:
            thr_cal = fits["balanced"][1]
            n_fallback += 1
        fits["thr_calibrado"] = (m_bal, thr_cal)   # mesmas probabilidades

        for strat, (m, thr) in fits.items():
            p = float(m.predict_proba(Xte)[:, 1][0])
            rows.append({"ano": t, "modelo": model_name, "estrategia": strat,
                         "y": int(y.loc[t]), "prob": p,
                         "pred05": int(p >= 0.5),
                         "predthr": int(p >= thr), "thr": thr})
    return pd.DataFrame(rows), n_fallback


def run_item_a(feats, targets):
    all_metrics, all_preds = [], []
    for cell, fam, N, ms, model_name, fs, papel in A_CONFIGS:
        X, y, _, _ = bl.cell_frame(feats, targets, fam, N, ms, fs)
        print(f"[A | {cell} | {model_name} | {fs}] anos {X.index.min()}–"
              f"{X.index.max()} ({len(X)}), prev={y.mean():.3f}")
        preds, n_fb = walk_forward_strategies(X, y, N, model_name)
        if n_fb:
            print(f"    thr_calibrado: {n_fb} folds com fallback p/ thr de treino")
        for strat, g in preds.groupby("estrategia"):
            met = bl.aggregate_metrics(g)
            met["estrategia"] = strat
            met["rodada"] = RODADA
            met["celula"], met["feature_set"] = cell, fs
            met["papel_modelo"] = papel
            met["thr_calibrado_fallbacks"] = n_fb if strat == "thr_calibrado" else 0
            all_metrics.append(met)
        preds["celula"], preds["feature_set"] = cell, fs
        all_preds.append(preds)
    met = pd.concat(all_metrics, ignore_index=True)
    cols = ["rodada", "celula", "feature_set", "modelo", "papel_modelo",
            "estrategia"]
    met = met[cols + [c for c in met.columns if c not in cols]]
    met.to_csv(cfg.PROCESSED_DIR / "resultados_desbalanceamento.csv", index=False)
    pd.concat(all_preds, ignore_index=True).to_csv(
        cfg.INTERIM_DIR / "predicoes_oos_desbalanceamento.csv", index=False)
    return met


# ---------------------------------------------------------------------------
# B – curvas Precision-Recall (previsões OOS do baseline, 300 DPI)
# ---------------------------------------------------------------------------
AZUL, VERMELHO, CINZA = "#0072B2", "#D55E00", "#595959"   # CVD-safe (validada)


def _pr_axes(ax):
    ax.set_xlabel("Revocação (recall)")
    ax.set_ylabel("Precisão")
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.05)
    ax.grid(True, linewidth=0.4, alpha=0.35)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def run_item_b():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    base = pd.read_csv(cfg.INTERIM_DIR / "predicoes_oos_baseline.csv")
    wf = base[base.protocolo == "walk_forward"]
    fig_dir = cfg.REPORTS_DIR / "figuras_modelos"
    fig_dir.mkdir(parents=True, exist_ok=True)
    out = []

    def _fmt(x):
        return f"{x:.3f}".replace(".", ",")   # vírgula decimal (PT-BR)

    def _curve(ax, g, cor, rotulo, ann_offset=5, ann_x=0.02):
        prec, rec, _ = precision_recall_curve(g["y"], g["prob"])
        from sklearn.metrics import average_precision_score
        ap = average_precision_score(g["y"], g["prob"])
        prev = g["y"].mean()
        ax.plot(rec, prec, color=cor, linewidth=2,
                label=f"{rotulo} (PR-AUC {_fmt(ap)})")
        ax.axhline(prev, color=cor, linewidth=1, linestyle="--", alpha=0.6)
        ax.annotate(f"prevalência = {_fmt(prev)}", xy=(ann_x, prev),
                    xytext=(0, ann_offset), textcoords="offset points",
                    ha="left", fontsize=8, color=cor, alpha=0.9)
        out.append({"figura": ax.get_title() or "pr", "serie": rotulo,
                    "pr_auc": round(float(ap), 3),
                    "prevalencia": round(float(prev), 3),
                    "n_anos": len(g), "positivos": int(g["y"].sum())})

    # Fig 1 – proxy_N1 RF, completo e sem_dipl
    fig, ax = plt.subplots(figsize=(7, 5))
    g1 = wf[(wf.celula == "proxy_N1_ge2") & (wf.modelo == "rf")
            & (wf.feature_set == "completo")]
    g2 = wf[(wf.celula == "proxy_N1_ge2") & (wf.modelo == "rf")
            & (wf.feature_set == "sem_dipl")]
    _curve(ax, g1, AZUL, "completo (63 features)", ann_offset=5)
    _curve(ax, g2, VERMELHO, "sem_dipl (46 features)", ann_offset=-12)
    _pr_axes(ax)
    ax.set_title("Curva Precision-Recall – proxy_N1 (RF)\n"
                 "previsões out-of-sample agregadas, walk-forward purgado; "
                 "tracejado = prevalência", fontsize=10)
    ax.legend(loc="upper right", frameon=False)
    p1 = fig_dir / "pr_curve_proxy_N1_rf.png"
    fig.tight_layout()
    fig.savefig(p1, dpi=300)
    plt.close(fig)

    # Fig 2 – war_N1, melhor modelo (logreg / nivel_delta, P4 por PR-AUC)
    fig, ax = plt.subplots(figsize=(7, 5))
    g3 = wf[(wf.celula == "war_N1_ge2") & (wf.modelo == "logreg")
            & (wf.feature_set == "nivel_delta")]
    _curve(ax, g3, AZUL, "logreg, nivel_delta (48 features)",
           ann_offset=-13, ann_x=0.30)
    _pr_axes(ax)
    ax.set_title("Curva Precision-Recall – war_N1 (melhor modelo do baseline)\n"
                 "previsões out-of-sample agregadas, walk-forward purgado; "
                 "tracejado = prevalência", fontsize=10)
    ax.legend(loc="upper right", frameon=False)
    p2 = fig_dir / "pr_curve_war_N1_logreg.png"
    fig.tight_layout()
    fig.savefig(p2, dpi=300)
    plt.close(fig)

    print(f"[B] figuras: {p1.name}, {p2.name} (300 DPI)")
    return pd.DataFrame(out), (p1, p2)


# ---------------------------------------------------------------------------
# C – sensibilidade ao corte CINC
# ---------------------------------------------------------------------------
_ORIG_STATES_IN_YEAR = annual.states_in_year


def _set_cinc(thr):
    """Aplica o corte em TODOS os pontos de entrada (annual + target, que
    importa a função por nome). None restaura o original (1%)."""
    fn = _ORIG_STATES_IN_YEAR if thr is None else (
        lambda year, threshold=thr: _ORIG_STATES_IN_YEAR(year, threshold))
    annual.states_in_year = fn
    tgt.states_in_year = fn


def rebuild_for_threshold(thr):
    """Reconstrói features E alvos sob CINC >= thr (pipeline idêntico)."""
    tag = f"{thr * 100:.1f}".replace(".", "p")
    _set_cinc(thr)
    try:
        feats = bf.build_feature_matrix()
        onsets = {"proxy": tgt.proxy_onsets(min_system_states=2)}
        targets = {2: tgt.build_targets(onsets)}
    finally:
        _set_cinc(None)
    feats.to_csv(cfg.INTERIM_DIR / f"features_cinc_{tag}.csv", index=False)
    return feats, targets


def run_item_c():
    rows = []

    def _run(thr, feats, targets, origem):
        n_est = float(feats["n_estados"].mean())
        for fs in ("completo", "sem_dipl"):
            X, y, _, _ = bl.cell_frame(feats, targets, "proxy", 1, 2, fs)
            wf = bl.walk_forward(X, y, 1, model_names=("rf",))
            met = bl.aggregate_metrics(wf).iloc[0]
            rows.append({
                "rodada": RODADA, "cinc_threshold": thr,
                "n_estados_medio_ano": round(n_est, 1),
                "feature_set": fs, "modelo": "rf",
                "anos_utilizaveis": len(X),
                "prevalencia_janela": round(float(y.mean()), 3),
                "n_anos_teste": int(met["n_anos_teste"]),
                "positivos_teste": int(met["positivos_teste"]),
                "prevalencia_teste": met["prevalencia_teste"],
                "auc_roc": met["auc_roc"], "pr_auc": met["pr_auc"],
                "origem": origem,
            })
            print(f"[C | CINC>={thr:.1%} | {fs}] n_est={n_est:.1f} "
                  f"prev={y.mean():.3f} AUC={met['auc_roc']} PR={met['pr_auc']}")

    # 1% base: reutiliza a matriz e os alvos existentes (mesmo pipeline)
    feats_base, targets_base = bl.load_data()
    _run(0.01, feats_base, {2: targets_base[2]}, "base (reutilizado)")

    for thr in CINC_GRID:
        print(f"\n[C] reconstruindo features+alvos sob CINC >= {thr:.1%} ...")
        feats, targets = rebuild_for_threshold(thr)
        _run(thr, feats, targets, "reconstruído (esta rodada)")

    df = pd.DataFrame(rows).sort_values(["cinc_threshold", "feature_set"])
    df.to_csv(cfg.PROCESSED_DIR / "resultados_cinc_sensibilidade.csv", index=False)
    return df


# ---------------------------------------------------------------------------
def run_all():
    feats, targets = bl.load_data()

    print("=== Item A – estratégias de desbalanceamento ===")
    met_a = run_item_a(feats, targets)
    show = met_a[met_a.modelo.isin(["rf", "logreg"])][
        ["celula", "feature_set", "modelo", "estrategia", "auc_roc", "pr_auc",
         "prec_thr", "rec_thr", "f1_thr", "prec_05", "rec_05", "f1_05"]]
    print(show.to_string(index=False))

    print("\n=== Item B – curvas PR (OOS agregado) ===")
    pr_info, _ = run_item_b()
    print(pr_info.to_string(index=False))

    print("\n=== Item C – sensibilidade ao corte CINC ===")
    df_c = run_item_c()
    print("\n" + df_c[["cinc_threshold", "feature_set", "n_estados_medio_ano",
                       "prevalencia_janela", "auc_roc", "pr_auc"]]
          .to_string(index=False))
    return met_a, pr_info, df_c


if __name__ == "__main__":
    run_all()
