"""
Figuras do TCC final – renormatização às normas ESALQ + figura F1
(delta pareado GNN−RF) + SHAP da célula com sinal (proxy_N1_ge2, RF).

Normas aplicadas (Manual de TCC 251/252, Tabelas 7–8):
  - sem linhas de grade, sem borda, sem título embutido no gráfico
    (o título vai na legenda "Figura N." do documento);
  - eixos principais em linha sólida preta 1,5 pt;
  - fonte Arial tamanho <= 11, cor preta; rótulos em PT-BR;
  - vírgula como separador decimal;
  - figuras multipainel identificadas por letras maiúsculas no canto
    superior esquerdo, sem pontuação.
Exceção "software específico" (Manual §15.1): gráficos gerados em
matplotlib/shap por não serem reproduzíveis em MS Excel.

Fontes de dados (Regra de Ouro – nenhum número digitado à mão):
  - Curvas PR:      data/interim/predicoes_oos_baseline.csv
  - Curvas GNN:     data/interim/gnn_curvas_treino.csv
  - F1 (delta IC):  data/processed/gnn_delta_ic.csv
  - SHAP:           recomputado fold a fold pelo MESMO protocolo do baseline
                    (src.models.baseline.shap_walk_forward, seed=42,
                    determinístico) – proxy_N1_ge2/RF/completo (novo) e
                    war_N1_ge2/XGB/completo (re-render PT-BR, suplementar).

Paleta CVD-safe do projeto (validada): #0072B2 (azul) / #D55E00 (vermelho) /
#595959 (cinza, reservado a linhas de referência – não é cor de série).

Uso: ~/.virtualenvs/datascience/bin/python scripts/gerar_figuras_tcc.py
+ figuras_redes() (re-render das figs. 1–2 do preliminar).
+ figura_grafos() (subcomando `grafos`): grafos de alianças e disputas em
  anos selecionados, lidos dos brutos COW pelos construtores do projeto.
Saídas: reports/figuras_modelos/*.png (300 DPI) +
        data/processed/shap_ranking_proxy_N1_ge2_rf.csv
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter
from sklearn.metrics import average_precision_score, precision_recall_curve

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config as cfg  # noqa: E402
from src.models.baseline import cell_frame, load_data, shap_walk_forward  # noqa: E402

AZUL, VERMELHO, CINZA = "#0072B2", "#D55E00", "#595959"
FIG_DIR = cfg.REPORTS_DIR / "figuras_modelos"

plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 11,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "axes.unicode_minus": False,
})


def fmt_pt(x, dec=1):
    """Número com vírgula decimal (PT-BR)."""
    return f"{x:.{dec}f}".replace(".", ",")


def tick_virgula(dec=1):
    return FuncFormatter(lambda v, _: fmt_pt(v, dec))


def tick_virgula_auto():
    """Formatador adaptativo (%g) com vírgula decimal – para faixas pequenas."""
    return FuncFormatter(lambda v, _: f"{v:g}".replace(".", ","))


def eixos_norma(ax):
    """Norma ESALQ: sem grade, sem moldura; eixos pretos 1,5 pt."""
    ax.grid(False)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("black")
        ax.spines[s].set_linewidth(1.5)


def letra_painel(ax, letra):
    """Identificação de painel (Manual §15.1): maiúscula, canto sup. esquerdo,
    sem pontuação."""
    ax.text(-0.10, 1.06, letra, transform=ax.transAxes, fontsize=11,
            fontweight="bold", va="top", ha="left", color="black")


# ---------------------------------------------------------------------------
# 1–2. Curvas Precision-Recall (sem título embutido)
# ---------------------------------------------------------------------------
def _curva_pr(ax, g, cor, rotulo, ann_offset=5, ann_x=0.02):
    prec, rec, _ = precision_recall_curve(g["y"], g["prob"])
    ap = average_precision_score(g["y"], g["prob"])
    prev = g["y"].mean()
    ax.plot(rec, prec, color=cor, linewidth=2,
            label=f"{rotulo} (PR-AUC {fmt_pt(ap, 3)})")
    ax.axhline(prev, color=cor, linewidth=1, linestyle="--", alpha=0.6)
    ax.annotate(f"prevalência = {fmt_pt(prev, 3)}", xy=(ann_x, prev),
                xytext=(0, ann_offset), textcoords="offset points",
                ha="left", fontsize=9, color=cor)


def _eixos_pr(ax):
    ax.set_xlabel("Revocação (recall)")
    ax.set_ylabel("Precisão")
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.05)
    ax.xaxis.set_major_formatter(tick_virgula(1))
    ax.yaxis.set_major_formatter(tick_virgula(1))
    eixos_norma(ax)


def curvas_pr():
    base = pd.read_csv(cfg.INTERIM_DIR / "predicoes_oos_baseline.csv")
    wf = base[base.protocolo == "walk_forward"]

    # Figura: proxy_N1 RF (completo × sem_dipl)
    fig, ax = plt.subplots(figsize=(7, 5))
    g1 = wf[(wf.celula == "proxy_N1_ge2") & (wf.modelo == "rf")
            & (wf.feature_set == "completo")]
    g2 = wf[(wf.celula == "proxy_N1_ge2") & (wf.modelo == "rf")
            & (wf.feature_set == "sem_dipl")]
    _curva_pr(ax, g1, AZUL, "completo (63 atributos)", ann_offset=5)
    _curva_pr(ax, g2, VERMELHO, "sem diplomacia (46 atributos)", ann_offset=-13)
    _eixos_pr(ax)
    ax.legend(loc="upper right", frameon=False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "pr_curve_proxy_N1_rf.png", dpi=300)
    plt.close(fig)

    # Figura: war_N1, melhor modelo do baseline (logreg / nivel_delta)
    fig, ax = plt.subplots(figsize=(7, 5))
    g3 = wf[(wf.celula == "war_N1_ge2") & (wf.modelo == "logreg")
            & (wf.feature_set == "nivel_delta")]
    _curva_pr(ax, g3, AZUL, "regressão logística, nível+delta (48 atributos)",
              ann_offset=-13, ann_x=0.30)
    _eixos_pr(ax)
    ax.legend(loc="upper right", frameon=False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "pr_curve_war_N1_logreg.png", dpi=300)
    plt.close(fig)
    print("[PR] pr_curve_proxy_N1_rf.png, pr_curve_war_N1_logreg.png")


# ---------------------------------------------------------------------------
# 3. Curvas de treino GNN – multipainel A–D
# ---------------------------------------------------------------------------
def curvas_gnn():
    cur = pd.read_csv(cfg.INTERIM_DIR / "gnn_curvas_treino.csv")
    cur = cur[(cur.celula == "proxy_N1_ge2") & (cur.variante == "sem_dipl")]
    assert not cur.empty, "curvas de proxy_N1/sem_dipl ausentes do CSV"
    folds = sorted(cur.fold.unique())
    paineis = [("gcn", folds[0], "A"), ("gcn", folds[1], "B"),
               ("gat", folds[0], "C"), ("gat", folds[1], "D")]

    fig, axes = plt.subplots(2, 2, figsize=(10, 6.5))
    for ax, (arch, fold, letra) in zip(axes.ravel(), paineis):
        g = cur[(cur.arch == arch) & (cur.fold == fold)].sort_values("epoca")
        ax.plot(g.epoca, g.loss_treino, color=AZUL, linewidth=2,
                label="treino (80%)")
        ax.plot(g.epoca, g.loss_val, color=VERMELHO, linewidth=2,
                label="validação interna (20%)")
        ax.set_xlabel("Época")
        ax.set_ylabel("Perda (entropia cruzada ponderada)")
        ax.yaxis.set_major_formatter(tick_virgula(2))
        eixos_norma(ax)
        letra_painel(ax, letra)
    axes[0, 0].legend(loc="center right", frameon=False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "gnn_curvas_treino.png", dpi=300)
    plt.close(fig)
    print("[GNN] gnn_curvas_treino.png (painéis A–D: GCN/GAT × folds "
          f"{folds[0]}/{folds[1]}, seed 42, proxy_N1 sem diplomacia)")


# ---------------------------------------------------------------------------
# 4. F1 – delta pareado (GNN − RF) com IC 95% (figura-chave da H2)
# ---------------------------------------------------------------------------
def delta_pareado():
    d = pd.read_csv(cfg.PROCESSED_DIR / "gnn_delta_ic.csv")
    ordem = [("proxy_N1_ge2", "com_dipl", "gcn"), ("proxy_N1_ge2", "com_dipl", "gat"),
             ("proxy_N1_ge2", "sem_dipl", "gcn"), ("proxy_N1_ge2", "sem_dipl", "gat"),
             ("war_N1_ge2", "com_dipl", "gcn"), ("war_N1_ge2", "com_dipl", "gat"),
             ("war_N1_ge2", "sem_dipl", "gcn"), ("war_N1_ge2", "sem_dipl", "gat")]
    rot_cel = {"proxy_N1_ge2": "Proxy MID", "war_N1_ge2": "Guerra oficial"}
    rot_var = {"com_dipl": "4 camadas", "sem_dipl": "3 camadas"}

    # posições com respiro extra entre células (grupo proxy × grupo war)
    ys, rotulos, linhas = [], [], []
    y = 0.0
    for i, (cel, var, arch) in enumerate(ordem):
        if i == 4:
            y -= 0.9
        ys.append(y)
        rotulos.append(f"{rot_cel[cel]} · {rot_var[var]} · {arch.upper()}")
        linhas.append(d[(d.celula == cel) & (d.variante == var)
                        & (d.arch == arch)].iloc[0])
        y -= 1.0

    cores = {"gcn": AZUL, "gat": VERMELHO}
    marcas = {"gcn": "o", "gat": "s"}   # forma = codificação secundária (CVD)
    paineis = [("A", "delta_auc_pontual", "delta_auc_ic95_lo",
                "delta_auc_ic95_hi", "ΔAUC-ROC (GNN − random forest)"),
               ("B", "delta_pr_auc_pontual", "delta_pr_auc_ic95_lo",
                "delta_pr_auc_ic95_hi", "ΔPR-AUC (GNN − random forest)")]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), sharey=True)
    for ax, (letra, c_pt, c_lo, c_hi, xlabel) in zip(axes, paineis):
        ax.axvline(0, color=CINZA, linewidth=1, linestyle="--")
        for yy, row in zip(ys, linhas):
            cor, marca = cores[row.arch], marcas[row.arch]
            ax.plot([row[c_lo], row[c_hi]], [yy, yy], color=cor, linewidth=2,
                    solid_capstyle="butt")
            for x_extremo in (row[c_lo], row[c_hi]):
                ax.plot([x_extremo, x_extremo], [yy - 0.16, yy + 0.16],
                        color=cor, linewidth=2)
            ax.plot(row[c_pt], yy, marca, color=cor, markersize=7,
                    markeredgecolor="white", markeredgewidth=0.8, zorder=3)
        ax.set_xlabel(xlabel)
        ax.xaxis.set_major_formatter(tick_virgula(1))
        eixos_norma(ax)
        letra_painel(ax, letra)
    axes[0].set_yticks(ys)
    axes[0].set_yticklabels(rotulos)
    axes[0].tick_params(axis="y", length=0)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "gnn_delta_pareado_ic.png", dpi=300)
    plt.close(fig)
    print("[F1] gnn_delta_pareado_ic.png (bootstrap pareado, blocos de 10 "
          "anos, 2.000 reamostras – gnn_delta_ic.csv)")


# ---------------------------------------------------------------------------
# 5. SHAP – summary plots normatizados (PT-BR) + ranking CSV
# ---------------------------------------------------------------------------
def _shap_summary_norma(sv_mat, x_mat, png_path):
    import shap

    # PT-BR na fonte: o shap fixa Low/High/labels via dicionário interno –
    # substituir texto depois do draw não persiste (FixedFormatter).
    try:
        from shap.plots import _labels
        _labels.labels["FEATURE_VALUE_LOW"] = "baixo"
        _labels.labels["FEATURE_VALUE_HIGH"] = "alto"
        _labels.labels["FEATURE_VALUE"] = "valor do atributo"
        _labels.labels["VALUE"] = "valor SHAP (impacto na saída do modelo)"
    except ImportError:
        pass

    shap.summary_plot(sv_mat.values, x_mat, max_display=15, show=False,
                      color_bar_label="valor do atributo")
    fig = plt.gcf()
    ax = fig.axes[0]
    ax.set_xlabel("valor SHAP (impacto na saída do modelo)", fontsize=11)
    ax.tick_params(labelsize=10)
    ax.xaxis.set_major_formatter(tick_virgula_auto())
    # fallback: barra de cor é o último eixo; rótulos são ticklabels fixos
    cbax = fig.axes[-1]
    if [t.get_text() for t in cbax.get_yticklabels()] == ["Low", "High"]:
        cbax.set_yticklabels(["baixo", "alto"])
    cbax.tick_params(labelsize=10)
    fig.set_size_inches(9, 6)
    fig.tight_layout()
    fig.savefig(png_path, dpi=300)
    plt.close("all")


def shap_figuras():
    feats, targets = load_data()

    # (a) NOVO – célula com sinal: proxy_N1_ge2, RF, completo (figura principal)
    X, y, _, _ = cell_frame(feats, targets, "proxy", 1, 2, "completo")
    sv, xm = shap_walk_forward(X, y, N=1, model_name="rf")
    rank = (sv.abs().mean().sort_values(ascending=False)
            .rename("mean_abs_shap").reset_index()
            .rename(columns={"index": "feature"}))
    rank.to_csv(cfg.PROCESSED_DIR / "shap_ranking_proxy_N1_ge2_rf.csv",
                index=False)
    _shap_summary_norma(sv, xm, FIG_DIR / "shap_summary_proxy_N1_ge2_rf.png")
    print("[SHAP] shap_summary_proxy_N1_ge2_rf.png + "
          "shap_ranking_proxy_N1_ge2_rf.csv "
          f"({len(sv)} anos OOS explicados)")
    print(rank.head(10).to_string(index=False))

    # (b) re-render PT-BR do existente (war_N1_ge2, XGB) – suplementar.
    #     Mesmo protocolo/seed do baseline; ranking NÃO é regravado (já existe
    #     em data/processed/shap_ranking_war_N1_ge2_xgb.csv e não muda).
    Xw, yw, _, _ = cell_frame(feats, targets, "war", 1, 2, "completo")
    svw, xmw = shap_walk_forward(Xw, yw, N=1, model_name="xgb")
    rank_old = pd.read_csv(cfg.PROCESSED_DIR / "shap_ranking_war_N1_ge2_xgb.csv")
    rank_new = (svw.abs().mean().sort_values(ascending=False)
                .rename("mean_abs_shap").reset_index()
                .rename(columns={"index": "feature"}))
    top15_igual = list(rank_old.feature.head(15)) == list(rank_new.feature.head(15))
    print(f"[SHAP] war_N1 re-render: top-15 idêntico ao baseline? {top15_igual}")
    assert top15_igual, "SHAP war_N1 divergiu do baseline – investigar antes de sobrescrever"
    _shap_summary_norma(svw, xmw, FIG_DIR / "shap_summary_war_N1_ge2_xgb.png")
    print("[SHAP] shap_summary_war_N1_ge2_xgb.png (re-render PT-BR, suplementar)")


# ---------------------------------------------------------------------------
# 6. Figuras da caracterização das redes – re-render nas normas
#    das figuras 1–2 do preliminar. As originais em figuras_prelim/ NÃO são
#    alteradas (documento já avaliado). Dados: os MESMOS CSVs do preliminar.
# ---------------------------------------------------------------------------
VERDE, LARANJA = "#009E73", "#E69F00"   # complementos Okabe-Ito (CVD-safe)
CAMADAS = [("alliance", "Alianças", AZUL),
           ("trade", "Comércio", VERDE),
           ("disputes", "Disputas (janela 3 anos)", VERMELHO),
           ("diplomacy", "Diplomacia", LARANJA)]
MARCOS = {1914: "1ª Guerra", 1939: "2ª Guerra", 1947: "Guerra Fria",
          1991: "Fim da URSS"}


def _marcos(ax, y_frac=0.97):
    for yr, lbl in MARCOS.items():
        ax.axvline(yr, color=CINZA, ls="--", lw=0.8, alpha=0.7)
        ax.text(yr, ax.get_ylim()[1] * y_frac, lbl, rotation=90, va="top",
                ha="right", fontsize=8, color=CINZA)


def figuras_redes():
    serie = pd.read_csv(cfg.INTERIM_DIR
                        / "prelim_series_densidade_modularidade.csv")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.0, 6.8), sharex=True)
    for ax, col, letra, ylab in ((ax1, "density", "A", "Densidade"),
                                 (ax2, "modularity", "B",
                                  "Modularidade (Louvain)")):
        for key, rotulo, cor in CAMADAS:
            s = serie[serie.layer == key].dropna(subset=[col])
            if key == "diplomacy":      # medida só em anos espaçados
                ax.plot(s.year, s[col], "o", ms=3.5, color=cor, label=rotulo)
            else:
                ax.plot(s.year, s[col], lw=1.4, color=cor, label=rotulo)
        ax.set_ylabel(ylab)
        ax.yaxis.set_major_formatter(tick_virgula(1))
        eixos_norma(ax)
        letra_painel(ax, letra)
    ax1.set_ylim(0, 1.05)
    _marcos(ax1)
    _marcos(ax2)
    ax2.set_xlabel("Ano")
    h, l = ax1.get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, frameon=False, fontsize=9,
               bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(FIG_DIR / "redes_series_densidade_modularidade.png", dpi=300)
    plt.close(fig)

    comp = pd.read_csv(cfg.INTERIM_DIR / "B2_composicao_sistema_por_ano.csv")
    fig, ax = plt.subplots(figsize=(8.0, 3.8))
    ax.bar(comp.year, comp.churn, color="#BBBBBB",
           label="Churn (entradas + saídas)")
    ax.set_ylabel("Churn (nº de Estados)")
    ax.set_xlabel("Ano")
    eixos_norma(ax)
    axr = ax.twinx()
    axr.plot(comp.year, comp.n_states, color=AZUL, lw=1.8,
             label="Nº de Estados (CINC ≥ 1%)")
    axr.set_ylabel("Nº de Estados no sistema")
    axr.grid(False)
    axr.spines["top"].set_visible(False)
    axr.spines["left"].set_visible(False)
    axr.spines["right"].set_color("black")
    axr.spines["right"].set_linewidth(1.5)
    for yr in (1940, 1946, 1991):
        ax.axvline(yr, color=CINZA, ls="--", lw=0.8, alpha=0.7)
        ax.text(yr, ax.get_ylim()[1] * 0.95, str(yr), rotation=90, va="top",
                ha="right", fontsize=8, color=CINZA)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = axr.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper left", frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "redes_composicao_sistema.png", dpi=300)
    plt.close(fig)
    print("[redes] redes_series_densidade_modularidade.png + "
          "redes_composicao_sistema.png")


# ---------------------------------------------------------------------------
# 7. Grafos das camadas esparsas em anos selecionados. Construídos pelos
#    MESMOS construtores usados nos atributos (src/network/annual.py), a
#    partir dos brutos COW. Layout circular determinístico: os Estados são
#    ordenados pela comunidade de Louvain da camada de alianças (semente 42)
#    e ocupam a mesma posição nos dois painéis de cada ano.
# ---------------------------------------------------------------------------
ANOS_GRAFOS = (1913, 1938, 1975, 2005)
NOME_PT = {
    "USA": "EUA", "CAN": "Canadá", "MEX": "México", "BRA": "Brasil",
    "UKG": "Reino Unido", "BEL": "Bélgica", "FRN": "França", "SPN": "Espanha",
    "GMY": "Alemanha", "GFR": "Alemanha Oc.", "GDR": "Alemanha Or.",
    "POL": "Polônia", "AUH": "Áustria-Hungria", "CZE": "Tchecoslováquia",
    "ITA": "Itália", "BUL": "Bulgária", "ROM": "Romênia", "RUS": "Rússia",
    "UKR": "Ucrânia", "SWD": "Suécia", "TUR": "Turquia", "EGY": "Egito",
    "IRN": "Irã", "IRQ": "Iraque", "SAU": "Arábia Saudita", "ETH": "Etiópia",
    "CHN": "China", "TAW": "Taiwan", "PRK": "Coreia do Norte",
    "ROK": "Coreia do Sul", "JPN": "Japão", "IND": "Índia", "PAK": "Paquistão",
    "BNG": "Bangladesh", "THI": "Tailândia", "DRV": "Vietnã",
    "INS": "Indonésia", "AUL": "Austrália",
}


def figura_grafos(anos=ANOS_GRAFOS):
    import community as community_louvain
    import networkx as nx
    from src.network import annual

    nmc = pd.read_csv(cfg.cow_raw_dir() / cfg.COW_FILES["nmc"])
    fig, axes = plt.subplots(len(anos), 2, figsize=(7.6, 3.05 * len(anos)))
    letras = iter("ABCDEFGHIJKL")
    for i, ano in enumerate(anos):
        nos = annual.states_in_year(ano)
        sub = nmc[(nmc.year == ano) & (nmc.ccode.isin(nos))].set_index("ccode")
        g_ali = annual.alliance_graph(ano, nos)
        g_dis = annual.dispute_graph(ano, nos)       # janela retroativa de 3 anos
        part = community_louvain.best_partition(g_ali, weight="weight",
                                                random_state=cfg.RANDOM_SEED)
        # blocos maiores primeiro; dentro do bloco, maior capacidade primeiro
        tam = pd.Series(part).value_counts()
        ordem = sorted(nos, key=lambda n: (-tam[part[n]], part[n],
                                           -float(sub.loc[n, "cinc"])))
        ang = {n: np.pi / 2 - 2 * np.pi * k / len(ordem)
               for k, n in enumerate(ordem)}
        pos = {n: (np.cos(a), np.sin(a)) for n, a in ang.items()}
        tamanhos = [60 + 2600 * float(sub.loc[n, "cinc"]) for n in ordem]
        for j, (g, cor, rotulo) in enumerate(((g_ali, AZUL, "Alianças"),
                                               (g_dis, VERMELHO, "Disputas"))):
            ax = axes[i, j]
            pesos = np.array([d.get("weight", 1) for *_, d in g.edges(data=True)],
                             dtype=float)
            if len(pesos):
                larg = 0.6 + 1.6 * pesos / pesos.max()
                nx.draw_networkx_edges(g, pos, ax=ax, edge_color=cor,
                                       width=larg, alpha=0.55)
            nx.draw_networkx_nodes(g, pos, nodelist=ordem, ax=ax,
                                   node_size=tamanhos, node_color=cor,
                                   edgecolors="white", linewidths=0.8)
            for n, t in zip(ordem, tamanhos):    # rótulos por fora do círculo,
                x, y = pos[n]                    # afastados conforme o nó
                r = 1.13 + 0.0042 * np.sqrt(t)
                ax.text(r * x, r * y,
                        NOME_PT.get(sub.loc[n, "stateabb"],
                                    sub.loc[n, "stateabb"]),
                        fontsize=6.5, color="black",
                        va=("bottom" if y > 0.9 else
                            "top" if y < -0.9 else "center"),
                        ha=("left" if x > 0.15 else
                            "right" if x < -0.15 else "center"))
            dens = nx.density(g)
            ax.text(0.5, -0.04,
                    f"{rotulo}, {ano} (N = {len(ordem)}; densidade = "
                    f"{fmt_pt(dens, 2)})",
                    transform=ax.transAxes, ha="center", va="top",
                    fontsize=8, color="black")
            ax.text(-0.02, 1.02, next(letras), transform=ax.transAxes,
                    fontsize=11, fontweight="bold", va="top", ha="left")
            ax.set_xlim(-1.75, 1.75)
            ax.set_ylim(-1.32, 1.32)
            ax.set_aspect("equal")
            ax.axis("off")
    fig.subplots_adjust(left=0.02, right=0.98, top=0.99, bottom=0.03,
                        wspace=0.05, hspace=0.16)
    fig.savefig(FIG_DIR / "redes_grafos_anos_selecionados.png", dpi=300,
                facecolor="white")
    plt.close(fig)
    print("[grafos] redes_grafos_anos_selecionados.png")


# ---------------------------------------------------------------------------
# 8. Comunidades de alianças com as disputas sobrepostas. Áreas sombreadas:
#    comunidades de Louvain da camada de alianças (mesma função e semente
#    do atributo de modularidade). Arestas vermelhas: disputas da janela
#    retroativa de três anos. Ilustra dois atributos do modelo: a
#    modularidade das alianças e a sobreposição entre alianças e disputas
#    (lidos de data/processed/features_targets_1890_2012.csv).
# ---------------------------------------------------------------------------
ANOS_COMUNIDADES = (1938, 2005)
# sem laranja: reservado às arestas de disputa
CORES_COMUNIDADE = (AZUL, "#009E73", "#CC79A7", "#56B4E9", "#F0E442")


def _envoltoria(pontos, raio):
    """Envoltória convexa com folga: polígono em torno dos nós do bloco."""
    from scipy.spatial import ConvexHull
    t = np.linspace(0, 2 * np.pi, 24, endpoint=False)
    nuvem = np.vstack([np.c_[x + raio * np.cos(t), y + raio * np.sin(t)]
                       for x, y in pontos])
    return nuvem[ConvexHull(nuvem).vertices]


def figura_comunidades(anos=ANOS_COMUNIDADES):
    import community as community_louvain
    import matplotlib.patheffects as pe
    import networkx as nx
    from matplotlib.lines import Line2D
    from src.network import annual

    nmc = pd.read_csv(cfg.cow_raw_dir() / cfg.COW_FILES["nmc"])
    feats = pd.read_csv(cfg.PROCESSED_DIR / "features_targets_1890_2012.csv")
    fig, axes = plt.subplots(1, len(anos), figsize=(7.6, 4.5))
    resumo = []
    for ax, ano, letra in zip(axes, anos, "ABCD"):
        nos = annual.states_in_year(ano)
        sub = nmc[(nmc.year == ano) & (nmc.ccode.isin(nos))].set_index("ccode")
        g_ali = annual.alliance_graph(ano, nos)
        g_dis = annual.dispute_graph(ano, nos)
        part = community_louvain.best_partition(g_ali, weight="weight",
                                                random_state=cfg.RANDOM_SEED)
        mod = community_louvain.modularity(part, g_ali, weight="weight")
        linha = feats[feats["ano"] == ano].iloc[0]
        assert abs(mod - linha["ali_modularidade"]) < 1e-9, (ano, mod)
        grupos = {}
        for n, c in part.items():
            grupos.setdefault(c, []).append(n)
        blocos = sorted((v for v in grupos.values() if len(v) > 1),
                        key=lambda v: (-len(v), min(v)))
        soltos = sorted((v[0] for v in grupos.values() if len(v) == 1),
                        key=lambda n: -float(sub.loc[n, "cinc"]))
        # posição: cada bloco tem um centro num círculo; dentro dele, layout
        # de Kamada-Kawai do subgrafo. Estados sem aliança ficam numa linha abaixo.
        pos = {}
        for k, membros in enumerate(blocos):
            a = np.pi / 2 + 2 * np.pi * k / max(len(blocos), 1)
            centro = np.array([1.75 * np.cos(a), 1.45 * np.sin(a)])
            local = nx.kamada_kawai_layout(g_ali.subgraph(membros),
                                           weight=None)   # determinístico
            escala = 0.42 + 0.085 * len(membros)
            for n, xy in local.items():
                pos[n] = centro + escala * np.asarray(xy)
        for k, n in enumerate(soltos):
            pos[n] = np.array([-2.2 + 4.4 * (k + 0.5) / len(soltos), -3.0])
        cor_no = {}
        for k, membros in enumerate(blocos):
            cor = CORES_COMUNIDADE[k % len(CORES_COMUNIDADE)]
            ax.fill(*_envoltoria([pos[n] for n in membros], 0.26).T,
                    color=cor, alpha=0.13, lw=0)
            cor_no.update({n: cor for n in membros})
        cor_no.update({n: "#BBBBBB" for n in soltos})
        solto = set(soltos)
        d_mesma = sum(1 for u, v in g_dis.edges()
                      if part[u] == part[v])
        d_solto = sum(1 for u, v in g_dis.edges() if u in solto or v in solto)
        resumo.append({
            "ano": ano, "n_estados": len(nos), "n_comunidades": len(blocos),
            "n_sem_alianca": len(soltos), "modularidade": round(mod, 6),
            "n_aliancas": g_ali.number_of_edges(),
            "n_disputas": g_dis.number_of_edges(),
            "disputas_mesma_comunidade": d_mesma,
            "disputas_com_estado_sem_alianca": d_solto,
            "disputas_entre_comunidades":
                g_dis.number_of_edges() - d_mesma - d_solto})
        intra = [(u, v) for u, v in g_ali.edges() if part[u] == part[v]]
        entre = [(u, v) for u, v in g_ali.edges() if part[u] != part[v]]
        nx.draw_networkx_edges(g_ali, pos, edgelist=entre, ax=ax, width=0.7,
                               edge_color="#8A8A8A", style=(0, (3, 3)))
        nx.draw_networkx_edges(g_ali, pos, edgelist=intra, ax=ax, width=1.0,
                               edge_color="#8A8A8A")
        nx.draw_networkx_edges(g_dis, pos, ax=ax, width=1.5,
                               edge_color=VERMELHO, alpha=0.85)
        ordem = list(pos)
        nx.draw_networkx_nodes(
            g_ali, pos, nodelist=ordem, ax=ax,
            node_size=[70 + 2300 * float(sub.loc[n, "cinc"]) for n in ordem],
            node_color=[cor_no[n] for n in ordem], edgecolors="black",
            linewidths=0.6)
        for n in ordem:
            x, y = pos[n]
            dy = 0.11 + 0.0042 * np.sqrt(70 + 2300 * float(sub.loc[n, "cinc"]))
            ax.text(x, y + dy, NOME_PT.get(sub.loc[n, "stateabb"],
                                           sub.loc[n, "stateabb"]),
                    fontsize=6.3, ha="center", va="bottom", color="black",
                    path_effects=[pe.withStroke(linewidth=1.6,
                                                foreground="white")])
        ax.text(0.5, -0.02,
                f"{ano} (N = {len(nos)}; modularidade = {fmt_pt(mod, 2)}; "
                f"Jaccard alianças–disputas = "
                f"{fmt_pt(linha['jaccard_ali_disp'], 2)})",
                transform=ax.transAxes, ha="center", va="top", fontsize=7.5)
        ax.text(0.0, 1.0, letra, transform=ax.transAxes, fontsize=11,
                fontweight="bold", va="top", ha="left", color="black")
        ax.set_xlim(-3.25, 3.25)
        ax.set_ylim(-3.55, 3.05)
        ax.set_aspect("equal")
        ax.axis("off")
    fig.legend(
        handles=[Line2D([], [], color="#8A8A8A", lw=1.0,
                        label="Aliança na mesma comunidade"),
                 Line2D([], [], color="#8A8A8A", lw=0.7, ls=(0, (3, 3)),
                        label="Aliança entre comunidades"),
                 Line2D([], [], color=VERMELHO, lw=1.5,
                        label="Disputa (janela de três anos)")],
        loc="lower center", ncol=3, frameon=False, fontsize=7.5,
        bbox_to_anchor=(0.5, 0.0))
    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.11,
                        wspace=0.04)
    fig.savefig(FIG_DIR / "redes_comunidades_aliancas_disputas.png", dpi=300,
                facecolor="white")
    plt.close(fig)
    pd.DataFrame(resumo).to_csv(
        cfg.INTERIM_DIR / "redes_comunidades_resumo.csv", index=False)
    print("[comunidades] redes_comunidades_aliancas_disputas.png + "
          "data/interim/redes_comunidades_resumo.csv")


if __name__ == "__main__":
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    if len(sys.argv) > 1 and sys.argv[1] == "redes":
        figuras_redes()          # só as figuras de caracterização
        sys.exit(0)
    if len(sys.argv) > 1 and sys.argv[1] == "grafos":
        figura_grafos()          # grafos das camadas esparsas em anos selecionados
        sys.exit(0)
    if len(sys.argv) > 1 and sys.argv[1] == "comunidades":
        figura_comunidades()     # blocos de alianças com disputas sobrepostas
        sys.exit(0)
    if len(sys.argv) > 1 and sys.argv[1] == "gnn":
        curvas_gnn()             # rótulos PT-BR (sem SHAP)
        delta_pareado()
        sys.exit(0)
    curvas_pr()
    curvas_gnn()
    delta_pareado()
    shap_figuras()
    figuras_redes()
    print("\nOK – figuras normatizadas em", FIG_DIR)
