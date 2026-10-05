"""
GNN (GCN/GAT) para classificação de GRAFO anual – teste da H2.

Protocolo idêntico ao baseline: walk-forward purgado (s + N < t, treino mín.
40 anos), censura respeitada, mesmos anos OOS do RF (pareamento exato por
variante: com_dipl <-> `completo`, sem_dipl <-> `sem_dipl`).

Arquiteturas e hiperparâmetros FIXOS, PRÉ-DECLARADOS em
reports/GNN_RESULTADOS.md ANTES de qualquer execução (uma config por
arquitetura; sem search):
  GCN: por camada 2x GCNConv(->16->16) c/ edge_weight; GAT: 2x GATConv
  (heads=2, 8/head -> 16, edge_dim=1, fill_value=1.0 p/ self-loops em grafos
  sem arestas). Global mean pooling por camada -> concat -> Linear(16) ->
  ReLU -> Linear(1). Dropout 0,4; Adam lr=5e-3, weight_decay=5e-3; máx. 300
  épocas; early stopping paciência 30 no split temporal interno do treino
  (primeiros 80% gradiente, últimos 20% validação; melhor estado de val é
  usado no teste; SEM re-treino no treino completo – decisão pré-declarada).
  BCEWithLogits com pos_weight = neg/pos do split de gradiente.

Normalização dependente de dados (única): pesos de aresta de comércio
(log1p) e disputas (contagem) divididos pelo MÁXIMO sobre os anos de TREINO
do fold (train = 80% + 20% interno; o teste nunca entra).

Seeds 42–46 por fold: previsão principal = média das probabilidades
(ensemble); métricas por seed reportadas como média ± dp.

Teste da H2: block bootstrap PAREADO (blocos de 10 anos, 2.000 reamostras,
seed=42) do delta (GNN_ensemble − RF) em AUC e PR-AUC sobre os MESMOS anos
OOS; RF lido de data/interim/predicoes_oos_baseline.csv.

Uso:  ~/.virtualenvs/dsa-gnn/bin/python scripts/rodar_gnn.py [--smoke]
Saídas: data/processed/resultados_gnn.csv, data/processed/gnn_delta_ic.csv,
        data/interim/predicoes_oos_gnn.csv,
        data/interim/gnn_curvas_treino.csv,
        reports/figuras_modelos/gnn_curvas_treino.png.
"""

import copy
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, roc_auc_score
from torch_geometric.data import Batch, Data
from torch_geometric.nn import GATConv, GCNConv, global_mean_pool

from .. import config as cfg
from . import gnn_data

SEED_LIST = (42, 43, 44, 45, 46)
MIN_TRAIN = 40
N_HORIZON = 1
MAX_EPOCHS = 300
PATIENCE = 30
LR = 5e-3
WEIGHT_DECAY = 5e-3
DROPOUT = 0.4
HIDDEN = 16
INNER_VAL_FRAC = 0.2
BLOCK, N_BOOT, BOOT_SEED = 10, 2000, 42
RODADA = "gnn"

CELLS = ("proxy", "war")
VARIANTS = ("com_dipl", "sem_dipl")
ARCHS = ("gcn", "gat")
FS_MAP = {"com_dipl": "completo", "sem_dipl": "sem_dipl"}
CURVE_FOLDS = (1970, 2000)   # folds representativos p/ curvas treino/val

torch.set_num_threads(max(1, (torch.get_num_threads() or 8) - 2))


# ---------------------------------------------------------------------------
# Modelos
# ---------------------------------------------------------------------------
class LayerGCN(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.c1 = GCNConv(in_dim, HIDDEN)
        self.c2 = GCNConv(HIDDEN, HIDDEN)
        self.drop = nn.Dropout(DROPOUT)

    def forward(self, x, ei, ew, batch):
        h = torch.relu(self.c1(x, ei, ew))
        h = self.drop(h)
        h = torch.relu(self.c2(h, ei, ew))
        return global_mean_pool(h, batch)


class LayerGAT(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        # fill_value=1.0: atributo dos self-loops (evita NaN em grafo sem arestas)
        self.c1 = GATConv(in_dim, HIDDEN // 2, heads=2, edge_dim=1,
                          fill_value=1.0)
        self.c2 = GATConv(HIDDEN, HIDDEN // 2, heads=2, edge_dim=1,
                          fill_value=1.0)
        self.drop = nn.Dropout(DROPOUT)

    def forward(self, x, ei, ew, batch):
        ea = ew.view(-1, 1)
        h = torch.relu(self.c1(x, ei, edge_attr=ea))
        h = self.drop(h)
        h = torch.relu(self.c2(h, ei, edge_attr=ea))
        return global_mean_pool(h, batch)


class MultiLayerGNN(nn.Module):
    """Um encoder por camada de rede -> concat -> MLP -> logit (1/grafo)."""

    def __init__(self, arch, layers, in_dim):
        super().__init__()
        enc = LayerGCN if arch == "gcn" else LayerGAT
        self.layers = list(layers)
        self.encoders = nn.ModuleDict({lay: enc(in_dim) for lay in layers})
        self.drop = nn.Dropout(DROPOUT)
        self.head = nn.Sequential(
            nn.Linear(HIDDEN * len(layers), HIDDEN), nn.ReLU(),
            nn.Dropout(DROPOUT), nn.Linear(HIDDEN, 1))

    def forward(self, batches):
        embs = [self.encoders[lay](b.x, b.edge_index, b.edge_weight, b.batch)
                for lay, b in batches.items()]
        return self.head(self.drop(torch.cat(embs, dim=1))).squeeze(-1)


# ---------------------------------------------------------------------------
# Montagem de batches (por fold, com normalização fitada no treino)
# ---------------------------------------------------------------------------
def _fold_scales(data, train_years, layers):
    """Máximo do peso de aresta sobre os anos de TREINO (com/disp)."""
    scales = {}
    for lay in ("com", "disp"):
        if lay not in layers:
            continue
        mx = 0.0
        for t in train_years:
            w = data[t]["edges"][lay][1]
            if len(w):
                mx = max(mx, float(w.max()))
        scales[lay] = mx if mx > 0 else 1.0
    return scales


def make_batches(data, years, layers, scales):
    """Um Batch PyG por camada (mesmo vetor batch em todas as camadas)."""
    batches = {}
    for lay in layers:
        dlist = []
        for t in years:
            d = data[t]
            ei, ew = d["edges"][lay]
            ew = ew / scales[lay] if lay in scales else ew
            dlist.append(Data(x=torch.from_numpy(d["x"]),
                              edge_index=torch.from_numpy(ei),
                              edge_weight=torch.from_numpy(ew),
                              num_nodes=d["n"]))
        batches[lay] = Batch.from_data_list(dlist)
    return batches


# ---------------------------------------------------------------------------
# Treino de um fold (uma seed)
# ---------------------------------------------------------------------------
def train_fold(arch, layers, in_dim, b_tr, y_tr, b_val, y_val, b_te, seed,
               record_curve=False):
    torch.manual_seed(seed)
    model = MultiLayerGNN(arch, layers, in_dim)
    opt = torch.optim.Adam(model.parameters(), lr=LR,
                           weight_decay=WEIGHT_DECAY)
    pos = float(y_tr.sum())
    neg = float(len(y_tr) - pos)
    pw = torch.tensor(neg / pos if pos > 0 else 1.0)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pw)

    best_val, best_state, best_ep, since = np.inf, None, 0, 0
    curve = []
    for ep in range(MAX_EPOCHS):
        model.train()
        opt.zero_grad()
        out = model(b_tr)
        loss = lossf(out, y_tr)
        loss.backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            vloss = float(lossf(model(b_val), y_val))
        if record_curve:
            curve.append((ep, float(loss), vloss))
        if vloss < best_val - 1e-6:
            best_val, best_ep, since = vloss, ep, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            since += 1
            if since >= PATIENCE:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        prob = float(torch.sigmoid(model(b_te))[0])
    return prob, best_ep, curve


# ---------------------------------------------------------------------------
# Walk-forward de uma célula x variante x arquitetura
# ---------------------------------------------------------------------------
def walk_forward_gnn(data, targets, layers, familia, arch, seeds=SEED_LIST,
                     max_folds=None):
    years = gnn_data.usable_years(data, targets, familia, N_HORIZON)
    ymap = dict(zip(targets["ano"], targets[f"y{N_HORIZON}_{familia}"]))
    in_dim = next(iter(data.values()))["x"].shape[1]
    rows, curves = [], []
    test_years = [t for t in years
                  if len([s for s in years if s + N_HORIZON < t]) >= MIN_TRAIN]
    if max_folds:
        test_years = test_years[-max_folds:]
    t0 = time.time()
    for k, t in enumerate(test_years):
        train_years = [s for s in years if s + N_HORIZON < t]
        n_val = max(int(round(len(train_years) * INNER_VAL_FRAC)), 5)
        tr, val = train_years[:-n_val], train_years[-n_val:]
        y_tr = torch.tensor([float(ymap[s]) for s in tr])
        if y_tr.sum() == 0 or y_tr.sum() == len(y_tr):
            continue   # guarda: gradiente de classe única
        y_val = torch.tensor([float(ymap[s]) for s in val])
        scales = _fold_scales(data, train_years, layers)
        b_tr = make_batches(data, tr, layers, scales)
        b_val = make_batches(data, val, layers, scales)
        b_te = make_batches(data, [t], layers, scales)
        probs = []
        for seed in seeds:
            rec = (t in CURVE_FOLDS) and seed == seeds[0]
            prob, best_ep, curve = train_fold(arch, layers, in_dim, b_tr,
                                              y_tr, b_val, y_val, b_te, seed,
                                              record_curve=rec)
            probs.append(prob)
            rows.append({"ano": t, "seed": seed, "prob": prob,
                         "best_epoch": best_ep, "y": int(ymap[t])})
            for ep, ltr, lval in curve:
                curves.append({"fold": t, "arch": arch, "epoca": ep,
                               "loss_treino": ltr, "loss_val": lval})
        if (k + 1) % 20 == 0:
            print(f"    fold {k + 1}/{len(test_years)} "
                  f"({time.time() - t0:.0f}s)")
    preds = pd.DataFrame(rows)
    return preds, pd.DataFrame(curves)


# ---------------------------------------------------------------------------
# Métricas e bootstrap pareado
# ---------------------------------------------------------------------------
def _metrics(y, p):
    y, p = np.asarray(y), np.asarray(p)
    auc = roc_auc_score(y, p) if len(np.unique(y)) == 2 else np.nan
    ap = average_precision_score(y, p) if len(np.unique(y)) == 2 else np.nan
    pred = (p >= 0.5).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return auc, ap, prec, rec, f1


def summarize(preds):
    """Métricas do ensemble (média de prob entre seeds) + média±dp por seed."""
    ens = preds.groupby("ano").agg(y=("y", "first"),
                                   prob=("prob", "mean")).reset_index()
    auc, ap, prec, rec, f1 = _metrics(ens["y"], ens["prob"])
    per_seed = []
    for seed, g in preds.groupby("seed"):
        a, p_, *_ = _metrics(g["y"], g["prob"])
        per_seed.append((a, p_))
    aucs = np.array([a for a, _ in per_seed])
    aps = np.array([p_ for _, p_ in per_seed])
    return ens, {
        "n_anos_teste": len(ens), "positivos_teste": int(ens["y"].sum()),
        "prevalencia_teste": round(float(ens["y"].mean()), 3),
        "auc_ensemble": round(auc, 3), "pr_auc_ensemble": round(ap, 3),
        "prec_05": round(prec, 3), "rec_05": round(rec, 3),
        "f1_05": round(f1, 3),
        "auc_seed_media": round(float(aucs.mean()), 3),
        "auc_seed_dp": round(float(aucs.std(ddof=1)), 3),
        "pr_auc_seed_media": round(float(aps.mean()), 3),
        "pr_auc_seed_dp": round(float(aps.std(ddof=1)), 3),
        "best_epoch_medio": round(float(preds["best_epoch"].mean()), 1),
    }


def paired_delta_ic(gnn_ens, rf_preds):
    """Block bootstrap PAREADO do delta (GNN − RF) em AUC e PR-AUC."""
    m = gnn_ens.merge(rf_preds[["ano", "y", "prob"]], on="ano",
                      suffixes=("_gnn", "_rf"))
    assert (m["y_gnn"] == m["y_rf"]).all(), "rótulos divergem no pareamento"
    m = m.sort_values("ano")
    y = m["y_gnn"].to_numpy(float)
    pg = m["prob_gnn"].to_numpy(float)
    pr = m["prob_rf"].to_numpy(float)
    n = len(y)
    eff = min(BLOCK, n)
    starts = np.arange(0, n - eff + 1)
    nb = int(np.ceil(n / eff))
    rng = np.random.default_rng(BOOT_SEED)
    d_auc, d_ap = [], []
    skipped = 0
    for _ in range(N_BOOT):
        st = rng.choice(starts, size=nb, replace=True)
        idx = np.concatenate([np.arange(s, s + eff) for s in st])[:n]
        yy = y[idx]
        if yy.min() == yy.max():
            skipped += 1
            continue
        d_auc.append(roc_auc_score(yy, pg[idx]) - roc_auc_score(yy, pr[idx]))
        d_ap.append(average_precision_score(yy, pg[idx])
                    - average_precision_score(yy, pr[idx]))
    out = {
        "n_anos_pareados": n,
        "delta_auc_pontual": round(float(roc_auc_score(y, pg)
                                         - roc_auc_score(y, pr)), 3),
        "delta_pr_auc_pontual": round(
            float(average_precision_score(y, pg)
                  - average_precision_score(y, pr)), 3),
        "n_reamostras_validas": len(d_auc),
        "n_descartadas_classe_unica": skipped,
    }
    for nome, vals in (("delta_auc", d_auc), ("delta_pr_auc", d_ap)):
        out[f"{nome}_ic95_lo"] = round(float(np.percentile(vals, 2.5)), 3)
        out[f"{nome}_ic95_hi"] = round(float(np.percentile(vals, 97.5)), 3)
    return out


# ---------------------------------------------------------------------------
# Orquestração
# ---------------------------------------------------------------------------
def run_all(smoke=False):
    rf_all = pd.read_csv(cfg.INTERIM_DIR / "predicoes_oos_baseline.csv")
    metrics_rows, delta_rows, all_preds, all_curves = [], [], [], []
    variants = ("sem_dipl",) if smoke else VARIANTS
    cells = ("proxy",) if smoke else CELLS
    archs = ("gcn",) if smoke else ARCHS
    seeds = (42,) if smoke else SEED_LIST
    for variant in variants:
        print(f"\n=== Variante {variant}: construindo grafos ===")
        data, targets, layers = gnn_data.build_dataset(variant)
        for familia in cells:
            for arch in archs:
                tag = f"{familia}_N1 | {variant} | {arch}"
                print(f"[{tag}] walk-forward ({len(seeds)} seeds)...")
                t0 = time.time()
                preds, curves = walk_forward_gnn(
                    data, targets, layers, familia, arch, seeds=seeds,
                    max_folds=5 if smoke else None)
                ens, summ = summarize(preds)
                rf = rf_all[(rf_all.celula == f"{familia}_N1_ge2")
                            & (rf_all.protocolo == "walk_forward")
                            & (rf_all.feature_set == FS_MAP[variant])
                            & (rf_all.modelo == "rf")]
                row = {"rodada": RODADA, "celula": f"{familia}_N1_ge2",
                       "variante": variant, "arch": arch, **summ,
                       "tempo_s": round(time.time() - t0, 1)}
                metrics_rows.append(row)
                if not smoke:
                    d = paired_delta_ic(ens, rf)
                    delta_rows.append({"celula": f"{familia}_N1_ge2",
                                       "variante": variant, "arch": arch,
                                       "vs": f"rf/{FS_MAP[variant]}", **d})
                preds["celula"] = f"{familia}_N1_ge2"
                preds["variante"], preds["arch"] = variant, arch
                all_preds.append(preds)
                if len(curves):
                    curves["celula"] = f"{familia}_N1_ge2"
                    curves["variante"] = variant
                    all_curves.append(curves)
                print(f"  AUC_ens={summ['auc_ensemble']} "
                      f"PR_ens={summ['pr_auc_ensemble']} | por seed AUC "
                      f"{summ['auc_seed_media']}±{summ['auc_seed_dp']} "
                      f"({row['tempo_s']}s)")

    met = pd.DataFrame(metrics_rows)
    if smoke:
        print("\n[SMOKE] métricas:\n", met.to_string(index=False))
        return met
    met.to_csv(cfg.PROCESSED_DIR / "resultados_gnn.csv", index=False)
    pd.DataFrame(delta_rows).to_csv(cfg.PROCESSED_DIR / "gnn_delta_ic.csv",
                                    index=False)
    pd.concat(all_preds, ignore_index=True).to_csv(
        cfg.INTERIM_DIR / "predicoes_oos_gnn.csv", index=False)
    cur = pd.concat(all_curves, ignore_index=True)
    cur.to_csv(cfg.INTERIM_DIR / "gnn_curvas_treino.csv", index=False)
    plot_curves(cur)
    print("\n== Métricas ==\n", met.to_string(index=False))
    print("\n== Delta pareado vs RF ==\n",
          pd.DataFrame(delta_rows).to_string(index=False))
    return met


def plot_curves(cur):
    """Treino vs validação interna nos folds representativos (proxy)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    AZUL, VERMELHO = "#0072B2", "#D55E00"
    sub = cur[(cur.celula == "proxy_N1_ge2") & (cur.variante == "sem_dipl")]
    folds = sorted(sub["fold"].unique())
    archs = sorted(sub["arch"].unique())
    fig, axes = plt.subplots(len(archs), len(folds),
                             figsize=(5 * len(folds), 3.4 * len(archs)),
                             squeeze=False)
    for i, arch in enumerate(archs):
        for j, f in enumerate(folds):
            ax = axes[i][j]
            g = sub[(sub.arch == arch) & (sub.fold == f)]
            ax.plot(g["epoca"], g["loss_treino"], color=AZUL, lw=1.6,
                    label="treino (80%)")
            ax.plot(g["epoca"], g["loss_val"], color=VERMELHO, lw=1.6,
                    label="validação interna (20%)")
            ax.set_title(f"{arch.upper()} – fold {f} (seed 42)", fontsize=10)
            ax.set_xlabel("época")
            ax.set_ylabel("BCE (pos_weight)")
            ax.grid(True, lw=0.4, alpha=0.35)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
            if i == 0 and j == 0:
                ax.legend(frameon=False, fontsize=9)
    fig.suptitle("GNN – curvas de treino vs validação interna "
                 "(proxy_N1, sem_dipl; early stopping paciência 30)",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = cfg.REPORTS_DIR / "figuras_modelos" / "gnn_curvas_treino.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"[curvas] {out.name}")


if __name__ == "__main__":
    import sys
    run_all(smoke="--smoke" in sys.argv)
