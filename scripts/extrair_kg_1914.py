#!/usr/bin/env python3
"""
Subseção didática de Grafo de Conhecimento .

Extrai dos CSVs brutos do COW um recorte REAL e rastreável de 1914 (Regra de
Ouro), no formato de triplos (sujeito, relação, objeto, atributos, fonte), e
desenha o esquema conceitual (entidades/relações/atributos) nas normas de
figura do Manual (sem título embutido, sem grade, Arial, preto).

Também mede, por ano, quantas díades do sistema têm comércio AUSENTE na
fonte (smoothtotrade == -9) – achado: nas guerras mundiais a
maioria dos pares está ausente, não zerada.

Saídas:
  data/interim/kg_triplos_1914.csv
  data/interim/comercio_ausentes_por_ano.csv
  reports/figuras_modelos/kg_esquema_conceitual.png

Uso: ~/.virtualenvs/datascience/bin/python scripts/extrair_kg_1914.py
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config as cfg  # noqa: E402
from src.network import annual  # noqa: E402

RAW = cfg.cow_raw_dir()
ANO = 1914
FIG_DIR = cfg.REPORTS_DIR / "figuras_modelos"

nmc = pd.read_csv(RAW / cfg.COW_FILES["nmc"])
ally = pd.read_csv(RAW / cfg.COW_FILES["alliances_yearly"])
trade = pd.read_csv(RAW / cfg.COW_FILES["trade_dyadic"])
midb = pd.read_csv(RAW / cfg.COW_FILES["midb"])
dip = pd.read_csv(RAW / cfg.COW_FILES["diplomatic"])
names = (nmc[["ccode", "stateabb"]].drop_duplicates()
         .set_index("ccode")["stateabb"].to_dict())
ns = annual.states_in_year(ANO)

rows = []


def add(s, r, o, atr, fonte):
    rows.append({"sujeito": s, "relacao": r, "objeto": o, "atributos": atr,
                 "fonte": fonte})


# --- atributos de Estado (CINC) --------------------------------------------
cinc = nmc[(nmc.year == ANO) & nmc.ccode.isin(ns)].set_index("ccode")["cinc"]
for c in cinc.sort_values(ascending=False).head(3).index:
    add(names[c], "tem_atributo", "CINC", f"ano={ANO}; cinc={cinc[c]:.3f}",
        "NMC_5_0.csv")

# --- alianças (tipo do compromisso) ----------------------------------------
a = ally[(ally.year == ANO) & ally.ccode1.isin(ns) & ally.ccode2.isin(ns)]


def tipo(r):
    return ("defesa" if r.defense else "neutralidade" if r.neutrality
            else "não agressão" if r.nonaggression else "entente")


for (c1, c2) in [(255, 300), (220, 365), (200, 220)]:
    sub = a[((a.ccode1 == c1) & (a.ccode2 == c2)) |
            ((a.ccode1 == c2) & (a.ccode2 == c1))]
    assert len(sub) > 0, (c1, c2)
    tipos = sorted({tipo(r) for r in sub.itertuples()})
    add(names[c1], "aliado_de", names[c2],
        f"ano={ANO}; tipo(s)={'/'.join(tipos)}", "alliance_v4.1_by_dyad_yearly.csv")

# --- disputa como ENTIDADE com atributos próprios (MID#257 = 1ª Guerra) -----
m = midb[(midb.dispnum == 257) & midb.ccode.isin(ns)]
for c in (255, 300, 220, 200):
    r = m[m.ccode == c].iloc[0]
    add(names[c], "participa_de", "Disputa MID#257",
        f"lado={'A' if r.sidea == 1 else 'B'}; styear={int(r.styear)}; "
        f"endyear={int(r.endyear)}; hostlev={int(r.hostlev)}; "
        f"fatality={int(r.fatality)}", "MIDB 5.0.csv")

# --- diplomacia (ano medido, nível de representação) -----------------------
d = dip[(dip.year == ANO) & (dip.ccode1 == 200) & (dip.ccode2 == 255)].iloc[0]
add(names[200], "intercambio_diplomatico", names[255],
    f"ano={ANO}; DE={int(d.DE)}; DR_at_1={int(d.DR_at_1)}; "
    f"DR_at_2={int(d.DR_at_2)}", "Diplomatic_Exchange_2006v1.csv")

# --- comércio: um par medido e um par AUSENTE na fonte (-9) ----------------
t = trade[(trade.year == ANO)]
tj = t[(t.ccode1 == 710) & (t.ccode2 == 740)]
if len(tj) == 0:
    tj = t[(t.ccode1 == 740) & (t.ccode2 == 710)]
tj = tj.iloc[0]
add(names[int(tj.ccode1)], "comercia_com", names[int(tj.ccode2)],
    f"ano={ANO}; smoothtotrade={tj.smoothtotrade:.1f}", "Dyadic_COW_4.0.csv")
tu = t[(t.ccode1 == 200) & (t.ccode2 == 255)].iloc[0]
add(names[200], "comercia_com", names[255],
    f"ano={ANO}; smoothtotrade={tu.smoothtotrade:.0f} (código de valor "
    "ausente na fonte)", "Dyadic_COW_4.0.csv")

kg = pd.DataFrame(rows)
kg.to_csv(cfg.INTERIM_DIR / "kg_triplos_1914.csv", index=False)
print(kg.to_string(index=False))

# --- comércio ausente por ano (todas as díades do sistema) -----------------
aus = []
for y in range(cfg.YEAR_MIN, 2013):
    nsy = annual.states_in_year(y)
    ty = trade[(trade.year == y) & trade.ccode1.isin(nsy) & trade.ccode2.isin(nsy)]
    n_pares = len(nsy) * (len(nsy) - 1) // 2
    aus.append({"ano": y, "n_estados": len(nsy), "pares": n_pares,
                "linhas_fonte": len(ty),
                "ausentes_menos9": int((ty.smoothtotrade == -9).sum()),
                "positivos": int((ty.smoothtotrade > 0).sum())})
aus = pd.DataFrame(aus)
aus["frac_ausente"] = aus.ausentes_menos9 / aus.pares
aus.to_csv(cfg.INTERIM_DIR / "comercio_ausentes_por_ano.csv", index=False)
print(aus[aus.frac_ausente > 0.4].to_string(index=False))

# --- esquema conceitual (figura) -------------------------------------------
plt.rcParams.update({"font.family": "Arial", "font.size": 10})
fig, ax = plt.subplots(figsize=(9.0, 6.2))
ax.set_xlim(0, 10)
ax.set_ylim(0, 7)
ax.axis("off")


def box(x, y, w, h, titulo, linhas, fill="#FFFFFF"):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                       linewidth=1.2, edgecolor="black", facecolor=fill)
    ax.add_patch(p)
    ax.text(x + w / 2, y + h - 0.3, titulo, ha="center", va="center",
            fontsize=10.5, fontweight="bold")
    ax.plot([x + 0.12, x + w - 0.12], [y + h - 0.56, y + h - 0.56],
            color="black", lw=0.8)
    for i, l in enumerate(linhas):
        ax.text(x + 0.15, y + h - 0.88 - i * 0.34, l, ha="left", va="center",
                fontsize=8.5)


def seta(x1, y1, x2, y2, rotulo, lx, ly):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=12, lw=1.0, color="black"))
    ax.text(lx, ly, rotulo, ha="center", va="center", fontsize=8.5,
            style="italic",
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none"))


box(3.6, 2.55, 2.8, 2.1, "Estado",
    ["ccode, nome", "ano (snapshot)", "CINC (share)", "entrada/saída"],
    fill="#EEF3F8")
box(0.3, 4.7, 3.0, 2.1, "Disputa militarizada",
    ["dispnum, início, fim", "hostlev (1–5)", "fatality (0–6)", "lado A / lado B"])
box(6.7, 4.7, 3.0, 2.1, "Aliança",
    ["identificador, início, fim", "tipo: defesa, neutralidade,",
     "não agressão, entente"])
box(0.3, 0.2, 3.0, 2.1, "Fluxo comercial",
    ["ano, díade", "smoothtotrade", "código −9 = ausente (≠ zero)"])
box(6.7, 0.2, 3.0, 2.1, "Missão diplomática",
    ["ano medido (espaçado)", "DE (há intercâmbio)",
     "nível de representação", "(DR_at_1, DR_at_2)"])
seta(3.6, 4.5, 3.3, 5.3, "participa_de (lado)", 2.6, 4.45)
seta(6.4, 4.5, 6.7, 5.3, "membro_de", 7.3, 4.45)
seta(3.6, 2.7, 3.3, 1.9, "origem/destino", 2.7, 2.65)
seta(6.4, 2.7, 6.7, 1.9, "envia/recebe", 7.3, 2.65)
ax.text(5.0, 1.15, "Projeção usada neste\ntrabalho: 4 camadas\nhomogêneas "
        "Estado–Estado,\num peso escalar por aresta",
        ha="center", va="center", fontsize=8.5,
        bbox=dict(boxstyle="round,pad=0.35", fc="#F2F2F2", ec="black", lw=0.8))
fig.tight_layout()
fig.savefig(FIG_DIR / "kg_esquema_conceitual.png", dpi=300)
print("[kg] figura salva")
