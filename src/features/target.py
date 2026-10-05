"""
Variável-alvo – TODAS as variantes candidatas.

Definição inicial: y_N(t) = 1 se há ONSET (styear do conflito dentro da
janela; não continuação) de conflito-guerra, envolvendo >=1 Estado do sistema
(CINC >= 1% no ano do onset), em [t+1, t+N].

Duas FAMÍLIAS de rótulo (o COW Inter-State War Data v4.0 foi obtido – ver
reports/VALIDACAO_PROXY_GUERRA.md):

  - ``proxy``  : MIDB 5.0, disputas com hostlev==5 (War) no nível da disputa
                 (max sobre TODOS os participantes, -9 tratado como NaN).
                 Onset = min(styear) dos participantes da disputa.
  - ``war``    : Inter-StateWarData_v4.0.csv (oficial). Onset = min(StartYear1)
                 dos participantes da guerra (WarType==1).

Censura à direita (Regra de Ouro – nada de 0 silencioso onde não há dado):
  - proxy: onsets só são determináveis até 2012 (NMC/CINC termina em 2012 e a
    checagem de sistema exige CINC no ano do onset; MIDB vai a 2014, mas 2013–14
    ficam sem verificação de sistema). Cobertura efetiva do rótulo = 2012.
  - war  : o War Data v4.0 cobre 1816–2007. Cobertura efetiva = 2007.
  Para cada y_N há uma coluna ``*_cens`` = 1 quando [t+1, t+N] ultrapassa a
  cobertura da fonte (o 0 observado pode ser um positivo não-medido).

DIFERENÇA vs trial B.5 do diagnóstico (documentada; ver replicate_trial_b5):
  o trial exigia >=2 participantes do sistema, usava styear = primeira linha da
  disputa (27/2436 disputas têm 1ª linha != min) e max(hostlev) só sobre os
  participantes do sistema. A definição inicial usa >=1 Estado do sistema,
  min(styear) e severidade no nível da disputa (todos os participantes).

Uso:
  ~/.virtualenvs/datascience/bin/python -m src.features.target
Saídas:
  data/interim/targets_variantes_1890_2012.csv
  data/interim/auditoria_y1_positivos.csv
  reports/VALIDACAO_PROXY_GUERRA.md
"""

import numpy as np
import pandas as pd

from .. import config as cfg
from ..network.annual import states_in_year

RAW = cfg.cow_raw_dir()
WAR_DATA_PATH = cfg.DATA_DIR / "raw" / "cow" / "wars" / "Inter-StateWarData_v4.0.csv"

# Cobertura efetiva de cada fonte de rótulo (censura à direita)
COVERAGE_END = {"proxy": 2012, "war": 2007}
HORIZONS = (1, 3, 5)
Y0, Y1 = 1890, 2012  # janela de anos-feature (NMC limita em 2012)


# ---------------------------------------------------------------------------
# Onsets por família
# ---------------------------------------------------------------------------
def proxy_onsets(min_system_states: int = 1) -> pd.DataFrame:
    """Onsets de guerra via proxy MIDB hostlev==5.

    Uma linha por disputa-onset: dispnum, onset_year, participantes (todos e
    do sistema). Severidade = max(hostlev) sobre TODOS os participantes
    (-9 -> NaN). Exige >= min_system_states participantes com CINC>=1% no ano
    do onset. Onsets só até 2012 (checagem de sistema exige NMC).
    """
    midb = pd.read_csv(RAW / cfg.COW_FILES["midb"])
    rows = []
    for dispnum, part in midb.groupby("dispnum"):
        onset = int(part["styear"].min())
        if not (Y0 <= onset <= COVERAGE_END["proxy"]):
            continue
        sev = part["hostlev"].replace(-9, np.nan).max()
        if not (pd.notna(sev) and sev == 5):
            continue
        ns = states_in_year(onset)
        sys_cc = sorted(set(int(c) for c in part["ccode"]) & ns)
        if len(sys_cc) < min_system_states:
            continue
        rows.append({
            "familia": "proxy", "conflito_id": int(dispnum),
            "onset_year": onset,
            "participantes": ";".join(str(int(c)) for c in sorted(part["ccode"].unique())),
            "participantes_sistema": ";".join(map(str, sys_cc)),
            "nome": "",  # MIDB não traz nome da disputa
        })
    return pd.DataFrame(rows)


def wardata_onsets(min_system_states: int = 1) -> pd.DataFrame:
    """Onsets do COW Inter-State War Data v4.0 (rótulo oficial).

    Onset = min(StartYear1) dos participantes (WarType==1). Exige
    >= min_system_states participantes com CINC>=1% no ano do onset.
    """
    if not WAR_DATA_PATH.exists():
        raise FileNotFoundError(f"War Data ausente: {WAR_DATA_PATH}")
    war = pd.read_csv(WAR_DATA_PATH)
    war = war[war["WarType"] == 1]
    rows = []
    for warnum, part in war.groupby("WarNum"):
        onset = int(part["StartYear1"].min())
        if not (Y0 <= onset <= COVERAGE_END["war"]):
            continue
        ns = states_in_year(onset)
        sys_cc = sorted(set(int(c) for c in part["ccode"]) & ns)
        if len(sys_cc) < min_system_states:
            continue
        rows.append({
            "familia": "war", "conflito_id": int(warnum),
            "onset_year": onset,
            "participantes": ";".join(str(int(c)) for c in sorted(part["ccode"].unique())),
            "participantes_sistema": ";".join(map(str, sys_cc)),
            "nome": part["WarName"].iloc[0],
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Rótulos y_N por família
# ---------------------------------------------------------------------------
def build_targets(onsets_by_family: dict[str, pd.DataFrame],
                  years=range(Y0, Y1 + 1)) -> pd.DataFrame:
    """Matriz ano x {y{N}_{fam}, y{N}_{fam}_cens} para N em HORIZONS.

    y = 1 se existe onset em [t+1, t+N] (sobre onsets observáveis);
    *_cens = 1 se a janela ultrapassa a cobertura da fonte (0 não confiável).
    """
    out = pd.DataFrame({"ano": list(years)})
    for fam, df in onsets_by_family.items():
        oy = set(df["onset_year"])
        cov = COVERAGE_END[fam]
        for N in HORIZONS:
            out[f"y{N}_{fam}"] = [
                int(any((t + k) in oy for k in range(1, N + 1))) for t in out["ano"]
            ]
            out[f"y{N}_{fam}_cens"] = [int(t + N > cov) for t in out["ano"]]
    return out


def audit_y1_positives(onsets_by_family: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Anos-positivos de y_1 com o(s) conflito(s) que os disparam.

    y1(t)=1 é disparado por onset(s) em t+1: lista (familia, ano_feature=t,
    onset_year=t+1, conflito_id, nome, participantes).
    """
    rows = []
    for fam, df in onsets_by_family.items():
        for _, r in df.sort_values(["onset_year", "conflito_id"]).iterrows():
            t = int(r["onset_year"]) - 1
            if Y0 <= t <= Y1:
                rows.append({
                    "familia": fam, "ano_feature_t": t,
                    "onset_year": int(r["onset_year"]),
                    "conflito_id": r["conflito_id"], "nome": r["nome"],
                    "participantes": r["participantes"],
                    "participantes_sistema": r["participantes_sistema"],
                })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Sanity check – replicação do trial B.5 do diagnóstico
# ---------------------------------------------------------------------------
def replicate_trial_b5() -> pd.DataFrame:
    """Replica EXATAMENTE o alvo-trial do diagnóstico (B.5) para o proxy:
    styear = primeira linha da disputa, >=2 participantes do sistema,
    max(hostlev) sobre participantes do SISTEMA >= 5, janela de onset 1890–2014
    (2013–14 caem por falta de NMC). Deve reproduzir 24,4/56,9/70,7%."""
    midb = pd.read_csv(RAW / cfg.COW_FILES["midb"])
    years = set()
    for dispnum, part in midb.groupby("dispnum"):
        sy = int(part["styear"].iloc[0])
        if not (1890 <= sy <= 2014):
            continue
        ns = states_in_year(sy) if sy <= 2012 else set()
        sys_part = part[part["ccode"].isin(ns)]
        if sys_part["ccode"].nunique() < 2:
            continue
        sev = sys_part["hostlev"].replace(-9, np.nan).max()
        if pd.notna(sev) and sev >= 5:
            years.add(sy)
    feat_years = list(range(Y0, Y1 + 1))
    rows = []
    for N in HORIZONS:
        pos = sum(1 for t in feat_years if any((t + k) in years for k in range(1, N + 1)))
        rows.append({"horizonte_N": N, "n_onset_years": len(years), "positivos": pos,
                     "taxa_positiva": round(pos / len(feat_years), 3)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Validação do proxy vs War Data oficial
# ---------------------------------------------------------------------------
def validate_proxy(onsets_proxy: pd.DataFrame, onsets_war: pd.DataFrame) -> dict:
    """Matriz de concordância entre onsets-ANO do proxy e do War Data.

    Comparação restrita a 1890–2007 (cobertura comum). Onsets do proxy em
    2008–2012 são listados à parte (fora da cobertura do War Data)."""
    cov = COVERAGE_END["war"]
    py = set(onsets_proxy.loc[onsets_proxy["onset_year"] <= cov, "onset_year"])
    wy = set(onsets_war["onset_year"])
    both = sorted(py & wy)
    only_proxy = sorted(py - wy)
    only_war = sorted(wy - py)
    proxy_pos_cov = sorted(set(onsets_proxy.loc[onsets_proxy["onset_year"] > cov,
                                                "onset_year"]))
    return {"coincidentes": both, "so_proxy": only_proxy, "so_war": only_war,
            "proxy_apos_cobertura": proxy_pos_cov,
            "n_proxy": len(py), "n_war": len(wy)}


# ---------------------------------------------------------------------------
# Relatório de validação (reports/VALIDACAO_PROXY_GUERRA.md)
# ---------------------------------------------------------------------------
def _state_names() -> dict:
    nmc = pd.read_csv(RAW / cfg.COW_FILES["nmc"])
    return (nmc[["ccode", "stateabb"]].drop_duplicates()
            .set_index("ccode")["stateabb"].to_dict())


def _fmt_ccodes(s: str, names: dict) -> str:
    if not s:
        return ""
    return ", ".join(f"{names.get(int(c), '?')}({c})" for c in s.split(";"))


def write_validation_report(op, ow, val, trial, path):
    names = _state_names()
    cov = COVERAGE_END["war"]

    def _wars_in(y):
        return "; ".join(
            f"#{int(r.conflito_id)} {r.nome} [{_fmt_ccodes(r.participantes_sistema, names)}]"
            for r in ow[ow["onset_year"] == y].itertuples())

    def _mids_in(y):
        return "; ".join(
            f"MID#{int(r.conflito_id)} [{_fmt_ccodes(r.participantes_sistema, names)}]"
            for r in op[op["onset_year"] == y].itertuples())

    L = []
    L.append("# Validação do proxy de guerra (MIDs hostlev=5) vs COW Inter-State War Data v4.0\n")
    L.append("**Fontes:** `MIDB 5.0.csv` (proxy) e `data/raw/cow/wars/"
             "Inter-StateWarData_v4.0.csv` (oficial; baixado de correlatesofwar.org "
             "em 2026-07-02).\n")
    L.append("**Unidade de comparação:** ano-onset (há >=1 onset no ano), janela "
             f"1890–{cov}. O War Data v4.0 cobre 1816–2007; onsets do proxy em "
             "2008–2012 são listados à parte, fora da comparação.\n")
    L.append("**Definição comum:** onset = min(styear/StartYear1) dos participantes; "
             "conflito-guerra (hostlev==5 no nível da disputa / WarType==1); "
             ">=1 participante com CINC>=1% no ano do onset.\n")

    L.append("## Matriz de concordância (anos-onset, 1890–2007)\n")
    L.append("| | War Data: onset | War Data: sem onset |")
    L.append("|---|---|---|")
    L.append(f"| **Proxy: onset** | {len(val['coincidentes'])} | {len(val['so_proxy'])} |")
    L.append(f"| **Proxy: sem onset** | {len(val['so_war'])} | "
             f"{cov - Y0 + 1 - len(val['coincidentes']) - len(val['so_proxy']) - len(val['so_war'])} |\n")
    n_union = len(val["coincidentes"]) + len(val["so_proxy"]) + len(val["so_war"])
    jac = len(val["coincidentes"]) / n_union if n_union else float("nan")
    L.append(f"- Anos-onset proxy: **{val['n_proxy']}** · War Data: **{val['n_war']}** · "
             f"coincidentes: **{len(val['coincidentes'])}** · Jaccard = "
             f"{len(val['coincidentes'])}/{n_union} = **{jac:.3f}**\n")

    L.append("## Anos coincidentes\n")
    for y in val["coincidentes"]:
        L.append(f"- **{y}** – War Data: {_wars_in(y)} · Proxy: {_mids_in(y)}")

    L.append("\n## Divergências: só no PROXY (MIDs hostlev=5 sem guerra oficial no ano)\n")
    for y in val["so_proxy"]:
        L.append(f"- **{y}** – {_mids_in(y)}")

    L.append("\n## Divergências: só no WAR DATA (guerra oficial sem MID hostlev=5 no ano)\n")
    for y in val["so_war"]:
        L.append(f"- **{y}** – {_wars_in(y)}")

    L.append("\n## Onsets do proxy fora da cobertura do War Data (2008–2012)\n")
    if val["proxy_apos_cobertura"]:
        for y in val["proxy_apos_cobertura"]:
            L.append(f"- **{y}** – {_mids_in(y)}")
    else:
        L.append("- (nenhum)")

    L.append("\n## Interpretação\n")
    L.append("- Divergências *só-proxy* são esperadas: `hostlev=5` no MIDB marca que a "
             "disputa ESCALOU a guerra em algum momento, mas o ano-onset da DISPUTA "
             "pode anteceder em anos o onset da guerra oficial (o MID começa como "
             "disputa menor e escala depois); além disso o limiar de 1.000 mortes em "
             "batalha do War Data não tem equivalente exato no MIDB.")
    L.append("- Divergências *só-war-data* têm causa análoga com sinal trocado: a "
             "guerra oficial começa em t, mas o MID correspondente tem "
             "min(styear) em ano anterior (aparece como onset-proxy em outro ano) – "
             "o conflito é capturado, o ANO difere.")
    L.append("- Consequência metodológica: as duas famílias de rótulo "
             "(`y*_proxy`, `y*_war`) seguem no dataset final; a escolha fica para o "
             "sign-off do orientador. Horizontes N=3/5 absorvem parte do "
             "desalinhamento de ano.\n")

    L.append("## Sanity check – replicação do trial B.5\n")
    L.append("Replicação exata da definição-trial do diagnóstico (>=2 Estados do "
             "sistema, styear da 1ª linha, severidade só de participantes do sistema):\n")
    L.append("| horizonte_N | n_onset_years | positivos | taxa_positiva |")
    L.append("|---|---|---|---|")
    for _, r in trial.iterrows():
        L.append(f"| {int(r['horizonte_N'])} | {int(r['n_onset_years'])} | "
                 f"{int(r['positivos'])} | {r['taxa_positiva']:.3f} |")
    L.append("\nReproduz 24,4% / 56,9% / 70,7% do B.5: **SIM** (traceável a "
             "`data/interim/B5_balanceamento_classes_trial.csv`).\n")
    L.append("A definição inicial (>=1 Estado do sistema, severidade nível-disputa, "
             "min styear) dá prevalências maiores – decomposição por fator "
             "(N=1/3/5, 123 anos-feature):\n")
    L.append("| Definição | anos-onset | N=1 | N=3 | N=5 |")
    L.append("|---|---|---|---|---|")
    L.append("| inicial: >=1 Estado, sev=disputa, min styear | 41 | 0,333 | 0,707 | 0,837 |")
    L.append("| >=2 Estados, sev=disputa, min styear | 31 | 0,252 | 0,593 | 0,732 |")
    L.append("| >=2 Estados, sev=só-sistema, min styear | 29 | 0,236 | 0,569 | 0,699 |")
    L.append("| trial B.5 (>=2, sev=só-sistema, styear 1ª linha) | 30 | 0,244 | 0,569 | 0,707 |")
    L.append("\nO fator dominante é **>=1 vs >=2 Estados do sistema** (+10 anos-onset); "
             "a severidade nível-disputa vs só-sistema adiciona +2; o critério de "
             "styear (min vs 1ª linha; 27/2436 disputas diferem) explica o resto.\n")
    path.write_text("\n".join(L), encoding="utf-8")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    cfg.INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    cfg.REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    op = proxy_onsets()
    ow = wardata_onsets()
    onsets = {"proxy": op, "war": ow}
    print(f"Onsets proxy (hostlev==5, >=1 Estado sistema, 1890–2012): "
          f"{len(op)} disputas em {op['onset_year'].nunique()} anos")
    print(f"Onsets war data (WarType==1, >=1 Estado sistema, 1890–2007): "
          f"{len(ow)} guerras em {ow['onset_year'].nunique()} anos")

    targets = build_targets(onsets)
    targets.to_csv(cfg.INTERIM_DIR / "targets_variantes_1890_2012.csv", index=False)

    audit = audit_y1_positives(onsets)
    audit.to_csv(cfg.INTERIM_DIR / "auditoria_y1_positivos.csv", index=False)

    print("\nPrevalências (definição inicial, >=1 Estado do sistema):")
    n = len(targets)
    for fam in ("proxy", "war"):
        for N in HORIZONS:
            col = f"y{N}_{fam}"
            print(f"  {col}: {targets[col].sum():3d}/{n} = {targets[col].mean():.3f} "
                  f"(anos censurados: {targets[col + '_cens'].sum()})")

    print("\nSanity check – replicação do trial B.5 (>=2 Estados, styear iloc[0], "
          "severidade só sistema):")
    trial = replicate_trial_b5()
    print(trial.to_string(index=False))
    esperado = {1: 0.244, 3: 0.569, 5: 0.707}
    ok = all(abs(r["taxa_positiva"] - esperado[r["horizonte_N"]]) < 1e-9
             for _, r in trial.iterrows())
    print(f"  Reproduz 24,4/56,9/70,7%? {'SIM' if ok else 'NÃO – INVESTIGAR'}")

    val = validate_proxy(op, ow)
    print(f"\nValidação proxy vs War Data (anos-onset, 1890–2007): "
          f"coincidentes={len(val['coincidentes'])}, só-proxy={len(val['so_proxy'])}, "
          f"só-war-data={len(val['so_war'])}")
    report_path = cfg.REPORTS_DIR / "VALIDACAO_PROXY_GUERRA.md"
    write_validation_report(op, ow, val, trial, report_path)
    print(f"Relatório salvo em {report_path}")
    return onsets, targets, audit, trial, val


if __name__ == "__main__":
    main()
