"""
Configuração central do projeto DSA – caminhos e constantes.

Regra de ouro: os dados brutos COW NÃO são versionados neste repo. Por ora são
lidos (read-only) do repositório do TCC BIA, que já os contém. Quando o pipeline
final for consolidado, os brutos podem ser copiados para ``data/raw/`` deste repo
(ver TODO em reports/DIAGNOSTICO_TEMPORAL.md).
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Caminhos
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent          # usp-dsa-tcc/
DATA_DIR = REPO_ROOT / "data"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
REPORTS_DIR = REPO_ROOT / "reports"

# Brutos COW: primeiro tenta data/raw/cow local; senão, o repo BIA (read-only).
_LOCAL_COW = DATA_DIR / "raw" / "cow"
_BIA_COW = (
    REPO_ROOT.parent.parent
    / "001 BIA TCC" / "usp-bia-tcc" / "data" / "raw" / "cow"
)


def cow_raw_dir() -> Path:
    """Resolve o diretório de dados brutos COW (local tem precedência)."""
    if (_LOCAL_COW / "nmc" / "NMC_5_0.csv").exists():
        return _LOCAL_COW
    if (_BIA_COW / "nmc" / "NMC_5_0.csv").exists():
        return _BIA_COW
    raise FileNotFoundError(
        "Dados brutos COW não encontrados.\n"
        f"  - local : {_LOCAL_COW}\n"
        f"  - BIA   : {_BIA_COW}\n"
        "Copie os brutos para data/raw/cow/ ou ajuste src/config.py."
    )


# Arquivos COW (relativos a cow_raw_dir())
COW_FILES = {
    "alliances_yearly": "alliances/alliance_v4.1_by_dyad_yearly.csv",
    "alliances_dyad": "alliances/alliance_v4.1_by_dyad.csv",
    "trade_dyadic": "trade/Dyadic_COW_4.0.csv",
    "mida": "mids/MIDA 5.0.csv",
    "midb": "mids/MIDB 5.0.csv",
    "nmc": "nmc/NMC_5_0.csv",
    "diplomatic": "diplomatic/Diplomatic_Exchange_2006v1.csv",
    "country_codes": "COW-country-codes.csv",
}

# ---------------------------------------------------------------------------
# Constantes (herdadas do TCC de BIA)
# ---------------------------------------------------------------------------
RANDOM_SEED = 42

# Limiar de seleção de Estados (CINC). 1% = referência do BIA (Opção A).
CINC_THRESHOLD = 0.01

# Pesos por tipo de aliança (commitment forte -> fraco)
ALLIANCE_WEIGHTS = {"defense": 4, "neutrality": 3, "nonaggression": 2, "entente": 1}

# Cobertura temporal-alvo do TCC DSA (proposta). Limites efetivos por dataset
# são medidos empiricamente em reports/DIAGNOSTICO_TEMPORAL.md.
YEAR_MIN = 1890
YEAR_MAX = 2014

# Níveis de hostilidade das MIDs (COW hostlev)
HOSTLEV_LABELS = {
    1: "No militarized action",
    2: "Threat to use force",
    3: "Display of force",
    4: "Use of force",
    5: "War",
}
