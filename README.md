# Predição de instabilidade sistêmica em redes interestatais multicamada com aprendizado de máquina (1890–2012)

**Trabalho de Conclusão de Curso – MBA em Data Science e Analytics (USP/Esalq)**
**Autor:** Maurício Gonçalves · **Orientador:** Edilson José Rodrigues

Código, dados derivados e figuras do TCC. O trabalho dá continuidade ao TCC do
MBA em Business Intelligence e Analytics
([usp-bia-tcc](https://github.com/mauricio-opus10/usp-bia-tcc)), que comparou
descritivamente as redes interestatais em duas janelas históricas. Aqui a
análise passa de descritiva a preditiva, com recortes anuais.

## Pergunta de pesquisa

É possível, a partir de métricas estruturais de redes interestatais
multicamada, treinar modelos de aprendizado de máquina capazes de distinguir
anos de estabilidade de anos que antecedem conflitos de alta intensidade?

## Desenho

- **Dados:** projeto Correlates of War (COW) – alianças, comércio, disputas
  militarizadas (MID), diplomacia, capacidades nacionais (CINC) e guerras
  interestatais.
- **Redes:** um grafo por camada e por ano (1890–2012), com os Estados de
  CINC ≥ 1% como vértices (isolados incluídos).
- **Atributos:** 63 por ano – métricas por camada em nível e variação anual,
  métricas entre camadas, composição do sistema e concentração de capacidade.
- **Alvos:** início de disputa no nível de guerra (proxy MID) e início de
  guerra do conjunto oficial, em horizontes de um e três anos.
- **Modelos:** regressão logística, random forest, XGBoost e redes neurais de
  grafos (GCN/GAT).
- **Validação:** walk-forward purgado, baselines ingênuos, bootstrap em
  blocos, testes de robustez definidos antes da execução. Semente 42.

## Resultado em uma frase

Apenas o proxy de disputas no horizonte de um ano apresentou sinal, com random
forest (AUC-ROC de 0,678 e 0,699), mas os intervalos de confiança de 95%
incluem o acaso e o sinal se concentra em 1931–1945. O rótulo oficial de
guerra ficou no nível do acaso e as redes neurais de grafos não superaram o
modelo clássico. Os resultados são indícios, não evidência, e a predição é
estrutural, não causal.

Os números vêm de `data/processed/resultados_baseline.csv`,
`data/processed/robustez_bootstrap_ic.csv` e
`tcc_branch_h1_operacionalizacao/analises/pos_hoc_periodos.csv`.

## Estrutura

```
usp-dsa-tcc/
├── src/
│   ├── config.py            # caminhos e constantes (CINC, semente, janela)
│   ├── network/annual.py    # construtores das redes anuais por camada
│   ├── features/            # matriz de atributos e variáveis-alvo
│   └── models/              # baseline, robustez, desbalanceamento, GNN
├── scripts/                 # pontos de entrada de cada etapa
├── data/
│   ├── raw/                 # brutos COW (não versionados; ver abaixo)
│   ├── interim/             # diagnósticos e previsões fora da amostra
│   └── processed/           # atributos, alvos e tabelas de resultados
├── reports/
│   ├── figuras_modelos/     # figuras do TCC (300 DPI)
│   └── figuras_prelim/      # figuras da etapa preliminar
└── tcc_branch_h1_operacionalizacao/analises/   # análise exploratória por período
```

## Como reproduzir

### 1. Dados brutos

Os arquivos do COW não são redistribuídos aqui. Baixe-os em
<https://correlatesofwar.org/data-sets/> e organize assim:

```
data/raw/cow/
├── alliances/alliance_v4.1_by_dyad_yearly.csv
├── trade/Dyadic_COW_4.0.csv
├── mids/MIDA 5.0.csv
├── mids/MIDB 5.0.csv
├── nmc/NMC_5_0.csv
├── diplomatic/Diplomatic_Exchange_2006v1.csv
├── wars/Inter-StateWarData_v4.0.csv
└── COW-country-codes.csv
```

### 2. Ambientes

```bash
# abordagem clássica (Python 3.12)
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# redes neurais de grafos (ambiente separado, Python 3.14)
python -m venv .venv-gnn && source .venv-gnn/bin/activate
pip install -r requirements-gnn.txt
```

### 3. Execução, na ordem

| Etapa | Comando | Saídas principais |
|---|---|---|
| Diagnóstico temporal | `python scripts/diagnostico_temporal.py` | `data/interim/B1…B5_*.csv` |
| Alvos | `python -m src.features.target` | `data/interim/targets_variantes_1890_2012.csv` |
| Atributos | `python -m src.features.build_features` | `data/processed/features_targets_1890_2012.csv` |
| Modelos clássicos | `python scripts/rodar_baseline.py` | `data/processed/resultados_baseline.csv` |
| Robustez | `python scripts/rodar_robustez.py` | `resultados_robustez.csv`, `robustez_bootstrap_ic.csv` |
| Desbalanceamento e corte de CINC | `python scripts/rodar_feedback.py` | `resultados_desbalanceamento.csv`, `resultados_cinc_sensibilidade.csv` |
| Redes neurais de grafos | `python scripts/rodar_gnn.py` (ambiente GNN) | `resultados_gnn.csv`, `gnn_delta_ic.csv` |
| Análise por período | `python tcc_branch_h1_operacionalizacao/analises/analise_pos_hoc.py` | `pos_hoc_*.csv` |
| Figuras | `python scripts/gerar_figuras_tcc.py` | `reports/figuras_modelos/` |

As previsões fora da amostra de cada modelo ficam em
`data/interim/predicoes_oos_*.csv`, o que permite auditar todas as métricas
sem reexecutar o treinamento.

## Convenções metodológicas

1. Todo número reportado é rastreável a um arquivo de origem e a um cálculo.
2. Estados com CINC ≥ 1% entram em todas as camadas, mesmo sem arestas.
3. A camada de comércio usa `smoothtotrade`.
4. A predição é funcional e estrutural. Nenhuma leitura causal é feita.

## Fonte dos dados

Correlates of War Project. <https://correlatesofwar.org>
