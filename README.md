# Causal Effect of Preventive Follow-up on Low Birth Weight

## Problema

Qual é o efeito estimado de um acompanhamento pré-natal adequado sobre a probabilidade de baixo peso ao nascer, considerando diferenças observáveis entre as gestantes?

## Contexto

Baixo peso ao nascer (BPN < 2.500g) é um importante preditor de morbidade e mortalidade neonatal. A literatura clínica sugere que o acompanhamento pré-natal reduz o risco de BPN, mas é necessário separar o efeito causal do acompanhamento de outros fatores que afetam tanto a probabilidade de receber acompanhamento quanto a probabilidade de BPN.

Este projeto aplica métodos de **inferência causal** para estimar esse efeito de forma robusta.

## Objetivo

Usar dados de nascimentos de uma base pública para:

1. Definir formalmente a pergunta causal;
2. Identificar e ajustar por fatores confundidores;
3. Estimar o efeito causal médio usando múltiplos métodos (incluindo o Estimador Duplamente Robusto);
4. Calcular o Intervalo de Confiança de 95% via estatística Bootstrap;
5. Avaliar a sensibilidade dos resultados a diferentes especificações e documentar limitações causais.

## Metodologia

### Pergunta Causal

**Tratamento:** Acompanhamento pré-natal adequado (≥ 6 consultas durante a gravidez)  
**Desfecho:** Baixo peso ao nascer (< 2.500g)  
**Estimand:** Efeito médio do tratamento sobre a população (ATE - Average Treatment Effect)

### Métodos

- **Propensity Score Matching (PSM);**
- **Inverse Probability Weighting (IPW);**
- **Doubly Robust Estimation (AIPW);**
- **Intervalos de Confiança (95%) calculados via Bootstrap não-paramétrico;**
- **Análise de sensibilidade para confundimento não-observado.**
- **Análise de Sensibilidade de Rosenbaum (`src/rosenbaum_bounds.py`):** Avaliação de robustez do efeito causal estimado frente a potenciais confundidores não-observados (omitted variable bias), calculando limites superiores e inferiores de p-valor para diferentes níveis de viés ($\Gamma \in [1.0, 2.0]$).

### Confundidores presumidos

- Idade materna;
- Paridade (número de gestações anteriores);
- Escolaridade;
- Renda/situação socioeconômica;
- Etnia/raça;
- Tabagismo durante gravidez;
- Consumo de álcool;
- Presença de comorbidades maternas (hipertensão, diabetes).

## Dataset

**Fonte:** CDC Natality Public Use Dataset ou equivalente público de nascimentos  
**Período:** [a definir conforme acesso aos dados]  
**Tamanho esperado:** ~10.000 a 100.000 registros  
**Licença:** Domínio público

## Tecnologias

- Python 3.10+
- pandas: manipulação e preparação de dados
- NumPy: cálculos numéricos e reamostragem bootstrap
- statsmodels: modelos estatísticos e propensity score
- scikit-learn: ML supervisionado e modelos de regressão/propensão
- matplotlib/seaborn: visualizações
- pytest: testes unitários

## Estrutura do Projeto
causal-prenatal-lowbw/
├── README.md                    # Este arquivo
├── requirements.txt             # Dependências Python
├── .gitignore                   # Arquivos a ignorar no Git
│
├── data/
│   ├── raw/                     # Dados brutos (não versionados)
│   └── processed/               # Dados processados e limpos
│
├── notebooks/
│   ├── 01_exploratory_analysis.ipynb
│   ├── 02_propensity_score.ipynb
│   ├── 03_causal_estimation.ipynb
│   └── 04_sensitivity_analysis.ipynb
│
├── src/
│   ├── init.py
│   ├── data_loader.py           # Carregamento e limpeza de dados
│   ├── causal_analysis.py       # Métodos de inferência causal (PSM, IPW, Doubly Robust, Bootstrap)
│   ├── diagnostics.py           # Diagnósticos de balanceamento
│   └── visualization.py         # Gráficos e tabelas
│
├── reports/
│   └── figures/                 # Gráficos e tabelas exportados
│
└── tests/
├── init.py
└── test_causal_analysis.py  # Testes unitários
## Como Executar

### 1. Clonar o repositório

```bash
git clone [https://github.com/jessicaandradeg-dot/causal-prenatal-lowbw.git](https://github.com/jessicaandradeg-dot/causal-prenatal-lowbw.git)
cd causal-prenatal-lowbw
