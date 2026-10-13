# 🧬 Causal Inference in Public Health: Preventive Care and Low Birth Weight

Projeto de inferência causal para estimar o impacto do acompanhamento pré-natal adequado sobre o risco de baixo peso ao nascer.

## Problema de negócio

Baixo peso ao nascer é um indicador central de risco neonatal e está associado a consequências clínicas graves. O acompanhamento pré-natal adequado costuma ser associado a melhor prognóstico, mas a simples correlação entre essas variáveis não basta para concluir causalidade.

O desafio é separar o efeito real do cuidado médico dos fatores de confusão, como renda, escolaridade, idade materna, tabagismo e condições de saúde. Em outras palavras, o problema é estimar o efeito causal de uma intervenção de saúde com rigor estatístico.

## O que foi construído

- definição formal da pergunta causal
- ajuste por confundidores observáveis
- comparação de estimadores causais
- cálculo de intervalos de confiança por bootstrap
- análise de sensibilidade para viés não observável
- apresentação dos resultados em linguagem acessível para profissionais e recrutadores

## Metodologia

A análise usa métodos bem consolidados de inferência causal:

- PSM (Propensity Score Matching)
- IPW (Inverse Probability Weighting)
- AIPW (Augmented Inverse Probability Weighting / Doubly Robust Estimation)
- bootstrap para intervalos de confiança de 95%
- sensibilidade por Rosenbaum para avaliar robustez frente a variáveis não observadas

### Pergunta causal

- Tratamento: acompanhamento pré-natal adequado
- Desfecho: baixo peso ao nascer
- Estimando: efeito médio do tratamento sobre a população (ATE)

## Resultados e KPI

### KPIs relevantes

- estimativa do efeito causal médio
- IC 95% do efeito
- comparação entre estimadores
- estabilidade do resultado sob diferentes especificações
- robustez em análise de sensibilidade

### Exemplo de interpretação

| Método | Efeito estimado | IC 95% | Interpretação |
|---|---:|---:|---|
| PSM | -0.045 | [-0.08, -0.01] | redução relevante do risco |
| IPW | -0.038 | [-0.07, -0.01] | resultado consistente |
| AIPW | -0.041 | [-0.07, -0.02] | estimativa robusta |

Essa estrutura de resultado comunica bem a ideia de causalidade e rigor analítico, além de deixar claro que o trabalho não é apenas descritivo.

## Benchmark comparativo

O projeto compara múltiplos métodos em vez de depender de uma única estratégia. Isso é importante porque, em inferência causal, o resultado pode variar conforme a especificação do modelo e os pressupostos assumidos.

A comparação entre PSM, IPW e AIPW mostra maturidade na seleção de estimadores e no cuidado com robustez.

## Stack

- Python 3.10+
- Pandas
- NumPy
- Scikit-learn
- Statsmodels
- Matplotlib / Seaborn
- Pytest
- Inferência causal e bootstrap

## Estrutura do projeto

```text
causal-birthweight-analysis/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
├── notebooks/
├── src/
│   ├── data_loader.py
│   ├── causal_analysis.py
│   ├── diagnostics.py
│   └── visualization.py
├── reports/
├── tests/
└── .
```

## Relevância para vagas

Este projeto é altamente relevante para vagas de:

- Causal Inference
- Data Science
- Estatística aplicada
- Health analytics
- Pesquisa aplicada em dados públicos
- ML com foco em inferência e interpretação


## Link para artigo / benchmark / tabela

- [Resumo executivo](#)
- [Comparativo de estimadores](#)
- [Notebook de inferência causal](#)
- [Análise de sensibilidade](#)

