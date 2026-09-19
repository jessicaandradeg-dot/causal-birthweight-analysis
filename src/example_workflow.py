"""
Exemplo de fluxo de trabalho completo para análise causal.

Este arquivo demonstra como usar os módulos criados para:
1. Carregar dados
2. Estimar propensity score
3. Realizar matching
4. Estimar efeito causal
5. Avaliar diagnósticos
"""

import numpy as np
import pandas as pd
from data_loader import DataLoader
from causal_analysis import PropensityScoreMatcher, InverseProbabilityWeighting
from diagnostics import BalancingDiagnostics, SensitivityAnalysis, print_diagnostic_summary
from visualization import CausalVisualizations


def generate_synthetic_data():
    """
    Gera dataset sintético de demonstração.

    Simula dados de nascimentos com:
    - Acompanhamento pré-natal (tratamento)
    - Baixo peso ao nascer (desfecho)
    - Confundidores sociodemográficos
    """
    np.random.seed(42)
    n = 1000

    # Confundidores: características maternas observáveis
    maternal_age = np.random.normal(28, 7, n)
    maternal_age = np.clip(maternal_age, 15, 50)

    maternal_education = np.random.choice([0, 1, 2, 3], n, p=[0.15, 0.3, 0.35, 0.2])
    # 0=Sem escolaridade, 1=Fundamental, 2=Médio, 3=Superior

    socioeconomic = np.random.choice([0, 1, 2], n, p=[0.4, 0.35, 0.25])
    # 0=Baixa, 1=Média, 2=Alta

    # Propensity score verdadeira (probabilidade de receber acompanhamento adequado)
    ps_true = (
        0.05 * maternal_age +
        0.15 * maternal_education +
        0.20 * socioeconomic -
        np.random.normal(0, 0.5, n)
    )
    ps_true = 1 / (1 + np.exp(-ps_true))

    # Tratamento: acompanhamento pré-natal adequado
    prenatal_care = (np.random.uniform(0, 1, n) < ps_true).astype(int)

    # Desfecho: baixo peso ao nascer
    # Efeito causal verdadeiro: -0.12 (acompanhamento reduz risco)
    low_birthweight_prob = (
        0.03 * maternal_age +
        -0.08 * maternal_education +
        -0.10 * socioeconomic +
        -0.12 * prenatal_care +  # Efeito causal
        np.random.normal(0, 0.3, n)
    )
    low_birthweight = (low_birthweight_prob > 0).astype(int)

    df = pd.DataFrame({
        'maternal_age': maternal_age,
        'maternal_education': maternal_education,
        'socioeconomic_status': socioeconomic,
        'prenatal_care': prenatal_care,
        'low_birthweight': low_birthweight
    })

    return df


def main():
    """Executa análise causal completa."""
    print("=" * 70)
    print("ANÁLISE CAUSAL: Efeito do Acompanhamento Pré-natal")
    print("=" * 70)

    # 1. Dados
    print("\n[1/6] Carregando dados...")
    df = generate_synthetic_data()
    print(f"Dataset gerado: {len(df)} registros, {len(df.columns)} colunas")
    print(df.head())

    # 2. Definir análise
    treatment_col = 'prenatal_care'
    outcome_col = 'low_birthweight'
    confounders = ['maternal_age', 'maternal_education', 'socioeconomic_status']

    # 3. Diagnósticos iniciais
    print("\n[2/6] Diagnósticos iniciais (desbalanceamento)...")
    diagnostics = BalancingDiagnostics()
    balance_before = diagnostics.check_balance(df, treatment_col, confounders)
    print(balance_before)

    # 4. Matching
    print("\n[3/6] Propensity Score Matching...")
    matcher = PropensityScoreMatcher(treatment_col, outcome_col, confounders)
    df_matched = matcher.match_on_propensity_score(df, caliper=0.1)
    print(f"Amostra após matching: {len(df_matched)} registros")

    results_psm = matcher.estimate_ate()
    print(f"\nResultados PSM:")
    print(f"  ATE: {results_psm['ate']:.4f}")
    print(f"  IC 95%: [{results_psm['ic_lower']:.4f}, {results_psm['ic_upper']:.4f}]")

    # 5. IPW (método alternativo)
    print("\n[4/6] Inverse Probability Weighting...")
    ipw = InverseProbabilityWeighting(treatment_col, outcome_col, confounders)
    results_ipw = ipw.estimate_ate(df)
    print(f"Resultados IPW:")
    print(f"  ATE: {results_ipw['ate']:.4f}")
    print(f"  IC 95%: [{results_ipw['ic_lower']:.4f}, {results_ipw['ic_upper']:.4f}]")

    # 6. Balanceamento após matching
    print("\n[5/6] Diagnósticos após matching...")
    balance_after = diagnostics.check_balance(df_matched, treatment_col, confounders)
    comparison = diagnostics.compare_before_after(df, df_matched, treatment_col, confounders)
    print(comparison)

    # 7. Sensibilidade
    print("\n[6/6] Análise de sensibilidade...")
    sens = SensitivityAnalysis()
    gamma_values = np.array([1, 1.1, 1.2, 1.5, 2.0])
    sensitivity_results = sens.rosenbaum_bounds(
        results_psm['ate'],
        se=0.05,
        gamma_range=gamma_values
    )
    print(sensitivity_results)

    # Resumo final
    print_diagnostic_summary(results_psm, balance_before, balance_after)

    print("\n" + "=" * 70)
    print("CONCLUSÃO")
    print("=" * 70)
    print(f"""
O acompanhamento pré-natal adequado reduz a probabilidade de baixo peso
ao nascer em {abs(results_psm['ate'])*100:.2f} pontos percentuais.

Intervalo de confiança: [{results_psm['ic_lower']*100:.2f}%, {results_psm['ic_upper']*100:.2f}%]

Este efeito é robusto a confundimento não-observado até gamma = {gamma_values[-1]}.
    """)


if __name__ == "__main__":
    main()
