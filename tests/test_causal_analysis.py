"""
Testes para módulos de análise causal.
"""

import pytest
import numpy as np
import pandas as pd
from src.causal_analysis import PropensityScoreMatcher, InverseProbabilityWeighting
from src.diagnostics import BalancingDiagnostics


@pytest.fixture
def synthetic_data():
    """
    Cria dataset sintético para testes.

    Simula dados de tratamento e confundidores com relacionamento conhecido.
    """
    np.random.seed(42)
    n = 500

    # Confundidores
    age = np.random.normal(30, 8, n)
    education = np.random.choice([0, 1, 2], n)
    income = np.random.normal(5000, 2000, n)

    # Propensity score verdadeira
    ps_true = 1 / (1 + np.exp(-(0.1 * age - 0.5 * education + 0.0001 * income)))

    # Tratamento (probabilístico baseado em PS)
    treatment = (np.random.uniform(0, 1, n) < ps_true).astype(int)

    # Desfecho (com efeito causal verdadeiro = -0.1)
    outcome = (
        (0.05 * age + 0.2 * education - 0.0001 * income) +
        (-0.1 * treatment) +  # Efeito causal verdadeiro
        np.random.normal(0, 0.3, n)
    ) > 0.5

    df = pd.DataFrame({
        'age': age,
        'education': education,
        'income': income,
        'treatment': treatment,
        'outcome': outcome.astype(int)
    })

    return df


def test_propensity_score_matcher_initialization():
    """Testa inicialização do PSM."""
    matcher = PropensityScoreMatcher(
        treatment_col='treatment',
        outcome_col='outcome',
        confounders=['age', 'education']
    )

    assert matcher.treatment_col == 'treatment'
    assert matcher.outcome_col == 'outcome'
    assert len(matcher.confounders) == 2


def test_propensity_score_estimation(synthetic_data):
    """Testa estimação de propensity score."""
    matcher = PropensityScoreMatcher(
        treatment_col='treatment',
        outcome_col='outcome',
        confounders=['age', 'education', 'income']
    )

    ps_scores = matcher.estimate_propensity_score(synthetic_data)

    # Propensity scores devem estar entre 0 e 1
    assert (ps_scores >= 0).all() and (ps_scores <= 1).all()
    # Deve ter tamanho igual ao dataset
    assert len(ps_scores) == len(synthetic_data)


def test_propensity_score_matching(synthetic_data):
    """Testa matching baseado em propensity score."""
    matcher = PropensityScoreMatcher(
        treatment_col='treatment',
        outcome_col='outcome',
        confounders=['age', 'education', 'income']
    )

    matched_data = matcher.match_on_propensity_score(synthetic_data, caliper=0.15)

    # Deve ter menos dados após matching (alguns descartados)
    assert len(matched_data) <= len(synthetic_data)
    # Deve manter número igual de tratados e controles (1:1 matching)
    n_treated = len(matched_data[matched_data['treatment'] == 1])
    n_control = len(matched_data[matched_data['treatment'] == 0])
    assert n_treated == n_control


def test_ate_estimation(synthetic_data):
    """Testa estimação de ATE."""
    matcher = PropensityScoreMatcher(
        treatment_col='treatment',
        outcome_col='outcome',
        confounders=['age', 'education', 'income']
    )

    matcher.match_on_propensity_score(synthetic_data, caliper=0.15)
    results = matcher.estimate_ate()

    # ATE deve ser numérico
    assert isinstance(results['ate'], (int, float, np.number))
    # IC deve conter ATE
    assert results['ic_lower'] <= results['ate'] <= results['ic_upper']
    # Riscos devem estar entre 0 e 1
    assert 0 <= results['risk_treated'] <= 1
    assert 0 <= results['risk_control'] <= 1


def test_ipw_estimation(synthetic_data):
    """Testa estimação via IPW."""
    ipw = InverseProbabilityWeighting(
        treatment_col='treatment',
        outcome_col='outcome',
        confounders=['age', 'education', 'income']
    )

    results = ipw.estimate_ate(synthetic_data)

    # ATE deve ser numérico
    assert isinstance(results['ate'], (int, float, np.number))
    # IC deve conter ATE
    assert results['ic_lower'] <= results['ate'] <= results['ic_upper']


def test_balancing_diagnostics(synthetic_data):
    """Testa diagnósticos de balanceamento."""
    diagnostics = BalancingDiagnostics()

    balance = diagnostics.check_balance(
        synthetic_data,
        'treatment',
        ['age', 'education', 'income']
    )

    # Deve retornar DataFrame com 3 variáveis
    assert len(balance) == 3
    # Deve ter colunas esperadas
    assert 'Variable' in balance.columns
    assert 'Std. Difference' in balance.columns


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
