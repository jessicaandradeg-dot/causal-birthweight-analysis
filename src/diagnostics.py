"""
Diagnósticos para avaliação de qualidade da análise causal.
"""

import numpy as np
import pandas as pd


class BalancingDiagnostics:
    """
    Diagnósticos de balanceamento antes e depois de ajuste causal.

    Mensurada a diferença padronizada entre grupos antes (bruta)
    e depois do ajuste (matching ou IPW).
    """

    @staticmethod
    def standardized_difference(
        df: pd.DataFrame,
        treatment_col: str,
        var: str
    ) -> float:
        """
        Calcula diferença padronizada de uma variável entre grupos.

        Fórmula: (mean_treated - mean_control) / sqrt((var_treated + var_control) / 2)

        Args:
            df: DataFrame com dados.
            treatment_col: Nome da coluna de tratamento.
            var: Nome da variável.

        Returns:
            Diferença padronizada (deve ser < 0.1 após balanceamento).
        """
        treated = df[df[treatment_col] == 1][var]
        control = df[df[treatment_col] == 0][var]

        mean_diff = treated.mean() - control.mean()
        pooled_var = (treated.var() + control.var()) / 2

        if pooled_var == 0:
            return 0.0

        std_diff = mean_diff / np.sqrt(pooled_var)
        return std_diff

    def check_balance(
        self,
        df: pd.DataFrame,
        treatment_col: str,
        confounders: list,
        threshold: float = 0.1
    ) -> pd.DataFrame:
        """
        Verifica balanceamento de confundidores.

        Args:
            df: DataFrame com dados.
            treatment_col: Nome da coluna de tratamento.
            confounders: Lista de variáveis a verificar.
            threshold: Limite de diferença padronizada aceitável.

        Returns:
            DataFrame com diferenças padronizadas e status (Balanceado/Desbalanceado).
        """
        balance_stats = []

        for var in confounders:
            std_diff = self.standardized_difference(df, treatment_col, var)
            balanced = "✓" if abs(std_diff) < threshold else "✗"

            balance_stats.append({
                'Variable': var,
                'Std. Difference': round(std_diff, 4),
                'Balanced': balanced
            })

        return pd.DataFrame(balance_stats)

    def compare_before_after(
        self,
        df_before: pd.DataFrame,
        df_after: pd.DataFrame,
        treatment_col: str,
        confounders: list
    ) -> pd.DataFrame:
        """
        Compara balanceamento antes e depois do ajuste.

        Args:
            df_before: DataFrame antes do ajuste.
            df_after: DataFrame depois do ajuste (matched ou weighted).
            treatment_col: Nome da coluna de tratamento.
            confounders: Lista de variáveis a verificar.

        Returns:
            DataFrame comparativo.
        """
        results = []

        for var in confounders:
            std_before = self.standardized_difference(df_before, treatment_col, var)
            std_after = self.standardized_difference(df_after, treatment_col, var)
            improvement = ((abs(std_before) - abs(std_after)) / abs(std_before)) * 100

            results.append({
                'Variable': var,
                'Before': round(std_before, 4),
                'After': round(std_after, 4),
                'Improvement (%)': round(improvement, 2)
            })

        return pd.DataFrame(results)


class SensitivityAnalysis:
    """
    Análise de sensibilidade para confundimento não-observado.

    Avalia como o efeito estimado mudaria se houvesse confundimento escondido.
    """

    @staticmethod
    def rosenbaum_bounds(ate: float, se: float, gamma_range: np.ndarray):
        """
        Limites de Rosenbaum para sensibilidade a confundimento não-observado.

        Gamma: razão de chances de dois indivíduos idênticos em observáveis
               diferirem em receber tratamento.

        Args:
            ate: Efeito causal estimado.
            se: Erro padrão do efeito.
            gamma_range: Array de valores de gamma a testar.

        Returns:
            DataFrame com limites inferiores e superiores.
        """
        results = []

        for gamma in gamma_range:
            # Limites aproximados baseados em Rosenbaum
            lower = ate - np.sqrt(gamma) * se
            upper = ate + np.sqrt(gamma) * se

            results.append({
                'Gamma': gamma,
                'Lower Bound': round(lower, 4),
                'Upper Bound': round(upper, 4),
                'Contains Zero': "Yes" if lower <= 0 <= upper else "No"
            })

        return pd.DataFrame(results)

    @staticmethod
    def interpret_gamma(gamma: float) -> str:
        """
        Interpreta o valor de gamma.

        Args:
            gamma: Razão de chances de tratamento.

        Returns:
            Interpretação em linguagem natural.
        """
        if gamma == 1:
            return "Sem confundimento não-observado."
        elif gamma < 1.5:
            return "Resultado robusto a confundimento pequeno."
        elif gamma < 2:
            return "Resultado sensível a confundimento moderado."
        else:
            return "Resultado muito sensível a confundimento não-observado."


def print_diagnostic_summary(
    ate_estimate: dict,
    balance_before: pd.DataFrame,
    balance_after: pd.DataFrame = None
) -> None:
    """
    Imprime resumo dos diagnósticos de forma legível.

    Args:
        ate_estimate: Dicionário com ATE e IC.
        balance_before: DataFrame com balanceamento antes.
        balance_after: DataFrame com balanceamento depois (opcional).
    """
    print("\n" + "="*60)
    print("RESUMO DA ANÁLISE CAUSAL")
    print("="*60)

    print(f"\nEfeito Causal Estimado (ATE):")
    print(f"  Estimativa: {ate_estimate['ate']:.4f}")
    print(f"  IC 95%: [{ate_estimate['ic_lower']:.4f}, {ate_estimate['ic_upper']:.4f}]")
    print(f"  Interpretação: O tratamento reduz o risco em "
          f"{abs(ate_estimate['ate'])*100:.2f} pontos percentuais.")

    print(f"\nRiscos por Grupo (Após Ajuste):")
    print(f"  Tratados: {ate_estimate.get('risk_treated', 'N/A')}")
    print(f"  Controles: {ate_estimate.get('risk_control', 'N/A')}")

    if 'n_treated' in ate_estimate:
        print(f"\nTamanho da Amostra:")
        print(f"  Tratados: {ate_estimate['n_treated']}")
        print(f"  Controles: {ate_estimate['n_control']}")

    print(f"\nBalanceamento de Confundidores (Antes do Ajuste):")
    print(balance_before.to_string(index=False))

    if balance_after is not None:
        print(f"\nBalanceamento Após Ajuste:")
        print(balance_after.to_string(index=False))

    print("\n" + "="*60 + "\n")
