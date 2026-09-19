"""
Visualizações para análise causal.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd


class CausalVisualizations:
    """Gráficos e visualizações para análise causal."""

    @staticmethod
    def plot_propensity_score_distribution(
        df: pd.DataFrame,
        treatment_col: str,
        ps_col: str = 'propensity_score',
        save_path: str = None
    ) -> None:
        """
        Plota distribuição de propensity scores por grupo de tratamento.

        Args:
            df: DataFrame com dados.
            treatment_col: Nome da coluna de tratamento.
            ps_col: Nome da coluna de propensity scores.
            save_path: Caminho para salvar figura (opcional).
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        treated = df[df[treatment_col] == 1][ps_col]
        control = df[df[treatment_col] == 0][ps_col]

        ax.hist(control, bins=30, alpha=0.6, label='Controle', color='steelblue')
        ax.hist(treated, bins=30, alpha=0.6, label='Tratado', color='coral')

        ax.set_xlabel('Propensity Score', fontsize=12)
        ax.set_ylabel('Frequência', fontsize=12)
        ax.set_title('Distribuição de Propensity Scores por Grupo', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(axis='y', alpha=0.3)

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figura salva em: {save_path}")

        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_balance_comparison(
        balance_df: pd.DataFrame,
        save_path: str = None
    ) -> None:
        """
        Plota comparação de balanceamento antes/depois.

        Args:
            balance_df: DataFrame com colunas 'Variable', 'Before', 'After'.
            save_path: Caminho para salvar figura (opcional).
        """
        fig, ax = plt.subplots(figsize=(10, len(balance_df) * 0.4))

        x = np.arange(len(balance_df))
        width = 0.35

        ax.barh(x - width/2, balance_df['Before'], width, label='Antes', color='lightcoral')
        ax.barh(x + width/2, balance_df['After'], width, label='Depois', color='lightgreen')

        ax.axvline(x=-0.1, color='gray', linestyle='--', linewidth=1, alpha=0.7)
        ax.axvline(x=0.1, color='gray', linestyle='--', linewidth=1, alpha=0.7)

        ax.set_yticks(x)
        ax.set_yticklabels(balance_df['Variable'])
        ax.set_xlabel('Diferença Padronizada', fontsize=12)
        ax.set_title('Diagnóstico de Balanceamento', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(axis='x', alpha=0.3)

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figura salva em: {save_path}")

        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_ate_with_ci(
        results: dict,
        method_name: str = "ATE",
        save_path: str = None
    ) -> None:
        """
        Plota estimativa de ATE com intervalo de confiança.

        Args:
            results: Dicionário com 'ate', 'ic_lower', 'ic_upper'.
            method_name: Nome do método para label.
            save_path: Caminho para salvar figura (opcional).
        """
        fig, ax = plt.subplots(figsize=(10, 4))

        ate = results['ate']
        ic_lower = results['ic_lower']
        ic_upper = results['ic_upper']

        # Plota ponto e intervalo
        ax.errorbar(
            ate, 0,
            xerr=[[ate - ic_lower], [ic_upper - ate]],
            fmt='o',
            markersize=10,
            color='steelblue',
            elinewidth=2,
            capsize=5,
            label=method_name
        )

        # Linha de efeito nulo
        ax.axvline(x=0, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Efeito Nulo')

        ax.set_ylim(-0.5, 0.5)
        ax.set_yticks([])
        ax.set_xlabel('Efeito Causal (ATE)', fontsize=12)
        ax.set_title(f'Estimativa de Efeito Causal com IC 95%', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(axis='x', alpha=0.3)

        # Adiciona texto com valores
        ax.text(
            ate, -0.35,
            f"ATE = {ate:.4f}\nIC: [{ic_lower:.4f}, {ic_upper:.4f}]",
            ha='center',
            fontsize=10,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        )

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figura salva em: {save_path}")

        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_outcome_distribution(
        df: pd.DataFrame,
        treatment_col: str,
        outcome_col: str,
        save_path: str = None
    ) -> None:
        """
        Plota distribuição de desfecho por grupo de tratamento.

        Args:
            df: DataFrame com dados.
            treatment_col: Nome da coluna de tratamento.
            outcome_col: Nome da coluna de desfecho.
            save_path: Caminho para salvar figura (opcional).
        """
        fig, ax = plt.subplots(figsize=(8, 6))

        treated_outcome = df[df[treatment_col] == 1][outcome_col].mean()
        control_outcome = df[df[treatment_col] == 0][outcome_col].mean()

        groups = ['Tratado', 'Controle']
        outcomes = [treated_outcome, control_outcome]
        colors = ['coral', 'steelblue']

        bars = ax.bar(groups, outcomes, color=colors, alpha=0.7, edgecolor='black')

        # Adiciona valores nas barras
        for bar, outcome in zip(bars, outcomes):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width()/2, height,
                f'{outcome:.2%}',
                ha='center', va='bottom',
                fontsize=12, fontweight='bold'
            )

        ax.set_ylabel('Proporção de Desfecho', fontsize=12)
        ax.set_title(f'Desfecho por Grupo (Antes do Ajuste)', fontsize=14, fontweight='bold')
        ax.set_ylim(0, max(outcomes) * 1.2)
        ax.grid(axis='y', alpha=0.3)

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figura salva em: {save_path}")

        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_sensitivity_analysis(
        sensitivity_df: pd.DataFrame,
        save_path: str = None
    ) -> None:
        """
        Plota análise de sensibilidade para confundimento não-observado.

        Args:
            sensitivity_df: DataFrame com colunas 'Gamma', 'Lower Bound', 'Upper Bound'.
            save_path: Caminho para salvar figura (opcional).
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        ax.fill_between(
            sensitivity_df['Gamma'],
            sensitivity_df['Lower Bound'],
            sensitivity_df['Upper Bound'],
            alpha=0.3,
            color='steelblue',
            label='Intervalo de Confiança'
        )

        ax.plot(
            sensitivity_df['Gamma'],
            sensitivity_df['Lower Bound'],
            'o--',
            color='steelblue',
            linewidth=2,
            markersize=6
        )

        ax.plot(
            sensitivity_df['Gamma'],
            sensitivity_df['Upper Bound'],
            'o--',
            color='steelblue',
            linewidth=2,
            markersize=6
        )

        # Linha de efeito nulo
        ax.axhline(y=0, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Efeito Nulo')

        ax.set_xlabel('Gamma (Razão de Chances de Tratamento)', fontsize=12)
        ax.set_ylabel('Efeito Causal', fontsize=12)
        ax.set_title('Análise de Sensibilidade: Confundimento Não-Observado',
                    fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figura salva em: {save_path}")

        plt.tight_layout()
        plt.show()
