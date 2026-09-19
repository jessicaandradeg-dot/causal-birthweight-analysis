"""
Métodos de inferência causal para estimação de efeito de tratamento.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from statsmodels.formula.api import logit
import warnings

warnings.filterwarnings('ignore')


class PropensityScoreMatcher:
    """
    Propensity Score Matching para estimação de efeito causal.

    Processa:
    1. Estima propensity score (probabilidade de receber tratamento dado confundidores)
    2. Faz matching entre tratados e controles com scores similares
    3. Estima efeito causal no amostra matcheada
    """

    def __init__(self, treatment_col: str, outcome_col: str, confounders: list):
        """
        Inicializa o matcher.

        Args:
            treatment_col: Nome da coluna de tratamento (0/1).
            outcome_col: Nome da coluna de desfecho (0/1).
            confounders: Lista de colunas de confundidores a ajustar.
        """
        self.treatment_col = treatment_col
        self.outcome_col = outcome_col
        self.confounders = confounders
        self.ps_model = None
        self.ps_scores = None
        self.matched_data = None

    def estimate_propensity_score(self, df: pd.DataFrame) -> np.ndarray:
        """
        Estima propensity score usando regressão logística.

        Args:
            df: DataFrame com tratamento e confundidores.

        Returns:
            Array com propensity scores.
        """
        X = df[self.confounders].copy()
        y = df[self.treatment_col].copy()

        # Padroniza features numéricas
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Treina modelo logístico
        self.ps_model = LogisticRegression(max_iter=1000, random_state=42)
        self.ps_model.fit(X_scaled, y)

        # Calcula scores (probabilidade de receber tratamento)
        self.ps_scores = self.ps_model.predict_proba(X_scaled)[:, 1]

        return self.ps_scores

    def match_on_propensity_score(
        self, df: pd.DataFrame, caliper: float = 0.1
    ) -> pd.DataFrame:
        """
        Faz 1:1 matching entre tratados e controles.

        Args:
            df: DataFrame com todos os dados.
            caliper: Distância máxima permitida entre scores (padrão: 0.1).

        Returns:
            DataFrame com pares matcheados.
        """
        if self.ps_scores is None:
            self.estimate_propensity_score(df)

        df_with_ps = df.copy()
        df_with_ps['ps'] = self.ps_scores

        treated = df_with_ps[df_with_ps[self.treatment_col] == 1]
        control = df_with_ps[df_with_ps[self.treatment_col] == 0]

        matched_pairs = []

        for idx, row in treated.iterrows():
            ps_treated = row['ps']
            # Encontra controles dentro do caliper
            eligible = control[
                np.abs(control['ps'] - ps_treated) <= caliper
            ]

            if len(eligible) > 0:
                # Seleciona o mais próximo
                closest_idx = (
                    (eligible['ps'] - ps_treated).abs().idxmin()
                )
                matched_pairs.append(idx)
                matched_pairs.append(closest_idx)

        self.matched_data = df_with_ps.loc[matched_pairs].reset_index(drop=True)
        return self.matched_data

    def estimate_ate(self) -> dict:
        """
        Estima efeito médio do tratamento (ATE) na amostra matcheada.

        Returns:
            Dicionário com ATE, IC 95% e tamanho das amostras.
        """
        if self.matched_data is None:
            raise ValueError("Execute match_on_propensity_score primeiro.")

        treated = self.matched_data[
            self.matched_data[self.treatment_col] == 1
        ][self.outcome_col]
        control = self.matched_data[
            self.matched_data[self.treatment_col] == 0
        ][self.outcome_col]

        ate = control.mean() - treated.mean()
        # Risco em controles - Risco em tratados
        # Se ATE < 0: tratamento reduz risco (efeito protetor)

        # Intervalo de confiança via bootstrap
        n_bootstrap = 1000
        ates_boot = []

        np.random.seed(42)
        for _ in range(n_bootstrap):
            idx_t = np.random.choice(treated.index, size=len(treated), replace=True)
            idx_c = np.random.choice(control.index, size=len(control), replace=True)
            ate_boot = (
                self.matched_data.loc[idx_c, self.outcome_col].mean()
                - self.matched_data.loc[idx_t, self.outcome_col].mean()
            )
            ates_boot.append(ate_boot)

        ates_boot = np.array(ates_boot)
        ic_lower = np.percentile(ates_boot, 2.5)
        ic_upper = np.percentile(ates_boot, 97.5)

        return {
            'ate': ate,
            'ic_lower': ic_lower,
            'ic_upper': ic_upper,
            'n_treated': len(treated),
            'n_control': len(control),
            'risk_treated': treated.mean(),
            'risk_control': control.mean()
        }


class InverseProbabilityWeighting:
    """
    Inverse Probability Weighting para estimação robusta de efeito causal.

    Pondera observações inversamente à probabilidade de sua posição observada
    no tratamento, criando uma população pseudo-aleatória.
    """

    def __init__(self, treatment_col: str, outcome_col: str, confounders: list):
        """
        Inicializa o IPW.

        Args:
            treatment_col: Nome da coluna de tratamento (0/1).
            outcome_col: Nome da coluna de desfecho (0/1).
            confounders: Lista de colunas de confundidores.
        """
        self.treatment_col = treatment_col
        self.outcome_col = outcome_col
        self.confounders = confounders
        self.ps_model = None
        self.ps_scores = None
        self.weights = None

    def estimate_propensity_score(self, df: pd.DataFrame) -> np.ndarray:
        """Estima propensity score (igual a PSM)."""
        X = df[self.confounders].copy()
        y = df[self.treatment_col].copy()

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        self.ps_model = LogisticRegression(max_iter=1000, random_state=42)
        self.ps_model.fit(X_scaled, y)
        self.ps_scores = self.ps_model.predict_proba(X_scaled)[:, 1]

        return self.ps_scores

    def calculate_weights(self, df: pd.DataFrame) -> np.ndarray:
        """
        Calcula pesos IPW.

        Peso = 1/PS para tratados, 1/(1-PS) para controles.

        Args:
            df: DataFrame com tratamento.

        Returns:
            Array com pesos.
        """
        if self.ps_scores is None:
            self.estimate_propensity_score(df)

        self.weights = np.where(
            df[self.treatment_col] == 1,
            1 / self.ps_scores,
            1 / (1 - self.ps_scores)
        )

        return self.weights

    def estimate_ate(self, df: pd.DataFrame) -> dict:
        """
        Estima ATE usando IPW.

        Args:
            df: DataFrame com tratamento, desfecho e confundidores.

        Returns:
            Dicionário com ATE e intervalo de confiança.
        """
        weights = self.calculate_weights(df)

        treated = df[df[self.treatment_col] == 1]
        control = df[df[self.treatment_col] == 0]

        w_treated = weights[df[self.treatment_col] == 1]
        w_control = weights[df[self.treatment_col] == 0]

        risk_treated = np.average(
            treated[self.outcome_col], weights=w_treated
        )
        risk_control = np.average(
            control[self.outcome_col], weights=w_control
        )

        ate = risk_control - risk_treated

        # IC por bootstrap
        n_bootstrap = 1000
        ates_boot = []
        np.random.seed(42)

        for _ in range(n_bootstrap):
            idx = np.random.choice(len(df), size=len(df), replace=True)
            df_boot = df.iloc[idx].reset_index(drop=True)
            weights_boot = self.calculate_weights(df_boot)

            t = df_boot[df_boot[self.treatment_col] == 1]
            c = df_boot[df_boot[self.treatment_col] == 0]
            wt = weights_boot[df_boot[self.treatment_col] == 1]
            wc = weights_boot[df_boot[self.treatment_col] == 0]

            if len(t) > 0 and len(c) > 0:
                rt = np.average(t[self.outcome_col], weights=wt)
                rc = np.average(c[self.outcome_col], weights=wc)
                ates_boot.append(rc - rt)

        ates_boot = np.array(ates_boot)
        ic_lower = np.percentile(ates_boot, 2.5)
        ic_upper = np.percentile(ates_boot, 97.5)

        return {
            'ate': ate,
            'ic_lower': ic_lower,
            'ic_upper': ic_upper,
            'risk_treated': risk_treated,
            'risk_control': risk_control
        }
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, LinearRegression

def estimador_duplamente_robusto(y, t, X):
    """
    Estimador Doubly Robust (AIPW): Combina o Modelo de Propensão (PSM/IPW)
    com a Regressão do Desfecho para estimar o ATE do pré-natal no peso ao nascer.
    """
    # Modelo de Propensão
    ps_model = LogisticRegression(max_iter=1000).fit(X, t)
    ps = np.clip(ps_model.predict_proba(X)[:, 1], 0.01, 0.99)
    
    # Modelos de Regressão do Desfecho
    m1 = LinearRegression().fit(X[t == 1], y[t == 1])
    m0 = LinearRegression().fit(X[t == 0], y[t == 0])
    
    mu1 = m1.predict(X)
    mu0 = m0.predict(X)
    
    # Equação Doubly Robust (AIPW)
    ate_dr = np.mean(
        (t * y / ps - (t - ps) / ps * mu1) - 
        ((1 - t) * y / (1 - ps) + (t - ps) / (1 - ps) * mu0)
    )
    return ate_dr

def bootstrap_intervalo_confianca(y, t, X, n_iterations=500, alpha=0.05):
    """Calcula Intervalo de Confiança de 95% via Bootstrap para o ATE."""
    ates = []
    n = len(y)
    np.random.seed(42)
    
    for _ in range(n_iterations):
        idx = np.random.choice(n, size=n, replace=True)
        ate = estimador_duplamente_robusto(
            y.iloc[idx].values if hasattr(y, 'iloc') else y[idx], 
            t.iloc[idx].values if hasattr(t, 'iloc') else t[idx], 
            X.iloc[idx].values if hasattr(X, 'iloc') else X[idx]
        )
        ates.append(ate)
        
    ic_lower = np.percentile(ates, 100 * (alpha / 2))
    ic_upper = np.percentile(ates, 100 * (1 - alpha / 2))
    return ic_lower, ic_upper