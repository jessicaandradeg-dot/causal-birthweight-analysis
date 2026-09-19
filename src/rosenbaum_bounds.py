import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

def rosenbaum_sensitivity_bounds(y_treated: np.ndarray, y_control: np.ndarray, gamma_range: list = [1.0, 1.25, 1.5, 1.75, 2.0]):
    """
    Calcula os limites de Rosenbaum (Rosenbaum Bounds) para avaliar a robustez
    da estimativa causal diante de um potencial confundidor não-observado.
    """
    differences = y_treated - y_control
    results = []
    
    # Teste de postos sinalizados de Wilcoxon baseline
    _, p_value_base = wilcoxon(differences)
    
    for gamma in gamma_range:
        # Limite superior e inferior aproximados para o p-valor sob presença de viés γ
        adj_p_upper = min(1.0, p_value_base * gamma)
        adj_p_lower = p_value_base / gamma
        
        results.append({
            "Gamma (γ)": gamma,
            "Odds Ratio Odds Difference": f"{gamma:.2f}x",
            "p-value Upper Bound": round(adj_p_upper, 5),
            "p-value Lower Bound": round(adj_p_lower, 5),
            "Robust at α=0.05": adj_p_upper < 0.05
        })
        
    df_bounds = pd.DataFrame(results)
    print("=== ROSENBAUM SENSITIVITY BOUNDS ANALYSIS ===")
    print(df_bounds.to_string(index=False))
    return df_bounds

if __name__ == "__main__":
    np.random.seed(42)
    sample_treated = np.random.normal(loc=3100, scale=400, size=100)
    sample_control = np.random.normal(loc=2850, scale=400, size=100)
    
    rosenbaum_sensitivity_bounds(sample_treated, sample_control)