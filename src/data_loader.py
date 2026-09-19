"""
Carregamento e preparação de dados para análise causal.
"""

import pandas as pd
import numpy as np
from pathlib import Path


class DataLoader:
    """Carregador e processador de dados de nascimentos."""

    def __init__(self, data_dir: str = None):
        """
        Inicializa o DataLoader.

        Args:
            data_dir: Caminho para o diretório de dados.
                     Se None, usa './data' por padrão.
        """
        if data_dir is None:
            data_dir = Path(__file__).parent.parent / "data"
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"

        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def load_raw_data(self, filename: str) -> pd.DataFrame:
        """
        Carrega dados brutos do diretório raw.

        Args:
            filename: Nome do arquivo (CSV, Excel, Parquet, etc.)

        Returns:
            DataFrame com dados carregados.
        """
        filepath = self.raw_dir / filename

        if not filepath.exists():
            raise FileNotFoundError(f"Arquivo não encontrado: {filepath}")

        if filename.endswith('.csv'):
            return pd.read_csv(filepath)
        elif filename.endswith('.xlsx'):
            return pd.read_excel(filepath)
        elif filename.endswith('.parquet'):
            return pd.read_parquet(filepath)
        else:
            raise ValueError(f"Formato não suportado: {filename}")

    def save_processed_data(self, df: pd.DataFrame, filename: str) -> None:
        """
        Salva dados processados no diretório processed.

        Args:
            df: DataFrame para salvar.
            filename: Nome do arquivo (sem extensão, será salvo como Parquet).
        """
        filepath = self.processed_dir / f"{filename}.parquet"
        df.to_parquet(filepath, index=False)
        print(f"Dados salvos em: {filepath}")

    def clean_missing_values(self, df: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
        """
        Remove colunas com taxa de valores faltantes acima do threshold.

        Args:
            df: DataFrame para limpar.
            threshold: Proporção máxima de NaN aceitável (padrão: 50%).

        Returns:
            DataFrame com colunas removidas.
        """
        missing_ratio = df.isnull().sum() / len(df)
        cols_to_drop = missing_ratio[missing_ratio > threshold].index.tolist()

        if cols_to_drop:
            print(f"Removidas colunas com >50% de valores faltantes: {cols_to_drop}")
            df = df.drop(columns=cols_to_drop)

        return df

    def validate_data(self, df: pd.DataFrame) -> dict:
        """
        Retorna estatísticas de qualidade dos dados.

        Args:
            df: DataFrame para validar.

        Returns:
            Dicionário com métricas de qualidade.
        """
        return {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'missing_values': df.isnull().sum().to_dict(),
            'duplicates': df.duplicated().sum(),
            'dtypes': df.dtypes.to_dict()
        }


def main():
    """Teste básico do DataLoader."""
    loader = DataLoader()
    print("DataLoader inicializado com sucesso!")
    print(f"Diretório de dados brutos: {loader.raw_dir}")
    print(f"Diretório de dados processados: {loader.processed_dir}")


if __name__ == "__main__":
    main()
