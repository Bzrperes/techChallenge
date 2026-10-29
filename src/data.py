"""Carga e limpeza determinística dos dados (sem aprender nada dos dados)."""
from pathlib import Path

import pandas as pd

from src.config import ALVO, CATEGORICAS, FEATURES


def limpar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica as regras de limpeza da Etapa 1 às colunas de entrada.

    - ``Total Charges`` vira número (textos em branco viram NaN);
    - tenure = 0 (cliente novo, ainda não cobrado) => ``Total Charges`` = 0;
    - remove espaços extras nas categorias.

    Retorna uma cópia; o DataFrame original não é alterado.
    """
    df = df.copy()
    df["Total Charges"] = pd.to_numeric(df["Total Charges"], errors="coerce")
    cliente_novo = df["Tenure Months"].eq(0)
    df.loc[df["Total Charges"].isna() & cliente_novo, "Total Charges"] = 0.0
    for coluna in CATEGORICAS:
        df[coluna] = df[coluna].map(lambda valor: valor.strip() if isinstance(valor, str) else valor)
    return df


def carregar_dados(caminho: str | Path) -> pd.DataFrame:
    """Lê .csv ou .xlsx (bruto ou já tratado) e devolve FEATURES + alvo limpos."""
    caminho = Path(caminho)
    if caminho.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(caminho)
    else:
        df = pd.read_csv(caminho)
    df = limpar_features(df)
    return df[FEATURES + [ALVO]].copy()


def separar_x_y(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separa a matriz de features e o alvo inteiro (0/1)."""
    if df[ALVO].isna().any() or not df[ALVO].isin([0, 1]).all():
        raise ValueError("Alvo deve conter somente 0 e 1, sem ausências.")
    if (df[["Tenure Months", "Monthly Charges", "Total Charges"]] < 0).any().any():
        raise ValueError("Features numéricas não podem ser negativas.")
    return df[FEATURES].copy(), df[ALVO].astype(int).copy()
