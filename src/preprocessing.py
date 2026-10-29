"""Pré-processamento: ColumnTransformer + pipeline do Scikit-Learn."""
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import CATEGORICAS, FEATURES, NUMERICAS
from src.data import limpar_features


def criar_preprocessador() -> ColumnTransformer:
    """Cria um pré-processador novo a cada chamada (sem objetos compartilhados)."""
    return ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]), NUMERICAS),
            ("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", drop="if_binary")),
            ]), CATEGORICAS),
        ],
        remainder="drop",
    )


def montar_pipeline(modelo: BaseEstimator) -> Pipeline:
    """Encadeia pré-processamento + estimador final."""
    return Pipeline([("prep", criar_preprocessador()), ("modelo", modelo)])


def preparar_entrada(registros: list[dict] | pd.DataFrame) -> pd.DataFrame:
    """Converte registros brutos (ex.: vindos da API) no DataFrame esperado pelo pipeline.

    Garante a ordem/colunas de ``FEATURES`` e aplica a mesma limpeza do treino.
    Levanta ``ValueError`` se faltar alguma coluna.
    """
    df = pd.DataFrame(registros) if not isinstance(registros, pd.DataFrame) else registros
    faltando = [c for c in FEATURES if c not in df.columns]
    if faltando:
        raise ValueError(f"Colunas ausentes na entrada: {faltando}")
    return limpar_features(df[FEATURES])
