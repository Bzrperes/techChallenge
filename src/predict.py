"""Inferência com o Pipeline original da Etapa 2, sem retreinamento."""
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.validation import check_is_fitted
from src.config import MODEL_PATH, FEATURES, LIMIAR, RF_PARAMS
from src.preprocessing import preparar_entrada


def carregar_artefato(caminho: str | Path = MODEL_PATH) -> dict:
    """Carrega somente arquivos locais confiáveis produzidos pelo grupo."""
    pipeline = joblib.load(Path(caminho))
    if not isinstance(pipeline, Pipeline):
        raise ValueError("Esperado Pipeline da Etapa 2, salvo diretamente com joblib.dump.")
    modelo = pipeline.named_steps.get("modelo")
    if not isinstance(modelo, RandomForestClassifier):
        raise ValueError("O campeão esperado é RandomForestClassifier.")
    check_is_fitted(pipeline)
    divergentes = [k for k, v in RF_PARAMS.items()
                   if k != "n_jobs" and modelo.get_params().get(k) != v]
    if divergentes:
        raise ValueError(f"Configuração diferente do campeão da Etapa 2: {divergentes}")
    if set(pipeline.feature_names_in_) != set(FEATURES):
        raise ValueError("Colunas do modelo incompatíveis com as 19 features do projeto.")
    if not np.array_equal(modelo.classes_, [0, 1]):
        raise ValueError("Classes esperadas: [0, 1].")
    return {"pipeline": pipeline, "limiar": LIMIAR, "modelo": "Random Forest (Etapa 2)"}


def prever(artefato: dict, registros: list[dict] | pd.DataFrame) -> list[dict]:
    X = preparar_entrada(registros)
    # Preserva inclusive a ordem registrada no notebook.
    pipeline = artefato["pipeline"]
    X = X.loc[:, pipeline.feature_names_in_]
    probabilidades = pipeline.predict_proba(X)[:, 1]
    # predict preserva o comportamento do notebook, inclusive empate em 0.5.
    classes = pipeline.predict(X)
    return [{"probabilidade_churn": float(p), "churn_previsto": bool(c)}
            for p, c in zip(probabilidades, classes)]
