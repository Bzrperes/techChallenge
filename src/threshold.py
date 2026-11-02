"""Seleção reproduzível do limiar operacional de classificação."""

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score


def _validar_vetores(y_true, probabilidades):
    y_true = np.asarray(y_true)
    probabilidades = np.asarray(probabilidades, dtype=float)
    if y_true.ndim != 1 or probabilidades.ndim != 1:
        raise ValueError("y_true e probabilidades devem ser vetores unidimensionais.")
    if len(y_true) != len(probabilidades) or len(y_true) == 0:
        raise ValueError("y_true e probabilidades devem ter o mesmo tamanho não vazio.")
    if not np.isin(y_true, [0, 1]).all():
        raise ValueError("y_true deve conter somente 0 e 1.")
    if not np.isfinite(probabilidades).all() or (
        (probabilidades < 0) | (probabilidades > 1)
    ).any():
        raise ValueError("As probabilidades devem ser finitas e estar entre 0 e 1.")
    return y_true, probabilidades


def selecionar_limiar_recall(
    y_true,
    probabilidades,
    meta_recall: float = 0.80,
) -> dict[str, float]:
    """Maximiza precision entre os limiares que atingem a meta de recall.

    Esta função deve receber previsões out-of-fold do conjunto de treino, e
    não previsões obtidas nos mesmos dados usados para ajustar o estimador.
    """
    y_true, probabilidades = _validar_vetores(y_true, probabilidades)
    if not 0 < meta_recall <= 1:
        raise ValueError("meta_recall deve estar no intervalo (0, 1].")

    candidatos = []
    for limiar in np.unique(probabilidades):
        previsoes = probabilidades >= limiar
        recall = recall_score(y_true, previsoes, zero_division=0)
        if recall >= meta_recall:
            candidatos.append({
                "limiar": float(limiar),
                "recall": float(recall),
                "precision": float(precision_score(y_true, previsoes, zero_division=0)),
                "f1": float(f1_score(y_true, previsoes, zero_division=0)),
                "taxa_alertas": float(previsoes.mean()),
            })

    if not candidatos:
        raise ValueError("Nenhum limiar atingiu a meta de recall.")

    return max(
        candidatos,
        key=lambda item: (item["precision"], item["f1"], item["limiar"]),
    )


def avaliar_top_percentual(
    y_true,
    probabilidades,
    proporcao: float = 0.20,
) -> dict[str, float]:
    """Avalia o grupo com as maiores probabilidades sem ajustar o modelo."""
    y_true, probabilidades = _validar_vetores(y_true, probabilidades)
    if not 0 < proporcao < 1:
        raise ValueError("proporcao deve estar no intervalo (0, 1).")
    quantidade = max(1, int(np.ceil(len(probabilidades) * proporcao)))
    corte = float(np.partition(probabilidades, -quantidade)[-quantidade])
    previsoes = probabilidades >= corte
    return {
        "corte": corte,
        "recall": float(recall_score(y_true, previsoes, zero_division=0)),
        "precision": float(precision_score(y_true, previsoes, zero_division=0)),
        "f1": float(f1_score(y_true, previsoes, zero_division=0)),
        "taxa_alertas": float(previsoes.mean()),
    }
