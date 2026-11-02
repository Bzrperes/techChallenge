"""Reproduz o campeão e calcula seu limiar apenas com dados de treino."""

import argparse
import json
import platform
from pathlib import Path

import joblib
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_predict,
    cross_validate,
    train_test_split,
)

from src.config import (
    CV_SPLITS,
    META_RECALL,
    METRICS_PATH,
    MODEL_PATH,
    RANDOM_STATE,
    RF_PARAMS,
    TEST_SIZE,
)
from src.data import carregar_dados, separar_x_y
from src.preprocessing import montar_pipeline
from src.threshold import avaliar_top_percentual, selecionar_limiar_recall

CRITERIO_SELECAO = "Maior PR-AUC média na validação cruzada"
OBSERVACAO = (
    "O limiar foi ajustado com previsões out-of-fold do treino. "
    "O conjunto de teste já havia sido observado na Etapa 1."
)
SCORING = {
    "pr_auc": "average_precision",
    "roc_auc": "roc_auc",
    "recall": "recall",
    "precision": "precision",
    "f1": "f1",
}


def _metricas_probabilidades(y_true, probabilidades, limiar):
    previsoes = probabilidades >= limiar
    return {
        "pr_auc": float(average_precision_score(y_true, probabilidades)),
        "roc_auc": float(roc_auc_score(y_true, probabilidades)),
        "recall": float(recall_score(y_true, previsoes, zero_division=0)),
        "precision": float(precision_score(y_true, previsoes, zero_division=0)),
        "f1": float(f1_score(y_true, previsoes, zero_division=0)),
    }


def _medias_folds(resultados_cv):
    return {
        metrica: float(resultados_cv[f"test_{metrica}"].mean())
        for metrica in SCORING
    }


def criar_relatorio(
    resultados_cv,
    y_train,
    probabilidades_oof,
    y_test,
    probabilidades_teste,
    ponto_operacional,
):
    """Monta o contrato JSON compartilhado pelo treino e pelo notebook."""
    limiar = ponto_operacional["limiar"]
    metricas_oof = _metricas_probabilidades(y_train, probabilidades_oof, limiar)
    metricas_teste = _metricas_probabilidades(y_test, probabilidades_teste, limiar)
    top_20_oof = avaliar_top_percentual(y_train, probabilidades_oof)
    top_20_teste = avaliar_top_percentual(y_test, probabilidades_teste)
    return {
        "modelo": "Random Forest (GridSearchCV)",
        "criterio_selecao": CRITERIO_SELECAO,
        "meta_recall": META_RECALL,
        "limiar_operacional": limiar,
        "parametros": RF_PARAMS,
        "n_treino": int(len(y_train)),
        "n_teste": int(len(y_test)),
        "versoes": {
            "python": platform.python_version(),
            "scikit_learn": sklearn.__version__,
        },
        "observacao": OBSERVACAO,
        "taxas_alertas": {
            "oof_limiar": ponto_operacional["taxa_alertas"],
            "teste_limiar": float((probabilidades_teste >= limiar).mean()),
            "oof_top_20": top_20_oof["taxa_alertas"],
            "teste_top_20": top_20_teste["taxa_alertas"],
        },
        "metricas_media_folds": _medias_folds(resultados_cv),
        "metricas_oof_agregado": metricas_oof,
        "metricas_teste": metricas_teste,
        "top_20_oof": top_20_oof,
        "top_20_teste": top_20_teste,
    }


def treinar_campeao(df):
    X, y = separar_x_y(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )
    pipeline = montar_pipeline(RandomForestClassifier(**RF_PARAMS))
    cv = StratifiedKFold(
        n_splits=CV_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )
    resultados_cv = cross_validate(
        pipeline,
        X_train,
        y_train,
        cv=cv,
        scoring=SCORING,
        n_jobs=1,
    )
    probabilidades_oof = cross_val_predict(
        pipeline,
        X_train,
        y_train,
        cv=cv,
        method="predict_proba",
        n_jobs=1,
    )[:, 1]
    ponto_operacional = selecionar_limiar_recall(
        y_train,
        probabilidades_oof,
        meta_recall=META_RECALL,
    )

    pipeline.fit(X_train, y_train)
    probabilidades_teste = pipeline.predict_proba(X_test)[:, 1]
    relatorio = criar_relatorio(
        resultados_cv,
        y_train,
        probabilidades_oof,
        y_test,
        probabilidades_teste,
        ponto_operacional,
    )
    return pipeline, relatorio


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", type=Path, default=MODEL_PATH)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and not args.overwrite:
        parser.error("O modelo já existe. Para substituí-lo explicitamente, use --overwrite.")

    pipeline, relatorio = treinar_campeao(carregar_dados(args.data))
    artefato = {
        "pipeline": pipeline,
        "limiar": relatorio["limiar_operacional"],
        "meta_recall": relatorio["meta_recall"],
        "modelo": "Random Forest (Etapa 2)",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artefato, args.output)
    caminho_metricas = (
        METRICS_PATH
        if args.output.resolve() == MODEL_PATH.resolve()
        else args.output.with_suffix(".metrics.json")
    )
    caminho_metricas.write_text(
        json.dumps(relatorio, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(relatorio, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
