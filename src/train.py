"""Reproduz o campeão e calcula seu limiar apenas com dados de treino."""
import argparse
import json
import platform
from pathlib import Path
import joblib
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.metrics import average_precision_score, roc_auc_score, precision_score, recall_score, f1_score
from src.config import CV_SPLITS, META_RECALL, METRICS_PATH, MODEL_PATH, RF_PARAMS, RANDOM_STATE, TEST_SIZE
from src.data import carregar_dados, separar_x_y
from src.preprocessing import montar_pipeline
from src.threshold import selecionar_limiar_recall


def treinar_campeao(df):
    X, y = separar_x_y(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)
    pipeline = montar_pipeline(RandomForestClassifier(**RF_PARAMS))
    cv = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    probabilidades_oof = cross_val_predict(
        pipeline, X_train, y_train, cv=cv, method="predict_proba", n_jobs=1
    )[:, 1]
    ponto_operacional = selecionar_limiar_recall(
        y_train, probabilidades_oof, meta_recall=META_RECALL
    )

    pipeline.fit(X_train, y_train)
    p = pipeline.predict_proba(X_test)[:, 1]
    pred = p >= ponto_operacional["limiar"]
    metricas = {"average_precision": average_precision_score(y_test, p),
                "roc_auc": roc_auc_score(y_test, p),
                "recall": recall_score(y_test, pred, zero_division=0),
                "precision": precision_score(y_test, pred, zero_division=0),
                "f1": f1_score(y_test, pred, zero_division=0)}
    return pipeline, {"metricas_teste": metricas, "n_treino": len(X_train), "n_teste": len(X_test),
                      "python": platform.python_version(), "sklearn": sklearn.__version__,
                      "parametros": RF_PARAMS, "meta_recall": META_RECALL,
                      "limiar_operacional": ponto_operacional["limiar"],
                      "metricas_oof_limiar": ponto_operacional,
                      "nota": "Limiar ajustado com previsões out-of-fold do treino. Não é uma validação independente."}


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
    caminho_metricas.write_text(json.dumps(relatorio, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(relatorio, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
