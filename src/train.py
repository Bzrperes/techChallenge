"""Reproduz apenas o campeão; não realiza nova seleção ou ajuste de limiar.
Preferir o .joblib original. Use este comando apenas se ele não estiver disponível.
"""
import argparse
import json
import platform
from pathlib import Path
import joblib
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score, roc_auc_score, precision_score, recall_score, f1_score
from src.config import MODEL_PATH, RF_PARAMS, RANDOM_STATE, TEST_SIZE
from src.data import carregar_dados, separar_x_y
from src.preprocessing import montar_pipeline


def treinar_campeao(df):
    X, y = separar_x_y(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)
    pipeline = montar_pipeline(RandomForestClassifier(**RF_PARAMS)).fit(X_train, y_train)
    p = pipeline.predict_proba(X_test)[:, 1]
    pred = pipeline.predict(X_test)
    metricas = {"average_precision": average_precision_score(y_test, p),
                "roc_auc": roc_auc_score(y_test, p),
                "recall": recall_score(y_test, pred, zero_division=0),
                "precision": precision_score(y_test, pred, zero_division=0),
                "f1": f1_score(y_test, pred, zero_division=0)}
    return pipeline, {"metricas_teste": metricas, "n_treino": len(X_train), "n_teste": len(X_test),
                      "python": platform.python_version(), "sklearn": sklearn.__version__,
                      "parametros": RF_PARAMS, "limiar": 0.5,
                      "nota": "Reprodução da configuração da Etapa 2. Não é uma validação independente."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", type=Path, default=MODEL_PATH)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and not args.overwrite:
        parser.error("O modelo já existe. Para substituí-lo explicitamente, use --overwrite.")
    pipeline, relatorio = treinar_campeao(carregar_dados(args.data))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, args.output)
    args.output.with_suffix(".metrics.json").write_text(json.dumps(relatorio, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(relatorio, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
