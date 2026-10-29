"""Verifica o modelo real e a API, sem retreiná-lo."""
import argparse
import json
import numpy as np
from fastapi.testclient import TestClient
from src.api import criar_app
from src.config import MODEL_PATH, FEATURES
from src.data import carregar_dados
from src.predict import carregar_artefato
from src.schemas import MAPA_COLUNAS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True)
    parser.add_argument('--model', default=str(MODEL_PATH))
    parser.add_argument('--n', type=int, default=20)
    args = parser.parse_args()
    if args.n < 1:
        parser.error('--n deve ser positivo')
    pipeline = carregar_artefato(args.model)['pipeline']
    X = carregar_dados(args.data)[FEATURES].head(args.n)
    X = X.loc[:, pipeline.feature_names_in_]
    esperado = pipeline.predict_proba(X)[:, 1]
    classes = pipeline.predict(X)
    inverso = {v: k for k, v in MAPA_COLUNAS.items()}
    with TestClient(criar_app(args.model)) as client:
        assert client.get('/health').json()['modelo_carregado']
        obtido, decisoes = [], []
        for registro in json.loads(X.to_json(orient='records')):
            payload = {inverso[k]: v for k, v in registro.items()}
            resposta = client.post('/predict', json=payload)
            assert resposta.status_code == 200, resposta.text
            obtido.append(resposta.json()['probabilidade_churn'])
            decisoes.append(resposta.json()['churn_previsto'])
    np.testing.assert_allclose(obtido, esperado, rtol=0, atol=1e-12)
    np.testing.assert_array_equal(decisoes, classes.astype(bool))
    print(f'OK: {len(X)} clientes; API e pipeline concordam. Modelo não foi retreinado.')


if __name__ == '__main__':
    main()
