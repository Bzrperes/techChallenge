import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.api import criar_app
from src.preprocessing import preparar_entrada
from src.schemas import Cliente


@pytest.fixture
def client(modelo_path):
    with TestClient(criar_app(modelo_path)) as client:
        yield client


def test_startup_carrega_arquivo(client):
    assert client.get("/health").json() == {"status": "ok", "modelo_carregado": True}


def test_api_equivale_pipeline(client, pipeline, payload_cliente):
    r = client.post("/predict", json=payload_cliente)
    assert r.status_code == 200
    X = preparar_entrada([Cliente(**payload_cliente).para_registro()])
    probabilidade = pipeline.predict_proba(X)[0, 1]
    assert r.json()["probabilidade_churn"] == pytest.approx(probabilidade, abs=1e-12)
    assert r.json()["churn_previsto"] == bool(probabilidade >= 0.30)
    assert r.json()["limiar"] == 0.30


@pytest.mark.parametrize("tenure", [0, 10])
def test_total_ausente_preserva_tratamento(client, payload_cliente, tenure):
    payload_cliente.pop("total_charges")
    payload_cliente["tenure_months"] = tenure
    registro = Cliente(**payload_cliente).para_registro()
    X = preparar_entrada([registro])
    assert X["Total Charges"].iloc[0] == 0 if tenure == 0 else np.isnan(X["Total Charges"].iloc[0])
    assert client.post("/predict", json=payload_cliente).status_code == 200


@pytest.mark.parametrize("campo,valor", [("contract", "Vitalício"), ("monthly_charges", -1), ("tenure_months", -1), ("extra", 1)])
def test_entrada_invalida(client, payload_cliente, campo, valor):
    payload_cliente[campo] = valor
    assert client.post("/predict", json=payload_cliente).status_code == 422


def test_campo_obrigatorio(client, payload_cliente):
    payload_cliente.pop("tenure_months")
    assert client.post("/predict", json=payload_cliente).status_code == 422


@pytest.mark.parametrize(
    "alteracoes",
    [
        {"phone_service": "No", "multiple_lines": "No"},
        {"phone_service": "Yes", "multiple_lines": "No phone service"},
        {"internet_service": "No", "online_security": "No"},
        {"internet_service": "DSL", "online_security": "No internet service"},
    ],
)
def test_rejeita_combinacoes_incoerentes(client, payload_cliente, alteracoes):
    payload_cliente.update(alteracoes)
    assert client.post("/predict", json=payload_cliente).status_code == 422


@pytest.mark.parametrize("corrompido", [False, True])
def test_indisponivel(tmp_path, payload_cliente, corrompido):
    path = tmp_path / "modelo.joblib"
    if corrompido:
        path.write_bytes(b"arquivo invalido")
    with TestClient(criar_app(path)) as client:
        assert client.get("/health").json()["modelo_carregado"] is False
        assert client.post("/predict", json=payload_cliente).status_code == 503
