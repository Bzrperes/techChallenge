"""Fixtures compartilhadas: dados sintéticos (sem depender do Kaggle) e modelo mínimo."""
import atexit
import shutil
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd
import pytest

from src.config import ALVO, RANDOM_STATE
from src.train import treinar_campeao

_PASTA_TEMPORARIA: Path | None = None


def _remover_temporarios(raiz: Path) -> None:
    """Tenta remover todos os temporários sem mascarar o resultado dos testes."""
    shutil.rmtree(raiz, ignore_errors=True)


def pytest_configure(config):
    """Isola arquivos temporários por sessão para evitar bloqueios no Windows."""
    global _PASTA_TEMPORARIA
    raiz_temporaria = Path(".pytest_runs").resolve()
    _remover_temporarios(raiz_temporaria)
    raiz_temporaria.mkdir(exist_ok=True)
    _PASTA_TEMPORARIA = raiz_temporaria / uuid4().hex
    config.option.basetemp = str(_PASTA_TEMPORARIA)
    atexit.register(_remover_temporarios, raiz_temporaria)


def pytest_sessionfinish(session, exitstatus):
    """Remove o temporário da sessão, tolerando arquivos bloqueados no Windows."""
    if _PASTA_TEMPORARIA is not None:
        shutil.rmtree(_PASTA_TEMPORARIA, ignore_errors=True)


def gerar_dados_sinteticos(n: int = 400, seed: int = RANDOM_STATE) -> pd.DataFrame:
    """Gera um DataFrame com o mesmo esquema do dataset Telco IBM."""
    rng = np.random.default_rng(seed)
    sn = ["Yes", "No"]
    sni = ["Yes", "No", "No internet service"]
    contrato = rng.choice(["Month-to-month", "One year", "Two year"], n, p=[0.55, 0.25, 0.20])
    tenure = rng.integers(0, 72, n)
    mensal = rng.uniform(20, 115, n).round(2)
    df = pd.DataFrame({
        "Tenure Months": tenure,
        "Monthly Charges": mensal,
        "Total Charges": (tenure * mensal).round(2),
        "Gender": rng.choice(["Male", "Female"], n),
        "Senior Citizen": rng.choice(sn, n),
        "Partner": rng.choice(sn, n),
        "Dependents": rng.choice(sn, n),
        "Phone Service": rng.choice(sn, n),
        "Multiple Lines": rng.choice(["Yes", "No", "No phone service"], n),
        "Internet Service": rng.choice(["DSL", "Fiber optic", "No"], n),
        "Online Security": rng.choice(sni, n),
        "Online Backup": rng.choice(sni, n),
        "Device Protection": rng.choice(sni, n),
        "Tech Support": rng.choice(sni, n),
        "Streaming TV": rng.choice(sni, n),
        "Streaming Movies": rng.choice(sni, n),
        "Contract": contrato,
        "Paperless Billing": rng.choice(sn, n),
        "Payment Method": rng.choice(["Mailed check", "Electronic check",
                                      "Bank transfer (automatic)", "Credit card (automatic)"], n),
    })
    # churn mais provável com contrato mensal e pouco tempo de casa
    risco = 0.15 + 0.35 * (contrato == "Month-to-month") + 0.25 * (tenure < 12)
    df[ALVO] = (rng.uniform(size=n) < risco).astype(int)
    return df


@pytest.fixture(scope="session")
def df_sintetico() -> pd.DataFrame:
    return gerar_dados_sinteticos()


@pytest.fixture(scope="session")
def treino_sintetico(df_sintetico):
    return treinar_campeao(df_sintetico)


@pytest.fixture(scope="session")
def pipeline(treino_sintetico):
    return treino_sintetico[0]


@pytest.fixture
def modelo_path(tmp_path, pipeline):
    import joblib
    path = tmp_path / "pipeline_churn_rf.joblib"
    joblib.dump({
        "pipeline": pipeline,
        "limiar": 0.30,
        "meta_recall": 0.80,
        "modelo": "Random Forest (teste)",
    }, path)
    return path


@pytest.fixture
def payload_cliente() -> dict:
    return {
        "gender": "Female", "senior_citizen": "No", "partner": "Yes", "dependents": "No",
        "tenure_months": 5, "phone_service": "Yes", "multiple_lines": "No",
        "internet_service": "Fiber optic", "online_security": "No", "online_backup": "No",
        "device_protection": "No", "tech_support": "No", "streaming_tv": "Yes",
        "streaming_movies": "Yes", "contract": "Month-to-month", "paperless_billing": "Yes",
        "payment_method": "Electronic check", "monthly_charges": 85.5, "total_charges": 420.0,
    }
