"""Configurações centrais do projeto (colunas, seeds e caminhos)."""
from pathlib import Path

RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_SPLITS = 5

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
MODEL_PATH = MODELS_DIR / "pipeline_churn_rf.joblib"
COMPARISON_PATH = MODELS_DIR / "comparacao_modelos.csv"
METRICS_PATH = MODELS_DIR / "metricas_teste.json"

# --- Colunas (mesmo contrato definido na EDA, Etapa 1) ---
ALVO = "Churn Value"  # 1 = cancelou, 0 = ficou
VAZAMENTO = ["Churn Label", "Churn Score", "CLTV", "Churn Reason"]
IDENT_GEO = ["CustomerID", "Count", "Country", "State", "City",
             "Zip Code", "Lat Long", "Latitude", "Longitude"]

NUMERICAS = ["Tenure Months", "Monthly Charges", "Total Charges"]
CATEGORICAS = ["Gender", "Senior Citizen", "Partner", "Dependents",
               "Phone Service", "Multiple Lines", "Internet Service",
               "Online Security", "Online Backup", "Device Protection",
               "Tech Support", "Streaming TV", "Streaming Movies",
               "Contract", "Paperless Billing", "Payment Method"]
FEATURES = NUMERICAS + CATEGORICAS

# Campeão registrado na Etapa 2; decisão original via predict().
LIMIAR = 0.5
RF_PARAMS = dict(n_estimators=100, max_depth=6, min_samples_leaf=10,
                 max_features=0.5, class_weight=None, random_state=42, n_jobs=1)
