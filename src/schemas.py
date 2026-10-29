"""Schemas Pydantic de entrada e saída da API."""
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

SimNao = Literal["Yes", "No"]
SimNaoSemInternet = Literal["Yes", "No", "No internet service"]

# snake_case da API -> nome da coluna usada no treino
MAPA_COLUNAS = {
    "gender": "Gender", "senior_citizen": "Senior Citizen", "partner": "Partner",
    "dependents": "Dependents", "tenure_months": "Tenure Months",
    "phone_service": "Phone Service", "multiple_lines": "Multiple Lines",
    "internet_service": "Internet Service", "online_security": "Online Security",
    "online_backup": "Online Backup", "device_protection": "Device Protection",
    "tech_support": "Tech Support", "streaming_tv": "Streaming TV",
    "streaming_movies": "Streaming Movies", "contract": "Contract",
    "paperless_billing": "Paperless Billing", "payment_method": "Payment Method",
    "monthly_charges": "Monthly Charges", "total_charges": "Total Charges",
}


class Cliente(BaseModel):
    """Dados de um cliente (valores categóricos iguais aos do dataset Telco IBM)."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, json_schema_extra={"example": {
        "gender": "Female", "senior_citizen": "No", "partner": "Yes", "dependents": "No",
        "tenure_months": 5, "phone_service": "Yes", "multiple_lines": "No",
        "internet_service": "Fiber optic", "online_security": "No", "online_backup": "No",
        "device_protection": "No", "tech_support": "No", "streaming_tv": "Yes",
        "streaming_movies": "Yes", "contract": "Month-to-month", "paperless_billing": "Yes",
        "payment_method": "Electronic check", "monthly_charges": 85.5, "total_charges": 420.0,
    }})

    gender: Literal["Male", "Female"]
    senior_citizen: SimNao
    partner: SimNao
    dependents: SimNao
    tenure_months: int = Field(ge=0, le=120)
    phone_service: SimNao
    multiple_lines: Literal["Yes", "No", "No phone service"]
    internet_service: Literal["DSL", "Fiber optic", "No"]
    online_security: SimNaoSemInternet
    online_backup: SimNaoSemInternet
    device_protection: SimNaoSemInternet
    tech_support: SimNaoSemInternet
    streaming_tv: SimNaoSemInternet
    streaming_movies: SimNaoSemInternet
    contract: Literal["Month-to-month", "One year", "Two year"]
    paperless_billing: SimNao
    payment_method: Literal["Mailed check", "Electronic check",
                            "Bank transfer (automatic)", "Credit card (automatic)"]
    monthly_charges: float = Field(ge=0)
    total_charges: float | None = Field(
        default=None, ge=0,
        description="Opcional. Ausência: zero para tenure=0; nos demais casos, mediana aprendida no treino.")

    def para_registro(self) -> dict:
        """Converte para os nomes de coluna usados no treino."""
        dados = self.model_dump()
        return {MAPA_COLUNAS[chave]: valor for chave, valor in dados.items()}


class Predicao(BaseModel):
    probabilidade_churn: float = Field(ge=0, le=1)
    churn_previsto: bool
    limiar: float
    modelo: str


class Saude(BaseModel):
    status: Literal["ok"]
    modelo_carregado: bool
