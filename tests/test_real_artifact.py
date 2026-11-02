import json

import pytest

from src.config import METRICS_PATH, MODEL_PATH
from src.predict import carregar_artefato, prever
from src.schemas import Cliente

LIMIAR_ESPERADO = 0.2898431181242871
PROBABILIDADE_ESPERADA = 0.7080131434658807


def test_artefato_real_contrato_limiar_e_probabilidade(payload_cliente):
    artefato = carregar_artefato(MODEL_PATH)
    registro = Cliente(**payload_cliente).para_registro()

    resultado = prever(artefato, [registro])[0]

    assert artefato["limiar"] == LIMIAR_ESPERADO
    assert resultado["probabilidade_churn"] == pytest.approx(
        PROBABILIDADE_ESPERADA,
        abs=1e-15,
    )
    assert resultado["churn_previsto"] is True


def test_relatorio_do_treino_e_json_versionado_tem_as_mesmas_chaves(
    treino_sintetico,
):
    relatorio_gerado = treino_sintetico[1]
    relatorio_versionado = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    assert relatorio_gerado.keys() == relatorio_versionado.keys()
