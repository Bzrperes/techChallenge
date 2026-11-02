import joblib
import numpy as np
import pytest

from src.predict import carregar_artefato, prever
from src.threshold import avaliar_top_percentual, selecionar_limiar_recall


def test_carregar_artefato_rejeita_contrato_invalido(tmp_path):
    caminho = tmp_path / "invalido.joblib"
    joblib.dump({"pipeline": "não é pipeline"}, caminho)
    with pytest.raises(ValueError, match="artefato com Pipeline"):
        carregar_artefato(caminho)


def test_carregar_artefato_rejeita_limiar_invalido(modelo_path):
    artefato = joblib.load(modelo_path)
    artefato["limiar"] = 1.5
    joblib.dump(artefato, modelo_path)
    with pytest.raises(ValueError, match="entre 0 e 1"):
        carregar_artefato(modelo_path)


def test_prever_rejeita_colunas_ausentes(modelo_path):
    artefato = carregar_artefato(modelo_path)
    with pytest.raises(ValueError, match="Colunas ausentes"):
        prever(artefato, [{"Gender": "Female"}])


@pytest.mark.parametrize(
    "y,probabilidades,mensagem",
    [
        ([0, 1], [0.2], "mesmo tamanho"),
        ([[0, 1]], [0.2, 0.8], "unidimensionais"),
        ([0, 2], [0.2, 0.8], "somente 0 e 1"),
        ([0, 1], [0.2, np.nan], "finitas"),
    ],
)
def test_selecao_limiar_rejeita_vetores_invalidos(y, probabilidades, mensagem):
    with pytest.raises(ValueError, match=mensagem):
        selecionar_limiar_recall(y, probabilidades)


@pytest.mark.parametrize("meta", [0, 1.1])
def test_selecao_limiar_rejeita_meta_invalida(meta):
    with pytest.raises(ValueError, match="meta_recall"):
        selecionar_limiar_recall([0, 1], [0.2, 0.8], meta_recall=meta)


@pytest.mark.parametrize("proporcao", [0, 1])
def test_top_percentual_rejeita_proporcao_invalida(proporcao):
    with pytest.raises(ValueError, match="proporcao"):
        avaliar_top_percentual([0, 1], [0.2, 0.8], proporcao=proporcao)
