import pandas as pd
import pytest

from src.config import ALVO, FEATURES
from src.data import carregar_dados, separar_x_y


def test_carregar_dados_csv_temporario(tmp_path, df_sintetico):
    caminho = tmp_path / "dados.csv"
    df_sintetico.to_csv(caminho, index=False)

    carregado = carregar_dados(caminho)

    assert list(carregado.columns) == FEATURES + [ALVO]
    assert carregado.shape == df_sintetico.shape


def test_carregar_dados_rejeita_formato(tmp_path):
    caminho = tmp_path / "dados.json"
    caminho.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="Formato não suportado"):
        carregar_dados(caminho)


def test_carregar_dados_rejeita_colunas_ausentes(tmp_path):
    caminho = tmp_path / "incompleto.csv"
    pd.DataFrame({"Churn Value": [0]}).to_csv(caminho, index=False)
    with pytest.raises(ValueError, match="Colunas obrigatórias ausentes"):
        carregar_dados(caminho)


@pytest.mark.parametrize(
    "coluna,valor,mensagem",
    [
        (ALVO, 2, "Alvo deve conter somente 0 e 1"),
        ("Monthly Charges", -1, "Features numéricas não podem ser negativas"),
    ],
)
def test_separar_x_y_rejeita_dados_invalidos(
    df_sintetico,
    coluna,
    valor,
    mensagem,
):
    dados = df_sintetico.copy()
    dados.loc[0, coluna] = valor
    with pytest.raises(ValueError, match=mensagem):
        separar_x_y(dados)
