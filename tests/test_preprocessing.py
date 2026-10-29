import numpy as np
import pandas as pd
import pytest

from src.config import ALVO, FEATURES
from src.data import limpar_features, separar_x_y
from src.preprocessing import montar_pipeline, preparar_entrada
from src.predict import carregar_artefato
from sklearn.linear_model import LogisticRegression


def test_limpar_features_trata_total_charges_em_branco(df_sintetico):
    df = df_sintetico.copy()
    df["Total Charges"] = df["Total Charges"].astype(object)
    df.loc[0, ["Tenure Months", "Total Charges"]] = [0, " "]
    df.loc[1, ["Tenure Months", "Total Charges"]] = [10, " "]  # branco inesperado: continua NaN

    limpo = limpar_features(df)

    assert pd.api.types.is_numeric_dtype(limpo["Total Charges"])
    assert limpo.loc[0, "Total Charges"] == 0.0      # cliente novo -> 0
    assert np.isnan(limpo.loc[1, "Total Charges"])   # fica para o imputer do pipeline
    assert df.loc[0, "Total Charges"] == " "         # original intacto


def test_limpar_features_remove_espacos_das_categorias(df_sintetico):
    df = df_sintetico.copy()
    df.loc[0, "Contract"] = " Month-to-month "
    assert limpar_features(df).loc[0, "Contract"] == "Month-to-month"


def test_preparar_entrada_exige_todas_as_colunas(df_sintetico):
    X, _ = separar_x_y(df_sintetico)
    with pytest.raises(ValueError, match="Colunas ausentes"):
        preparar_entrada(X.drop(columns=["Contract"]))


def test_pipeline_treina_e_gera_probabilidades_validas(df_sintetico):
    X, y = separar_x_y(df_sintetico)
    pipe = montar_pipeline(LogisticRegression(max_iter=500)).fit(X, y)
    proba = pipe.predict_proba(preparar_entrada(X.head(10)))[:, 1]
    assert proba.shape == (10,)
    assert ((proba >= 0) & (proba <= 1)).all()


def test_pipeline_ignora_categoria_desconhecida(df_sintetico):
    X, y = separar_x_y(df_sintetico)
    pipe = montar_pipeline(LogisticRegression(max_iter=500)).fit(X, y)
    novo = X.head(1).copy()
    novo["Payment Method"] = "Pix"
    assert 0 <= pipe.predict_proba(novo)[0, 1] <= 1


def test_modelo_salvo_preserva_previsoes(modelo_path, pipeline, df_sintetico):
    carregado = carregar_artefato(modelo_path)["pipeline"]
    X, _ = separar_x_y(df_sintetico)
    np.testing.assert_allclose(carregado.predict_proba(X), pipeline.predict_proba(X), atol=0, rtol=0)


def test_rejeita_modelo_divergente(tmp_path, df_sintetico):
    import joblib
    X, y = separar_x_y(df_sintetico)
    modelo = montar_pipeline(LogisticRegression(max_iter=500)).fit(X, y)
    path = tmp_path / "invalido.joblib"
    joblib.dump(modelo, path)
    with pytest.raises(ValueError, match="RandomForest"):
        carregar_artefato(path)


def test_configuracao_fiel_etapa2(pipeline):
    from src.config import RF_PARAMS
    parametros = pipeline.named_steps["modelo"].get_params()
    assert all(parametros[k] == v for k, v in RF_PARAMS.items())
