import json
import sys

import joblib
import pytest

from src import train, verify
from src.config import MODEL_PATH


def test_main_train_com_saida_temporaria(
    monkeypatch,
    tmp_path,
    df_sintetico,
):
    dados = tmp_path / "dados.csv"
    saida = tmp_path / "modelo.joblib"
    df_sintetico.to_csv(dados, index=False)
    monkeypatch.setattr(
        sys,
        "argv",
        ["train", "--data", str(dados), "--output", str(saida)],
    )

    train.main()

    assert saida.exists()
    assert saida.with_suffix(".metrics.json").exists()
    assert joblib.load(saida)["limiar"] > 0
    relatorio = json.loads(saida.with_suffix(".metrics.json").read_text(encoding="utf-8"))
    assert relatorio["n_treino"] + relatorio["n_teste"] == len(df_sintetico)


def test_main_train_impede_sobrescrita(monkeypatch, tmp_path):
    dados = tmp_path / "dados.csv"
    saida = tmp_path / "modelo.joblib"
    dados.write_text("não deve ser lido", encoding="utf-8")
    saida.write_bytes(b"existente")
    monkeypatch.setattr(
        sys,
        "argv",
        ["train", "--data", str(dados), "--output", str(saida)],
    )
    with pytest.raises(SystemExit):
        train.main()


def test_main_verify_com_artefato_real(monkeypatch, capsys):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "verify",
            "--data",
            "data/processed/telco_tratado.csv",
            "--model",
            str(MODEL_PATH),
            "--n",
            "1",
        ],
    )

    verify.main()

    assert "OK: 1 clientes" in capsys.readouterr().out


def test_main_verify_rejeita_n_invalido(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["verify", "--data", "dados.csv", "--n", "0"],
    )
    with pytest.raises(SystemExit):
        verify.main()
