# Validação final

Validação realizada em 2 de novembro de 2026 com Python 3.13.15 e uv 0.12.13, seguindo os comandos documentados no README. O ambiente local padrão `.venv` estava bloqueado por um processo do Windows; por isso, a validação de desenvolvimento usou `.venv-audit`, criado do mesmo `uv.lock` com `UV_PROJECT_ENVIRONMENT=.venv-audit`. A reprodução limpa descrita ao final usa um clone temporário e um ambiente novo.

## Ambiente isolado

```powershell
uv --version
uv python install 3.13.15
uv sync --frozen
```

O lockfile selecionou Python 3.13.15 e instalou 132 pacotes sem alterar `pyproject.toml`, `uv.lock` ou `requirements.txt`.

## Testes automatizados e qualidade

```powershell
uv run python -m pytest -q
uv run --with pytest-cov python -m pytest -q --cov=src --cov-report=term-missing
uv run --with ruff ruff check --select F,I src tests
```

Resultados no projeto: 46 testes aprovados em duas execuções consecutivas, um aviso esperado de categoria desconhecida tratada por `OneHotEncoder(handle_unknown="ignore")`, cobertura total de 97% em `src/` e Ruff sem achados. O Pytest usa artefatos sintéticos isolados nos testes de treino, API e caminhos de erro, e também carrega diretamente o artefato real `models/pipeline_churn_rf.joblib`, sem retreiná-lo, para verificar contrato, limiar e probabilidade do exemplo do README.

A fixture de sessão usa uma pasta exclusiva dentro de `.pytest_runs/` e tenta removê-la no encerramento, com tolerância às travas de arquivos do Windows. Duas execuções consecutivas não acumularam novas pastas; uma pasta antiga, já bloqueada por outro processo local, não pôde ser removida nessa máquina e não faz parte do Git.

## Modelo, dados e relatório

```powershell
uv run python -m src.verify --data data/processed/telco_tratado.csv
uv run python -m src.verify --data data/processed/telco_tratado.csv --n 7043
uv run python -m src.train --data data/processed/telco_tratado.csv --output CAMINHO_TEMPORARIO
```

Resultados: o verificador confirmou a concordância entre API e pipeline para 20 e para 7.043 clientes. O treino dirigido a um caminho temporário gerou relatório com as mesmas 15 chaves e os mesmos valores de `models/metricas_teste.json`.

O CSV versionado no Git, com fim de linha LF, tem SHA-256 `d8ed77b4ad0bd8dd15deff8b748d921a9e98af99f7800ef10986d0e6cb41b5a5`. O SHA-256 `31d61b29480418d7aa7bf20f27d16dfacc87587ef941211e7a035afc497f695c` corresponde à cópia de trabalho no Windows, com CRLF. A mudança de pasta foi registrada pelo Git como uma renomeação 100%, sem alteração do conteúdo versionado. Os dois arquivos Joblib permaneceram byte a byte idênticos após a reexecução:

- `models/pipeline_churn_rf.joblib`: `5f0bcde82a78429cc6bae3008a83e2515313285435ce4d3f9d1db1c4a483c60a`;
- `notebooks/artefatos/baseline_churn.joblib`: `965e097bd291ceb086e021cfd65432d97ed85f91f6c1aa313d1579fe74cfc963`.

Não foi necessário recorrer apenas à equivalência funcional dos Joblib, pois os binários não mudaram.

## API em execução

A porta 8000 estava ocupada, então a API real foi validada nas portas 8767 e 8768:

- `GET /health`: HTTP 200, com modelo carregado;
- `POST /predict` com o payload do README: HTTP 200, probabilidade `0.7080131434658807` e limiar `0.2898431181242871`;
- combinação incoerente de telefonia: HTTP 422;
- `POST /predict` em instância sem artefato: HTTP 503.

## Notebooks

Os notebooks foram executados na ordem documentada:

```powershell
uv run python -m nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=600 notebooks/01_entendimento_e_preparacao.ipynb
uv run python -m nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=2400 notebooks/02_modelagem_avaliacao.ipynb
```

Resultados:

- notebook 1: 31 células no total, 23 de código, contagens sequenciais de 1 a 23 e nenhuma célula com erro;
- notebook 2: 51 células no total, 37 de código, contagens sequenciais de 1 a 37 e nenhuma célula com erro;
- nenhuma saída contém caminho local ou nome de usuário;
- não foram encontradas células duplicadas;
- o aviso de não convergência aparece apenas na configuração inicial da MLP; a configuração comparada usa parada antecipada;
- o limiar, as divisões e todas as métricas obrigatórias foram reproduzidos sem alteração.

## Download real do Kaggle

Em 2 de novembro de 2026, o notebook 1 foi executado pelo comando do README em um clone temporário fora do repositório. O clone não continha `.env`; as variáveis de credencial do processo foram removidas sem leitura dos respectivos valores; e `KAGGLEHUB_CACHE` apontou para uma pasta temporária nova. O download anônimo e a execução terminaram com sucesso, sem necessidade de credencial ou da alternativa com `.env`.

## Resultado geral

No clone local limpo do commit `fa862b0`, foram executados `uv sync --frozen`, os 46 testes duas vezes, o verificador com 20 clientes, `src.train` com saída temporária e os dois notebooks na ordem documentada. O relatório temporário de treino apresentou as mesmas chaves e os mesmos valores do JSON versionado; os notebooks terminaram sem erro e reproduziram os mesmos resultados, desconsiderando somente timestamps, identificadores HTML aleatórios do pandas e endereços voláteis de objetos.
