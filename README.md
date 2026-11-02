# Tech Challenge — Previsão de Churn com Scikit-Learn e FastAPI

Projeto acadêmico da Fase 1 do curso 12MLET (FIAP). A solução explora o dataset **Telco Customer Churn**, compara modelos de classificação e disponibiliza a propensão de churn por uma API REST.

## Entregas

- [`01_entendimento_e_preparacao.ipynb`](./notebooks/01_entendimento_e_preparacao.ipynb): EDA e baseline de Regressão Logística;
- [`02_modelagem_avaliacao.ipynb`](./notebooks/02_modelagem_avaliacao.ipynb): Random Forest, MLP, validação cruzada, comparação e avaliação;
- [`src/`](./src): pré-processamento, treino, inferência e FastAPI;
- [`tests/`](./tests): testes automatizados;
- [`ML_CANVAS.md`](./docs/ML_CANVAS.md) e [`MODEL_CARD.md`](./docs/MODEL_CARD.md): contexto, resultados, limitações e possíveis vieses.

## Estrutura

```text
techChallenge/
├── data/
│   ├── README.md
│   └── processed/telco_tratado.csv
├── docs/
├── models/
│   ├── pipeline_churn_rf.joblib
│   ├── comparacao_modelos.csv
│   ├── experimentos_modelos.csv
│   └── metricas_teste.json
├── notebooks/
│   ├── 01_entendimento_e_preparacao.ipynb
│   ├── 02_modelagem_avaliacao.ipynb
│   └── artefatos/baseline_churn.joblib
├── src/
├── tests/
├── pyproject.toml
├── requirements.txt
└── uv.lock
```

Dados brutos, `.env`, ambientes virtuais, caches e arquivos de IDE não são versionados.

## Instalação reproduzível

### uv — recomendado

Requer [uv](https://docs.astral.sh/uv/getting-started/installation/) **0.11.33 ou superior**. Confira antes de instalar:

```powershell
uv --version
git clone https://github.com/Bzrperes/techChallenge.git
cd techChallenge
uv python install 3.13.15
uv sync --frozen
```

Os mesmos comandos funcionam em Linux e macOS. O projeto fixa Python 3.13.15; `uv.lock` é a fonte de verdade do ambiente e `--frozen` impede sua alteração silenciosa.

### pip no Windows

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m src.verify --data data/processed/telco_tratado.csv
.\.venv\Scripts\python.exe -m uvicorn src.api:app --host 127.0.0.1 --port 8000
```

### pip no Linux ou macOS

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest -q
python -m src.verify --data data/processed/telco_tratado.csv
python -m uvicorn src.api:app --host 127.0.0.1 --port 8000
```

## Dados e notebooks

O notebook da Etapa 1 baixa o dataset público pelo `kagglehub`. O download anônimo foi validado em cache vazio e sem `.env`. Se o Kaggle passar a exigir autenticação, use um `.env` apenas local e nunca versione ou exiba credenciais.

```powershell
uv run jupyter lab
uv run python -m nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=600 notebooks/01_entendimento_e_preparacao.ipynb
uv run python -m nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=2400 notebooks/02_modelagem_avaliacao.ipynb
```

A base original possui 7.043 registros e 33 colunas, com churn de 26,54%. O modelo usa 19 features. As 11 ausências de `Total Charges` pertencem a clientes com permanência zero e recebem zero por regra determinística; o imputador do pipeline protege apenas contra ausências inesperadas. Identificadores, geografia, constantes e variáveis que revelam o alvo são excluídos.

## Seleção e resultados

A Average Precision, denominada PR-AUC no projeto, é a métrica principal. Os três candidatos usam a mesma divisão estratificada, as mesmas features, cinco folds e seed 42.

| Modelo finalista | PR-AUC média | ROC-AUC média | Recall a 0,5 |
|---|---:|---:|---:|
| Regressão Logística | 0,6788 | 0,8588 | 0,8094 |
| Random Forest | **0,6862** | **0,8627** | 0,5445 |
| MLPClassifier | 0,6782 | 0,8587 | 0,5472 |

A vantagem da Random Forest é pequena e não demonstra superioridade estatística. O recall 0,8094 da Regressão Logística a 0,5 decorre do uso de `class_weight="balanced"`; a Random Forest usa `class_weight=None`. Portanto, esses recalls não representam o mesmo ponto operacional.

A Random Forest foi escolhida exclusivamente pela maior PR-AUC média. Depois, previsões *out-of-fold* (OOF) do treino definiram o limiar **0,2898431181242871**, maximizando precision entre pontos com recall mínimo de 80%.

| Métrica da Random Forest | Média dos folds | OOF agregado | Teste |
|---|---:|---:|---:|
| PR-AUC | 0,6862 | 0,6776 | 0,6656 |
| ROC-AUC | 0,8627 | 0,8609 | 0,8537 |
| Recall | 0,5445 | 0,8020 | 0,8021 |
| Precision | 0,6746 | 0,5445 | 0,5376 |
| F1 | 0,6022 | 0,6486 | 0,6438 |

PR-AUC e ROC-AUC OOF são calculadas sobre todas as previsões OOF agregadas; recall, precision e F1 OOF usam o limiar operacional. A coluna de média dos folds usa as métricas médias das cinco validações e, para as métricas dependentes de classe, o corte padrão de cada estimador.

### Dois pontos operacionais diferentes

- `churn_previsto`: triagem ampla pelo limiar 0,289843, com taxa de alertas de **39,08% no OOF** e **39,60% no teste**;
- piloto de negócio: os **20% de maior probabilidade**, com corte aproximado de **0,515347 no OOF** e **0,516267 no teste**. Nesse ponto, o recall foi **51,77% no OOF** e **51,34% no teste**.

O top 20% não substitui `churn_previsto`: ele define o orçamento de um futuro piloto controlado. Os resultados completos estão em [`comparacao_modelos.csv`](./models/comparacao_modelos.csv), [`experimentos_modelos.csv`](./models/experimentos_modelos.csv) e [`metricas_teste.json`](./models/metricas_teste.json).

## Métricas de negócio

A métrica oficial proposta é a **redução incremental do churn em 90 dias** no top 20%, medida por experimento aleatório entre tratamento e controle:

**Redução incremental = churn do controle − churn do tratamento**

A meta é reduzir o churn em 3 pontos percentuais absolutos em 90 dias, com intervalo de confiança de 95%. Como métrica financeira futura, propõe-se ROI incremental de retenção mínimo de **1,25 em 180 dias**, com recuperação do investimento em até 90 dias. Nenhuma dessas metas foi medida neste projeto.

## Testes, verificador e API

```powershell
uv run python -m pytest -q
uv run python -m src.verify --data data/processed/telco_tratado.csv
uv run python -m src.verify --data data/processed/telco_tratado.csv --n 7043
uv run uvicorn src.api:app --host 127.0.0.1 --port 8000
```

Com a API ativa, abra <http://127.0.0.1:8000/docs>. Em outro PowerShell:

```powershell
Invoke-RestMethod -Method Get -Uri 'http://127.0.0.1:8000/health'

$payload = @{
  gender = 'Female'; senior_citizen = 'No'; partner = 'Yes'; dependents = 'No'
  tenure_months = 5; phone_service = 'Yes'; multiple_lines = 'No'
  internet_service = 'Fiber optic'; online_security = 'No'; online_backup = 'No'
  device_protection = 'No'; tech_support = 'No'; streaming_tv = 'Yes'
  streaming_movies = 'Yes'; contract = 'Month-to-month'; paperless_billing = 'Yes'
  payment_method = 'Electronic check'; monthly_charges = 85.5; total_charges = 420.0
} | ConvertTo-Json

Invoke-RestMethod -Method Post -Uri 'http://127.0.0.1:8000/predict' `
  -ContentType 'application/json' -Body $payload
```

Em Linux ou macOS:

```bash
curl -s http://127.0.0.1:8000/health
curl -s -X POST http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"gender":"Female","senior_citizen":"No","partner":"Yes","dependents":"No","tenure_months":5,"phone_service":"Yes","multiple_lines":"No","internet_service":"Fiber optic","online_security":"No","online_backup":"No","device_protection":"No","tech_support":"No","streaming_tv":"Yes","streaming_movies":"Yes","contract":"Month-to-month","paperless_billing":"Yes","payment_method":"Electronic check","monthly_charges":85.5,"total_charges":420.0}'
```

`GET /health` informa a carga do modelo. `POST /predict` retorna probabilidade, classe, limiar e modelo. Combinações incoerentes de telefone ou internet recebem 422; modelo ausente ou inválido produz 503 em `/predict`.

## Reprodução deliberada do campeão

O artefato versionado já está pronto. Para reproduzi-lo:

```powershell
uv run python -m src.train --data data/processed/telco_tratado.csv --overwrite
```

O comando reproduz a configuração vencedora, o limiar OOF e o mesmo esquema de `models/metricas_teste.json`. O teste já havia sido observado durante o projeto e não constitui amostra inédita.

## Documentação

- [`ML_CANVAS.md`](./docs/ML_CANVAS.md): stakeholders, métricas e riscos;
- [`MODEL_CARD.md`](./docs/MODEL_CARD.md): performance, calibração, grupos, limitações e vieses;
- [`VALIDACAO.md`](./docs/VALIDACAO.md): comandos e evidências de reprodução.

O vídeo STAR é uma entrega separada e não está versionado neste repositório.
