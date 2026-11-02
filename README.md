# Tech Challenge — Previsão de Churn com Scikit-Learn e FastAPI

Projeto acadêmico da Fase 1 do curso 12MLET (FIAP). A solução usa o dataset **Telco Customer Churn** para explorar os fatores associados ao cancelamento, comparar modelos de classificação e disponibilizar a propensão de churn por uma API REST.

## Status e entregas

| Etapa | Entrega | Status |
|---|---|---|
| 1 — Entendimento e preparação | EDA, métricas e baseline de Regressão Logística | Concluída |
| 2 — Modelagem e avaliação | Random Forest, MLP, validação cruzada e campeão | Concluída |
| 3 — Engenharia de software | Módulos em `src/`, testes e API FastAPI | Concluída |
| 4 — Documentação | README, ML Canvas e Model Card | Concluída |

Notebooks executados, com saídas visíveis e sem células com erro:

- [`01_entendimento_e_preparacao.ipynb`](./notebooks/01_entendimento_e_preparacao.ipynb): EDA e baseline;
- [`02_modelagem_avaliacao.ipynb`](./notebooks/02_modelagem_avaliacao.ipynb): comparação, escolha do campeão e ajuste do limiar.

## Estrutura

```text
techChallenge/
├── data/                           # Origem e tratamento dos dados
├── docs/                           # ML Canvas, Model Card e validações
├── models/
│   ├── pipeline_churn_rf.joblib    # Pipeline, limiar e metadados operacionais
│   ├── comparacao_modelos.csv      # Registro comparativo dos experimentos
│   └── metricas_teste.json         # Métricas, versões e limiar final
├── notebooks/
│   ├── 01_entendimento_e_preparacao.ipynb
│   ├── 02_modelagem_avaliacao.ipynb
│   └── artefatos/                  # Base tratada e baseline
├── src/                            # Código produtivo, treinamento e API
├── tests/                          # Testes automatizados com Pytest
├── pyproject.toml                  # Dependências diretas exatas
├── uv.lock                         # Lockfile completo e reproduzível
├── requirements.txt                # Exportação exata para alternativa com pip
└── README.md
```

O arquivo `.env`, ambientes virtuais, caches e dados brutos não são versionados.

## Reprodução do ambiente

### Opção recomendada: uv

Pré-requisitos: Git e [uv](https://docs.astral.sh/uv/getting-started/installation/). O projeto fixa **Python 3.13.15** e todas as dependências; o próprio uv pode instalar essa versão do Python.

```powershell
git clone https://github.com/Bzrperes/techChallenge.git
cd techChallenge
uv python install 3.13.15
uv sync --frozen
```

O `uv.lock` é a fonte de verdade do ambiente. `--frozen` impede que a instalação altere o lockfile silenciosamente.

### Alternativa com pip

Use exatamente Python 3.13.15. O `requirements.txt` é exportado do mesmo lockfile:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Dados e notebooks

O notebook da Etapa 1 tenta baixar o dataset público pelo `kagglehub`. Em condições normais, uma chave não é necessária para esse dataset. Se o Kaggle exigir autenticação ou aplicar limite de acesso, crie um `.env` local com as variáveis indicadas pela sua conta; nunca versione esse arquivo nem exponha os valores.

Para abrir os notebooks:

```powershell
uv run jupyter lab
```

Para reexecutá-los na ordem usada pelo projeto:

```powershell
uv run python -m nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=600 notebooks/01_entendimento_e_preparacao.ipynb
uv run python -m nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=1800 notebooks/02_modelagem_avaliacao.ipynb
```

A base original possui **7.043 registros e 33 colunas**, com taxa de churn de **26,54%**. Foram selecionadas 19 features (3 numéricas e 16 categóricas); identificadores, geografia, constantes e colunas que revelam o alvo foram excluídos. Imputação, padronização e codificação são ajustadas somente no treino, dentro dos pipelines.

## Métricas do projeto

### Métrica técnica e decisão do campeão

A **Average Precision**, chamada de PR-AUC no projeto, é a métrica técnica principal. Ela resume o compromisso entre precision e recall da classe positiva e é apropriada ao desbalanceamento observado. ROC-AUC, recall, precision e F1 são métricas complementares.

Todos os candidatos usam a mesma divisão estratificada, as mesmas features e validação cruzada com cinco folds. A Random Forest do `GridSearchCV` foi escolhida porque obteve a maior PR-AUC média:

| Modelo | PR-AUC média | ROC-AUC média | Recall a 0,5 |
|---|---:|---:|---:|
| Regressão Logística | 0,6788 | 0,8588 | 0,8094 |
| Random Forest | **0,6862** | **0,8627** | 0,5445 |
| MLPClassifier | 0,6782 | 0,8587 | 0,5472 |

A Regressão Logística realmente identifica mais churns no limiar padrão de 0,5, mas isso não a torna automaticamente o melhor modelo: recall depende do limiar. Para separar os objetivos, o projeto usa PR-AUC para escolher a capacidade geral de ordenação e ajusta o limiar do campeão somente com previsões *out-of-fold* do treino.

Entre os limiares com recall de validação de pelo menos 80%, foi escolhido o de maior precision (desempate por F1 e pelo maior limiar). O limiar final é **0,289843**. No teste, ele produziu PR-AUC **0,6656**, ROC-AUC **0,8537**, recall **0,8021**, precision **0,5376** e F1 **0,6438**. O teste não foi usado para selecionar o limiar.

Os valores completos estão em [`comparacao_modelos.csv`](./models/comparacao_modelos.csv) e [`metricas_teste.json`](./models/metricas_teste.json).

### Métrica de negócio

A métrica oficial de negócio é a **redução incremental da taxa de churn em 90 dias** entre os 20% de clientes com maior propensão. Em um futuro piloto, esses clientes seriam divididos aleatoriamente entre tratamento e controle:

**Redução incremental = taxa de churn do controle − taxa de churn do tratamento**

A meta inicial é reduzir o churn em pelo menos **3 pontos percentuais absolutos em 90 dias**, reportando intervalo de confiança de 95%. A escolha:

- conecta o modelo diretamente ao objetivo de retenção da operadora;
- mede o efeito da ação, não apenas a previsão;
- evita atribuir ao modelo cancelamentos que ocorreriam sem intervenção;
- pode ser compreendida e acompanhada pelo negócio;
- não exige inventar custos ou margens ausentes no dataset.

O projeto não executou uma campanha real. A meta é uma proposta para um piloto futuro, não um resultado alcançado.

Como métrica financeira futura, propõe-se o **ROI incremental de retenção**, quando houver dados de margem, ARPU, contato e ofertas:

**ROI = margem incremental líquida da campanha ÷ custo total da campanha**

A meta inicial é **ROI mínimo de 1,25 em 180 dias**, com recuperação do investimento em até 90 dias. Ela complementa, mas não substitui, a redução incremental de churn.

## Executar e testar

Na raiz do repositório:

```powershell
uv run python -m pytest -q
uv run python -m src.verify --data notebooks/artefatos/telco_tratado.csv
uv run uvicorn src.api:app --host 127.0.0.1 --port 8000
```

O verificador compara, para 20 clientes, as probabilidades e classes retornadas diretamente pelo artefato e pela API. Para verificar toda a base, acrescente `--n 7043`.

Com a API em execução, a documentação interativa fica em <http://127.0.0.1:8000/docs>. Em outro terminal PowerShell, teste os endpoints:

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

`GET /health` retorna o estado da aplicação. `POST /predict` retorna `probabilidade_churn`, `churn_previsto`, `limiar` e `modelo`. Entradas inválidas recebem 422; se o artefato estiver ausente ou incompatível, `/predict` recebe 503 e `/health` informa `modelo_carregado: false`.

## Reproduzir o artefato do campeão

O arquivo versionado já está pronto para a API. Para reproduzi-lo deliberadamente:

```powershell
uv run python -m src.train --data notebooks/artefatos/telco_tratado.csv --overwrite
```

O comando treina a configuração vencedora, calcula o limiar com previsões *out-of-fold* do treino e salva um dicionário com o pipeline, o limiar e os metadados. Também atualiza `models/metricas_teste.json`. O teste já foi observado durante o projeto e não deve ser tratado como uma nova amostra inédita.

## Documentação complementar

- [`ML_CANVAS.md`](./docs/ML_CANVAS.md): stakeholders, métricas e riscos de negócio;
- [`MODEL_CARD.md`](./docs/MODEL_CARD.md): performance, uso, limitações e possíveis vieses;
- [`ALTERACOES.md`](./docs/ALTERACOES.md): decisões técnicas da integração;
- [`VALIDACAO.md`](./docs/VALIDACAO.md): comandos e evidências da validação final.

O vídeo STAR é uma entrega separada e não está versionado neste repositório.
