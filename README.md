# Tech Challenge — Pipeline Preditivo de Churn com Modelagem Clássica e API REST

Este projeto utiliza o dataset **Telco Customer Churn**, disponibilizado no Kaggle, para analisar os fatores relacionados ao cancelamento de clientes, treinar modelos de classificação com Scikit-Learn e servir o modelo campeão por meio de uma API FastAPI.

## Status do projeto

| Etapa | Conteúdo | Situação |
|---|---|---|
| 1 — Entendimento e preparação | EDA, definição de métricas e baseline de Regressão Logística | Concluída |
| 2 — Modelagem e avaliação | Árvore/Random Forest, MLP, validação cruzada e escolha do campeão | Concluída |
| 3 — Engenharia de software e API | Refatoração para `src/`, testes com pytest e API FastAPI | Concluída |
| 4 — Documentação e entrega final | Model Card, README final e vídeo STAR | Em andamento |

Notebooks: [`TechChallenge.ipynb`](./notebooks/TechChallenge.ipynb) (EDA e baseline) e [`02_modelagem_avaliacao.ipynb`](./notebooks/02_modelagem_avaliacao.ipynb) (modelagem e avaliação).

## Estrutura do projeto

```text
techChallenge-grupo/
├── data/                  # Orientações e futuros artefatos de dados
├── docs/                  # ML Canvas, Model Card e documentação técnica
│   ├── ALTERACOES.md      # Decisões e rastreabilidade da Etapa 3
│   └── VALIDACAO.md       # Verificações executadas e pendências
├── models/
│   └── pipeline_churn_rf.joblib   # Pipeline do modelo campeão usado pela API
├── notebooks/
│   ├── TechChallenge.ipynb        # EDA e baseline
│   ├── 02_modelagem_avaliacao.ipynb
│   └── artefatos/                 # Saídas geradas pelos notebooks (ex.: telco_tratado.csv)
├── src/                   # Código produtivo do pipeline e da API
│   ├── api.py             # Aplicação FastAPI (/health e /predict)
│   ├── config.py          # Configurações e constantes
│   ├── data.py            # Carga e limpeza dos dados
│   ├── preprocessing.py   # Pré-processamento
│   ├── predict.py         # Carga do modelo e geração de predições
│   ├── schemas.py         # Esquemas de entrada e saída (Pydantic)
│   ├── train.py           # Reprodução do treino do campeão
│   └── verify.py          # Verificação API x pipeline
├── tests/                 # Testes automatizados (pytest)
├── .gitignore
├── pytest.ini
├── README.md
└── requirements.txt
```

O arquivo `.env`, o ambiente virtual `.venv` e arquivos temporários não são publicados no GitHub. O dataset bruto é obtido diretamente do Kaggle para que o download seja reproduzível e não seja necessário versionar uma cópia local.

## Configuração do ambiente

### Pré-requisitos

- Python 3.11 ou superior.
- `pip` disponível no mesmo interpretador que executará o notebook e a API.
- Credencial válida do Kaggle configurada no ambiente local (necessária apenas para baixar o dataset nos notebooks).

### Configuração da credencial do Kaggle

Cada integrante deverá receber a chave da API por um canal seguro e criar manualmente um arquivo chamado `.env` na raiz do projeto:

```dotenv
KAGGLE_API_TOKEN=cole_aqui_a_chave_recebida
```

O `.env` está listado no `.gitignore` e não deve ser adicionado ao Git nem compartilhado no repositório. O nome da variável pode ser documentado; somente o valor da chave deve permanecer restrito aos integrantes autorizados.

### Instalação das dependências

Execute os comandos a partir da pasta raiz do projeto. A utilização de um ambiente virtual evita conflitos com outros projetos.

```bash
python -m venv .venv
```

Ative o ambiente virtual no Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

No Linux ou macOS, utilize:

```bash
source .venv/bin/activate
```

Com o ambiente ativado, instale todos os pacotes declarados no arquivo [`requirements.txt`](./requirements.txt):

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

O uso de `python -m pip` garante que os pacotes sejam instalados no mesmo interpretador Python utilizado para executar o notebook e a API.

**Sobre versões:** para carregar o modelo original, priorize as mesmas versões de Python, Scikit-Learn, NumPy, SciPy e joblib usadas para salvá-lo. O `requirements.txt` contém faixas de compatibilidade e não é um lockfile comprovado do ambiente do grupo. Depois de validar no ambiente original, é possível registrá-lo com `python -m pip freeze > requirements-lock.txt`. Avisos de incompatibilidade de versão do modelo não devem ser tratados como validação concluída.

### Dependências do projeto

| Pacote | Finalidade |
|---|---|
| `kagglehub` | Autenticação no Kaggle e download do dataset. |
| `python-dotenv` | Carregamento das credenciais armazenadas localmente no arquivo `.env`. |
| `numpy` | Operações numéricas utilizadas durante a EDA e a modelagem. |
| `pandas` | Leitura, tratamento e análise dos dados tabulares. |
| `openpyxl` | Mecanismo utilizado pelo Pandas para ler o arquivo Excel do dataset. |
| `matplotlib` | Construção dos gráficos básicos da análise exploratória. |
| `seaborn` | Visualizações estatísticas das distribuições e relações com churn. |
| `scikit-learn` | Pré-processamento, baselines, métricas e modelos de Machine Learning. |
| `joblib` | Serialização e carga do pipeline treinado. |
| `fastapi` | Framework da API de inferência. |
| `uvicorn` | Servidor ASGI para executar a API. |
| `pytest` | Testes automatizados. |
| `ipykernel` | Kernel Python utilizado para executar o notebook no VS Code ou Jupyter. |
| `jupyterlab` | Ambiente independente para abrir e executar arquivos `.ipynb`. |

Para iniciar o JupyterLab depois da instalação:

```bash
python -m jupyterlab
```

---

# Etapa 1 — Análise Exploratória de Dados (EDA)

As células 1 e 2 do notebook `TechChallenge.ipynb` são responsáveis pela autenticação no Kaggle e pelo carregamento do dataset. A seguir estão os resumos das células de análise exploratória, da 3 à 13.

## Célula 3 — Definição das variáveis

**Variáveis analisadas:** todas as 33 colunas do dataset.

**O que apresenta:** separa as colunas em alvo, numéricas, categóricas, identificação/geografia e possíveis vazamentos de dados.

**Principais conclusões:**

- O alvo do modelo é `Churn Value`.
- Foram selecionadas 3 variáveis numéricas e 16 categóricas.
- `Churn Label`, `Churn Score`, `CLTV` e `Churn Reason` não devem entrar no baseline por apresentarem risco de vazamento de informação.

## Célula 4 — Estrutura e qualidade inicial

**Variáveis analisadas:** todas as colunas.

**O que apresenta:** formato da base, tipos de dados, valores ausentes, quantidade de valores únicos, duplicidades e colunas constantes.

**Principais conclusões:**

- A base possui 7.043 clientes e 33 colunas.
- Não há linhas duplicadas nem valores duplicados em `CustomerID`.
- `Count`, `Country` e `State` são constantes.
- `Churn Reason` possui muitos valores ausentes porque só é preenchida para clientes que cancelaram.

## Célula 5 — Validação de `Total Charges`

**Variáveis analisadas:** `Total Charges`, `Tenure Months`, `Monthly Charges` e `Churn Value`.

**O que apresenta:** converte `Total Charges` para o formato numérico, identifica valores inválidos e compara o total informado com a estimativa `Tenure Months × Monthly Charges`.

**Principais conclusões:**

- Existem 11 valores ausentes em `Total Charges`.
- Esses registros pertencem a clientes com zero meses de permanência.
- O desvio mediano entre o total informado e o valor estimado é de aproximadamente 2%.

## Célula 6 — Volume e papel das variáveis

**Variáveis analisadas:** todas as colunas, com destaque para `Churn Value` e as features candidatas.

**O que apresenta:** volume da base, memória utilizada, cardinalidade, valores ausentes e função de cada coluna no projeto.

**Principais conclusões:**

- Existem 19 features candidatas ao baseline: 3 numéricas e 16 categóricas.
- A taxa de churn é de 26,54%.
- A distribuição do alvo apresenta desbalanceamento moderado.

## Célula 7 — Qualidade, cardinalidade e outliers

**Variáveis analisadas:** todas as colunas; os outliers são verificados em `Tenure Months`, `Monthly Charges` e `Total Charges`.

**O que apresenta:** duplicidades, ausências, colunas constantes, maiores cardinalidades e quantidade de outliers pelo método do intervalo interquartil, ou IQR.

**Principais conclusões:**

- Não existem linhas ou identificadores duplicados.
- Existem 11 ausências em `Total Charges` e três colunas constantes.
- As três variáveis numéricas não apresentam outliers pelo critério do IQR.
- Colunas de identificação e geografia possuem alta cardinalidade e não devem entrar no baseline inicial.

## Célula 8 — Distribuição da variável alvo

**Variável analisada:** `Churn Value`.

**O que apresenta:** quantidade e percentual de clientes com e sem churn.

**Principais conclusões:**

- 26,54% dos clientes cancelaram o serviço.
- 73,46% dos clientes não cancelaram.
- Um classificador que previsse sempre "não churn" teria 73,46% de acurácia. Por isso, a acurácia isolada não é suficiente para avaliar o modelo.

## Célula 9 — Distribuição das variáveis numéricas

**Variáveis analisadas:** `Tenure Months`, `Monthly Charges` e `Total Charges`, comparadas com `Churn Value`.

**O que apresenta:** estatísticas descritivas, histogramas, curvas de densidade e boxplots separados por classe.

**Principais conclusões:**

- Clientes com churn possuem menor tempo de permanência e mensalidades maiores.
- A mediana de permanência é de 10 meses entre clientes com churn e de 38 meses entre os demais.
- `Total Charges` tende a ser menor entre clientes com churn, principalmente por causa do menor tempo de permanência.

## Célula 10 — Distribuição das variáveis categóricas

**Variáveis analisadas:** as 16 features categóricas, como `Gender`, `Contract`, `Internet Service`, `Payment Method` e os serviços contratados.

**O que apresenta:** gráficos com a frequência de cada categoria.

**Principais conclusões:**

- Algumas categorias possuem frequências bastante diferentes.
- A análise permite identificar categorias pouco representadas.
- As variáveis categóricas precisarão ser transformadas antes do treinamento, por exemplo, com One-Hot Encoding.

## Célula 11 — Taxa de churn por categoria

**Variáveis analisadas:** as 16 features categóricas em relação a `Churn Value`.

**O que apresenta:** taxa de churn de cada categoria, quantidade de clientes em cada grupo e comparação com a taxa geral de churn.

**Principais conclusões:**

- Clientes com contrato mensal apresentam churn de aproximadamente 42,7%.
- Clientes com fibra óptica apresentam churn de aproximadamente 41,9%.
- Clientes que pagam por cheque eletrônico apresentam churn de aproximadamente 45,3%.
- Ausência de segurança online e de suporte técnico também está associada a taxas maiores de churn.
- Contratos de dois anos apresentam taxas de churn consideravelmente menores.

## Célula 12 — Correlações numéricas

**Variáveis analisadas:** `Tenure Months`, `Monthly Charges`, `Total Charges` e `Churn Value`.

**O que apresenta:** matriz de correlação e gráfico das correlações entre as features numéricas e o alvo.

**Principais conclusões:**

- `Tenure Months` possui correlação de aproximadamente -0,35 com o churn, sendo a relação linear mais forte entre as variáveis numéricas.
- `Total Charges` possui correlação de aproximadamente -0,20.
- `Monthly Charges` possui correlação de aproximadamente +0,19.
- Maior tempo de permanência está associado a menor churn, enquanto mensalidades mais altas estão associadas a maior churn.

## Célula 13 — Conclusões e preparação do baseline

**Variáveis analisadas:** principais resultados da EDA e as 19 features selecionadas para o baseline.

**O que apresenta:** resumo dos achados, exclusões, tratamentos recomendados, estratégia de validação e métricas de avaliação.

**Principais conclusões:**

- Identificadores, variáveis geográficas, colunas constantes e variáveis com vazamento de informação não devem entrar no modelo inicial.
- As features numéricas devem receber imputação pela mediana e padronização, ajustadas somente nos dados de treino.
- As features categóricas devem ser transformadas com One-Hot Encoding.
- O notebook cria `X_eda`, com 7.043 registros e 19 features, e `y_eda`, contendo a variável alvo.
- Para avaliar a Regressão Logística, são recomendadas ROC-AUC e PR-AUC como métricas principais, complementadas por recall, precision, F1 e matriz de confusão.

## Métricas técnicas

A EDA mostrou que 26,54% dos clientes pertencem à classe positiva (`Churn Value = 1`). Portanto, existe um desbalanceamento moderado e a avaliação não deve depender somente da acurácia.

### Métricas adotadas

| Métrica | Papel | Descrição |
|---|---|---|
| **PR-AUC** | Métrica principal | Resume a relação entre precision e recall da classe de churn. É mais informativa quando a classe positiva é menos frequente e possui referência natural de aproximadamente 0,2654 neste dataset. |
| **Recall** | Métrica de negócio complementar | Mede quantos clientes que realmente cancelariam foram identificados. É importante quando deixar um cliente de risco sem ação de retenção possui custo elevado. |
| **Precision** | Métrica de negócio complementar | Mede quantos clientes classificados como risco realmente cancelariam. Ajuda a controlar o custo de oferecer campanhas de retenção para clientes sem risco. |
| **F1-score** | Equilíbrio | Combina precision e recall e é útil quando falsos positivos e falsos negativos possuem importância semelhante. |
| **F2-score** | Alternativa ao F1 | Dá mais peso ao recall. É apropriada caso o negócio conclua que perder um cliente é mais caro do que abordar desnecessariamente um cliente sem risco. |
| **ROC-AUC** | Métrica complementar | Avalia a capacidade de ordenação do modelo em diferentes limiares. Deve ser reportada, mas não usada sozinha porque pode parecer otimista em bases desbalanceadas. |
| **Matriz de confusão** | Diagnóstico obrigatório | Mostra diretamente verdadeiros positivos, verdadeiros negativos, falsos positivos e falsos negativos no limiar escolhido. |
| **Log Loss** | Diagnóstico de probabilidade | Avalia a qualidade das probabilidades previstas e penaliza previsões erradas feitas com confiança excessiva. É útil porque a API retorna propensão de churn. |

### Avaliação

- Utilizar **PR-AUC como métrica técnica principal** para comparar modelos.
- Definir o limiar de classificação usando **recall e precision**, de acordo com o custo da campanha de retenção e o custo de perder clientes.
- Reportar também ROC-AUC, F1, F2, Log Loss e matriz de confusão.
- Comparar os resultados com a taxa base da classe positiva e com um classificador ingênuo.
- Calcular as métricas em validação cruzada estratificada no conjunto de treino e realizar a avaliação final uma única vez no conjunto de teste.

### Métricas descartadas

| Métrica | Motivo |
|---|---|
| **Acurácia** | Um modelo que sempre prevê "não churn" já alcança 73,46% sem identificar nenhum cliente que cancelará. Pode ser informada apenas como referência. |
| **Especificidade isolada** | Pode favorecer excessivamente a classe majoritária e esconder um recall baixo para churn. |
| **ROC-AUC isolada** | Considera o desempenho sobre ambas as classes e pode ocultar resultados fracos de precision na classe positiva. Deve permanecer somente como métrica complementar. |
| **R², MAE, MSE e RMSE** | São métricas de regressão e não são adequadas para um problema de classificação binária. |
| **MAPE** | Além de ser uma métrica de regressão, não é definida adequadamente quando o valor real pode ser zero, como ocorre no alvo binário. |

## Baselines

### 1. Regressão Logística

Baseline principal e obrigatório do projeto. Utiliza imputação pela mediana nas variáveis numéricas, padronização com `StandardScaler` e `OneHotEncoder(handle_unknown="ignore")` nas categóricas. É rápida, interpretável, fornece probabilidades e permite analisar a direção dos coeficientes. Recomenda-se testar inicialmente sem balanceamento e depois comparar com `class_weight="balanced"`.

### 2. Árvore de Decisão simples

Utiliza um `DecisionTreeClassifier` com profundidade e tamanho mínimo das folhas limitados. Esse baseline verifica se relações não lineares e interações, como contrato mensal combinado com baixa permanência, produzem ganho sobre a Regressão Logística. A árvore deve ser regularizada para evitar sobreajuste e não substitui a avaliação de Random Forest e MLP.

Os modelos utilizam a mesma divisão estratificada, as mesmas features permitidas e o mesmo procedimento de validação para que a comparação seja justa e reproduzível.

---

# Etapa 2 — Modelagem e avaliação

O notebook [`02_modelagem_avaliacao.ipynb`](./notebooks/02_modelagem_avaliacao.ipynb) compara os modelos Linear (Regressão Logística), baseado em árvores (Random Forest) e rede neural simples (`MLPClassifier`), usando validação cruzada e as métricas definidas acima.

O modelo campeão é a **Random Forest do GridSearchCV**, escolhida por Average Precision (chamada PR-AUC no notebook). O pipeline completo (pré-processamento + modelo) é salvo em `models/pipeline_churn_rf.joblib`, gerado pela célula 55 do notebook.

---

# Etapa 3 — Engenharia de software e API

A API coloca em produção o campeão escolhido nas etapas 1 e 2. Ela **não realiza nova competição entre modelos**: mantém `predict()` e o limiar original de 0,5, inclusive a regra do Scikit-Learn para empate, e não altera os notebooks das etapas anteriores.

## Executar a API

Na raiz do projeto, com as dependências instaladas:

```powershell
.\.venv\Scripts\python.exe -m uvicorn src.api:app --reload
```

Linux/macOS: use `.venv/bin/python` no lugar de `.\.venv\Scripts\python.exe`.

Com o servidor em execução, acesse a documentação interativa em http://127.0.0.1:8000/docs e use `POST /predict` > *Try it out* > *Execute*. O exemplo completo aparece automaticamente no formulário. Para encerrar o servidor, pressione `Ctrl + C` no terminal.

## Endpoints

- `GET /health`: o código 200 indica que a aplicação está no ar; confira o campo `modelo_carregado`.
- `POST /predict`: retorna probabilidade, classificação, limiar e identificação do modelo.

Comportamento e regras de entrada:

- Sem modelo, arquivo inválido ou configuração divergente, `/predict` retorna **503**; consulte o log do terminal para a causa. Reinicie a API após trocar o arquivo do modelo.
- Dados inválidos retornam **422**. Campos extras são rejeitados.
- `total_charges` pode ser omitido ou `null`: clientes novos recebem zero; os demais usam a mediana aprendida no pipeline. Não há estimativa por mensalidade × tempo.
- As categorias devem ser enviadas exatamente como no exemplo da documentação, em inglês.
- Não há faixas arbitrárias de risco nem promessa de probabilidade calibrada.

### Exemplo de resposta

Uma requisição de teste ao endpoint `POST /predict` retornou o código HTTP **200**, com a seguinte resposta:

```json
{
  "probabilidade_churn": 0.7080131434658807,
  "churn_previsto": true,
  "limiar": 0.5,
  "modelo": "Random Forest (Etapa 2)"
}
```

Para o cliente utilizado nesse teste, o modelo estimou aproximadamente 70,8% de probabilidade de churn. Como esse valor supera o limiar de 50%, a classificação retornada foi `churn_previsto: true`. A previsão representa uma estimativa do modelo, não uma certeza de que o cliente cancelará o serviço; outros dados de entrada podem produzir resultados diferentes.

## Testes automatizados

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Resultado observado: **19 testes aprovados, dois avisos e nenhuma falha**.

Os testes usam dados sintéticos e cobrem, entre outros pontos, a inicialização real da aplicação e a carga do modelo em disco. O sucesso neles não comprova o desempenho do modelo no dataset Telco.

## Verificação com o modelo real

Para confirmar que a API retorna as mesmas previsões do pipeline, com o CSV tratado produzido pela Etapa 1:

```powershell
.\.venv\Scripts\python.exe -m src.verify --data notebooks/artefatos/telco_tratado.csv
```

O verificador também aceita o XLSX bruto. Ele inicia a aplicação, carrega o arquivo real e compara probabilidades e classes para 20 clientes com as do pipeline direto, sem treinar. Use `--n 7043` para verificar toda a base.

Resultado observado:

```text
OK: 20 clientes; API e pipeline concordam. Modelo não foi retreinado.
```

Essa verificação confirma a consistência da integração na amostra avaliada; não representa uma nova avaliação da qualidade preditiva do modelo e, por ser feita em amostra, não certifica todos os casos.

## Reproduzir o modelo (somente se o arquivo original não estiver disponível)

```powershell
.\.venv\Scripts\python.exe -m src.train --data notebooks/artefatos/telco_tratado.csv
```

Reproduz a configuração vencedora: 100 árvores, profundidade 6, folha mínima 10, `max_features=0.5`, `class_weight=None`, seed 42 e split estratificado 80/20. Não ajusta hiperparâmetros, não altera o limiar e não usa o teste para selecionar modelo. Salva o Pipeline diretamente e um JSON com métricas e versões. Para sobrescrever um modelo existente é necessário usar `--overwrite`.

Diferenças de versões de bibliotecas ou de dados podem alterar os resultados, portanto esse caminho não garante identidade com o arquivo original. A avaliação usa o mesmo teste já observado na Etapa 2, não uma amostra inédita.

## Documentação complementar

- [`docs/ALTERACOES.md`](./docs/ALTERACOES.md): decisões e rastreabilidade da Etapa 3.
- [`docs/VALIDACAO.md`](./docs/VALIDACAO.md): verificações executadas e pendências.

---

# Etapa 4 — Pendências da entrega final

- Model Card (performance, limitações e possíveis vieses) em `docs/`.
- Vídeo de 5 minutos no método STAR.
- Revisão final deste README.
