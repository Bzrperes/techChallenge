# Model Card — Previsão de Churn

## Visão geral

O modelo estima a propensão de cancelamento de clientes de telecomunicações a partir do dataset Telco Customer Churn. O problema é uma classificação binária; foram comparados Regressão Logística, Random Forest e `MLPClassifier` com pipelines do Scikit-Learn e validação cruzada estratificada.

A Random Forest ajustada com `GridSearchCV` obteve a maior PR-AUC média e foi escolhida como modelo final. A API FastAPI carrega `models/pipeline_churn_rf.joblib`, que reúne pipeline, limiar e metadados operacionais.

## Dados e features

A base possui 7.043 registros e 33 colunas, com 26,54% de churn. O modelo usa 19 variáveis: `Tenure Months`, `Monthly Charges`, `Total Charges` e 16 categóricas sobre perfil, serviços, contrato e pagamento.

Foram removidos identificadores, informações geográficas, colunas constantes e variáveis que revelam o alvo. O pré-processamento fica dentro do pipeline e é ajustado somente nos dados de treino.

## Treinamento e seleção

Os dados foram divididos de forma estratificada em 80% para treino (5.634 registros) e 20% para teste (1.409), com `random_state=42`. A comparação usa cinco folds e a Average Precision, denominada PR-AUC no projeto, como métrica principal.

| Modelo | PR-AUC média | ROC-AUC média | Recall a 0,5 |
|---|---:|---:|---:|
| Regressão Logística | 0,6788 | 0,8588 | 0,8094 |
| Random Forest | **0,6862** | **0,8627** | 0,5445 |
| MLPClassifier | 0,6782 | 0,8587 | 0,5472 |

A Regressão Logística tem maior recall no limiar padrão, enquanto a Random Forest tem a melhor PR-AUC. Esses resultados não são contraditórios: PR-AUC avalia o ranking de probabilidades ao longo dos limiares; recall é uma decisão em um limiar específico.

Configuração final da Random Forest:

| Parâmetro | Valor |
|---|---|
| `n_estimators` | 100 |
| `max_depth` | 6 |
| `min_samples_leaf` | 10 |
| `max_features` | 0,5 |
| `class_weight` | `None` |
| `random_state` | 42 |

## Limiar operacional e performance

Depois da escolha do algoritmo, probabilidades *out-of-fold* do treino foram usadas para selecionar um limiar com recall mínimo de 80%. Entre os pontos elegíveis, maximizou-se precision, com desempate por F1 e maior limiar. O valor escolhido foi **0,289843**.

| Métrica | Validação *out-of-fold* | Teste |
|---|---:|---:|
| PR-AUC | 0,6862 | 0,6656 |
| ROC-AUC | 0,8627 | 0,8537 |
| Recall no limiar | 0,8020 | 0,8021 |
| Precision no limiar | 0,5445 | 0,5376 |
| F1 no limiar | 0,6486 | 0,6438 |

O teste não foi usado para escolher o limiar. Entretanto, essa divisão já havia sido observada na Etapa 1 e, portanto, não deve ser apresentada como uma avaliação totalmente inédita ao projeto.

## Uso previsto

A API oferece:

- `GET /health`: estado da aplicação e carga do modelo;
- `POST /predict`: probabilidade, classe, limiar e identificação do modelo.

As previsões podem apoiar a priorização de clientes para análise ou para um futuro piloto controlado de retenção. Elas não devem ser interpretadas como certeza de cancelamento, nem usadas como única base para decisões adversas ao cliente.

## Métricas de negócio

A métrica oficial proposta é a **redução incremental do churn em 90 dias** nos 20% de clientes com maior risco, medida por experimento aleatório entre tratamento e controle. A meta inicial é uma redução absoluta de 3 pontos percentuais, com intervalo de confiança de 95%.

Como métrica financeira futura, propõe-se ROI incremental de retenção mínimo de **1,25 em 180 dias**, com *payback* em até 90 dias. Nenhuma das metas foi medida neste projeto, pois não houve campanha real nem dados de custos e margens.

## Limitações

- O dataset histórico pode não representar outra operadora, região ou período.
- O conjunto de teste já havia sido observado na Etapa 1.
- O limiar prioriza a meta de recall, mas não incorpora custos reais de contato, oferta, falso positivo ou falso negativo.
- As probabilidades não passaram por avaliação específica de calibração.
- Não há validação temporal, monitoramento de *drift* ou teste em produção.
- A disponibilidade das 19 features antes da decisão operacional não foi validada com uma operadora.
- O `MLPClassifier` emitiu aviso de não convergência no limite de iterações; seu resultado deve ser interpretado com essa ressalva.

## Possíveis vieses e uso responsável

Variáveis como gênero e indicação de cliente idoso podem refletir desigualdades ou produzir desempenhos distintos entre grupos. O projeto não mediu paridade de recall, precision, calibração ou impacto entre grupos protegidos. Antes de uso real, é necessário avaliar essas métricas, justificar a necessidade de cada feature e envolver áreas jurídica, de privacidade e negócio.

As previsões devem apoiar revisão humana e ações benéficas de retenção. Não se recomenda usá-las para negar serviço, aplicar cobrança diferenciada ou tomar decisão automatizada de alto impacto.

## Reprodutibilidade e validação

O ambiente fixa Python 3.13.15 e dependências exatas por `pyproject.toml` e `uv.lock`. A comparação está em `models/comparacao_modelos.csv`; métricas, versões e limiar estão em `models/metricas_teste.json`; os procedimentos executados estão em `docs/VALIDACAO.md`.

O artefato Joblib deve ser carregado somente de fonte confiável, pois formatos baseados em Pickle podem executar código durante a desserialização.
