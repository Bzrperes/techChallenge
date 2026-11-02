# Model Card — Previsão de Churn

## Visão geral

O modelo estima a propensão de cancelamento de clientes de telecomunicações. Regressão Logística, Random Forest e `MLPClassifier` foram comparados com pipelines do Scikit-Learn, divisão estratificada, cinco folds e seed 42. A Random Forest do `GridSearchCV` foi escolhida pela maior PR-AUC média.

O artefato `models/pipeline_churn_rf.joblib` reúne pré-processamento, modelo, limiar e metadados operacionais e é consumido pela API FastAPI.

## Dados

A base tem 7.043 registros, 33 colunas e 26,54% de churn. O modelo usa 19 features: 3 numéricas e 16 categóricas. Identificadores, geografia, constantes e variáveis que revelam o alvo foram removidos.

As 11 ausências de `Total Charges` pertencem a clientes com permanência zero e recebem zero por regra determinística. O imputador do pipeline permanece como proteção contra ausências inesperadas. Todo pré-processamento aprendido é ajustado somente no treino.

## Seleção do modelo

O treino contém 5.634 registros e o teste 1.409. A Average Precision, chamada PR-AUC no projeto, é o critério principal.

| Modelo | PR-AUC média | ROC-AUC média | Recall a 0,5 |
|---|---:|---:|---:|
| Regressão Logística | 0,6788 | 0,8588 | 0,8094 |
| Random Forest | **0,6862** | **0,8627** | 0,5445 |
| MLPClassifier | 0,6782 | 0,8587 | 0,5472 |

A vantagem da Random Forest é pequena e não demonstra superioridade estatística. A Regressão Logística usa `class_weight="balanced"`, enquanto a Random Forest usa `class_weight=None`; por isso, os recalls a 0,5 não comparam o mesmo ponto operacional.

O aviso de não convergência ocorreu apenas na configuração **inicial** da MLP, sem parada antecipada. A configuração final comparada foi a do `RandomizedSearchCV`, com `early_stopping=True` e `n_iter_no_change=15`; ela não apresentou esse aviso nas saídas executadas.

Configuração do campeão:

| Parâmetro | Valor |
|---|---|
| `n_estimators` | 100 |
| `max_depth` | 6 |
| `min_samples_leaf` | 10 |
| `max_features` | 0,5 |
| `class_weight` | `None` |
| `random_state` | 42 |
| `n_jobs` | 1 |

## Métricas técnicas

Depois da escolha do algoritmo, probabilidades OOF do treino definiram o limiar **0,2898431181242871**, maximizando precision entre pontos com recall mínimo de 80%.

| Métrica | Média dos cinco folds | OOF agregado | Teste |
|---|---:|---:|---:|
| PR-AUC | 0,6862 | 0,6776 | 0,6656 |
| ROC-AUC | 0,8627 | 0,8609 | 0,8537 |
| Recall | 0,5445 | 0,8020 | 0,8021 |
| Precision | 0,6746 | 0,5445 | 0,5376 |
| F1 | 0,6022 | 0,6486 | 0,6438 |

PR-AUC e ROC-AUC na primeira coluna são médias dos folds. Na coluna OOF agregado, essas métricas são recalculadas sobre todas as previsões OOF reunidas; recall, precision e F1 usam o limiar operacional. As métricas de classe na média dos folds refletem o corte padrão do estimador.

O teste não escolheu modelo, configuração ou limiar. Contudo, essa divisão já havia sido observada na Etapa 1 e não é uma avaliação completamente inédita.

## Pontos operacionais

`churn_previsto` representa triagem ampla:

- taxa de alertas OOF: 39,08%;
- taxa de alertas no teste: 39,60%;
- recall no teste: 80,21%.

O futuro piloto usa os 20% de maior probabilidade:

| Conjunto | Corte aproximado | Taxa de alertas | Recall | Precision |
|---|---:|---:|---:|---:|
| OOF | 0,515347 | 20,00% | 51,77% | 68,68% |
| Teste | 0,516267 | 20,01% | 51,34% | 68,09% |

Esses pontos cumprem papéis diferentes. O limiar amplo não significa que 39% dos clientes receberão oferta; o top 20% representa o orçamento proposto para o piloto.

## Calibração

O Brier score no teste foi **0,133146**. A taxa observada cresce de 0,00% no primeiro decil de risco para 75,89% no décimo, enquanto as probabilidades médias vão de 0,67% a 74,64%. A tabela completa por decil está no notebook da Etapa 2.

Essa análise é exploratória: não houve calibração adicional, avaliação temporal ou comparação formal de métodos de calibração. A probabilidade deve ser interpretada como escore estimado, não garantia individual.

## Análise exploratória por grupos

Resultados no teste, usando o limiar operacional:

| Feature | Grupo | N | Churn | Alertas | Recall | Precision |
|---|---|---:|---:|---:|---:|---:|
| Gender | Female | 687 | 28,09% | 39,45% | 0,7876 | 0,5609 |
| Gender | Male | 722 | 25,07% | 39,75% | 0,8177 | 0,5157 |
| Senior Citizen | No | 1.187 | 23,25% | 35,05% | 0,7681 | 0,5096 |
| Senior Citizen | Yes | 222 | 44,14% | 63,96% | 0,8980 | 0,6197 |
| Partner | No | 736 | 33,70% | 53,40% | 0,8629 | 0,5445 |
| Partner | Yes | 673 | 18,72% | 24,52% | 0,6825 | 0,5212 |
| Dependents | No | 1.083 | 32,04% | 48,11% | 0,8184 | 0,5451 |
| Dependents | Yes | 326 | 8,28% | 11,35% | 0,5926 | 0,4324 |

A análise é exploratória, contém grupos de tamanhos diferentes e não comprova ausência de viés. Diferenças podem refletir prevalência, composição da amostra e incerteza elevada em grupos pequenos. Nenhuma feature foi removida com base nessa leitura; antes de uso real, são necessários intervalos de confiança, análise de impacto e revisão de privacidade.

## Uso previsto e negócio

A API oferece `GET /health` e `POST /predict`. As previsões podem apoiar priorização humana e um piloto controlado, mas não devem ser tratadas como certeza nem usadas isoladamente em decisões adversas.

A métrica de negócio proposta é a redução incremental do churn em 90 dias no top 20%, com meta de 3 pontos percentuais absolutos. O ROI incremental futuro tem meta inicial de 1,25 em 180 dias. Nenhuma meta foi medida neste projeto.

## Limitações

- Dados históricos podem não representar outra operadora, região ou período.
- O teste já havia sido observado na Etapa 1.
- A diferença de PR-AUC entre candidatos é pequena e sem teste de significância.
- O limiar não incorpora custos reais de contato, oferta ou erros.
- Calibração e grupos foram avaliados apenas de forma exploratória no teste.
- Alguns grupos são pequenos; a análise não demonstra ausência de viés.
- Não há validação temporal, externa, monitoramento de *drift* ou teste em produção.
- A disponibilidade operacional das 19 features não foi validada com uma operadora.

## Reprodutibilidade e segurança

Python 3.13.15 e dependências são fixados por `pyproject.toml` e `uv.lock`. Resultados completos estão em `models/comparacao_modelos.csv`, `models/experimentos_modelos.csv` e `models/metricas_teste.json`.

Joblib é baseado em Pickle e pode executar código durante a carga; use somente o artefato produzido e versionado pelo grupo.
