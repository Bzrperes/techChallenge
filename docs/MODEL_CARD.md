
# Model Card | Previsão de Churn

## 1. Descrição do modelo

Desenvolvemos um modelo de Machine Learning para prever o cancelamento de clientes de uma operadora de telecomunicações, utilizando o dataset Telco Customer Churn.

O problema foi tratado como classificação binária. Comparamos Regressão Logística, Random Forest e MLPClassifier, utilizando pipelines do Scikit-Learn e validação cruzada.

A Random Forest otimizada com GridSearchCV apresentou a maior PR-AUC média na validação cruzada e foi escolhida como modelo final.

O pipeline está salvo em `models/pipeline_churn_rf.joblib` e é utilizado pela API FastAPI para gerar as previsões.

## 2. Dados utilizados

O dataset contém 7.043 registros e 33 colunas, com informações sobre os clientes, serviços contratados e pagamentos.

A variável alvo é `Churn Value`:

- `0`: cliente que não cancelou o serviço.
- `1`: cliente que cancelou o serviço.

Na análise exploratória, identificamos uma taxa de churn de 26,54%, enquanto 73,46% dos clientes permaneceram na operadora.

Selecionamos 19 variáveis para a modelagem, sendo 3 numéricas e 16 categóricas.

As variáveis numéricas são `Tenure Months`, `Monthly Charges` e `Total Charges`. As categóricas incluem informações sobre serviços, contratos, pagamentos e características dos clientes.

Identificadores, informações geográficas, colunas constantes e variáveis com risco de vazamento foram excluídos da modelagem.

## 3. Treinamento

Utilizamos um pipeline do Scikit-Learn para reunir o pré-processamento e o modelo.

Os dados foram divididos de forma estratificada, com 80% para treinamento e 20% para teste. Também utilizamos validação cruzada com cinco folds para comparar os modelos.

A configuração final da Random Forest foi:

| Parâmetro | Valor |
|---|---|
| Modelo | RandomForestClassifier |
| n_estimators | 100 |
| max_depth | 6 |
| min_samples_leaf | 10 |
| max_features | 0.5 |
| class_weight | None |
| random_state | 42 |
| Limiar de classificação | 0.5 |

O pré-processamento faz parte do pipeline, permitindo aplicar os mesmos tratamentos durante o treinamento e a inferência.

O limiar de classificação foi mantido em 0,5, conforme definido na Etapa 2.

## 4. Resultados

Utilizamos a Average Precision, chamada de PR-AUC no projeto, como métrica principal para comparar os modelos.

### Validação cruzada

| Métrica | Resultado |
|---|---:|
| PR-AUC média | 0,6862 |
| ROC-AUC | 0,8627 |

### Conjunto de teste

| Métrica | Resultado |
|---|---:|
| PR-AUC | 0,6656 |
| ROC-AUC | 0,8537 |
| Recall | 0,5455 |
| Precision | 0,6645 |
| F1-score | 0,5991 |

A Random Forest apresentou a maior PR-AUC média na validação cruzada, mas a diferença entre os modelos foi pequena.

A Regressão Logística apresentou recall superior, mostrando que a escolha do modelo depende também da métrica considerada e do objetivo de negócio.

Esses resultados foram obtidos na Etapa 2 e não foram recalculados durante a integração com a API.

## 5. Integração com a API

O modelo foi disponibilizado por meio de uma API REST desenvolvida com FastAPI.

Foram implementados dois endpoints:

- `GET /health`: verifica o funcionamento da aplicação e informa se o modelo foi carregado.
- `POST /predict`: recebe os dados de um cliente e retorna a probabilidade estimada de churn e a classificação prevista.

A API também retorna o limiar de classificação e a identificação do modelo utilizado.

Mantivemos o pipeline original, sem realizar novo treinamento ou alterar o limiar definido na modelagem.

As probabilidades retornadas representam estimativas do modelo. Não foi realizada uma avaliação específica de calibração dessas probabilidades.

## 6. Limitações

Identificamos algumas limitações que precisam ser consideradas na utilização do modelo:

- O treinamento e a avaliação utilizaram dados históricos do dataset Telco Customer Churn. Não verificamos o desempenho com dados de outras operadoras ou períodos.
- O conjunto de teste já havia sido utilizado na Etapa 1, portanto não representa uma amostra totalmente inédita ao projeto.
- O recall de 0,5455 mostra que o modelo deixou de identificar parte dos clientes que cancelaram o serviço.
- O limiar de 0,5 foi mantido sem uma análise específica dos custos envolvidos nas ações de retenção.
- Não realizamos avaliações de calibração das probabilidades ou monitoramento do modelo em produção.
- O funcionamento da API depende do envio de dados compatíveis com as variáveis utilizadas no treinamento.

Essas limitações devem ser consideradas antes de utilizar as previsões para orientar decisões reais de retenção de clientes.

## 7. Possíveis vieses

O dataset contém informações sobre características dos clientes, serviços contratados e formas de pagamento.

Algumas dessas variáveis podem apresentar distribuições diferentes entre grupos de clientes, influenciando os resultados do modelo.

Variáveis como gênero e indicação de cliente idoso merecem atenção, principalmente em uma possível utilização comercial.

Não realizamos uma avaliação específica de vieses ou uma comparação de desempenho entre diferentes grupos.

Por isso, não podemos afirmar que o modelo apresenta o mesmo desempenho para todos os perfis de clientes.

## 8. Validação da integração

Realizamos testes automatizados e verificamos a consistência das previsões entre a API e o pipeline.

Os resultados registrados foram:

- 19 testes aprovados.
- 2 avisos e nenhuma falha.
- Resultados equivalentes entre a API e o pipeline para 20 clientes.
- Nenhum retreinamento durante a verificação.

Esses testes confirmam o funcionamento dos componentes avaliados e a consistência das previsões na amostra utilizada. Eles não representam uma nova avaliação do desempenho preditivo.

Os procedimentos estão documentados em `docs/VALIDACAO.md`.

## 9. Conclusão

Selecionamos a Random Forest por apresentar a maior PR-AUC média na comparação dos modelos.

O pipeline foi integrado à API FastAPI, mantendo as configurações e o comportamento definidos na Etapa 2.

Conseguimos estruturar o processo de previsão de churn, desde a preparação dos dados até a disponibilização do modelo para inferência.

Como próximos passos, seria necessário avaliar o modelo com novos dados, analisar possíveis vieses, verificar a calibração das probabilidades e acompanhar seu desempenho antes de uma utilização em produção.
