
# Modelo de Machine Learning

Esta pasta contém o pipeline da Random Forest selecionada na Etapa 2 do Tech Challenge.

## Modelo final

O arquivo `pipeline_churn_rf.joblib` contém o pré-processamento e o modelo treinado, sendo utilizado pela API FastAPI para realizar as previsões de churn.

A Random Forest foi escolhida após a comparação com Regressão Logística e MLPClassifier, considerando a maior PR-AUC média na validação cruzada.

## Resultados

| Métrica | Validação cruzada | Teste |
|---|---:|---:|
| PR-AUC | 0,6862 | 0,6656 |
| ROC-AUC | 0,8627 | 0,8537 |

Os resultados foram obtidos na Etapa 2 e não foram recalculados durante a integração com a API.

## Utilização

O pipeline é carregado pelo módulo `src/predict.py`, mantendo as configurações e o limiar de classificação de 0,5 definidos na modelagem.

Não é necessário retreinar o modelo para executar a API, desde que o arquivo original esteja disponível e seja compatível com o ambiente.

## Documentação

As informações sobre treinamento, avaliação e limitações estão disponíveis no [Model Card](../docs/MODEL_CARD.md).

Os testes de integração estão documentados em [VALIDACAO.md](../docs/VALIDACAO.md).