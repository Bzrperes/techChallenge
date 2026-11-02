# Artefatos do modelo

Esta pasta contém os artefatos reproduzidos na Etapa 2:

- `pipeline_churn_rf.joblib`: dicionário com o pipeline treinado, o limiar operacional (0,289843), a meta de recall e o nome do modelo;
- `comparacao_modelos.csv`: métricas médias de validação cruzada da Regressão Logística, Random Forest e MLPClassifier;
- `metricas_teste.json`: critério de seleção, métricas de validação e teste, tamanhos das divisões e versões de Python e Scikit-Learn.

A Random Forest foi selecionada pela maior PR-AUC média (0,6862). O limiar foi escolhido com previsões *out-of-fold* do treino para atingir recall mínimo de 80% e maximizar precision. No teste, foram observados PR-AUC 0,6656, ROC-AUC 0,8537, recall 0,8021, precision 0,5376 e F1 0,6438.

O módulo `src/predict.py` carrega e valida o artefato. Arquivos Joblib/Pickle podem executar código durante a desserialização; carregue somente o arquivo versionado e produzido pelo grupo.

Detalhes de treinamento, uso e limitações estão no [Model Card](../docs/MODEL_CARD.md).
