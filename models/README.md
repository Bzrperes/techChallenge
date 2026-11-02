# Artefatos do modelo

- `pipeline_churn_rf.joblib`: pipeline treinado, limiar 0,2898431181242871, meta de recall e rótulo do modelo;
- `comparacao_modelos.csv`: comparação dos três finalistas;
- `experimentos_modelos.csv`: todas as configurações de Regressão Logística, Random Forest e MLP avaliadas na Etapa 2;
- `metricas_teste.json`: contrato estruturado compartilhado pelo notebook e por `src/train.py`.

O JSON registra critério de seleção, meta e limiar, parâmetros, tamanhos das divisões, versões, observação metodológica, taxas de alertas, métricas médias dos folds, métricas OOF agregadas, teste e pontos top 20%.

A Random Forest foi selecionada pela maior PR-AUC média (0,6861634012594544). A diferença para os outros finalistas é pequena e não demonstra superioridade estatística. O limiar foi escolhido somente com previsões OOF do treino. No teste: PR-AUC 0,6655546107759017, ROC-AUC 0,853650313880493, recall 0,8021390374331551, precision 0,5376344086021505 e F1 0,6437768240343348.

O módulo `src/predict.py` valida o contrato antes da inferência. Joblib/Pickle pode executar código durante a desserialização; carregue somente o artefato produzido e versionado pelo grupo.

Consulte o [Model Card](../docs/MODEL_CARD.md) e a [validação](../docs/VALIDACAO.md).
