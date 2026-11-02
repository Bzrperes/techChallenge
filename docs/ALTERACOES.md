# Decisões técnicas da entrega

1. A **Random Forest** permanece como campeã por apresentar a maior PR-AUC média na validação cruzada: 0,6862, contra 0,6788 da Regressão Logística e 0,6782 da MLP. A diferença é pequena e não demonstra superioridade estatística.
2. O recall de 0,8094 da Regressão Logística no limiar 0,5 decorre de `class_weight="balanced"`, enquanto a Random Forest usa `class_weight=None`. Esses valores não representam o mesmo ponto operacional; a seleção do algoritmo permanece baseada em PR-AUC.
3. O limiar da Random Forest foi calculado com probabilidades *out-of-fold* do treino. Entre os pontos com recall mínimo de 80%, escolheu-se o de maior precision, resultando em 0,2898431181242871.
4. O teste foi usado apenas para avaliar decisões já fixadas. Ele não participou da seleção do modelo nem do limiar; ressalva-se que essa divisão já havia sido observada na Etapa 1.
5. `churn_previsto` representa a triagem ampla no limiar de recall mínimo de 80%. O piloto de negócio usa separadamente os 20% de clientes com maior probabilidade; os dois pontos operacionais não são equivalentes.
6. `pipeline_churn_rf.joblib` armazena pipeline, limiar, meta de recall e nome do modelo. A API valida esse contrato antes de aceitar o artefato.
7. Todas as configurações comparadas foram exportadas para `models/experimentos_modelos.csv`. Métricas por média dos folds, OOF agregado, teste, versões e limiar estão em `models/metricas_teste.json`, usando o mesmo esquema do notebook e de `src.train`.
8. A limpeza é compartilhada por treino e inferência. As 11 ausências de `Total Charges` pertencem a clientes com `tenure=0` e recebem zero por regra determinística; o imputador permanece como proteção para ausências inesperadas.
9. A aplicação é criada por fábrica. Os testes cobrem o artefato real e artefatos sintéticos isolados, carga de dados, comandos, endpoints, validações de entrada, pré-processamento e limiar.
10. A Etapa 1 foi renomeada para `01_entendimento_e_preparacao.ipynb`; as 48 células copiadas da Etapa 2 foram removidas, preservando uma sequência sem duplicação.
11. O ML Canvas, o Model Card e o README distinguem a métrica oficial de negócio, a métrica financeira futura, a triagem ampla e o piloto top 20%.
12. Python 3.13.15 e as dependências são reproduzidos pelo `uv.lock`; o projeto exige uv 0.11.33 ou superior e também documenta a instalação alternativa por `requirements.txt`.
