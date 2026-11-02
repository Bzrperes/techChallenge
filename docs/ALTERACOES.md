# Decisões técnicas da entrega

1. A **Random Forest** permanece como campeã por apresentar a maior PR-AUC média na validação cruzada: 0,6862, contra 0,6788 da Regressão Logística e 0,6782 da MLP.
2. O maior recall da Regressão Logística no limiar 0,5 não foi ignorado. Como recall depende do limiar, a seleção do algoritmo permanece baseada em PR-AUC e a decisão operacional foi tratada separadamente.
3. O limiar da Random Forest foi calculado com probabilidades *out-of-fold* do treino. Entre os pontos com recall mínimo de 80%, escolheu-se o de maior precision, resultando em 0,289843.
4. O teste foi usado apenas para a avaliação final do ponto já escolhido. Ele não participou da seleção do modelo ou do limiar; ressalva-se que essa divisão já havia sido observada na Etapa 1.
5. `pipeline_churn_rf.joblib` passou a armazenar um dicionário com pipeline, limiar, meta de recall e nome do modelo. A API valida esse contrato antes de aceitar o artefato.
6. A comparação completa foi exportada para `models/comparacao_modelos.csv`, e métricas, versões e limiar foram registrados em `models/metricas_teste.json`.
7. A limpeza é compartilhada por treino e inferência. `Total Charges` ausente recebe zero somente para `tenure=0`; nos demais casos, a imputação usa a mediana aprendida no treino.
8. A aplicação é criada por fábrica. Os testes exercitam o ciclo de vida, a carga do arquivo, os endpoints, o pré-processamento e o limiar.
9. A Etapa 1 foi renomeada para `01_entendimento_e_preparacao.ipynb`; as 48 células copiadas da Etapa 2 foram removidas, preservando uma sequência sem duplicação.
10. O ML Canvas e o Model Card foram atualizados com a métrica oficial de negócio, a métrica financeira futura, o ponto operacional e as limitações.
11. Python 3.13.15 e dependências diretas foram fixados em `pyproject.toml`; `uv.lock` registra o ambiente completo e `requirements.txt` é uma exportação exata para uso alternativo.
