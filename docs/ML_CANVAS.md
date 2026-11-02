# ML Canvas — Previsão de Churn

## Problema e objetivo

Uma operadora de telecomunicações precisa identificar clientes com maior propensão ao cancelamento para priorizar análises e futuras ações de retenção. O projeto estima essa propensão com classificação binária e disponibiliza o resultado por uma API REST. A execução de campanhas reais não faz parte do escopo acadêmico.

## Stakeholders

| Stakeholder | Interesse ou responsabilidade |
|---|---|
| Diretoria | Avaliar risco de perda, resultado incremental e viabilidade financeira de retenção. |
| Retenção e relacionamento | Priorizar análises e contatos em um futuro piloto. |
| Equipe de dados e tecnologia | Preparar dados, avaliar, implantar e monitorar o pipeline e a API. |
| Jurídico e privacidade | Avaliar base legal, minimização de dados, transparência e tratamento justo. |
| Clientes | Pessoas afetadas pelas ações; precisam de privacidade e decisões não discriminatórias. |

Esses papéis são propostos para o cenário acadêmico e não foram validados com uma operadora real.

## Dados

| Item | Definição |
|---|---|
| Dataset | Telco Customer Churn (Kaggle) |
| Volume | 7.043 registros e 33 colunas |
| Alvo | `Churn Value` (0 = permaneceu; 1 = cancelou) |
| Distribuição | 26,54% churn; 73,46% não churn |
| Features | 19: 3 numéricas e 16 categóricas |
| Divisão | 80% treino e 20% teste, estratificada, seed 42 |

Identificadores, geografia, constantes e variáveis que revelam o alvo são excluídos. Imputação, padronização e codificação ficam dentro dos pipelines e são ajustadas somente no treino. A disponibilidade operacional de todas as features ainda precisa ser confirmada em um cenário real.

## Modelagem e critério técnico

- Baseline obrigatório: Regressão Logística.
- Candidatos: Regressão Logística, Random Forest e `MLPClassifier`.
- Validação: cinco folds estratificados, com as mesmas features e divisão.
- Métrica principal: Average Precision, chamada PR-AUC no projeto.
- Complementares: ROC-AUC, recall, precision e F1.
- Campeão: Random Forest do `GridSearchCV`, PR-AUC média 0,6862.

A Regressão Logística alcançou recall 0,8094 no limiar 0,5, acima da Random Forest (0,5445), mas recall é dependente do limiar. A Random Forest foi mantida por ter melhor PR-AUC e ROC-AUC médias; depois, seu limiar foi tratado como decisão operacional separada.

Com probabilidades *out-of-fold* do treino, selecionou-se o ponto de maior precision entre os que atingem recall mínimo de 80%. O limiar resultante foi 0,289843.

| Métrica | Validação *out-of-fold* | Teste |
|---|---:|---:|
| PR-AUC | 0,6862 | 0,6656 |
| ROC-AUC | 0,8627 | 0,8537 |
| Recall | 0,8020 | 0,8021 |
| Precision | 0,5445 | 0,5376 |
| F1 | 0,6486 | 0,6438 |

O teste não escolheu o limiar, mas já havia sido observado na Etapa 1 e não é uma amostra totalmente inédita.

## Métrica oficial de negócio

A métrica oficial é a **redução incremental da taxa de churn em 90 dias** entre os 20% de clientes com maior risco. Em um piloto futuro, essa população seria dividida aleatoriamente entre tratamento e controle:

**Redução incremental = churn do controle − churn do tratamento**

A meta é uma redução de pelo menos **3 pontos percentuais absolutos em 90 dias**, com intervalo de confiança de 95%. A expansão dependeria de evidência de efeito positivo.

Essa métrica conecta o modelo à retenção, mede o efeito da ação em vez da simples previsão, evita atribuir ao modelo cancelamentos que ocorreriam sem intervenção, é compreensível para o negócio e não exige inventar custos ou margens ausentes no dataset.

Não houve campanha nem experimento controlado; a meta é futura, não um resultado do projeto.

## Métrica financeira futura

Quando houver margem, ARPU, custo de contato e custo de oferta, será medido o **ROI incremental de retenção**:

**ROI = margem incremental líquida da campanha ÷ custo total da campanha**

A meta inicial é **ROI mínimo de 1,25 em 180 dias**, com recuperação do investimento em até 90 dias. Ela complementa a redução incremental de churn e ainda não foi medida.

## Uso, riscos e controles

A API retorna probabilidade e classe de churn para apoiar priorização. O resultado não é certeza e não deve produzir decisão adversa automatizada.

Principais riscos: mudança de população, baixa calibração, custos de falsos positivos, perda de churns, indisponibilidade de features, vieses entre grupos, vazamento de dados e degradação temporal. Antes de uso real, devem ser feitos piloto controlado, análise de calibração e vieses, revisão de privacidade, validação temporal e monitoramento de dados e performance.

## Evidências

| Entrega | Caminho |
|---|---|
| EDA e baseline | `notebooks/01_entendimento_e_preparacao.ipynb` |
| Comparação e limiar | `notebooks/02_modelagem_avaliacao.ipynb` |
| Resultados dos experimentos | `models/comparacao_modelos.csv` e `models/metricas_teste.json` |
| Modelo final | `models/pipeline_churn_rf.joblib` |
| API e módulos | `src/` |
| Testes | `tests/` e `docs/VALIDACAO.md` |
| Limitações e vieses | `docs/MODEL_CARD.md` |
