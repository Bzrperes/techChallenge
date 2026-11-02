# ML Canvas — Previsão de Churn

## Problema e objetivo

Uma operadora de telecomunicações precisa identificar clientes com maior propensão ao cancelamento para priorizar análises e futuras ações de retenção. O projeto estima essa propensão por classificação binária e a disponibiliza por API. Campanhas reais não fazem parte do escopo executado.

## Stakeholders

| Stakeholder | Interesse ou responsabilidade |
|---|---|
| Diretoria | Avaliar perda, efeito incremental e viabilidade financeira. |
| Retenção e relacionamento | Priorizar contatos em um futuro piloto. |
| Dados e tecnologia | Preparar, avaliar, implantar e monitorar modelo e API. |
| Jurídico e privacidade | Rever base legal, minimização, transparência e tratamento justo. |
| Clientes | Receber tratamento respeitoso, privado e não discriminatório. |

Os papéis são propostos para o cenário acadêmico e não foram validados com uma operadora real.

## Dados

| Item | Definição |
|---|---|
| Dataset | Telco Customer Churn (Kaggle) |
| Volume | 7.043 registros e 33 colunas |
| Alvo | `Churn Value` |
| Distribuição | 26,54% churn; 73,46% não churn |
| Features | 19: 3 numéricas e 16 categóricas |
| Divisão | 5.634 treino e 1.409 teste, estratificada, seed 42 |

Identificadores, geografia, constantes e variáveis que revelam o alvo são excluídos. As 11 ausências de `Total Charges` com permanência zero recebem zero por regra determinística. Pré-processamento aprendido fica no pipeline e usa somente o treino.

## Modelagem

- Baseline: Regressão Logística com `class_weight="balanced"`.
- Candidatos: Regressão Logística, Random Forest e `MLPClassifier`.
- Validação: cinco folds estratificados.
- Métrica principal: PR-AUC/Average Precision.
- Campeão: Random Forest do `GridSearchCV`, PR-AUC média 0,6862.

A diferença para LR (0,6788) e MLP (0,6782) é pequena e não demonstra superioridade estatística. O recall da LR a 0,5 é influenciado pelo balanceamento de classes e não é diretamente comparável ao da RF com `class_weight=None`.

## Métricas técnicas

| Métrica da Random Forest | Média dos folds | OOF agregado | Teste |
|---|---:|---:|---:|
| PR-AUC | 0,6862 | 0,6776 | 0,6656 |
| ROC-AUC | 0,8627 | 0,8609 | 0,8537 |
| Recall | 0,5445 | 0,8020 | 0,8021 |
| Precision | 0,6746 | 0,5445 | 0,5376 |
| F1 | 0,6022 | 0,6486 | 0,6438 |

PR-AUC e ROC-AUC OOF são calculadas sobre todas as previsões OOF agregadas. Recall, precision e F1 OOF usam o limiar 0,289843, escolhido no treino para recall mínimo de 80%. As métricas médias de classe por fold usam o corte padrão do estimador.

## Decisões operacionais

Há dois públicos distintos:

1. **Triagem ampla da API:** `churn_previsto` usa limiar 0,289843. A taxa de alertas é 39,08% no OOF e 39,60% no teste.
2. **Piloto de negócio:** usa somente os 20% de maior probabilidade. O corte foi 0,515347 no OOF e 0,516267 no teste, com recall de 51,77% e 51,34%, respectivamente.

O primeiro prioriza identificar churns; o segundo respeita um orçamento de contatos. Um não substitui o outro.

## Métrica oficial de negócio

A métrica é a **redução incremental da taxa de churn em 90 dias** no top 20%. Essa população seria dividida aleatoriamente entre tratamento e controle:

**Redução incremental = churn do controle − churn do tratamento**

A meta é redução absoluta de pelo menos 3 pontos percentuais, com intervalo de confiança de 95%. A métrica mede o efeito da ação, evita atribuir ao modelo cancelamentos que ocorreriam sem intervenção e não exige inventar custos ou margens ausentes no dataset.

Não houve campanha real; a meta é futura.

## Métrica financeira futura

Quando houver margem, ARPU, custos de contato e oferta:

**ROI = margem incremental líquida da campanha ÷ custo total da campanha**

A meta inicial é ROI mínimo de 1,25 em 180 dias, com recuperação em até 90 dias.

## Uso, riscos e controles

A API apoia priorização humana e não deve produzir decisão adversa automática. Riscos incluem mudança de população, custos de falsos positivos, perda de churns, indisponibilidade de features, calibração, vieses, privacidade e degradação temporal.

No teste, o Brier score foi 0,133146. A análise exploratória de `Gender`, `Senior Citizen`, `Partner` e `Dependents` encontrou métricas diferentes entre grupos. Como alguns grupos são pequenos e não foram calculados intervalos de confiança, isso não comprova ausência de viés nem justifica remover features. Uso real exige avaliação temporal, calibração, equidade, privacidade e monitoramento.

## Evidências

| Entrega | Caminho |
|---|---|
| EDA e baseline | `notebooks/01_entendimento_e_preparacao.ipynb` |
| Modelagem e análises | `notebooks/02_modelagem_avaliacao.ipynb` |
| Base tratada | `data/processed/telco_tratado.csv` |
| Registro dos experimentos | `models/experimentos_modelos.csv` |
| Comparação final | `models/comparacao_modelos.csv` |
| Métricas estruturadas | `models/metricas_teste.json` |
| Modelo | `models/pipeline_churn_rf.joblib` |
| API e módulos | `src/` |
| Testes e validação | `tests/` e `docs/VALIDACAO.md` |
