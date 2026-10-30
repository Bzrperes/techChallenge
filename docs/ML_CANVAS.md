# ML Canvas | Previsão de Churn

## 1. Contexto e problema de negócio

Uma operadora de telecomunicações enfrenta o cancelamento de clientes e precisa identificar quais deles apresentam maior propensão ao churn. O desafio da FIAP propõe desenvolver um pipeline de Machine Learning que apoie essa identificação e disponibilize as previsões por meio de uma API REST.

No projeto, churn corresponde ao cancelamento do serviço. O objetivo é estimar a propensão ao cancelamento a partir das características disponíveis no dataset, sem afirmar que as previsões, por si só, reduzem a perda de clientes.

## 2. Objetivo do projeto

Desenvolver e avaliar um modelo supervisionado de classificação binária para prever churn de clientes de telecomunicações. A solução deve reunir preparação dos dados, comparação de modelos, escolha de um pipeline final e disponibilização das previsões por uma API FastAPI.

O resultado esperado é oferecer uma informação que possa apoiar a priorização de análises e eventuais ações de retenção. A implementação de campanhas de retenção não faz parte do escopo executado.

## 3. Stakeholders

| Stakeholder | Interesse ou responsabilidade |
|---|---|
| Diretoria da operadora | Compreender o risco de perda de clientes e avaliar possíveis decisões de retenção, conforme o contexto apresentado pela FIAP. |
| Equipe de retenção/relacionamento | Possível usuária das previsões para priorizar análises e contatos. **Papel proposto**, não validado com uma operadora real. |
| Equipe técnica e de dados | Preparar os dados, desenvolver, avaliar, integrar e manter o pipeline e a API no contexto do projeto acadêmico. |
| Clientes da operadora | Pessoas potencialmente afetadas por decisões orientadas pelas previsões; exigem atenção à privacidade e ao tratamento justo. |

Os stakeholders operacionais apresentados são uma interpretação do cenário acadêmico. Não houve levantamento de requisitos com uma empresa real.

## 4. Dados disponíveis

| Item | Definição registrada no projeto |
|---|---|
| Dataset | Telco Customer Churn, disponibilizado no Kaggle |
| Natureza | Dados tabulares históricos de clientes, serviços contratados e pagamentos |
| Base original | 7.043 registros e 33 colunas |
| Alvo | `Churn Value` |
| Classe 0 | Cliente que não cancelou o serviço |
| Classe 1 | Cliente que cancelou o serviço |
| Distribuição observada | 26,54% com churn e 73,46% sem churn |
| Variáveis preditoras | 19, sendo 3 numéricas e 16 categóricas |
| Arquivo tratado | `notebooks/artefatos/telco_tratado.csv` |

As variáveis numéricas são `Tenure Months`, `Monthly Charges` e `Total Charges`. As categóricas tratam de características dos clientes, serviços, contratos e meios de pagamento.

Na preparação foram excluídos identificadores, informações geográficas, colunas constantes e variáveis consideradas com risco de vazamento de informação. O pré-processamento está incorporado ao pipeline do Scikit-Learn.

A possibilidade de obter as 19 variáveis **antes do momento da previsão, em um cenário operacional real**, não foi verificada com uma operadora. Essa validação seria necessária para uso em produção.

## 5. Abordagem de Machine Learning

O problema foi formulado como **classificação binária supervisionada**.

- **Baseline:** Regressão Logística.
- **Modelos comparados:** Regressão Logística, Random Forest e `MLPClassifier`.
- **Separação dos dados:** 80% para treinamento e 20% para teste, de forma estratificada.
- **Validação cruzada:** cinco folds.
- **Métrica principal da seleção:** Average Precision, denominada PR-AUC no projeto.
- **Métricas complementares:** ROC-AUC, precision, recall e F1-score.
- **Modelo selecionado:** Random Forest ajustada com `GridSearchCV`.
- **Limiar mantido:** 0,5.

O pipeline final reúne pré-processamento e modelo e está salvo em `models/pipeline_churn_rf.joblib`.

## 6. Métricas técnicas e resultados observados

A PR-AUC foi utilizada como critério principal de comparação porque o projeto busca identificar a classe de churn, que corresponde a aproximadamente 26,54% dos registros.

| Métrica | Validação cruzada | Teste |
|---|---:|---:|
| PR-AUC / Average Precision | 0,6862 | 0,6656 |
| ROC-AUC | 0,8627 | 0,8537 |
| Precision | Não registrada aqui | 0,6645 |
| Recall | Não registrado aqui | 0,5455 |
| F1-score | Não registrado aqui | 0,5991 |

A Random Forest apresentou a maior PR-AUC média na validação cruzada. A Regressão Logística apresentou recall superior, o que mostra que a escolha também depende do objetivo de negócio. Esses resultados foram obtidos na Etapa 2 e não representam desempenho comprovado em produção.

## 7. Métricas de negócio

A FIAP exige a definição de uma métrica de negócio. Como não há no projeto uma operação real de retenção, custos de campanha ou resultados posteriores às previsões, **não foi possível medir impacto de negócio**.

| Indicador | Finalidade | Situação |
|---|---|---|
| Taxa de churn | Acompanhar a proporção de clientes que cancelam em determinado período | **Referência histórica do dataset:** 26,54%; não é uma taxa operacional atual |
| Redução da taxa de churn após ações de retenção | Avaliar se uma estratégia de retenção produz resultado | **Métrica de negócio proposta**, ainda sem meta, período ou medição |
| Custo por cliente retido | Avaliar a relação entre gastos de retenção e resultados alcançados | **Indicador complementar proposto**, sem dados para cálculo |

Para medir redução de churn de forma confiável, seria preciso definir período de acompanhamento, população-alvo, ações realizadas e uma forma de comparação que permita estimar seu efeito. O projeto não executou essa avaliação.

**Decisão pendente da equipe:** confirmar qual métrica de negócio será adotada formalmente e se a proposta de redução de churn representa o objetivo pretendido.

## 8. Utilização prevista da solução

A API FastAPI disponibiliza dois endpoints:

- `GET /health`: informa o estado da aplicação e se o modelo foi carregado.
- `POST /predict`: recebe as características de um cliente e retorna a probabilidade estimada de churn, a classe prevista, o limiar e o modelo utilizado.

Uma utilização potencial seria ordenar clientes por risco estimado para apoiar a análise de uma equipe responsável pela retenção. Essa aplicação é **uma possibilidade de negócio**, não uma funcionalidade de campanhas implementada ou testada no projeto.

## 9. Restrições, riscos e limitações

- Os dados são históricos e não representam necessariamente clientes de outra operadora ou período.
- O conjunto de teste já havia sido observado na Etapa 1; portanto, não é uma amostra inteiramente inédita ao projeto.
- O recall de 0,5455 indica que parte dos clientes que cancelaram não foi identificada pelo modelo.
- O limiar de 0,5 não foi escolhido com base em custos reais de ações de retenção.
- Não foram realizadas análise de calibração das probabilidades, auditoria de vieses por grupo nem avaliação em produção.
- Características dos clientes, como gênero e indicação de cliente idoso, exigem atenção em eventual uso comercial.
- As previsões devem apoiar análises humanas, e não ser tratadas como certeza de cancelamento.

## 10. Entregas e evidências do projeto

| Entrega | Evidência no repositório |
|---|---|
| Exploração de dados e baseline | `notebooks/TechChallenge.ipynb` |
| Comparação e avaliação dos modelos | `notebooks/02_modelagem_avaliacao.ipynb` |
| Modelo final | `models/pipeline_churn_rf.joblib` |
| Pré-processamento e inferência | `src/` |
| API FastAPI | `src/api.py` |
| Testes automatizados | `tests/` e `docs/VALIDACAO.md` |
| Desempenho, limitações e possíveis vieses | `docs/MODEL_CARD.md` |
| Instalação e execução | `README.md` |

Foram registrados 19 testes aprovados, dois avisos e nenhuma falha. A API e o pipeline retornaram resultados equivalentes para uma amostra de 20 clientes. Essas verificações avaliam a integração, não o impacto comercial do projeto.

## 11. Síntese e decisões pendentes

O projeto entregou um pipeline de classificação de churn com Random Forest e inferência por API. A seleção do modelo foi feita por PR-AUC média na validação cruzada, mantendo o limiar de 0,5.

Para completar o planejamento de negócio, ainda é necessário confirmar com a equipe a **métrica de negócio oficialmente adotada**. Também não houve validação de stakeholders, metas comerciais ou ações reais de retenção. Esses pontos permanecem identificados como propostas ou limitações, sem serem apresentados como resultados alcançados.

---

**Referências internas:** `README.md`, `data/README.md`, `docs/MODEL_CARD.md`, `docs/VALIDACAO.md`, `src/config.py` e os notebooks das Etapas 1 e 2.

**Referência do desafio:** PDF oficial *12MLET - Tech Challenge Fase 1*, Etapa 1: preenchimento do ML Canvas com stakeholders e métricas de negócio.
