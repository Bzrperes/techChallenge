# Decisões de integração

1. Campeão preservado: Random Forest da etapa 2; PR-AUC/AP = 0,6862 na CV registrada.
   Valores históricos de teste: AP 0,6656; ROC-AUC 0,8537; recall 0,5455;
   precision 0,6645; F1 0,5991. Não foram recalculados nesta entrega.
2. Removida a seleção por ROC-AUC e as configurações alternativas de RF/MLP.
3. Carregamento direto de `pipeline_churn_rf.joblib`, com validação de tipo,
   parâmetros críticos, colunas e classes. Essas verificações não provam a origem
   do arquivo: preservar o artefato original e sua versão continua necessário.
4. Limiar 0,5 e uso de `predict()` preservados. Em empate exato, a RF escolhe classe 0,
   coerente com o notebook. Nenhum novo ajuste de limiar foi feito.
5. Limpeza reaproveitada no treino e na inferência. Ausências categóricas preservadas
   para o imputador, sem virar texto "nan"; não altera os dados originais sem ausências.
6. Total Charges ausente: zero apenas para tenure=0, mediana aprendida para demais casos.
7. Removidas faixas de risco 0,4/0,7 sem sustentação experimental.
8. Aplicação criada por fábrica; testes executam lifespan, carga do arquivo e previsão.
9. Verificador opcional com dados reais compara API e pipeline sem retreinar.
10. Não foi adicionado ML Canvas nem refeita a modelagem: essas pendências das etapas
    anteriores permanecem separadas. Desempenho e impacto de negócio não são garantidos
    pela existência de uma API.
