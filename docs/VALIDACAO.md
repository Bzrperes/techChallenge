# Validação desta revisão

Verificações executadas com sucesso em dados sintéticos de 400 linhas:

- Treinamento com a configuração fixa da Random Forest da etapa 2.
- Salvamento em arquivo temporário e recarga do Pipeline.
- Igualdade exata de probabilidades e classes em 20 registros entre o pipeline direto
  e a função de inferência; nenhum arredondamento introduzido.
- Entrada Pydantic sem Total Charges: zero para tenure=0, NaN para tenure=10;
  ambos os casos geraram previsões após o tratamento.
- Rejeição de um modelo com profundidade divergente.
- Sintaxe de todos os módulos Python.

Ambiente dessas verificações: Python do runtime; scikit-learn 1.8.0, pandas 2.2.3,
NumPy 2.3.5, joblib 1.5.3, Pydantic 2.14.1. Não é certificação do ambiente do notebook.

Não executados neste ambiente:

- Suíte pytest e chamadas HTTP com FastAPI: pytest/FastAPI/httpx não estavam instalados;
  a tentativa de instalar as dependências falhou por timeout de rede.
- Paridade com o campeão real: o arquivo pipeline_churn_rf.joblib e a base completa
  não estavam incluídos nos anexos. Nenhum desempenho Telco foi recalculado.

Antes da entrega do grupo, executar os testes do README, copiar o modelo original,
executar src.verify com os dados reais e conferir /health e /predict em /docs.
Os testes de API fornecidos usam TestClient com contexto para executar o lifespan.
A verificação real é um comando separado, sem skip silencioso se faltar o modelo.
