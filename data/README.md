# Dados do projeto

O projeto utiliza o dataset público **Telco Customer Churn**, disponibilizado no Kaggle, com informações de clientes, serviços e pagamentos de uma operadora de telecomunicações.

O notebook [`01_entendimento_e_preparacao.ipynb`](../notebooks/01_entendimento_e_preparacao.ipynb) faz o download pelo `kagglehub`. Em condições normais, o dataset público não exige chave; se o Kaggle solicitar autenticação ou limitar o acesso, as credenciais devem ficar somente em um `.env` local e nunca ser versionadas.

A base original possui 7.043 registros e 33 colunas. O projeto usa 19 features e o alvo binário `Churn Value`; identificadores, informações geográficas, constantes e colunas que revelam o alvo são excluídos.

O arquivo tratado e versionado fica em `notebooks/artefatos/telco_tratado.csv`. A conversão de `Total Charges` e a seleção inicial ocorrem na Etapa 1; imputação, padronização e codificação são aprendidas dentro dos pipelines usando somente os dados de treino.

O arquivo bruto não é versionado. O código produtivo de carga e limpeza está em `src/data.py`.
