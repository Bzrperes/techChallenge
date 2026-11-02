# Dados do projeto

O projeto usa o dataset público **Telco Customer Churn**, obtido do Kaggle pelo notebook [`01_entendimento_e_preparacao.ipynb`](../notebooks/01_entendimento_e_preparacao.ipynb).

A base original contém 7.043 registros e 33 colunas. O modelo utiliza 19 features e o alvo binário `Churn Value`; identificadores, geografia, constantes e colunas que revelam o alvo são excluídos.

A base tratada e versionada fica em:

`data/processed/telco_tratado.csv`

Seu SHA-256 na movimentação foi `31D61B29480418D7AA7BF20F27D16DFACC87587EF941211E7A035AFC497F695C`, confirmando que o conteúdo não mudou.

As 11 ausências de `Total Charges` de clientes com permanência zero recebem zero por regra determinística. Imputação para ausências inesperadas, padronização e codificação são aprendidas dentro dos pipelines somente com dados de treino.

Dados brutos são ignorados pelo Git. O download anônimo foi validado em cache vazio; se o Kaggle exigir autenticação no futuro, credenciais devem permanecer em `.env` local.
