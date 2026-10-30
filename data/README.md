
# Dados do projeto

O projeto utiliza o dataset Telco Customer Churn, disponibilizado no Kaggle, com informações sobre clientes de uma operadora de telecomunicações.

## Origem dos dados

O dataset original é obtido pelo notebook `TechChallenge.ipynb`, utilizando a biblioteca `kagglehub`.

Para realizar o download, é necessário configurar a credencial do Kaggle no arquivo `.env`, conforme as instruções do README principal.

O arquivo bruto não foi incluído no repositório.

## Dados tratados

O arquivo utilizado nas etapas de modelagem e validação está disponível em:

`notebooks/artefatos/telco_tratado.csv`

A base original contém 7.043 registros e 33 colunas. Após a seleção das variáveis, foram utilizadas 19 features e a variável alvo `Churn Value`.

O alvo possui duas classes:

- `0`: cliente que não cancelou o serviço.
- `1`: cliente que cancelou o serviço.

As variáveis selecionadas incluem informações sobre tempo de permanência, valores cobrados, serviços contratados e características dos clientes.

## Tratamento dos dados

Na preparação dos dados, foram considerados os seguintes tratamentos:

- Conversão de `Total Charges` para formato numérico.
- Tratamento de valores ausentes em `Total Charges`.
- Exclusão de identificadores, informações geográficas, colunas constantes e variáveis com risco de vazamento.
- Separação das variáveis numéricas e categóricas para o pré-processamento.

O pipeline do Scikit-Learn aplica os tratamentos necessários durante o treinamento e a inferência.

## Utilização

O arquivo `telco_tratado.csv` foi utilizado na Etapa 2 para treinamento e avaliação dos modelos.

Na Etapa 3, também foi utilizado para verificar a consistência das previsões entre a API e o pipeline original.

O módulo `src/data.py` contém as funções de carregamento e limpeza utilizadas no código produtivo.

As análises e os resultados estão documentados nos notebooks, enquanto as instruções de execução estão disponíveis no [README principal](../README.md).
