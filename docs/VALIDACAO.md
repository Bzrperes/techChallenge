
# Validação do projeto

Este documento reúne as verificações realizadas durante a integração do modelo de churn com a API FastAPI, incluindo os testes iniciais e as validações posteriores.

## 1. Verificações iniciais

As primeiras verificações foram realizadas com dados sintéticos de 400 registros.

Foram verificados:

- Treinamento da Random Forest com a configuração definida na Etapa 2.
- Salvamento e carregamento do pipeline utilizando um arquivo temporário.
- Comparação das probabilidades e classes previstas pelo pipeline e pela função de inferência em 20 registros, sem diferenças nos resultados.
- Tratamento de `Total Charges` ausente: valor zero para clientes com `tenure=0` e `NaN` para clientes com `tenure=10`, permitindo a previsão após o tratamento.
- Rejeição de modelos com configuração diferente da Random Forest selecionada.
- Verificação da sintaxe dos módulos Python.

### Ambiente utilizado

As verificações foram realizadas com as seguintes versões:

| Biblioteca | Versão |
|---|---|
| Scikit-learn | 1.8.0 |
| Pandas | 2.2.3 |
| NumPy | 2.3.5 |
| Joblib | 1.5.3 |
| Pydantic | 2.14.1 |

Essas versões correspondem ao ambiente das verificações iniciais, não necessariamente ao ambiente utilizado nos notebooks.

### Limitações iniciais

Nessa primeira execução, não foi possível realizar os testes com Pytest e as chamadas HTTP da FastAPI, pois as dependências necessárias não estavam instaladas. A tentativa de instalação também não foi concluída devido a um timeout de rede.

O arquivo `pipeline_churn_rf.joblib` e a base completa não estavam disponíveis nesse ambiente. Por isso, não foi possível verificar a integração com o modelo original nem recalcular as métricas do dataset Telco.

## 2. Testes automatizados

Posteriormente, foram registrados no README os resultados da execução dos testes automatizados.

Comando utilizado:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Resultados:

- 19 testes aprovados.
- 2 avisos.
- Nenhuma falha.

Os testes verificam o pré-processamento, o carregamento do modelo, a validação dos dados de entrada e o funcionamento da API.

Os testes da API utilizam `TestClient` e verificam o carregamento do modelo, as respostas dos endpoints e o comportamento diante de entradas inválidas.

## 3. Verificação com o modelo original

Também foi registrada uma verificação utilizando o pipeline salvo na Etapa 2 e o arquivo `telco_tratado.csv`.

Comando utilizado:

```powershell
.\.venv\Scripts\python.exe -m src.verify --data notebooks/artefatos/telco_tratado.csv
```

Resultado registrado:

```text
OK: 20 clientes; API e pipeline concordam. Modelo não foi retreinado.
```

A comparação confirmou que a API e o pipeline retornaram os mesmos resultados para os 20 clientes avaliados.

Essa verificação foi realizada sem retreinar o modelo e não representa uma nova avaliação do desempenho preditivo.

## 4. Limitações das validações

Os testes automatizados utilizam dados sintéticos e verificam o funcionamento dos componentes do projeto. Eles não substituem a avaliação dos modelos realizada na Etapa 2.

A comparação entre a API e o pipeline utilizou 20 clientes. Portanto, os resultados confirmam a consistência nessa amostra, mas não garantem o funcionamento em todos os cenários possíveis.

Os resultados dos testes posteriores foram recuperados do README principal. As saídas completas do terminal e os detalhes dos dois avisos não foram incluídos neste documento.

As limitações das verificações iniciais foram mantidas para registrar o histórico da integração.
