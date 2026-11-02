# Validação final

Validação realizada em 1º de novembro de 2026, seguindo os comandos do README e usando Python 3.13.15.

## Ambiente isolado

O conteúdo entregável foi copiado para uma pasta temporária, sem `.git`, ambientes virtuais ou caches. O comando abaixo criou um ambiente novo usando apenas os arquivos do projeto:

```powershell
uv sync --frozen
```

Resultado: Python 3.13.15 selecionado, `.venv` criado e 132 pacotes instalados conforme `uv.lock`, sem alteração do lockfile.

Durante a primeira simulação, o Pytest encontrou pastas temporária e de cache sem permissão no Windows. O projeto foi ajustado para criar um diretório temporário local e único por sessão em `.pytest_runs/` e desativar apenas o plugin de cache. As pastas temporárias são ignoradas pelo Git. Execuções consecutivas confirmaram a correção.

## Testes automatizados

```powershell
uv run python -m pytest -q
```

Resultado na reprodução limpa:

```text
20 passed, 1 warning
```

O aviso é intencional: um teste envia categoria desconhecida ao `OneHotEncoder` e verifica que ela é tratada com `handle_unknown="ignore"`. Não houve falha.

Os testes cobrem carga e validação do artefato, equivalência entre API e pipeline, endpoints, entradas inválidas, indisponibilidade do modelo, tratamento de dados, categoria desconhecida e seleção do limiar.

## Modelo e integração

```powershell
uv run python -m src.verify --data notebooks/artefatos/telco_tratado.csv
```

Resultado:

```text
OK: 20 clientes; API e pipeline concordam. Modelo não foi retreinado.
```

O arquivo `models/pipeline_churn_rf.joblib` carregou sem erro ou aviso de incompatibilidade. Ele contém um dicionário com `pipeline`, `limiar`, `meta_recall` e `modelo`. O limiar carregado foi 0,2898431181242871.

## API em execução

A API foi iniciada com Uvicorn e chamada por HTTP real. A porta 8000 já estava ocupada na máquina de validação; por isso a evidência final foi coletada na porta alternativa 8766. O comando documentado no README permanece correto quando a porta padrão está livre.

`GET /health` retornou HTTP 200:

```json
{"status":"ok","modelo_carregado":true}
```

`POST /predict`, usando o payload completo do README, retornou HTTP 200:

```json
{
  "probabilidade_churn": 0.7080131434658807,
  "churn_previsto": true,
  "limiar": 0.2898431181242871,
  "modelo": "Random Forest (Etapa 2)"
}
```

## Notebooks

Os dois notebooks foram reexecutados com o ambiente fixado e na ordem documentada:

```powershell
uv run python -m nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=600 notebooks/01_entendimento_e_preparacao.ipynb
uv run python -m nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=1800 notebooks/02_modelagem_avaliacao.ipynb
```

Resultados:

- Etapa 1: 23 células de código, contagens de 1 a 23, sem erro;
- Etapa 2: 33 células de código, contagens de 1 a 33, sem erro;
- nenhuma saída contém caminho local ou nome de usuário;
- o dataset público foi obtido sem chave do Kaggle na validação;
- o aviso de não convergência da configuração inicial da MLP permanece visível, sem caminho local, e está registrado como limitação no Model Card;
- `n_jobs=1` evita falhas e rastros de limpeza do Joblib no Windows sem mudar o protocolo experimental.

A reexecução reproduziu o limiar 0,289843, as métricas registradas e os três arquivos da pasta `models/`.

## Script de reprodução do campeão

O módulo `src.train` também foi executado na cópia isolada, direcionando a saída para um arquivo temporário para não substituir o artefato entregue. Ele reproduziu Python 3.13.15, Scikit-Learn 1.9.1, limiar 0,2898431181242871 e todas as métricas de validação e teste registradas no notebook.

## Resultado geral

Instalação bloqueada pelo lockfile, testes, verificador, endpoints e notebooks foram reproduzidos com sucesso. As limitações remanescentes são metodológicas e estão descritas no Model Card: teste previamente observado, ausência de validação externa/temporal, calibração e vieses não avaliados, e métricas de negócio ainda não medidas em campanha real.
