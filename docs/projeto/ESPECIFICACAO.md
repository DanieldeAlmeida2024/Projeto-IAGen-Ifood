# Recomendação de itens — especificação do projeto iFood

> **REC-DOC v1.0.0** · 11/11/2026 · referência `2ea2ca89830b115f77b6af6ab0b5e70a88781e4e`. Código analisado estaticamente; evitar afirmar treinamento de ML que não foi executado.

## Objetivo real e duas estratégias de recomendação
O repositório investiga recomendações de itens a partir de interações de pedidos simuladas. Existem **dois caminhos diferentes** que não devem ser confundidos:
1. **Baseline funcional por popularidade:** `src/recomendacao.py` usa biblioteca padrão, lê interações em CSV e recomenda itens mais pedidos, descartando os já consumidos pelo usuário.
2. **Protótipo de fatoração de matrizes:** `src/model/mf_model.py` define `MFModel` com Surprise/SVD, mas apresenta inconsistências de importação/caminhos e usa `pd.DataFrame` sem importar pandas. Não é válido chamá-lo de pipeline treinado e testado.

O repositório contém também documentos próprios em `Documentação/` (arquitetura, estratégia de testes, dados/features, MLOps, negócio). Esta versão em Markdown descreve a implementação verificada e registra os pontos divergentes para facilitar o alinhamento.

## Baseline de popularidade: contrato
`carregar_interacoes(caminho)` exige cabeçalhos `user_id,item_id,order_count`, usa `utf-8-sig`, converte três campos para inteiros e rejeita linhas com usuário/item abaixo de 1 ou contagem negativa. A saída é uma lista de tuplas `(usuario,item,quantidade)`.

`recomendar(interacoes,usuario,limite=5)` requer `usuario>0` inteiro e `1<=limite<=100`. Constrói conjunto de itens com pedido positivo pelo usuário, soma quantidades por item globalmente, remove itens já consumidos e ordena por pontuação descendente e `item_id` ascendente como desempate. Retorna `[{item_id,pontuacao}]`.

**Cold start:** usuário sem histórico recebe itens populares globalmente. **Limitações:** não é personalização por similaridade; não usa janela temporal, preço, disponibilidade, estoque, privacidade de dados nem avaliação offline no ranking.

## SVD (pesquisa, não produção)
`MFModel` usa `SVD(n_factors=50,n_epochs=20,lr_all=0.005,reg_all=0.02)`, tenta converter matriz esparsa em DataFrame, gerar `Reader`, construir `Dataset`, treinar e prever scores, além de persistir modelo via `joblib`. A inspeção encontrou `src.model_layer.interfaces` e `src.data_layer.data_loader` referenciados sem correspondência com `src/model/` e `src/data/` atuais, e `pd` ausente. Antes de rodar treinamento, corrigir interfaces e teste de integração do pipeline.

## Dados e arquitetura
O caminho de dados declarado no repositório é `src/data/ifood_sim_data.csv`. O `DataLoader` existente pivoteia por usuário/item e monta uma matriz CSR, mas referencia `from interfaces import IDataLoader` e tem caminho padrão `data/ifood_sim_data.csv` divergente. Evitar registrar pedidos pessoais reais até existir política explícita de privacidade/retensão.

## Comandos para o baseline
```bash
python -m unittest discover -s tests -v
python -m compileall -q src/recomendacao.py tests
```

## Próximas decisões de produto
Definir objetivo mensurável (CTR, conversão, diversidade), conjuntos treino/validação/teste sem vazamento, critérios de cold start, esquema de dados e política de re-treino. Não aprovar o SVD com base nos testes do baseline.
