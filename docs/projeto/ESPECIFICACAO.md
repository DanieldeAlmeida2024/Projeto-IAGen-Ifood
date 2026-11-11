# Recomendação de itens — especificação do projeto iFood

> **REC-DOC v1.1.0 proposta** · 11/11/2026 · referência `2ea2ca89830b115f77b6af6ab0b5e70a88781e4e`. Código analisado estaticamente; evitar afirmar treinamento de ML que não foi executado.

## Objetivo real e duas estratégias de recomendação
O repositório investiga recomendações de itens a partir de interações de pedidos simuladas. Existem **dois caminhos diferentes** que não devem ser confundidos:
1. **Baseline funcional por popularidade:** `src/recomendacao.py` usa biblioteca padrão, lê interações em CSV e recomenda itens mais pedidos, descartando os já consumidos pelo usuário.
2. **Protótipo de fatoração de matrizes:** `src/model/mf_model.py` define `MFModel` com Surprise/SVD, mas a implementação histórica tinha inconsistências de importação/caminhos e usava `pd.DataFrame` sem importar pandas. A branch 1.1.0 propõe correção e testes; antes de CI aprovado ainda não se deve chamá-lo validado.

O repositório contém também documentos próprios em `Documentação/` (arquitetura, estratégia de testes, dados/features, MLOps, negócio). Esta versão em Markdown descreve a implementação verificada e registra os pontos divergentes para facilitar o alinhamento.

## Baseline de popularidade: contrato
`carregar_interacoes(caminho)` exige cabeçalhos `user_id,item_id,order_count`, usa `utf-8-sig`, converte três campos para inteiros e rejeita linhas com usuário/item abaixo de 1 ou contagem negativa. A saída é uma lista de tuplas `(usuario,item,quantidade)`.

`recomendar(interacoes,usuario,limite=5)` requer `usuario>0` inteiro e `1<=limite<=100`. Constrói conjunto de itens com pedido positivo pelo usuário, soma quantidades por item globalmente, remove itens já consumidos e ordena por pontuação descendente e `item_id` ascendente como desempate. Retorna `[{item_id,pontuacao}]`.

**Cold start:** usuário sem histórico recebe itens populares globalmente. **Limitações:** não é personalização por similaridade; não usa janela temporal, preço, disponibilidade, estoque, privacidade de dados nem avaliação offline no ranking.

## SVD (pesquisa, não produção)
`MFModel` usa `SVD(n_factors=50,n_epochs=20,lr_all=0.005,reg_all=0.02)`, tenta converter matriz esparsa em DataFrame, gerar `Reader`, construir `Dataset`, treinar e prever scores, além de persistir modelo via `joblib`. A inspeção encontrou `src.model_layer.interfaces` e `src.data_layer.data_loader` referenciados sem correspondência com `src/model/` e `src/data/` atuais, e `pd` ausente. Antes de rodar treinamento, corrigir interfaces e teste de integração do pipeline.

## Dados e arquitetura
O caminho de dados declarado no repositório é `src/data/ifood_sim_data.csv`. A branch 1.1.0 corrige o import para `src.data.interfaces`, faz caminho padrão absoluto ao lado do arquivo `src/data/ifood_sim_data.csv`, identifica explicitamente cabeçalhos repetidos no CSV de simulação e **agrega** duplicatas `(user_id,item_id)` por soma, sem aceitar texto arbitrário/contagens negativas. Monta CSR com `user_ids` e `item_ids` alinhados. Evitar registrar pedidos pessoais reais até existir política explícita de privacidade/retensão.

## Comandos para o baseline
```bash
python -m unittest discover -s tests -v
python -m compileall -q src/recomendacao.py tests
```

## Próximas decisões de produto
Definir objetivo mensurável (CTR, conversão, diversidade), conjuntos treino/validação/teste sem vazamento, critérios de cold start, esquema de dados e política de re-treino. Não aprovar o SVD com base nos testes do baseline.


## Pipeline SVD experimental — revisão 1.1.0 proposta

| Etapa | Contrato concreto | Falha corrigida / limite |
|---|---|---|
| `DataLoader.load_raw_data` | Arquivo CSV, colunas `user_id,item_id,order_count`, inteiros não negativos e IDs positivos | Descarta **apenas** cabeçalhos exatamente repetidos dentro do CSV, agrega pares duplicados (soma), rejeita valores inválidos |
| `get_interaction_matrix` | Matriz CSR, `user_ids` e `item_ids` ordenados e estáveis | Alinhamento entre índices internos e IDs de negócio documentado |
| `MFModel.train` | CSR + `user_ids` + `item_ids` (mesmas dimensões) | `COO` evita densificar matriz, valida finitude e não negatividade; Surprise SVD com `random_state=42` |
| `predict_score` | Score `float` para IDs conhecidos | IDs desconhecidos retornam **média global do treino**, explicitamente; sem modelo treinado há `RuntimeError` |
| `save_model`/`load_model` | `joblib` inclui modelo, IDs, média global e seed | Carrega/replica score em dataset sintético; usar apenas arquivos locais confiáveis (pickle/joblib não seguro para dados externos) |
| CI | Job `baseline` só stdlib; job `svd` com numpy<2, pandas, scipy, scikit-surprise, joblib | Condição técnica de compatibilidade da ABI; sucesso de testes de implementação **não** prova qualidade de ranking |

**Não confundir com IA generativa.** O SVD aprende fatores latentes dos pedidos simulados e tenta aproximar `order_count` como rating. Sem holdout temporal, ranking offline, baseline equivalente, disponibilidade de itens e dados com autorização, este modelo não tem evidência de precisão ou benefício real. Ver [UML específico](../uml/04_pipeline_svd.puml).


**Resultado comprovado da proposta 1.1.0:** [CI #43110171433](https://github.com/DanieldeAlmeida2024/Projeto-IAGen-Ifood/actions/runs/43110171433) aprovou 5 testes do baseline e 7 do SVD com Python 3.11/Surprise 1.1.4. O SVD foi exercitado em dataset sintético, mas **nenhuma métrica comparativa de recomendação foi calculada**; a expressão "modelo validado" refere-se apenas à execução técnica dos contratos de software.
