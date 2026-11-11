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


## 1.2.0 proposta — comparação offline sem vazamento

O novo módulo `src/evaluation.py` compara SVD e popularidade sobre **exatamente o mesmo treino** e mesmo conjunto de candidatos não consumidos, com `leave_one_out` por usuário elegível e seed 42. Ao retirar um item, preserva pelo menos uma interação do usuário no treinamento e ao menos uma ocorrência global do item, evitando casos impossíveis na avaliação de ranking. Nenhum dado real de cliente é processado.

A métrica `hit_rate@5` identifica quantos usuários têm seu único item retido no Top 5. `catalog_coverage@5` é a fração dos itens de catálogo que aparece em ao menos uma lista de recomendação. Ambas são registradas para baseline e SVD; **não há meta de vitória artificial do SVD**, e o CI falha apenas se invariantes/contratos falharem. Este CSV não contém timestamps confiáveis: holdout aleatório reprodutível **não demonstra generalização temporal**, avaliação online ou ausência de outros vieses. Ponderação por número de pedidos e disponibilidade do produto também não é considerada.

**Comando:** `python -m src.evaluation --k 5 --seed 42` gera JSON; CI preserva o resumo como artefato. Não há treinamento de IA generativa. [Rastreabilidade](../testes/PLANO_DE_TESTES.md) · [UML](../uml/05_avaliacao_offline.puml).

## Evidência de avaliação offline — 11/11/2026

**Fonte reproduzível:** [GitHub Actions #43113533934](https://github.com/DanieldeAlmeida2024/Projeto-IAGen-Ifood/actions/runs/43113533934), commit `8c42c0e78948dfa37594bea499d773debd11a52d`, artefato `offline-ranking-evaluation/evaluation-summary.json`. O CI registrou **5 testes de baseline, 7 testes SVD e 5 testes da avaliação — 17 testes aprovados**.

| Medida | Popularidade | SVD | Leitura |
|---|---:|---:|---|
| Hit Rate@5 | **0.15 (15%)** | 0.12 (12%) | Popularidade teve maior taxa de acerto no conjunto sintético; diferença de 3 p.p., sem inferência estatística |
| Cobertura de catálogo@5 | 0.220779 (22,1%) | **0.415584 (41,6%)** | SVD recomendou maior proporção de itens distintos |
| Usuários / holdouts | 100 | 100 | Mesmo conjunto de avaliação, um par por usuário |
| Itens no catálogo treinado | 77 | 77 | Mesmos candidatos por usuário |
| Interações de treino | 1689 | 1689 | Mesmo treino, com seed 42 |

**Decisão técnica:** não adotar o SVD como substituto da popularidade somente por esta avaliação. Os dados são inteiramente simulados, sem tempo de pedidos e sem teste online; esses resultados não demonstram conversão, precisão em ambiente real nem diferenças estatisticamente significativas. Os próximos experimentos devem comparar métricas em mais seeds, intervalos de confiança e, caso haja dados adequados e autorizados, partição temporal.


## Experimento multisseed — contrato 1.3.0 proposto

A análise compara ambas as estratégias em **cada uma** das seeds 42, 43, 44, 45 e 46 com o mesmo split para os modelos naquela rodada. `src/evaluation_multi_seed.py` reutiliza `evaluate_ranking` e fornece métricas individuais, média, desvio-padrão amostral, mínimo e máximo para HitRate@K, cobertura@K e diferença `SVD - popularidade`, sem esconder resultados desfavoráveis. A CLI rejeita lista vazia, seeds duplicadas ou menos de duas seeds.

**Validação:** o CI produz `evaluation-robustness.json` no job SVD, separado do artefato de seed única. Reportar também a faixa de usuários elegíveis por seed para impedir comparações equivocadas de denominadores. Esta análise ainda usa os mesmos pedidos **simulados e sem timestamp**, de modo que não fornece significância estatística nem teste temporal/online.


## Evidência do experimento multisseed (11/11/2026)

**Run:** [CI #43117017363](https://github.com/DanieldeAlmeida2024/Projeto-IAGen-Ifood/actions/runs/43117017363) sobre o commit `23be1ffbd5fff77f5c93d10dcb42ef85b4967b79`. Passaram **5 testes de baseline + 7 SVD + 5 avaliação de seed única + 5 avaliação multisseed = 22 testes**. Artefato `offline-ranking-multi-seed/evaluation-robustness.json` com 100 usuários avaliados em cada seed e catálogo de 77 itens.

| Métrica @5 (média de seeds 42–46) | Popularidade | SVD |
|---|---:|---:|
| Hit Rate@5 | **0,176 (17,6%)** | 0,134 (13,4%) |
| Desvio-padrão amostral de Hit Rate | 0,021909 | 0,021909 |
| Cobertura de catálogo@5 | 0,220779 (22,1%) | **0,376623 (37,7%)** |
| Desvio-padrão de cobertura | 0,009183 | 0,031812 |

**Diferença SVD − popularidade no Hit Rate@5:** média **−0,042 (−4,2 p.p.)**, mínimo −0,09, máximo 0,0, desvio-padrão 0,034205. Em **nenhuma das cinco seeds o SVD superou a popularidade em acerto**; houve empate em uma. Cobertura permaneceu maior para SVD. Os resultados são descritivos e dependentes: **não** significam teste de hipótese, intervalo de confiança ou qualidade real do produto. O histórico do PR #4 fica preservado.


## 1.4.0 proposta — contrato para validação temporal estrita

A versão documental 1.3.0 foi integrada em [PR #5](https://github.com/DanieldeAlmeida2024/Projeto-IAGen-Ifood/pull/5) pelo squash `88331ebfdc0405f4ab98ff44538ddabb56628a13`. O experimento de cinco seeds (42–46) mostrou HitRate@5 médio **17,6% popularidade vs 13,4% SVD** usando o mesmo CSV sintético, mas os dados **não têm data de pedido**.

Nova branch: `feat/temporal-data-validation-20261111`, que **não inventa timestamps** no CSV legado. Adiciona contrato separado de CSV com `user_id,item_id,order_count,occurred_at` e timestamps ISO 8601 com timezone explícito; o corte de treino/avaliação é **global** (treino estritamente anterior ao cutoff, teste a partir dele). A validação indica quantas linhas de teste são elegíveis ao ranking, quantas são cold-start de usuário, item ou repetição já consumida, sem treinar modelos com o futuro.

**Dados:** apenas fixture artificial `tests/fixtures/temporal_interactions.csv`. Não existe integração com iFood real. Não declara HitRate temporal ou benefício em negócio.

**Gates:** `tests/test_temporal_validation.py` e CLI `python -m src.temporal_validation --arquivo ... --cutoff ...` no CI com relatório `synthetic-temporal-validation`; execução real pendente no primeiro commit. O próximo passo será avaliação de ranking pós-cutoff apenas sobre dados cronológicos autorizados. Retenção, pseudonimização e LGPD devem ser definidos antes de importar qualquer registro pessoal.
