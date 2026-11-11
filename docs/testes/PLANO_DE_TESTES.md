# Plano de testes — recomendações iFood

> **REC-DOC v1.1.0 proposta** · 11/11/2026 · referência `2ea2ca89830b115f77b6af6ab0b5e70a88781e4e`. Código analisado estaticamente; evitar afirmar treinamento de ML que não foi executado.

## Casos já representados em código
| ID | Origem | Resultado esperado |
|---|---|---|
| REC-01 | `test_carregamento` | CSV com 4 interações aceito |
| REC-02 | `test_exclui_itens_ja_consumidos` | Usuário 1 recebe itens 12 e 13, não 10/11 |
| REC-03 | `test_cold_start` | Usuário novo recebe populares 12 e 13 |
| REC-04 | `test_limite_invalido` | `limite=0` rejeitado |
| REC-05 | `test_csv_incorreto` | CSV sem colunas obrigatórias rejeitado |

Os testes estão em `tests/test_recomendacao.py` e usam dados temporários. Eles **não exercitam `MFModel`**, nem demonstram métricas de qualidade de recomendação.

## Backlog de testes por risco
| ID | Prioridade | Cenário / hipótese | Critério |
|---|---|---|---|
| REC-06 | P0 | `user_id=0`, `item_id=-1`, `order_count=-1` | `ValueError` indicando linha |
| REC-07 | P1 | BOM UTF-8 e acentos | Arquivo parseado corretamente |
| REC-08 | P1 | Empate de popularidade | `item_id` menor vem primeiro |
| REC-09 | P1 | Tudo já consumido pelo usuário | Lista vazia |
| REC-10 | P0 | Importar e instanciar `MFModel` | Sem `ModuleNotFoundError/NameError` |
| REC-11 | P0 | Treinar SVD com CSV mínimo e CSR | Modelo ajustado sem erro |
| REC-12 | P0 | Salvar/carregar modelo com joblib | Scores reproduzíveis dentro de tolerância |
| REC-13 | P0 | Predição em usuário/item desconhecido | Regra de fallback explícita e testada |
| REC-14 | P1 | Divisão treino/teste por tempo/usuário | Sem vazamento de dados |
| REC-15 | P1 | Baseline versus SVD | Métrica comparável, sem inventar melhora |
| REC-16 | P1 | Privacidade dos dados de pedidos | Dados sintéticos e identificadores pseudonimizados |

## Estratégia de avaliação futura
Registrar precisão@K, recall@K, cobertura, diversidade e baseline, com sementes determinísticas e divisão adequada. `RMSE` por si só não certifica ranking útil. Definir limite para dados ausentes e cold start.

## Execução e rastreabilidade
```bash
python -m unittest discover -s tests -p "test_*.py" -v
```
Documentar versão de dependências, tamanho do conjunto, seed, commit, duração e métricas. Os casos REC-10–16 são **pendentes**; publicar relatórios somente após execução real.


## Revisão 1.1.0 — casos automatizados adicionais do SVD

| ID | Teste automático | Esperado | Estado inicial |
|---|---|---|---|
| REC-10 | Importar `DataLoader` e `MFModel` | Sem erro de import e interfaces corretas | Aprovado CI #43110171433 (suíte SVD) |
| REC-11 | Cabeçalho duplicado + tuplas duplicadas e matrix CSR | Soma `(1,101)` = 7, IDs [1,2] × [101,102,103] | Aprovado CI #43110171433 (suíte SVD) |
| REC-12 | Treinar SVD em CSR e salvar/carregar via joblib | Score reproduzível após load; modelo com `random_state=42` | Aprovado CI #43110171433 (suíte SVD) |
| REC-13 | Cold start e modelo sem treino | Desconhecido → média global; não treinado → `RuntimeError` | Aprovado CI #43110171433 (suíte SVD) |
| REC-17 | CSV com contagem negativa, texto ou decimal | `ValueError`, não criar matriz silenciosamente | Aprovado CI #43110171433 (suíte SVD) |
| REC-18 | Matriz vazia ou dimensões divergentes | Falha explícita antes de treinar | Aprovado CI #43110171433 (suíte SVD) |
| REC-19 | CSV de simulação `src/data/ifood_sim_data.csv` | Loader ignora só cabeçalhos repetidos conhecidos e carrega mais de 100 interações positivas | Aprovado CI #43110171433 (suíte SVD) |

**Isolamento de CI:** `test_recomendacao.py` permanece baseline sem terceiros; `test_mf_model.py` usa pandas/SciPy/Surprise. Não atribuir execução positiva aos novos casos sem verificar GitHub Actions. Os cinco testes baseline não são cobertura dos componentes SVD. **Validação de qualidade em aberto:** REC-14–16 exigem divisão sem vazamento e comparação com popularidade (por exemplo Recall@5 e cobertura). Arquivos `joblib` devem ser carregados só de fonte confiável.


### Evidência da suíte SVD (commit `57ed8bc6`)

[Run 43110171433](https://github.com/DanieldeAlmeida2024/Projeto-IAGen-Ifood/actions/runs/43110171433): `baseline` 5/5 e `svd` 7/7, ambos **success**. Esses testes não são equivalentes a cenários de ranking em dados reais ou E2E. REC-14 (split sem vazamento), REC-15 (baseline versus SVD) e REC-16 (política de dados reais) permanecem **PENDENTES**.


## Incremento REC-14 e REC-15 — avaliação offline em CI (1.2.0 proposta)

| Teste | Asserção | Código e evidência |
|---|---|---|
| REC-14a | Nenhuma tupla `(user_id,item_id)` do holdout aparece no treino | `tests/test_evaluation.py` |
| REC-14b | Mesmo seed → mesmos exemplos retidos | `tests/test_evaluation.py` |
| REC-14c | Cada usuário no holdout mantém histórico e cada item retido existe no catálogo de treino | `tests/test_evaluation.py` |
| REC-15a | Baseline e SVD usam mesmo treino/candidatos e scores variam entre [0,1] | `tests/test_evaluation.py` |
| REC-15b | Rodar `python -m src.evaluation --k 5 --seed 42` | Artefato `evaluation-summary.json` do GitHub Actions |
| REC-15c | Não afirmar ganho de SVD apenas por treinar/gerar métrica | Revisão manual de JSON e limites estatísticos |
| REC-20 | Dados degenerados, duplicados ou K inválido | `ValueError` no split/evaluator |

**Atenção:** métricas que aparecerem no CI são exclusivamente de teste sobre dados simulados. Deixar resultados em branco neste documento até execução, anotando SHA, data e JSON exato. REC-16 — privacidade de dados reais — permanece aberto.

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


## REC-21 — variabilidade entre seeds (1.3.0 proposta)

| Verificação | Critério | Evidência prevista |
|---|---|---|
| CLI seeds 42–46 | Um relatório por seed, mesmo split entre modelos na seed | `evaluation-robustness.json` |
| Repetibilidade | Mesma lista de seeds, mesma amostra → JSON igual | Unitário `test_evaluation_multi_seed.py` |
| Resumo | Média, std amostral, mínimo e máximo em métricas de ambos | Teste numérico e validação JSON |
| Comparação de métodos | `delta = SVD - popularidade` calculado por seed, não apenas em médias arredondadas | Teste e revisão do JSON |
| Dados degenerados | Seeds duplicadas, não numéricas, sem 2 replicações | `ValueError`, não produzir relato incorreto |

Essas verificações **não** substituem REC-16 (governança dos dados) ou teste em pedidos reais/partição temporal. Não publicar a hipótese de ganho antes de ler o relatório.


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

**REC-22**: rejeitar data sem timezone, arquivo legado sem `occurred_at`, IDs/contagens inválidos e split vazio; provar fronteira temporal e classificação de cold-start. Não marcar como testado antes do CI.
