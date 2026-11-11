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
