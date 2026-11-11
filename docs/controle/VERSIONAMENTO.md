# Versionamento documental — recomendador iFood

**REC-DOC v1.1.0 proposta**, 11/11/2026, código `2ea2ca89830b115f77b6af6ab0b5e70a88781e4e`.

| ID | Documento | Versão |
|---|---|---|
| REC-PROJ | [Especificação](../projeto/ESPECIFICACAO.md) | 1.1.0 proposta |
| REC-UML | [Diagramas](../uml/README.md) | 1.1.0 proposta |
| REC-TST | [Testes](../testes/PLANO_DE_TESTES.md) | 1.1.0 proposta |

**MAJOR:** trocar contrato de dados ou estratégia em produção; **MINOR:** nova feature/algoritmo/avaliação; **PATCH:** ajuste editorial. Cada mudança deve registrar commit, dados sintéticos, execução e métricas. Não converter hipótese de MLOps em evidência de resultado.


## Histórico da correção de pipeline

| Versão | Base de código | Mudança | Evidência |
|---|---|---|---|
| 1.0.0 | `2dacf1df0e850a1de2e2e50cc94a73e0c9b4bf7c` | Baseline de popularidade + SVD não executável | 5 testes simples do baseline |
| 1.1.0 (proposta) | branch `fix/svd-data-contract-and-ci-20261111` | Corrige importações `src.data`/`src.model`, CSV com cabeçalhos repetidos, CSR sem densificar, mapeamento externo, cold start, save/load, testes SVD e CI segregado | Jobs `baseline` e `svd` aprovados no CI #43110171433; ainda sem métrica de melhoria |

**SemVer documental:** MINOR por habilitar o pipeline experimental antes indisponível; **não** é release de serviço, não há integração com iFood nem modelo generativo. O conjunto de simulação foi preservado; nenhuma linha bruta foi ocultada ou alterada.


### Registro de execução verificada

- [GitHub Actions **43110171433**](https://github.com/DanieldeAlmeida2024/Projeto-IAGen-Ifood/actions/runs/43110171433), commit `57ed8bc61bad1ca99f9293eb0e33c3fa8e5181bd`.
- **Baseline sem dependências:** `Ran 5 tests ... OK` — **5/5** aprovados.
- **SVD com SciPy/pandas/Surprise:** `Ran 7 tests ... OK` — **7/7** aprovados. O job compilou `scikit-surprise==1.1.4` no Python 3.11 com NumPy 1.26.4, sem acesso a serviços externos.
- Os 12 testes confirmam loader/matriz, treinamento, score, cold start e persistência experimental; **não medem precision@K, recall@K nem superação do baseline**. Nenhum modelo de produção foi treinado, publicado ou usado com pedidos reais.


## Registro de integração e novo experimento (11/11/2026)

| Marco | Commit / PR | Execução | Resultado |
|---|---|---|---|
| Versão 1.1.0 integrada na `main` | PR #3 · squash `ca2dae06ee0de076cc47c6b412f065f846c6049b` | [CI main #43112834496](https://github.com/DanieldeAlmeida2024/Projeto-IAGen-Ifood/actions/runs/43112834496) | Baseline e SVD aprovados; sem deploy |
| 1.2.0 **proposta** | `feat/offline-ranking-evaluation-20261111` | Job CI inclui novos testes offline e artefato JSON | Avaliação e métricas aguardando CI |

**Novos contratos para 1.2.0:** holdout leave-one-out por usuário com seed fixa; nenhum par usuário-item repetido entre treino e teste; cada item retido aparece no catálogo de treino; popularidade e SVD recebem os mesmos candidatos filtrados; avaliação de `hit_rate@K` e `catalog_coverage@K`. Isso é **uma comparação em CSV sintético**, não performance real.

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


## Versão 1.3.0 proposta — estabilidade por seed (11/11/2026)

| Marco | Commit e contexto | Resultado |
|---|---|---|
| **1.2.0 integrada** | [PR #4](https://github.com/DanieldeAlmeida2024/Projeto-IAGen-Ifood/pull/4), squash `78f0ef5eda3683cca618be5efb005fcc898ce5d5` | Relatório com seed 42, HitRate@5 0,15 popularidade vs 0,12 SVD; métricas do CSV sintético |
| **1.3.0 proposta** | `feat/multi-seed-ranking-robustness-20261111` | Relatório JSON de **5 seeds (42–46)**, média/desvio-padrão/faixa de HitRate@5 e cobertura; teste da comparação, CI pendente |

**Interpretação:** repetições com sementes diferentes no mesmo dataset sintético ajudam a identificar instabilidade de ranking. Não são experimentos independentes nem intervalos de confiança estatística; não autorizam concluir superioridade, causalidade ou generalização.


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


### Evidência de execução da revisão 1.4.0
- [CI #43119222792](https://github.com/DanieldeAlmeida2024/Projeto-IAGen-Ifood/actions/runs/43119222792): **28 testes PASS** (baseline 5, SVD 7, avaliação única 5, multisseed 5, temporal 6).
- Relatório `synthetic-temporal-validation`: 5 linhas pré-cutoff, 5 pós-cutoff, 2 elegíveis ao ranking, 1 cold-start de usuário, 1 cold-start de item e 1 item já consumido. Todos os dados são fictícios.
- Esse resultado valida separação temporal do exemplo, **não** avalia HitRate temporal do SVD, e não permite inferir desempenho sobre pedidos reais.

O SHA do commit documental posterior deve voltar a passar no CI antes do merge.


### Errata de contagem — 11/11/2026

A referência inicial a 23 testes no [CI #43119222792](https://github.com/DanieldeAlmeida2024/Projeto-IAGen-Ifood/actions/runs/43119222792) era um **erro aritmético de documentação**. O total correto é **28 testes**: 5 baseline + 7 SVD + 5 avaliação seed única + 5 multisseed + 6 temporais. Todos os cinco grupos tiveram execução `OK` no log. Não houve mudança nas regras do modelo, dados de teste nem resultados; a errata não implica execução de novos testes. O CI na main do merge #6 também foi concluído com sucesso no [run #43120014628](https://github.com/DanieldeAlmeida2024/Projeto-IAGen-Ifood/actions/runs/43120014628).
