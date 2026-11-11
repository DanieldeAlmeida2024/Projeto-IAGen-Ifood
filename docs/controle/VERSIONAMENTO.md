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
