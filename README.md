# Protótipo de recomendação de pedidos — projeto de estudo

Repositório acadêmico de recomendação de itens alimentícios a partir de dados simulados de pedidos. Inclui documentos de arquitetura, levantamento de dados e um modelo SVD experimental.

## O que é executável neste PR

Um **recomendador por popularidade**, feito com Python 3.10+ e apenas a biblioteca padrão. Ele ordena produtos pela quantidade simulada de pedidos e, para um usuário informado, não sugere os itens que já aparecem em seu histórico.

```powershell
python principal.py --usuario 1 --quantidade 5
python principal.py --usuario 999 --quantidade 5
python -m unittest discover -s tests -v
```

O CSV original é `src/data/ifood_sim_data.csv`. Para outro arquivo, use `--arquivo caminho.csv`. Resultados são educativos; **não representam um modelo treinado em pedidos reais, nem integração com o iFood**.

## Avaliação offline do baseline

```bash
python principal.py --avaliar --quantidade 5
```

O relatório mostra `usuarios_avaliados`, `acertos` e **HitRate@5**. Para cada usuário com pelo menos dois itens distintos, um item é separado (maior ID, critério determinístico) e excluído das interações daquele usuário antes de gerar recomendações. O resultado é um indicador didático em dados simulados, **não** uma estimativa de desempenho em produção ou um comparativo com SVD. O workflow `baseline.yml` valida apenas o algoritmo de popularidade, sem instalar pacotes científicos.

## Arquitetura
- `principal.py`: CLI.
- `src/recomendacao.py`: leitura e classificação determinística.
- `src/data/`: CSV de interações simulado e carregador legado.
- `src/model/`: protótipo SVD não validado nesta entrega.
- `Documentação/`: documentos acadêmicos pré-existentes.
- `docs/REQUISITOS.md`, `docs/ARQUITETURA.md`, `docs/UML.puml`, `docs/PLANO_TESTES.md`: documentação nova.
- `AGENTS.md`: instruções para ferramentas de IA.

## Limitações
O modelo SVD original precisa corrigir importações, treinar e avaliar com ambiente de bibliotecas compatível; não foi testado ou publicado como API neste PR. A CLI de popularidade é um baseline simples e reproduzível.

## Documentação técnica versionada

- [Especificação: popularidade × SVD](docs/projeto/ESPECIFICACAO.md)
- [UML dos algoritmos e dados](docs/uml/README.md)
- [Plano de testes e validação](docs/testes/PLANO_DE_TESTES.md)
- [Histórico de versões](docs/controle/VERSIONAMENTO.md)


## Pipeline experimental SVD (PR separado do baseline)

A branch `fix/svd-data-contract-and-ci-20261111` corrige o módulo SVD que antes não podia ser importado/treinado. Requer Python 3.11 com dependências em `requirements.txt`. O CSV de simulação inclui cabeçalhos repetidos; o carregador novo os remove explicitamente e soma interações duplicadas do mesmo par.

```bash
python -m pip install "numpy>=1.24,<2" "pandas>=2,<3" "scipy>=1.10,<2" "scikit-surprise==1.1.4" "joblib>=1.3"
python -m unittest discover -s tests -p 'test_mf_model.py' -v
python -m unittest discover -s tests -p 'test_recomendacao.py' -v
```

**Status:** implementação/validação do pipeline em andamento; nenhum resultado de ranking comparativo, integração com iFood ou produção. [Especificação](docs/projeto/ESPECIFICACAO.md) · [Testes versionados](docs/testes/PLANO_DE_TESTES.md).


## Avaliação offline reproduzível (1.2.0 proposta)
```bash
python -m src.evaluation --k 5 --seed 42
python -m unittest discover -s tests -p 'test_evaluation.py' -v
```

O resultado JSON compara **hit rate@5** e **cobertura de catálogo@5** contra o baseline, a partir de um holdout sem pares usuário-item duplicados entre treino/teste. Não existem timestamps na simulação: não é avaliação temporal, e as métricas não demonstram desempenho real. [Plano](docs/testes/PLANO_DE_TESTES.md) · [UML](docs/uml/05_avaliacao_offline.puml).

### Resultado atual da comparação (dataset sintético)

[CI #43113533934](https://github.com/DanieldeAlmeida2024/Projeto-IAGen-Ifood/actions/runs/43113533934) aprovou **17 testes** e produziu métricas com seed 42: **Hit Rate@5 de 15% para popularidade e 12% para SVD**, com cobertura de **22,1%** e **41,6%**, respectivamente. O SVD apresentou maior variedade, **não maior taxa de acerto**. Esses números não são resultados reais de pedidos nem provam significância estatística.


## Sensibilidade por seeds (1.3.0 proposta)
```bash
python -m src.evaluation_multi_seed --k 5 --seeds 42,43,44,45,46
python -m unittest discover -s tests -p 'test_evaluation_multi_seed.py' -v
```
Os resultados incluem a diferença `SVD - popularidade`, média, std, mínimo e máximo. **Não** são estimativas de eficácia real nem intervalos de confiança; nenhum serviço iFood é chamado.


## Evidência do experimento multisseed (11/11/2026)

**Run:** [CI #43117017363](https://github.com/DanieldeAlmeida2024/Projeto-IAGen-Ifood/actions/runs/43117017363) sobre o commit `23be1ffbd5fff77f5c93d10dcb42ef85b4967b79`. Passaram **5 testes de baseline + 7 SVD + 5 avaliação de seed única + 5 avaliação multisseed = 22 testes**. Artefato `offline-ranking-multi-seed/evaluation-robustness.json` com 100 usuários avaliados em cada seed e catálogo de 77 itens.

| Métrica @5 (média de seeds 42–46) | Popularidade | SVD |
|---|---:|---:|
| Hit Rate@5 | **0,176 (17,6%)** | 0,134 (13,4%) |
| Desvio-padrão amostral de Hit Rate | 0,021909 | 0,021909 |
| Cobertura de catálogo@5 | 0,220779 (22,1%) | **0,376623 (37,7%)** |
| Desvio-padrão de cobertura | 0,009183 | 0,031812 |

**Diferença SVD − popularidade no Hit Rate@5:** média **−0,042 (−4,2 p.p.)**, mínimo −0,09, máximo 0,0, desvio-padrão 0,034205. Em **nenhuma das cinco seeds o SVD superou a popularidade em acerto**; houve empate em uma. Cobertura permaneceu maior para SVD. Os resultados são descritivos e dependentes: **não** significam teste de hipótese, intervalo de confiança ou qualidade real do produto. O histórico do PR #4 fica preservado.

### Contrato temporal (1.4.0 proposta)
O arquivo de simulação original não possui data/hora; o contrato novo exige `occurred_at` em ISO 8601 com fuso. A CLI valida cortes cronológicos **sem treinar SVD** ou afirmar métricas reais: `python -m src.temporal_validation --arquivo tests/fixtures/temporal_interactions.csv --cutoff 2026-01-02T00:00:00Z`. Veja [plano](docs/testes/PLANO_DE_TESTES.md).
