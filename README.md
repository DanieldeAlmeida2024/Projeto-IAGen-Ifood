# Sistemas de recomendação — avaliação offline de pedidos simulados

Laboratório de recomendação de itens alimentícios com **Python**, utilizando interações sintéticas para estudar ranking por popularidade, fatoração de matrizes (SVD) e avaliação offline. O foco é comparar qualidade de recomendação, cobertura de catálogo e comportamento em diferentes divisões dos dados.

## Abordagens

- **Popularidade:** ranking determinístico pela frequência de itens, excluindo itens já presentes no histórico do usuário.
- **SVD:** protótipo de filtragem colaborativa para explorar fatores latentes e recomendações personalizadas.
- **Avaliação offline:** divisão de dados para treino e teste, Hit Rate@K e cobertura de catálogo.
- **Análise multisseed:** comparação da sensibilidade dos resultados entre diferentes seeds.
- **Validação temporal:** formato de interações com `occurred_at` em ISO 8601 e corte cronológico para experimentos separados.

Todos os dados são **simulados**. Não há integração com a plataforma iFood nem utilização de informações reais de clientes.

## Executar baseline

Python 3.10+ para a CLI baseada na biblioteca padrão:

```bash
python principal.py --usuario 1 --quantidade 5
python principal.py --usuario 999 --quantidade 5
python principal.py --avaliar --quantidade 5
```

O arquivo de dados é `src/data/ifood_sim_data.csv`. A opção `--arquivo` permite selecionar outro CSV compatível.

## Avaliar os modelos

Para os componentes científicos, utilize Python 3.11 e as dependências declaradas em `requirements.txt`.

```bash
python -m pip install -r requirements.txt
python -m src.evaluation --k 5 --seed 42
python -m src.evaluation_multi_seed --k 5 --seeds 42,43,44,45,46
python -m unittest discover -s tests -p 'test_*.py' -v
```

O modelo temporal aceita dados com timestamps e realiza validação de corte, sem extrapolar os resultados para o comportamento de produção:

```bash
python -m src.temporal_validation --arquivo tests/fixtures/temporal_interactions.csv --cutoff 2026-01-02T00:00:00Z
```

## Resultados experimentais

O [workflow de avaliação offline](https://github.com/DanieldeAlmeida2024/Projeto-IAGen-Ifood/actions/runs/43117017363) registrou as seguintes métricas sobre dados sintéticos, com cinco seeds (42–46):

| Métrica @5, média das seeds | Popularidade | SVD |
| --- | ---: | ---: |
| Hit Rate@5 | **17,6%** | 13,4% |
| Cobertura do catálogo@5 | 22,1% | **37,7%** |

O SVD apresentou maior cobertura, mas menor Hit Rate médio nesse experimento. Os números descrevem o conjunto sintético utilizado; não constituem validação em ambiente real, estimativa de eficácia de produto ou conclusão estatística generalizável.

## Estrutura do projeto

- `principal.py`: interface de linha de comando e baseline.
- `src/recomendacao.py`: leitura de dados e regras do ranking por popularidade.
- `src/model/`: componentes de fatoração de matrizes.
- `src/evaluation.py`, `src/evaluation_multi_seed.py`: avaliação de ranking.
- `src/temporal_validation.py`: contrato e divisão temporal.
- `tests/`: testes de dados, modelos, ranking e avaliação.
- `Documentação/`: material acadêmico e especificações do domínio.

## Documentação

[Especificação do experimento](docs/projeto/ESPECIFICACAO.md) · [Arquitetura](docs/ARQUITETURA.md) · [UML](docs/uml/README.md) · [Plano de testes](docs/testes/PLANO_DE_TESTES.md) · [Versionamento](docs/controle/VERSIONAMENTO.md).

**Escopo:** estudo de recomendação e avaliação científica com dados simulados, sem publicação de API para usuários finais.
