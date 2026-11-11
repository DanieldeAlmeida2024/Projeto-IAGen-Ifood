# Plano de testes — recomendações iFood

> **REC-DOC v1.0.0** · 11/11/2026 · referência `2ea2ca89830b115f77b6af6ab0b5e70a88781e4e`. Código analisado estaticamente; evitar afirmar treinamento de ML que não foi executado.

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
