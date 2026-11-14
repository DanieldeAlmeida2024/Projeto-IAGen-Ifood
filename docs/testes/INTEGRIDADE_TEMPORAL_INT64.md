# IAG-INT64-01 — Integridade na ingestão temporal

14/11/2026 · Escopo: CSV temporal e higiene do código Python.

## Defeito
O contrato aceitava dígitos positivos mas não limitava user_id, item_id e order_count a int64 antes da coerção pandas. Um valor acima de 9223372036854775807 poderia falhar fora da validação definida.

## Correção e teste
- Verificar limites de int64 com ValueError explícito antes de converter os dados.
- Testar overflow das três colunas separadamente em CSV temporário.
- Eliminar bytecode compilado do Git; o .gitignore já bloqueia novos artefatos.

## Escopo
Não modifica os resultados da avaliação sintética, o modelo SVD ou a validade externa dos dados.
