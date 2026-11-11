# UML — recomendadores iFood

> **REC-DOC v1.1.0 proposta** · 11/11/2026 · referência `2ea2ca89830b115f77b6af6ab0b5e70a88781e4e`. Código analisado estaticamente; evitar afirmar treinamento de ML que não foi executado.

Estes diagramas distinguem baseline funcional e SVD experimental:

- [Dados e duas estratégias](01_visao_dados.puml)
- [Cálculo da popularidade](02_sequencia_baseline.puml)
- [Fluxo cold start](03_decisao_cold_start.puml)

Não confundir pipeline SVD desenhado com treinamento validado. Atualizar ao corrigir o `MFModel` e adicionar testes de treino e predição.

- [Pipeline CSR → Surprise/SVD → predição/cold start](04_pipeline_svd.puml) — fonte PlantUML adicional da revisão 1.1.0 proposta, não demonstra qualidade de recomendação.
