# Arquitetura — recomendador baseline

```mermaid
flowchart LR
  Operador --> CLI["principal.py"]
  CLI --> Regras["src/recomendacao.py"]
  CSV[("CSV simulado")] --> Regras
  Regras --> Ranking["Ranking de itens"]
```

Sem serviços externos ou SDKs para o baseline. O serviço SVD original permanece em estudos, requer dependências e correções próprias. A regra de popularidade é apenas uma referência reprodutível e não um motor de IA generativa.
