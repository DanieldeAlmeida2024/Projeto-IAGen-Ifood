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
