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


## Pipeline experimental SVD (PR separado do baseline)

A branch `fix/svd-data-contract-and-ci-20261111` corrige o módulo SVD que antes não podia ser importado/treinado. Requer Python 3.11 com dependências em `requirements.txt`. O CSV de simulação inclui cabeçalhos repetidos; o carregador novo os remove explicitamente e soma interações duplicadas do mesmo par.

```bash
python -m pip install "numpy>=1.24,<2" "pandas>=2,<3" "scipy>=1.10,<2" "scikit-surprise==1.1.4" "joblib>=1.3"
python -m unittest discover -s tests -p 'test_mf_model.py' -v
python -m unittest discover -s tests -p 'test_recomendacao.py' -v
```

**Status:** implementação/validação do pipeline em andamento; nenhum resultado de ranking comparativo, integração com iFood ou produção. [Especificação](docs/projeto/ESPECIFICACAO.md) · [Testes versionados](docs/testes/PLANO_DE_TESTES.md).
