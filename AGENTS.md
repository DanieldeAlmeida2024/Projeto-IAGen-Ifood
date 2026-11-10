# Instruções para agentes — projeto de recomendação

Distinga o baseline de popularidade (src/recomendacao.py) do modelo SVD experimental (src/model/mf_model.py). Não alegar IA generativa, treinamento de dados reais, integração com iFood ou métricas sem evidência.

- Leia o CSV e a documentação antes de alterar o contrato de dados.
- Não utilizar dados pessoais reais no repositório.
- Manter baseline Python padrão isolado de dependências opcionais.
- Para mudanças SVD, corrigir paths/imports e medir desempenho antes de publicar.
- Testar com `python -m unittest discover -s tests -v`.
- Nomes de domínio em português, com compatibilidade do contrato CSV existente.
