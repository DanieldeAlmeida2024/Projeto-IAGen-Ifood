# Plano de testes

- CT-01 Leitura de CSV válido.
- CT-02 Rejeição de CSV com cabeçalho inválido.
- CT-03 Exclusão dos itens já pedidos pelo usuário.
- CT-04 Cold start do usuário sem histórico.
- CT-05 Rejeição de limite fora de 1..100.
- CT-06 (futuro) Avaliar SVD real com partição treino/teste, RMSE, precisão e vieses.

Executar: `python -m unittest discover -s tests -v`. O baseline só usa dados de simulação.
