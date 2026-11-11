# Requisitos iniciais

RF-01 Carregar interações CSV com usuário, item e quantidade.
RF-02 Recomendar itens pela popularidade acumulada.
RF-03 Não recomendar itens já consumidos pelo usuário informado.
RF-04 Prever cold start com ranking global.
RF-05 Expor CLI com parâmetros de usuário, limite e CSV.
RNF-01 Não usar dados reais e não acessar serviços externos.
RNF-02 Garantir resultado determinístico sob os mesmos dados.
RNF-03 Testar erros de formato/entrada e regras de ordenação.

O modelo SVD em src/model ainda não é coberto pelo MVP.
