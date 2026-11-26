# Arquitetura — recomendadores iFood

Leia [duas estratégias de recomendação](projeto/ESPECIFICACAO.md) e [fluxos UML](uml/README.md). O baseline por popularidade não comprova o SVD.

O carregador SVD valida IDs e contagens do CSV sem conversão intermediária para ponto flutuante. A soma de cada par usuário/item é feita com inteiros do Python e validada antes de armazenar em `int64`. O CSV temporal usa um carregador separado.
