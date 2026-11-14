"""Recomendador simples, reprodutível e independente de APIs externas."""
import csv
from collections import Counter
from pathlib import Path

COLUNAS = {'user_id', 'item_id', 'order_count'}

def carregar_interacoes(caminho):
    arquivo = Path(caminho)
    with arquivo.open('r', encoding='utf-8-sig', newline='') as entrada:
        leitor = csv.DictReader(entrada)
        if not leitor.fieldnames or not COLUNAS.issubset(set(leitor.fieldnames)):
            raise ValueError('CSV sem colunas user_id, item_id e order_count')
        resultado = []
        for numero, linha in enumerate(leitor, start=2):
            if all((linha.get(coluna) or '').strip() == coluna for coluna in COLUNAS):
                continue  # cabeçalho repetido em bases CSV exportadas/concatenadas
            try:
                usuario = int(linha['user_id'])
                item = int(linha['item_id'])
                quantidade = int(linha['order_count'])
            except (TypeError, ValueError) as erro:
                raise ValueError(f'Linha {numero} com identificador ou contagem inválida') from erro
            if usuario < 1 or item < 1 or quantidade < 0:
                raise ValueError(f'Linha {numero} possui valor negativo ou identificador inválido')
            resultado.append((usuario, item, quantidade))
    return resultado

def recomendar(interacoes, usuario, limite=5):
    if not isinstance(usuario, int) or usuario <= 0:
        raise ValueError('Informe usuário inteiro positivo')
    if not isinstance(limite, int) or not 1 <= limite <= 100:
        raise ValueError('Limite deve estar entre 1 e 100')
    consumidos = {item for dono, item, quantidade in interacoes if dono == usuario and quantidade > 0}
    pontuacao = Counter()
    for _, item, quantidade in interacoes:
        if quantidade > 0:
            pontuacao[item] += quantidade
    itens = [(item, total) for item, total in pontuacao.items() if item not in consumidos]
    itens.sort(key=lambda x: (-x[1], x[0]))
    return [{'item_id': item, 'pontuacao': total} for item, total in itens[:limite]]
