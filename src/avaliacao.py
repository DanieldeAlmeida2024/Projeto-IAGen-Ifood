"""Avaliação determinística do baseline: leave-one-out por usuário.

Não é validação temporal nem estima qualidade em produção. Usa dados simulados.
"""
from collections import defaultdict
from src.recomendacao import recomendar


def avaliar_leave_one_out(interacoes, limite=5):
    if not isinstance(limite, int) or isinstance(limite, bool) or not 1 <= limite <= 100:
        raise ValueError('Limite deve estar entre 1 e 100')

    por_usuario = defaultdict(set)
    for usuario, item, quantidade in interacoes:
        if quantidade > 0:
            por_usuario[usuario].add(item)

    elegiveis = {
        usuario: max(itens) for usuario, itens in por_usuario.items()
        if len(itens) >= 2
    }
    acertos = 0
    for usuario, retido in sorted(elegiveis.items()):
        treino = [
            (dono, item, quantidade)
            for dono, item, quantidade in interacoes
            if not (dono == usuario and item == retido)
        ]
        ids = {item['item_id'] for item in recomendar(treino, usuario, limite)}
        acertos += int(retido in ids)

    total = len(elegiveis)
    return {
        'metodo': 'leave-one-out-deterministico',
        'usuarios_avaliados': total,
        'acertos': acertos,
        'limite': limite,
        'hit_rate_at_k': acertos / total if total else None,
    }
