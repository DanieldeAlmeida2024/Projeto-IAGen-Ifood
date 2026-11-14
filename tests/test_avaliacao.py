import unittest
from src.avaliacao import avaliar_leave_one_out

class TestesAvaliacao(unittest.TestCase):
    def test_usuarios_sem_historico_suficiente_nao_inflam_metrica(self):
        self.assertEqual(
            {'metodo': 'leave-one-out-deterministico', 'usuarios_avaliados': 0,
             'acertos': 0, 'limite': 5, 'hit_rate_at_k': None},
            avaliar_leave_one_out([(1, 10, 1)], 5),
        )

    def test_recomendacao_usa_apenas_dados_de_treino_por_usuario(self):
        dados = [(1, 10, 2), (1, 20, 1), (2, 20, 4)]
        relatorio = avaliar_leave_one_out(dados, limite=1)
        self.assertEqual(1, relatorio['usuarios_avaliados'])
        self.assertEqual(1, relatorio['acertos'])
        self.assertEqual(1.0, relatorio['hit_rate_at_k'])

    def test_limited_parameter(self):
        for limite in [0, 101, True]:
            with self.subTest(limite=limite):
                with self.assertRaises(ValueError):
                    avaliar_leave_one_out([], limite)

    def test_metrica_nao_usa_item_considerado_consumido(self):
        dados = [(1, 10, 2), (1, 20, 1), (2, 30, 5)]
        relatorio = avaliar_leave_one_out(dados, limite=1)
        self.assertEqual(0, relatorio['acertos'])
