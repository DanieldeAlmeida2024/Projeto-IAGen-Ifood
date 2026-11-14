import tempfile
import unittest
from pathlib import Path
from src.recomendacao import carregar_interacoes, recomendar

class TestesRecomendacao(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.arquivo = Path(self.tmp.name) / 'pedidos.csv'
        self.arquivo.write_text('user_id,item_id,order_count\n1,10,5\n1,11,2\n2,12,8\n3,13,7\n', encoding='utf-8')
    def tearDown(self): self.tmp.cleanup()
    def test_carregamento(self):
        self.assertEqual(4, len(carregar_interacoes(self.arquivo)))
    def test_exclui_itens_ja_consumidos(self):
        recomendacoes = recomendar(carregar_interacoes(self.arquivo), usuario=1, limite=3)
        self.assertEqual([12, 13], [x['item_id'] for x in recomendacoes])
    def test_cold_start(self):
        recomendacoes = recomendar(carregar_interacoes(self.arquivo), usuario=99, limite=2)
        self.assertEqual([12, 13], [x['item_id'] for x in recomendacoes])
    def test_limite_invalido(self):
        with self.assertRaises(ValueError): recomendar([], 1, 0)
    def test_csv_incorreto(self):
        self.arquivo.write_text('nome,valor\nA,1\n', encoding='utf-8')
        with self.assertRaises(ValueError): carregar_interacoes(self.arquivo)

    def test_csv_com_cabecalhos_repetidos(self):
        self.arquivo.write_text(
            'user_id,item_id,order_count\n1,10,2\n'
            'user_id,item_id,order_count\n2,20,3\n', encoding='utf-8'
        )
        self.assertEqual([(1, 10, 2), (2, 20, 3)], carregar_interacoes(self.arquivo))
