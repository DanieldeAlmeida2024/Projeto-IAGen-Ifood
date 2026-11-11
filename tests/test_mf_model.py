"""Integração da matriz CSR com Surprise/SVD, sem APIs ou dados pessoais."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix

from src.data.data_loader import DataLoader
from src.model.mf_model import MFModel


class DataLoaderAndSVDTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "sample.csv"
        self.path.write_text(
            "user_id,item_id,order_count\n"
            "1,101,3\n2,101,1\n1,102,2\n"
            "user_id,item_id,order_count\n"
            "1,101,4\n2,103,5\n",
            encoding="utf-8"
        )
        self.loader = DataLoader(self.path)

    def tearDown(self):
        self.tmp.cleanup()

    def test_duplicated_header_and_aggregation(self):
        df = self.loader.load_raw_data(self.path)
        self.assertEqual(4, len(df))
        self.assertEqual(7, int(df.loc[
            (df["user_id"] == 1) & (df["item_id"] == 101),
            "order_count"
        ].iloc[0]))

    def test_matrix_ids_and_history(self):
        matrix = self.loader.get_interaction_matrix()
        self.assertEqual((2, 3), matrix.shape)
        self.assertEqual([1, 2], self.loader.user_ids)
        self.assertEqual([101, 102, 103], self.loader.item_ids)
        self.assertEqual(7, matrix[0, 0])
        self.assertEqual(([101, 102], [7, 2]), self.loader.get_user_history(1))

    def test_invalid_negative_and_bad_text(self):
        for value in ("-1", "abc", "1.5"):
            with self.subTest(value=value):
                self.path.write_text("user_id,item_id,order_count\n1,101,"+value+"\n")
                with self.assertRaises(ValueError):
                    self.loader.load_raw_data(self.path)

    def test_train_predict_and_cold_start(self):
        matrix = self.loader.get_interaction_matrix()
        model = MFModel()
        with self.assertRaises(RuntimeError):
            model.predict_score(1, 101)
        model.train(matrix, user_ids=self.loader.user_ids, item_ids=self.loader.item_ids)
        self.assertEqual([101, 102, 103], model.get_all_item_ids())
        self.assertTrue(np.isfinite(model.predict_score(1, 101)))
        self.assertAlmostEqual(model.global_mean, model.predict_score(999, 101))
        self.assertAlmostEqual(model.global_mean, model.predict_score(1, 999))

    def test_save_load_and_reproducible_score(self):
        model = MFModel()
        model.train(self.loader.get_interaction_matrix(),
                    user_ids=self.loader.user_ids, item_ids=self.loader.item_ids)
        path = str(Path(self.tmp.name) / "model.joblib")
        before = model.predict_score(1, 101)
        model.save_model(path)
        after = MFModel.load_model(path)
        self.assertAlmostEqual(before, after.predict_score(1, 101), places=8)

    def test_shape_and_empty_training_errors(self):
        model = MFModel()
        with self.assertRaises(ValueError):
            model.train(csr_matrix((2, 2)), user_ids=[1], item_ids=[101, 102])
        with self.assertRaises(ValueError):
            model.train(csr_matrix((2, 2)), user_ids=[1, 2], item_ids=[101, 102])

    def test_simulated_dataset_loads_without_header_crash(self):
        real_sample = DataLoader()
        matrix = real_sample.get_interaction_matrix()
        self.assertGreater(matrix.nnz, 100)
        self.assertTrue(all(i > 0 for i in real_sample.item_ids))


if __name__ == "__main__":
    unittest.main()
