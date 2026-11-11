"""Garantias de holdout, fairness do conjunto de candidatos e reprodutibilidade."""
import unittest

import pandas as pd

from src.evaluation import evaluate_ranking, leave_one_out


class OfflineEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame([
            {"user_id": uid, "item_id": item, "order_count": score}
            for uid in range(1, 5)
            for item, score in ((101, 1 + uid), (102, 2 + uid), (103, 3 + uid))
        ])

    def test_no_user_item_leakage(self):
        train, holdout = leave_one_out(self.frame, seed=42)
        training_pairs = set(zip(train.user_id, train.item_id))
        withheld_pairs = set(zip(holdout.user_id, holdout.item_id))
        self.assertFalse(training_pairs & withheld_pairs)
        self.assertEqual(set(holdout.user_id), {1, 2, 3, 4})
        for user in holdout.user_id:
            self.assertGreaterEqual(len(train[train.user_id == user]), 1)
        self.assertTrue(set(holdout.item_id).issubset(set(train.item_id)))

    def test_same_seed_returns_same_holdout(self):
        _, one = leave_one_out(self.frame, seed=42)
        _, two = leave_one_out(self.frame, seed=42)
        self.assertEqual(one.to_dict("records"), two.to_dict("records"))

    def test_comparable_metrics_and_reproducibility(self):
        one = evaluate_ranking(self.frame, k=2, seed=42)
        two = evaluate_ranking(self.frame, k=2, seed=42)
        self.assertEqual(one, two)
        self.assertEqual(one.evaluated_users, 4)
        self.assertEqual(one.holdout_interactions, 4)
        self.assertEqual(one.train_interactions, 8)
        self.assertTrue(0 <= one.svd_hit_rate_at_k <= 1)
        self.assertTrue(0 <= one.popularity_hit_rate_at_k <= 1)
        self.assertTrue(0 <= one.svd_catalog_coverage_at_k <= 1)

    def test_unsupported_dataset_and_invalid_k_fail_explicitly(self):
        with self.assertRaises(ValueError):
            evaluate_ranking(self.frame, k=0)
        with self.assertRaises(ValueError):
            leave_one_out(pd.DataFrame([
                {"user_id": 1, "item_id": 101, "order_count": 1}
            ]))

    def test_duplicate_pair_must_be_aggregated_first(self):
        invalid = pd.concat([self.frame, self.frame.iloc[:1]], ignore_index=True)
        with self.assertRaises(ValueError):
            leave_one_out(invalid)


if __name__ == "__main__":
    unittest.main()
