"""Testes para variabilidade descritiva por seed, com CSV inteiramente sintético."""
import unittest

import pandas as pd

from src.evaluation_multi_seed import compare_seeds, parse_seeds


class MultiSeedRankingTests(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame([
            {"user_id": u, "item_id": i, "order_count": (u + i) % 5 + 1}
            for u in range(1, 7)
            for i in (101, 102, 103, 104)
        ])

    def test_parse_seeds_contract(self):
        self.assertEqual((42, 43, 44), parse_seeds("42,43, 44"))
        for invalid in ("", "42", "42,42", "-1,2", "abc,42", "42,"):
            with self.subTest(input=invalid):
                with self.assertRaises(ValueError):
                    parse_seeds(invalid)

    def test_metrics_have_per_seed_and_descriptive_ranges(self):
        report = compare_seeds(self.frame, seeds=(42, 43, 44), k=2)
        self.assertEqual(3, report["replications"])
        self.assertEqual([42, 43, 44], report["seeds"])
        self.assertEqual(3, len(report["per_seed"]))
        self.assertTrue(all(users > 0 for users in report["evaluated_user_counts"]))
        for name, summary in report["descriptive_summary"].items():
            self.assertLessEqual(summary["min"], summary["mean"])
            self.assertLessEqual(summary["mean"], summary["max"])
            self.assertGreaterEqual(summary["sample_stddev"], 0)

    def test_deterministic_report(self):
        self.assertEqual(
            compare_seeds(self.frame, seeds=(42, 43), k=2),
            compare_seeds(self.frame, seeds=(42, 43), k=2)
        )

    def test_delta_uses_per_seed_scores(self):
        report = compare_seeds(self.frame, seeds=(42, 43), k=2)
        for row in report["per_seed"]:
            self.assertAlmostEqual(
                row["svd_minus_popularity_hit_rate_at_k"],
                row["svd_hit_rate_at_k"] - row["popularity_hit_rate_at_k"],
                places=6
            )

    def test_degenerate_seed_list_is_rejected(self):
        for seeds in ((42,), (42, 42), (-1, 42), (42, 1.5)):
            with self.subTest(seeds=seeds):
                with self.assertRaises(ValueError):
                    compare_seeds(self.frame, seeds=seeds, k=2)


if __name__ == "__main__":
    unittest.main()
