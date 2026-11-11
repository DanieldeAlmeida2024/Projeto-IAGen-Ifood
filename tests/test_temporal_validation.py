"""Prova de corte global sem futuro e contagem explícita de cold start."""
import tempfile
import unittest
from pathlib import Path

from src.temporal_validation import (
    load_temporal_rows, parse_instant, validate_temporal_split,
)

FIXTURE = Path(__file__).parent / "fixtures" / "temporal_interactions.csv"
CUTOFF = "2026-01-02T00:00:00Z"


class TemporalValidationTests(unittest.TestCase):
    def test_global_cutoff_counts_and_cold_start(self):
        report = validate_temporal_split(load_temporal_rows(FIXTURE), CUTOFF)
        self.assertEqual(5, report.train_rows)
        self.assertEqual(5, report.future_test_rows)
        self.assertEqual(2, report.eligible_test_rows)
        self.assertEqual(1, report.cold_start_user_rows)
        self.assertEqual(1, report.cold_start_item_rows)
        self.assertEqual(1, report.already_consumed_test_rows)
        self.assertEqual(3, report.train_users)
        self.assertEqual(3, report.train_items)

    def test_timezones_are_normalized_to_utc(self):
        self.assertEqual(
            parse_instant("2026-01-02T03:00:00+03:00"),
            parse_instant(CUTOFF),
        )

    def test_missing_or_naive_timestamp_rejected(self):
        for text in ("2026-01-02", "2026-01-02T00:00:00", "2026/01/02", ""):
            with self.subTest(text=text):
                with self.assertRaises(ValueError):
                    parse_instant(text)

    def test_missing_timestamp_column_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "events.csv"
            path.write_text("user_id,item_id,order_count\n1,101,2\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_temporal_rows(path)

    def test_invalid_counts_and_ids_rejected(self):
        for first, second, third in (("0", "101", "1"), ("1", "-1", "1"),
                                     ("1", "101", "0"), ("1", "101", "xyz")):
            with self.subTest(values=(first, second, third)):
                with tempfile.TemporaryDirectory() as folder:
                    path = Path(folder) / "bad.csv"
                    path.write_text(
                        "user_id,item_id,order_count,occurred_at\n"
                        + f"{first},{second},{third},2026-01-01T10:00:00Z\n",
                        encoding="utf-8",
                    )
                    with self.assertRaises(ValueError):
                        load_temporal_rows(path)

    def test_empty_side_of_cutoff_is_error(self):
        with self.assertRaises(ValueError):
            validate_temporal_split(load_temporal_rows(FIXTURE), "2027-01-01T00:00:00Z")


if __name__ == "__main__":
    unittest.main()
