"""Estabilidade do ranking em seeds diferentes, exclusivamente sobre dados sintéticos.

As seeds geram reamostragens dependentes do mesmo conjunto histórico:
desvio-padrão descritivo NÃO é intervalo de confiança nem teste estatístico.
"""
from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import asdict

import pandas as pd

from .data.data_loader import DataLoader
from .evaluation import evaluate_ranking


METRICS = (
    "popularity_hit_rate_at_k", "svd_hit_rate_at_k",
    "popularity_catalog_coverage_at_k", "svd_catalog_coverage_at_k",
)


def parse_seeds(text: str) -> tuple[int, ...]:
    try:
        seeds = tuple(int(value.strip()) for value in text.split(","))
    except (ValueError, TypeError) as error:
        raise ValueError("Seeds devem ser inteiros separados por vírgula") from error
    if len(seeds) < 2 or len(set(seeds)) != len(seeds) or any(seed < 0 for seed in seeds):
        raise ValueError("Informe pelo menos duas seeds inteiras, únicas e não negativas")
    return seeds


def compare_seeds(frame: pd.DataFrame, *, seeds: tuple[int, ...], k: int = 5) -> dict:
    if len(seeds) < 2 or len(set(seeds)) != len(seeds) or any(
        not isinstance(s, int) or s < 0 for s in seeds
    ):
        raise ValueError("Seeds devem ser pelo menos duas inteiras únicas e positivas ou zero")

    result = [asdict(evaluate_ranking(frame, k=k, seed=seed)) for seed in seeds]
    for row in result:
        row["svd_minus_popularity_hit_rate_at_k"] = round(
            row["svd_hit_rate_at_k"] - row["popularity_hit_rate_at_k"], 6
        )
    keys = (*METRICS, "svd_minus_popularity_hit_rate_at_k")
    summary = {}
    for metric in keys:
        values = [float(row[metric]) for row in result]
        summary[metric] = {
            "mean": round(statistics.mean(values), 6),
            "sample_stddev": round(statistics.stdev(values), 6),
            "min": min(values),
            "max": max(values),
        }

    return {
        "experiment": "synthetic_leave_one_out_multi_seed",
        "k": k, "seeds": list(seeds), "replications": len(result),
        "evaluated_user_counts": [int(row["evaluated_users"]) for row in result],
        "per_seed": result, "descriptive_summary": summary,
        "limitations": "Dependent synthetic resamples; no timestamps; not confidence intervals",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Análise descritiva de sensibilidade por seed")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--seeds", default="42,43,44,45,46")
    args = parser.parse_args()
    loader = DataLoader()
    frame = loader.load_raw_data(loader.data_path)
    summary = compare_seeds(
        frame.loc[frame["order_count"] > 0],
        seeds=parse_seeds(args.seeds), k=args.k
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
