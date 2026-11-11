"""Validação temporal rigorosa de pedidos (não é avaliação de ranking).

A amostra histórica original não possui timestamp. Somente datasets com
ocorrência temporal explícita e timezone podem ser usados neste contrato.
Não inventar datas a partir de IDs ou da posição das linhas.
"""
from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

REQUIRED = ("user_id", "item_id", "order_count", "occurred_at")
ID = re.compile(r"^[1-9]\d*$")
COUNT = re.compile(r"^[1-9]\d*$")
TZ_SUFFIX = re.compile(r"(?:Z|[+-]\d{2}:\d{2})$")


def parse_instant(value: str) -> datetime:
    if not isinstance(value, str) or not TZ_SUFFIX.search(value):
        raise ValueError("Timestamp deve usar ISO 8601 com timezone explícito")
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError("Timestamp ISO 8601 inválido") from error
    if dt.tzinfo is None or dt.utcoffset() is None:
        raise ValueError("Timezone obrigatório")
    return dt.astimezone(timezone.utc)


@dataclass(frozen=True)
class TemporalReport:
    cutoff_utc: str
    train_rows: int
    future_test_rows: int
    eligible_test_rows: int
    cold_start_user_rows: int
    cold_start_item_rows: int
    already_consumed_test_rows: int
    train_users: int
    train_items: int


def load_temporal_rows(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise ValueError("Arquivo temporal não encontrado")
    frame = pd.read_csv(path, dtype=str, encoding="utf-8-sig")
    if not set(REQUIRED).issubset(frame.columns):
        raise ValueError("Dataset exige user_id,item_id,order_count,occurred_at")
    frame = frame.loc[:, list(REQUIRED)].copy()
    if frame.empty:
        raise ValueError("Dataset temporal vazio")
    for name, pattern in (("user_id", ID), ("item_id", ID), ("order_count", COUNT)):
        if not frame[name].map(lambda x: isinstance(x, str) and bool(pattern.fullmatch(x))).all():
            raise ValueError(f"{name} inválido: inteiro positivo obrigatório")
        frame[name] = frame[name].astype("int64")
    frame["timestamp_utc"] = frame["occurred_at"].map(parse_instant)
    return frame


def validate_temporal_split(frame: pd.DataFrame, cutoff: str) -> TemporalReport:
    limit = parse_instant(cutoff)
    if "timestamp_utc" not in frame.columns:
        raise ValueError("Datas precisam ser validadas por load_temporal_rows")
    train = frame.loc[frame["timestamp_utc"] < limit]
    test = frame.loc[frame["timestamp_utc"] >= limit]
    if train.empty or test.empty:
        raise ValueError("Corte precisa ter eventos positivos antes e depois")
    if not all(x < limit for x in train["timestamp_utc"]) or not all(
        x >= limit for x in test["timestamp_utc"]
    ):
        raise AssertionError("Vazamento temporal: treino contém futuro")
    history = set(zip(train["user_id"], train["item_id"]))
    users = set(train["user_id"])
    items = set(train["item_id"])

    user_cold = item_cold = already_consumed = eligible = 0
    for row in test.itertuples(index=False):
        if row.user_id not in users:
            user_cold += 1
        elif row.item_id not in items:
            item_cold += 1
        elif (row.user_id, row.item_id) in history:
            already_consumed += 1
        else:
            eligible += 1

    return TemporalReport(
        cutoff_utc=limit.isoformat(),
        train_rows=len(train),
        future_test_rows=len(test),
        eligible_test_rows=eligible,
        cold_start_user_rows=user_cold,
        cold_start_item_rows=item_cold,
        already_consumed_test_rows=already_consumed,
        train_users=len(users),
        train_items=len(items),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Checagem de dados temporais, sem treino")
    parser.add_argument("--arquivo", type=Path, required=True)
    parser.add_argument("--cutoff", required=True, help="ISO 8601, ex: 2026-01-02T00:00:00Z")
    args = parser.parse_args()
    report = validate_temporal_split(load_temporal_rows(args.arquivo), args.cutoff)
    print(json.dumps(asdict(report), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
