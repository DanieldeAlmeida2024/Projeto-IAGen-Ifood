"""Carregador de interações sintéticas para treino SVD e avaliação offline."""
from decimal import Decimal, InvalidOperation
from pathlib import Path

import pandas as pd
from scipy.sparse import csr_matrix

from .interfaces import IDataLoader


DEFAULT_DATA = Path(__file__).resolve().parent / "ifood_sim_data.csv"
COLUMNS = ("user_id", "item_id", "order_count")
MAX_INT64 = (1 << 63) - 1


def _parse_int64(value: str, field: str, minimum: int) -> int:
    try:
        number = Decimal(value)
    except (InvalidOperation, TypeError) as error:
        raise ValueError(f"{field} contém valor inválido") from error
    if not number.is_finite() or number != number.to_integral_value():
        raise ValueError(f"{field} precisa ser inteiro não nulo")
    if number < minimum:
        raise ValueError(f"{field} deve ser maior ou igual a {minimum}")
    if number > MAX_INT64:
        raise ValueError(f"{field} excede limite de inteiro int64")
    return int(number)


class DataLoader(IDataLoader):
    def __init__(self, data_path: str | Path = DEFAULT_DATA):
        self.data_path = Path(data_path)
        self._raw_df: pd.DataFrame | None = None
        self._interaction_matrix: csr_matrix | None = None
        self.user_ids: list[int] = []
        self.item_ids: list[int] = []

    def load_raw_data(self, source_path: str | Path) -> pd.DataFrame:
        """Valida inteiros positivos e agrega pedidos duplicados por usuário/item.

        A amostra legada contém cabeçalhos repetidos; removê-los explicitamente,
        sem tolerar texto arbitrário ou negativos em linhas de interação.
        """
        frame = pd.read_csv(source_path, dtype=str, encoding="utf-8-sig")
        if not set(COLUMNS).issubset(frame.columns):
            raise ValueError("CSV deve conter user_id, item_id e order_count")
        frame = frame.loc[:, list(COLUMNS)]
        repeated = (
            frame["user_id"].eq("user_id")
            & frame["item_id"].eq("item_id")
            & frame["order_count"].eq("order_count")
        )
        frame = frame.loc[~repeated].copy()
        if frame.empty:
            raise ValueError("CSV sem interações para treino")
        totals: dict[tuple[int, int], int] = {}
        for raw_user, raw_item, raw_count in frame.itertuples(index=False, name=None):
            user = _parse_int64(raw_user, "user_id", minimum=1)
            item = _parse_int64(raw_item, "item_id", minimum=1)
            count = _parse_int64(raw_count, "order_count", minimum=0)
            key = (user, item)
            total = totals.get(key, 0) + count
            if total > MAX_INT64:
                raise ValueError(f"Contagem agregada excede int64 para usuário {user}, item {item}")
            totals[key] = total
        rows = [(user, item, total) for (user, item), total in sorted(totals.items())]
        self._raw_df = pd.DataFrame(rows, columns=COLUMNS, dtype="int64")
        self._interaction_matrix = None
        self.user_ids = []
        self.item_ids = []
        return self._raw_df.copy()

    def get_interaction_matrix(self) -> csr_matrix:
        if self._raw_df is None:
            self.load_raw_data(self.data_path)
        assert self._raw_df is not None
        if self._interaction_matrix is None:
            table = self._raw_df.pivot_table(
                index="user_id", columns="item_id", values="order_count",
                aggfunc="sum", fill_value=0, sort=True
            )
            self.user_ids = [int(uid) for uid in table.index]
            self.item_ids = [int(iid) for iid in table.columns]
            self._interaction_matrix = csr_matrix(table.to_numpy(dtype=float))
        return self._interaction_matrix

    def get_user_history(self, user_id: int) -> tuple[list[int], list[int]]:
        if self._raw_df is None:
            self.load_raw_data(self.data_path)
        assert self._raw_df is not None
        rows = self._raw_df.loc[
            (self._raw_df["user_id"] == user_id) &
            (self._raw_df["order_count"] > 0)
        ].sort_values("item_id")
        return (
            [int(v) for v in rows["item_id"]],
            [int(v) for v in rows["order_count"]]
        )
