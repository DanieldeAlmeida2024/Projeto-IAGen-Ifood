"""Avaliação offline justa do SVD contra popularidade, usando dados sintéticos.

- Um item retido por usuário elegível, sem vazamento para treino.
- Nunca remover a última aparição de um item do catálogo de treino.
- Mesmos candidatos e mesmas interações visíveis para ambos os métodos.
- Não extrapolar resultados sintéticos para produção ou IA generativa.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from .data.data_loader import DataLoader
from .model.mf_model import MFModel


@dataclass(frozen=True)
class EvaluationResult:
    k: int
    seed: int
    evaluated_users: int
    train_interactions: int
    holdout_interactions: int
    catalog_items: int
    popularity_hit_rate_at_k: float
    svd_hit_rate_at_k: float
    popularity_catalog_coverage_at_k: float
    svd_catalog_coverage_at_k: float


def leave_one_out(frame: pd.DataFrame, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Separa 1 interação positiva por usuário sem tirar item do catálogo treinado.

    Usa contagem de linhas positivas, não order_count, para preservar
    a cobertura de itens; não altera os dados de entrada.
    """
    if not {"user_id", "item_id", "order_count"}.issubset(frame.columns):
        raise ValueError("Dataframe sem colunas de interação")
    if frame.empty:
        raise ValueError("Sem interações")
    if frame.duplicated(["user_id", "item_id"]).any():
        raise ValueError("Agregue pares duplicados antes do split")
    if (frame["order_count"] <= 0).any():
        raise ValueError("Split recebe somente interações positivas")
    df = frame.sort_values(["user_id", "item_id"]).reset_index(drop=True)
    generator = np.random.default_rng(seed)
    remaining = Counter(int(item) for item in df["item_id"])
    holdouts: list[int] = []
    for _, group in df.groupby("user_id", sort=True):
        if len(group) < 2:
            continue
        candidates = [int(idx) for idx in group.index
                      if remaining[int(df.at[idx, "item_id"])] > 1]
        if not candidates:
            continue
        selected = candidates[int(generator.integers(len(candidates)))]
        holdouts.append(selected)
        remaining[int(df.at[selected, "item_id"])] -= 1

    if not holdouts:
        raise ValueError("Dados insuficientes para holdout sem cold start de item")
    train = df.drop(index=holdouts).reset_index(drop=True)
    test = df.loc[holdouts].reset_index(drop=True)
    return train, test


def evaluate_ranking(frame: pd.DataFrame, k: int = 5, seed: int = 42) -> EvaluationResult:
    if not isinstance(k, int) or not 1 <= k <= 100:
        raise ValueError("k deve ser um inteiro de 1 a 100")
    train, test = leave_one_out(frame, seed=seed)
    pivot = train.pivot_table(
        index="user_id", columns="item_id",
        values="order_count", aggfunc="sum", fill_value=0, sort=True,
    )
    model = MFModel(random_state=seed)
    items = [int(v) for v in pivot.columns]
    model.train(csr_matrix(pivot.to_numpy(dtype=float)),
                user_ids=[int(v) for v in pivot.index], item_ids=items)
    popularity = {
        int(item): float(score) for item, score
        in train.groupby("item_id")["order_count"].sum().items()
    }
    history = {
        int(user): set(int(v) for v in group["item_id"])
        for user, group in train.groupby("user_id")
    }
    hits_pop = hits_svd = 0
    recommended_pop: set[int] = set()
    recommended_svd: set[int] = set()

    for row in test.itertuples(index=False):
        user = int(row.user_id)
        true_item = int(row.item_id)
        candidates = [item for item in items if item not in history.get(user, set())]
        # Todos os itens do holdout permanecem no catálogo de treinamento.
        if true_item not in candidates:
            raise AssertionError("Holdout vazou para treino ou catálogo inválido")
        by_pop = sorted(candidates, key=lambda item: (-popularity[item], item))[:k]
        by_svd = sorted(candidates, key=lambda item: (-model.predict_score(user, item), item))[:k]
        hits_pop += int(true_item in by_pop)
        hits_svd += int(true_item in by_svd)
        recommended_pop.update(by_pop)
        recommended_svd.update(by_svd)

    n = len(test)
    return EvaluationResult(
        k=k, seed=seed, evaluated_users=n,
        train_interactions=len(train), holdout_interactions=n,
        catalog_items=len(items),
        popularity_hit_rate_at_k=round(hits_pop / n, 6),
        svd_hit_rate_at_k=round(hits_svd / n, 6),
        popularity_catalog_coverage_at_k=round(len(recommended_pop) / len(items), 6),
        svd_catalog_coverage_at_k=round(len(recommended_svd) / len(items), 6),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Avaliação offline com holdout, sem dados reais")
    parser.add_argument("--arquivo", type=Path, default=None)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    loader = DataLoader(args.arquivo) if args.arquivo else DataLoader()
    df = loader.load_raw_data(loader.data_path)
    # Não avaliar linhas de interação zero como exemplos relevantes.
    output = evaluate_ranking(df.loc[df["order_count"] > 0], k=args.k, seed=args.seed)
    print(json.dumps(asdict(output), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
