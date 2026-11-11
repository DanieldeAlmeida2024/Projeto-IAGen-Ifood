"""SVD experimental sobre feedback implícito sintético (contagem de pedidos)."""
from pathlib import Path
from typing import Any, List

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import issparse, csr_matrix
from surprise import Dataset, Reader, SVD

from .interfaces import IRecommenderModel


class MFModel(IRecommenderModel):
    """Fatoração de matrizes usando IDs externos estáveis e sem densificar CSR.

    Este é um experimento de rating por volume de pedidos, não um ranking
    validado ou modelo de IA generativa.
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.algo = SVD(n_factors=20, n_epochs=20, lr_all=0.005,
                        reg_all=0.02, random_state=random_state)
        self.user_ids: list[int] = []
        self.item_ids: list[int] = []
        self.global_mean: float | None = None
        self._trained = False

    def train(self, interaction_matrix: Any, **kwargs) -> None:
        user_ids = [int(v) for v in kwargs["user_ids"]]
        item_ids = [int(v) for v in kwargs["item_ids"]]
        matrix = csr_matrix(interaction_matrix, dtype=float)
        if matrix.shape != (len(user_ids), len(item_ids)):
            raise ValueError("Dimensões da matriz não correspondem aos IDs externos")
        if len(set(user_ids)) != len(user_ids) or len(set(item_ids)) != len(item_ids):
            raise ValueError("IDs de usuários e itens devem ser únicos")
        if not np.isfinite(matrix.data).all() or (matrix.data < 0).any():
            raise ValueError("Contagens devem ser não negativas e finitas")
        if not matrix.nnz or not np.any(matrix.data > 0):
            raise ValueError("É necessária pelo menos uma interação positiva")
        # COO permite construir o treino diretamente de elementos esparsos.
        coo = matrix.tocoo()
        entries = [(str(user_ids[int(r)]), str(item_ids[int(c)]), float(score))
                   for r, c, score in zip(coo.row, coo.col, coo.data)
                   if score > 0]
        if not entries:
            raise ValueError("Nenhuma interação positiva")
        df = pd.DataFrame(entries, columns=["user_id", "item_id", "order_count"])
        reader = Reader(rating_scale=(1, max(1.0, float(df["order_count"].max()))))
        trainset = Dataset.load_from_df(
            df[["user_id", "item_id", "order_count"]], reader
        ).build_full_trainset()
        self.algo.fit(trainset)
        self.user_ids = user_ids
        self.item_ids = item_ids
        self.global_mean = float(trainset.global_mean)
        self._trained = True

    def predict_score(self, user_id: int, item_id: int) -> float:
        if not self._trained or self.global_mean is None:
            raise RuntimeError("Treine ou carregue o modelo antes de prever")
        if user_id not in self.user_ids or item_id not in self.item_ids:
            return self.global_mean
        return float(self.algo.predict(str(user_id), str(item_id)).est)

    def save_model(self, path: str) -> None:
        if not self._trained:
            raise RuntimeError("Não é possível salvar um modelo não treinado")
        joblib.dump({
            "algo": self.algo, "user_ids": self.user_ids,
            "item_ids": self.item_ids, "global_mean": self.global_mean,
            "random_state": self.random_state
        }, Path(path))

    @classmethod
    def load_model(cls, path: str) -> "MFModel":
        payload = joblib.load(path)
        model = cls(random_state=payload["random_state"])
        model.algo = payload["algo"]
        model.user_ids = list(payload["user_ids"])
        model.item_ids = list(payload["item_ids"])
        model.global_mean = float(payload["global_mean"])
        model._trained = True
        return model

    def get_all_item_ids(self) -> List[int]:
        return list(self.item_ids)
