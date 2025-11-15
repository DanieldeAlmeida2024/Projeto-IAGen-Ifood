# Arquivo: src/model_layer/mf_model.py
import joblib
from surprise import Reader, Dataset, SVD
from surprise.model_selection import train_test_split
from src.model_layer.interfaces import IRecommenderModel
from src.data_layer.data_loader import DataLoader
from typing import List, Any

class MFModel(IRecommenderModel):
    """
    Implementação da Fatoração de Matrizes (SVD).
    Adere à interface IRecommenderModel.
    """
    
    def __init__(self):
        self.algo = SVD(n_factors=50, n_epochs=20, lr_all=0.005, reg_all=0.02)
        self.item_map = None # Mapeamento de IDs para o modelo SVD
        self.global_mean = None
        
    def train(self, interaction_matrix: Any, **kwargs) -> None:
        """
        Treina o modelo SVD. O modelo Surprise requer um formato específico (DataFrame).
        O 'interaction_matrix' é a matriz esparsa da tarefa US-001.
        """
        
        # 1. Transformar a Matriz Esparsa de volta em DataFrame no formato (user, item, score)
        # Primeiro, converte a matriz densa de volta (o que é ok para simulação)
        df_dense = pd.DataFrame(interaction_matrix.todense(), 
                                index=kwargs['user_ids'], 
                                columns=kwargs['item_ids'])
        
        # Melt para o formato longo (user, item, score)
        df_long = df_dense.reset_index().melt(
            id_vars=['user_id'], 
            value_vars=df_dense.columns, 
            var_name='item_id', 
            value_name='order_count'
        )
        
        # Filtrar apenas interações reais (order_count > 0)
        df_train = df_long[df_long['order_count'] > 0].copy()

        # 2. Carregar no formato do Surprise
        reader = Reader(rating_scale=(1, df_train['order_count'].max()))
        data = Dataset.load_from_df(df_train[['user_id', 'item_id', 'order_count']], reader)
        trainset = data.build_full_trainset()
        
        # 3. Treinamento
        self.algo.fit(trainset)
        self.item_map = trainset.to_inner_uid
        self.global_mean = trainset.global_mean
        print(f"Modelo SVD treinado. RMSE (simulado) será calculado na avaliação.")

    def predict_score(self, user_id: int, item_id: int) -> float:
        """Prevê a pontuação para uma interação."""
        # O SVD usa IDs internos (inner_uid/iid). Esta é uma simplificação.
        # Em produção, você precisaria de um mapeamento robusto.
        try:
            prediction = self.algo.predict(str(user_id), str(item_id))
            return prediction.est
        except Exception:
            # Caso o usuário/item não exista (Cold Start), retorna a média global
            return self.global_mean if self.global_mean else 3.0

    def save_model(self, path: str) -> None:
        """Salva o modelo treinado usando joblib."""
        joblib.dump(self.algo, path)
        print(f"Modelo MF salvo em: {path}")

    def get_all_item_ids(self) -> List[int]:
        """Retorna todos os IDs de itens (simplificação)."""
        # Em produção, viria de um serviço de catálogo.
        # Aqui, vamos assumir que os IDs do DataLoader são os únicos válidos.
        return list(self.item_map.keys()) if self.item_map else []