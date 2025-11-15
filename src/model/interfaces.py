from typing import Dict, Any, List

class IRecommenderModel:
    """
    Interface para qualquer modelo de recomendação.
    Aplica o OCP/LSP: Permite estender o sistema adicionando novos modelos.
    """
    
    def train(self, interaction_matrix: Any, **kwargs) -> None:
        """Treina o modelo com a matriz de interação."""
        raise NotImplementedError
        
    def predict_score(self, user_id: int, item_id: int) -> float:
        """Calcula a pontuação (score) para uma interação específica."""
        raise NotImplementedError
        
    def save_model(self, path: str) -> None:
        """Salva o modelo treinado em um local persistente (ex: S3/local)."""
        raise NotImplementedError
        
    def get_all_item_ids(self) -> List[int]:
        """Retorna todos os IDs de itens que o modelo conhece."""
        raise NotImplementedError