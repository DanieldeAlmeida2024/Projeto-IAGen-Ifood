# Arquivo: src/data_layer/data_loader.py
import pandas as pd
from scipy.sparse import csr_matrix
from interfaces import IDataLoader

class DataLoader(IDataLoader):
    """
    Implementação concreta para carregar e transformar dados de pedidos.
    Aplica o SRP: Focado unicamente no manuseio e transformação de dados.
    """
    def __init__(self, data_path: str = 'data/ifood_sim_data.csv'):
        self.data_path = data_path
        self._interaction_matrix = None
        self._raw_df = None
        
    def load_raw_data(self, source_path: str) -> pd.DataFrame:
        """Carrega o CSV e armazena o DataFrame bruto."""
        # Em um projeto real, buscaria dados do S3, Redshift, etc.
        self._raw_df = pd.read_csv(source_path)
        return self._raw_df

    def get_interaction_matrix(self) -> csr_matrix:
        """
        Executa o pivoteamento e retorna a Matriz de Interação Escassa (CSR).
        """
        if self._interaction_matrix is None:
            if self._raw_df is None:
                # Se não foi carregado, carrega antes
                self.load_raw_data(self.data_path) 
            

            interaction_matrix_df = self._raw_df.pivot_table(
                index='user_id', 
                columns='item_id', 
                values='order_count', 
                fill_value=0 
            )
            
            # Conversão para Matriz Esparsa (Formato ideal para o modelo MF)
            self._interaction_matrix = csr_matrix(interaction_matrix_df.values)
        
        return self._interaction_matrix

    def get_user_history(self, user_id: int) -> tuple[list, list]:
        """Busca o histórico de pedidos de um usuário específico."""
        if self._raw_df is None:
            self.load_raw_data(self.data_path)
            
        history = self._raw_df[self._raw_df['user_id'] == user_id]['item_id'].tolist()
        return history