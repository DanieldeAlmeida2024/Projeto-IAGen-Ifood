from typing import Tuple
from scipy.sparse import csr_matrix
from pandas import DataFrame

class IDataLoader:
    """
    Interface para carregamento de dados. 
    Aplica o ISP: Garante que os consumidores dependam apenas dos métodos necessários.
    """
    
    def load_raw_data(self, source_path: str) -> DataFrame:
        """Carrega o DataFrame bruto (CSV, etc.) a partir de um caminho."""
        raise NotImplementedError
        
    def get_interaction_matrix(self) -> csr_matrix:
        """
        Retorna a Matriz de Interação no formato esparso (CSR), pronta para o modelo MF.
        (Conclusão do Card US-001)
        """
        raise NotImplementedError

    def get_user_history(self, user_id: int) -> Tuple[list, list]:
        """Retorna os itens já consumidos por um usuário, para a lógica de filtragem da API."""
        raise NotImplementedError