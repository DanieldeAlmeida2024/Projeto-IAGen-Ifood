"""CLI de recomendação por popularidade; não requer dependências externas."""
import argparse
import json
from src.recomendacao import carregar_interacoes, recomendar

def main():
    parser = argparse.ArgumentParser(description='Laboratório de recomendação de pedidos simulados')
    parser.add_argument('--arquivo', default='src/data/ifood_sim_data.csv')
    parser.add_argument('--usuario', type=int, required=True)
    parser.add_argument('--quantidade', type=int, default=5)
    args = parser.parse_args()
    try:
        resultado = recomendar(carregar_interacoes(args.arquivo), args.usuario, args.quantidade)
    except (OSError, ValueError) as erro:
        parser.error(str(erro))
    print(json.dumps({'usuario': args.usuario, 'recomendacoes': resultado}, ensure_ascii=False, indent=2))

if __name__ == '__main__': main()
