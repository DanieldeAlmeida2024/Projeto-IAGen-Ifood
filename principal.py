"""CLI de recomendação por popularidade; não requer dependências externas."""
import argparse
import json
from src.recomendacao import carregar_interacoes, recomendar
from src.avaliacao import avaliar_leave_one_out

def main():
    parser = argparse.ArgumentParser(description='Laboratório de recomendação de pedidos simulados')
    parser.add_argument('--arquivo', default='src/data/ifood_sim_data.csv')
    parser.add_argument('--usuario', type=int)
    parser.add_argument('--avaliar', action='store_true', help='Avalia HitRate@K em dados simulados')
    parser.add_argument('--quantidade', type=int, default=5)
    args = parser.parse_args()
    try:
        interacoes = carregar_interacoes(args.arquivo)
        if args.avaliar:
            resposta = avaliar_leave_one_out(interacoes, limite=args.quantidade)
        else:
            if args.usuario is None:
                parser.error('Informe --usuario para receber recomendações')
            resultado = recomendar(interacoes, args.usuario, args.quantidade)
            resposta = {'usuario': args.usuario, 'recomendacoes': resultado}
    except (OSError, ValueError) as erro:
        parser.error(str(erro))
    print(json.dumps(resposta, ensure_ascii=False, indent=2))

if __name__ == '__main__': main()
