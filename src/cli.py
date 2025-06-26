#!/usr/bin/env python3
from __future__ import annotations
import argparse
import sys
from pathlib import Path

from src.ga_word import GAWord
from src.vocab import decode  # para convertir IDs → palabra

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="rai",
        description="R.A.I. – Chat evolutivo word-level"
    )
    parser.add_argument(
        "-p", "--pop-size",
        type=int,
        default=300,
        help="Tamaño de la población"
    )
    parser.add_argument(
        "-g", "--max-gens",
        type=int,
        default=2000,
        help="Número máximo de generaciones"
    )
    args = parser.parse_args()

    print("🗣️  Chat evolutivo word-level (ENTER sin texto para salir)\n")

    while True:
        prompt = input("Tú: ").strip()
        if not prompt:
            print("Adiós 👋")
            sys.exit(0)

        # Instanciamos GAWord (sin prompt en constructor)
        ga = GAWord(
            pop_size=args.pop_size,
            max_gens=args.max_gens,
            runs_dir=Path("runs")
        )

        # Si quieres que tu fitness tenga en cuenta el prompt,
        # asegúrate de que GAWord use `self.prompt` dentro de _fitness().
        ga.prompt = prompt  # opcional, si tu GAWord lo espera

        # Ejecutamos la evolución y guardamos el último genome
        last_genome = None
        for gen, best_fit, best_genome in ga.run():
            last_genome = best_genome

        # Decodificamos IDs → palabras, saltando BOS/EOS
        response = decode(last_genome, skip_special=True)

        print("Bot:", response, "\n")


if __name__ == "__main__":
    main()
