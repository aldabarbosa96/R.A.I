#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path

def main() -> None:
    import sys
    from src.ga_word import GAWord

    p = argparse.ArgumentParser(
        prog="rai",
        description="R.A.I. – Chat evolutivo word-level"
    )
    p.add_argument(
        "-p", "--pop-size",
        type=int,
        default=300,
        help="Tamaño de la población"
    )
    p.add_argument(
        "-g", "--max-gens",
        type=int,
        default=2000,
        help="Número máximo de generaciones"
    )
    args = p.parse_args()

    print("🗣️  Chat evolutivo word-level (ENTER sin texto para salir)\n")

    while True:
        prompt = input("Tú: ").strip()
        if not prompt:
            print("Adiós 👋")
            sys.exit(0)

        # Instanciamos GAWord
        ga = GAWord(
            pop_size=args.pop_size,
            max_gens=args.max_gens,
            runs_dir=Path("runs")
        )

        # Ejecutamos toda la evolución y guardamos la última población
        last_population = None
        for population in ga.run():
            last_population = population

        # Sacamos el mejor de la última población por fitness
        best = max(last_population, key=lambda ind: ind.fitness)

        # Generamos la respuesta uniendo palabras
        response = " ".join(best.genes)

        print("Bot:", response, "\n")


if __name__ == "__main__":
    main()
