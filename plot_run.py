"""
Genera un gráfico con:
- Mejor fitness (línea sólida)
- Fitness promedio (línea sólida)
- Diversidad (línea discontinua, eje secundario)

Uso:
    python plot_run.py <CSV_FILE>
Si no se encuentra el archivo, se muestran los CSV disponibles en ./runs.
"""

import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt


def load_csv(path: Path):
    gens, best, avg, div = [], [], [], []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            gens.append(int(row["generation"]))
            best.append(float(row["best_fitness"]))
            avg.append(float(row["avg_fitness"]))
            div.append(float(row["diversity"]))
    return gens, best, avg, div


def show_available_runs(runs_dir: Path):
    print("\n⮕ CSV disponibles en", runs_dir)
    for p in runs_dir.rglob("*_run_*.csv"):
        print("  •", p)
    print()


def main(csv_arg: str):
    csv_path = Path(csv_arg).expanduser().resolve()

    if not csv_path.exists():
        print(f"❌ Archivo no encontrado: {csv_path}")
        runs_dir = Path("runs").resolve()
        if runs_dir.exists():
            show_available_runs(runs_dir)
        sys.exit(1)

    gens, best, avg, div = load_csv(csv_path)

    fig, ax1 = plt.subplots()
    ax1.plot(gens, best, label="Best fitness")
    ax1.plot(gens, avg, label="Avg fitness")
    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Fitness")
    ax1.legend(loc="upper left")

    ax2 = ax1.twinx()
    ax2.plot(gens, div, linestyle="--", label="Diversity")
    ax2.set_ylabel("Diversity")
    ax2.legend(loc="upper right")

    plt.title(csv_path.name)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python plot_run.py <CSV_FILE>")
        show_available_runs(Path("runs"))
        sys.exit(1)

    main(sys.argv[1])
