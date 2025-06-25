"""
main.py · Lanzador del GA word-level (GAWord)

Ejecución básica:
    python -m src.main                          # usa parámetros por defecto
Ejemplo con overrides:
    python -m src.main --pop-size 400 --max-gens 5000
"""
from __future__ import annotations

import argparse
from datetime      import datetime
from pathlib       import Path
from typing        import Sequence

from ga_word import GAWord         # el GA que acabamos de ajustar
from vocab   import decode         # utilidades ids→texto


# ───────────────────────────────────────────
# CLI
# ───────────────────────────────────────────
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GA a nivel palabra")
    p.add_argument("--pop-size", type=int, default=300,
                   help="Tamaño de la población")
    p.add_argument("--max-gens", type=int, default=2_000,
                   help="Número máximo de generaciones")
    p.add_argument("--runs-dir", type=str,  default="runs",
                   help="Carpeta donde guardar los CSV de métricas")
    return p.parse_args()


# ───────────────────────────────────────────
# Helpers
# ───────────────────────────────────────────
def _to_str(candidate: str | Sequence[int]) -> str:
    """Normaliza posible lista de ids → string legible."""
    if isinstance(candidate, str):
        return candidate
    return decode(candidate, skip_special=True)


def _save_metrics(path: Path, rows: list[tuple[int, float, str]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        fh.write("generation,fitness,best\n")
        for gen, fit, best in rows:
            fh.write(f"{gen},{fit},{best}\n")


# ───────────────────────────────────────────
# Programa principal
# ───────────────────────────────────────────
def main() -> None:
    args = _parse_args()

    ga = GAWord(
        pop_size   = args.pop_size,
        max_gens   = args.max_gens,
        runs_dir   = args.runs_dir,
    )

    history: list[tuple[int, float, str]] = []

    for gen, fit, best in ga.run():
        best_str = _to_str(best)
        print(f"Gen {gen:<4d} · Mejor: {best_str} (fit={fit:.3f})")
        history.append((gen, fit, best_str))

    # guardar CSV
    ts       = datetime.now().isoformat(timespec="seconds").replace(":", "-")
    csv_path = Path(args.runs_dir) / f"{ts}_run_word.csv"
    _save_metrics(csv_path, history)
    print(f"\n📄 Métricas guardadas en «{csv_path.resolve()}»")


if __name__ == "__main__":
    main()
