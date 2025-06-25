"""
RunLogger
=========

Pequeño registrador de métricas evolutivas.

• Siempre guarda los CSV en <repo-root>/runs/ sin importar
  desde dónde ejecutes `python`.
• Nombre de archivo:  <YYYY-MM-DD>T<HH-MM-SS>_run_<Scenario>.csv
"""

from __future__ import annotations

import csv
import datetime as _dt
from pathlib import Path
from typing import List, Tuple


class RunLogger:
    """
    Acumula estadísticas (best, avg, diversity) por generación y
    las vuelca a un CSV al final de la ejecución.
    """

    def __init__(self, scenario_name: str) -> None:
        self.scenario_name = scenario_name
        self.start_time: _dt.datetime = _dt.datetime.now()
        self._records: List[Tuple[int, float, float, int]] = []

    # ------------------------------------------------------------------ #
    # Registro
    # ------------------------------------------------------------------ #
    def log(self, generation: int, best: float, avg: float, diversity: int) -> None:
        """Añade una fila de métricas al buffer."""
        self._records.append((generation, best, avg, diversity))

    # ------------------------------------------------------------------ #
    # Persistencia
    # ------------------------------------------------------------------ #
    def save(self, output_dir: str = "runs") -> str:
        """
        Guarda el CSV y devuelve la ruta absoluta del archivo.
        El directorio <repo-root>/runs/ se crea si no existe.
        """
        # ➊ Raíz del proyecto: …/aldabarbosa96-r.a.i/
        repo_root = Path(__file__).resolve().parents[2]

        # ➋ Directorio destino absoluto
        out_dir = repo_root / output_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        # ➌ Nombre de archivo único
        timestamp = self.start_time.strftime("%Y-%m-%dT%H-%M-%S")
        filename  = f"{timestamp}_run_{self.scenario_name}.csv"
        path      = out_dir / filename

        # ➍ Escritura del CSV
        with path.open("w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["generation", "best_fitness", "avg_fitness", "diversity"])
            writer.writerows(self._records)

        return str(path)     # para imprimir o registrar
