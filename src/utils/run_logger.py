"""
RunLogger
=========

Pequeño registrador de métricas evolutivas.

Uso:
-----
logger = RunLogger(scenario_name="dictionary_scenario")
logger.log(gen, best_fitness, avg_fitness, diversity)
csv_path = logger.save()   # ⇒ runs/2025-06-25T12-34-56_run_dictionary_scenario.csv
"""

from __future__ import annotations

import csv
import datetime as _dt
import os
from typing import List, Tuple


class RunLogger:
    def __init__(self, scenario_name: str):
        self.scenario_name = scenario_name
        self.start_time: _dt.datetime = _dt.datetime.now()
        self._records: List[Tuple[int, float, float, int]] = []

    # ------------------------------------------------------------------ #
    # Registro
    # ------------------------------------------------------------------ #
    def log(self, generation: int, best: float, avg: float, diversity: int) -> None:
        """Añade una fila de métricas al buffer en memoria."""
        self._records.append((generation, best, avg, diversity))

    # ------------------------------------------------------------------ #
    # Persistencia
    # ------------------------------------------------------------------ #
    def save(self, output_dir: str = "runs") -> str:
        """
        Guarda el CSV con todos los registros y devuelve la ruta completa.
        Crea la carpeta *runs/* si no existe.
        """
        os.makedirs(output_dir, exist_ok=True)

        timestamp = self.start_time.strftime("%Y-%m-%dT%H-%M-%S")
        filename = f"{timestamp}_run_{self.scenario_name}.csv"
        path = os.path.join(output_dir, filename)

        with open(path, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["generation", "best_fitness", "avg_fitness", "diversity"])
            writer.writerows(self._records)

        return path
