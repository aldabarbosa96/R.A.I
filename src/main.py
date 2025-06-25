# ================================================================
# FILE: src/main.py   (configuración en un único bloque CONFIG)
# ================================================================
"""
Ejecución directa del motor evolutivo SIN archivos YAML.

– Modifica el diccionario CONFIG para probar distintos escenarios
  y parámetros del motor.
– El archivo ScenarioManager ya sabe leer las claves que necesite
  cada escenario (order, length, target_word, …).
"""

from pathlib import Path  # sólo para mensajes de control opcionales
from core.engine import EvolutionEngine
from scenarios.scenarios_manager import ScenarioManager

# ------------------------- CONFIG ------------------------------- #
CONFIG: dict = {
    # ─ Escenario a ejecutar ───────────────────────────────────── #
    #   • "language_fluency"
    #   • "ngram_fluency"
    #   • "target_sentence"
    #   • "dictionary_scenario"
    #   • "language_adaptive", "simple_maximization", ...
    "scenario": "language_fluency",

    # ─ Parámetros del escenario (solo si aplica) ─────────────── #
    "length": 40,  # language_/ngram_fluency
    "order": 2,  # ngram_fluency (2=bigramas, 3=trigramas…)
    "target_word": "HOUSE",  # dictionary_scenario

    # ─ Parámetros del motor evolutivo ────────────────────────── #
    "population_size": 600,
    "generations": 4000,
    "stagnation_patience": 400,

    # Mutación adaptativa
    "base_mutation_rate": 0.05,
    "low_diversity_factor": 0.20,
    "high_diversity_factor": 0.45,
    "anneal_factor": 0.998,

    # Micro-mutación focalizada
    "micro_threshold": 5,
    "micro_mutation_rate": 0.30,
}


# ---------------------------------------------------------------- #


def main() -> None:
    """Punto de entrada principal."""
    # 1. Instanciar el escenario adecuado
    scenario = ScenarioManager.get_scenario(CONFIG["scenario"], CONFIG)

    # «Hack» opcional: cambiar la frase objetivo al vuelo
    if CONFIG["scenario"] == "target_sentence" and "target_sentence" in CONFIG:
        scenario.target_sentence = CONFIG["target_sentence"].upper()
        scenario.gene_length = len(scenario.target_sentence)

    # 2. Crear el motor evolutivo con los parámetros de CONFIG
    engine = EvolutionEngine(
        scenario=scenario,
        population_size=CONFIG["population_size"],
        generations=CONFIG["generations"],
        stagnation_patience=CONFIG["stagnation_patience"],
        base_mutation_rate=CONFIG["base_mutation_rate"],
        low_diversity_factor=CONFIG["low_diversity_factor"],
        high_diversity_factor=CONFIG["high_diversity_factor"],
        anneal_factor=CONFIG["anneal_factor"],
        micro_threshold=CONFIG["micro_threshold"],
        micro_mutation_rate=CONFIG["micro_mutation_rate"],
    )

    # 3. ¡A rodar!
    engine.run()


if __name__ == "__main__":
    main()
