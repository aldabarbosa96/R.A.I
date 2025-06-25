# ================================================================
# FILE: src/main.py   (sin YAML, todo en este archivo)
# ================================================================
"""
Ejemplo de ejecución directa del motor evolutivo sin depender
de configuraciones externas.  Modifica las constantes de la
sección CONFIG para probar distintos escenarios o parámetros.
"""

from core.engine import EvolutionEngine
from scenarios.scenarios_manager import ScenarioManager

# ------------------------- CONFIG ------------------------------- #
CONFIG = dict(
    # nombre del escenario disponible en scenarios_manager.py
    #   • "language_fluency"  (fluidez con modelo n-gram simple)
    #   • "ngram_fluency"     (fluidez con bigramas-pickle)
    #   • "target_sentence"   (evoluciona hasta una frase fija)
    #   • "dictionary_scenario", "language_adaptive", etc.
    scenario="language_fluency",

    # parámetros ESPECÍFICOS del escenario (solo si lo usa)
    length=40,      # para language_fluency / ngram_fluency
    order=2,        # solo para ngram_fluency

    # parámetros del motor
    population_size=600,
    generations=4000,
    stagnation_patience=400,

    # mutación adaptativa
    base_mutation_rate=0.05,
    low_diversity_factor=0.20,
    high_diversity_factor=0.45,
    anneal_factor=0.998,
    micro_threshold=5,
    micro_mutation_rate=0.30,
)
# ---------------------------------------------------------------- #


def main():
    # instancia el escenario
    scenario = ScenarioManager.get_scenario(CONFIG["scenario"], CONFIG)

    # caso especial: cambiar la frase objetivo de target_sentence
    if CONFIG["scenario"] == "target_sentence" and "target_sentence" in CONFIG:
        scenario.target_sentence = CONFIG["target_sentence"]
        scenario.gene_length = len(scenario.target_sentence)

    # lanza el motor
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
    engine.run()


if __name__ == "__main__":
    main()
