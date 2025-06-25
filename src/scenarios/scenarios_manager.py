# ================================================================
#  FILE: src/scenarios/scenarios_manager.py
# ================================================================
from scenarios.simple_maximization import SimpleMaximizationScenario
from scenarios.target_search        import TargetSearchScenario
from scenarios.target_sentence      import TargetSentenceScenario
from scenarios.language_adaptive    import LanguageAdaptiveScenario
from scenarios.dictionary_scenario  import DictionaryScenario
from scenarios.ngram_fluency        import NgramFluencyScenario
from scenarios.language_fluency     import LanguageFluencyScenario


class ScenarioManager:
    """
    Devuelve una instancia de escenario a partir de su nombre.

    cfg: diccionario opcional con parámetros adicionales
         (viene del YAML o de los defaults del main).
    """

    @staticmethod
    def get_scenario(name: str, cfg: dict | None = None):
        cfg = cfg or {}

        if name == "simple_maximization":
            return SimpleMaximizationScenario()

        elif name == "target_search":
            return TargetSearchScenario()

        elif name == "target_sentence":
            return TargetSentenceScenario()

        elif name == "language_adaptive":
            return LanguageAdaptiveScenario()

        elif name == "dictionary_scenario":
            return DictionaryScenario(
                dictionary_file="data/words.txt",
                target_word="WORLD",
            )

        # --- escenarios basados en n-gramas --------------------- #
        elif name == "ngram_fluency":
            return NgramFluencyScenario(
                order  = cfg.get("order", 2),
                length = cfg.get("length", 30),
            )

        elif name == "language_fluency":
            return LanguageFluencyScenario(
                length = cfg.get("length", 40),
            )

        # --------------------------------------------------------- #
        else:
            raise ValueError(f"Scenario '{name}' not recognized.")
