from scenarios.simple_maximization import SimpleMaximizationScenario
from scenarios.target_search import TargetSearchScenario
from scenarios.target_sentence import TargetSentenceScenario
from scenarios.language_adaptive import LanguageAdaptiveScenario
from scenarios.dictionary_scenario import DictionaryScenario


class ScenarioManager:
    @staticmethod
    def get_scenario(name):
        if name == "simple_maximization":
            return SimpleMaximizationScenario()
        elif name == "target_search":
            return TargetSearchScenario()
        elif name == "target_sentence":
            return TargetSentenceScenario()
        elif name == "language_adaptive":
            return LanguageAdaptiveScenario()
        elif name == "dictionary_scenario":
            return DictionaryScenario(dictionary_file="data/words.txt", target_word="WORLD")
        else:
            raise ValueError(f"Scenario '{name}' not recognized.")
