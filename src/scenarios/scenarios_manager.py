from pathlib import Path

from .simple_maximization import SimpleMaximizationScenario
from .target_search import TargetSearchScenario
from .target_sentence import TargetSentenceScenario
from .language_adaptive import LanguageAdaptiveScenario
from .dictionary_scenario import DictionaryScenario
from .ngram_fluency import NgramFluencyScenario
from .language_fluency import LanguageFluencyScenario
from .language_adaptative_fluency import LanguageAdaptiveFluencyScenario


class ScenarioManager:
    @staticmethod
    def get_scenario(name: str, cfg: dict | None = None):
        cfg = cfg or {}
        data_dir = Path(__file__).resolve().parents[2] / "data" / "processed"

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
                dictionary_file=str(data_dir / "words.txt"),
                target_word="WORLD",
            )

        elif name == "ngram_fluency":
            return NgramFluencyScenario(
                order=cfg.get("order", 2),
                length=cfg.get("length", 30),
                ngram_file=str(data_dir / "bigrams.pkl"),
            )

        elif name == "language_fluency":
            return LanguageFluencyScenario(
                length=cfg.get("length", 40),
                bigram_file=str(data_dir / "bigrams.pkl"),
            )

        elif name == "language_adaptive_fluency":
            # cfg["prompt"] → texto de usuario
            prompt = cfg.get("prompt", "")
            length = cfg.get("length", max(len(prompt), 40))
            return LanguageAdaptiveFluencyScenario(
                prompt=prompt,
                length=length,
                bigram_file=str(data_dir / "es_bigrams.pkl"),
                unigram_file=str(data_dir / "es_unigrams.pkl"),
            )

        else:
            raise ValueError(f"Scenario '{name}' not recognized.")
