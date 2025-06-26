import string
import pickle
import re
from abc import ABC
from pathlib import Path

from .base_scenario import Scenario


class LanguageAdaptiveFluencyScenario(Scenario, ABC):
    """
    Combina:
      • Fluidez de bigramas en español
      • Reconocimiento de palabras reales (unigramas)
      • Estructura básica (mayúscula inicial, punto final)
      • Penalización por repeticiones (AA, BB…)
      • Adaptación al prompt (bonus por palabras compartidas)
    """

    def __init__(
            self,
            prompt: str,
            length: int | None = None,
            bigram_file: str = "data/processed/es_bigrams.pkl",
            unigram_file: str = "data/processed/es_unigrams.pkl",
    ):
        # si no dan longitud, al menos tan largo como el prompt o 40 chars
        if length is None:
            length = max(len(prompt), 40)
        super().__init__(gene_length=length)

        # guardamos prompt en mayúsculas y sus palabras
        self.prompt = prompt.upper()
        self.prompt_words = set(re.findall(r"[A-ZÁÉÍÓÚÜÑ]+", self.prompt))

        # alfabeto: A–Z + espacio
        self.charset = string.ascii_uppercase + " "

        # cargamos bigramas
        root = Path(__file__).resolve().parents[2]
        bg_path = root / bigram_file
        if not bg_path.exists():
            raise FileNotFoundError(f"Bigram file not found → {bg_path}")
        with bg_path.open("rb") as f:
            self.bigram_freq: dict[str, int] = pickle.load(f)

        # cargamos unigramas (frecuencia de palabras reales)
        ug_path = root / unigram_file
        if not ug_path.exists():
            # si no tienes un pickle de unigrams, inicializa vacío
            self.unigram_freq = {}
        else:
            with ug_path.open("rb") as f:
                self.unigram_freq: dict[str, int] = pickle.load(f)

        # pesos: ajusta a tu gusto
        self.W_BG = 1.0  # bigramas
        self.W_UG = 2.0  # unigramas/palabras
        self.W_STR = 5.0  # estructura (mayúscula, punto)
        self.W_REP = 1.0  # penalización repeticiones
        self.W_ADAPT = 3.0  # adaptación al prompt

    def evaluate(self, genes: list[str]) -> float:
        text = "".join(genes)
        # 1) Score bigramas
        bg_score = sum(
            self.bigram_freq.get(text[i:i + 2], 0)
            for i in range(len(text) - 1)
        )

        # 2) Score unigramas / palabras reales
        words = text.split()
        ug_score = sum(
            self.unigram_freq.get(w, 0)
            for w in words
        )

        # 3) Bonus de estructura
        str_bonus = 0
        if text and text[0].isalpha():
            str_bonus += 1
        if text.endswith("."):
            str_bonus += 1

        # 4) Penalización por repeticiones dobles
        rep_penalty = sum(
            1 for _ in re.finditer(r"(.)\1", text)
        )

        # 5) Adaptación al prompt: palabras en común
        adapt_score = sum(
            1 for w in words
            if w in self.prompt_words
        )

        # combinación lineal
        fitness = (
                self.W_BG * bg_score
                + self.W_UG * ug_score
                + self.W_STR * str_bonus
                - self.W_REP * rep_penalty
                + self.W_ADAPT * adapt_score
        )
        return fitness
