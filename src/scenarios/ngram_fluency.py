# ================================================================
# FILE: src/scenarios/ngram_fluency.py
# ================================================================
import os
import pickle
import random
import string
from scenarios.base_scenario import Scenario


class NgramFluencyScenario(Scenario):
    """
    Puntúa una frase por la suma de las frecuencias de sus n-gramas
    (bigramas por defecto) en un corpus.

    - order = 2 → bigramas, order = 3 → trigramas.
    - length = longitud fija de la frase.
    - ngram_file = pickle con dict{ngram: frecuencia}.
    """

    def __init__(self, order: int = 2, length: int = 30,
                 ngram_file: str = "data/bigrams.pkl"):
        super().__init__(gene_length=length)
        self.order = order
        self.charset = string.ascii_uppercase + " "
        # carga el dict de frecuencias
        with open(os.path.join("src", ngram_file), "rb") as f:
            self.ngram_freqs: dict[str, int] = pickle.load(f)

    # --------- API obligatoria ---------------------------------- #
    def random_genes(self):
        return [random.choice(self.charset) for _ in range(self.gene_length)]

    def evaluate(self, genes):
        s = "".join(genes)
        score = 0
        for i in range(len(s) - self.order + 1):
            ngram = s[i:i + self.order]
            if " " in ngram:           # ignoramos n-gramas con espacios
                continue
            score += self.ngram_freqs.get(ngram, 0)
        return score

    # fitness máxima desconocida
    @property
    def max_fitness(self):
        return None
