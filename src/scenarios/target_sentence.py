import random
import string
from .base_scenario import Scenario


class TargetSentenceScenario(Scenario):
    """
    Evoluciona una frase de 26 caracteres hasta que coincida con
    'HELLO FROM THE EVOLUTION LAB'.
    Fitness = nº de coincidencias / longitud  (máx = 1.0).
    """

    def __init__(self):
        self.target_sentence = "HELLO FROM THE EVOLUTION LAB"
        super().__init__(gene_length=len(self.target_sentence))
        self.charset = string.ascii_uppercase + " "

    # ---------- fitness máxima --------------------------------------- #
    @property
    def max_fitness(self):
        return 1.0

    # ---------- API obligatoria -------------------------------------- #
    def random_genes(self):
        return [random.choice(self.charset) for _ in range(self.gene_length)]

    def evaluate(self, genes):
        matches = sum(
            1 for g, t in zip(genes, self.target_sentence) if g == t
        )
        return matches / self.gene_length

    # ---------- helper para mutación focalizada ---------------------- #
    def incorrect_positions(self, genes):
        return [
            i for i, (g, t) in enumerate(zip(genes, self.target_sentence)) if g != t
        ]
