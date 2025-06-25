import random
import string
from scenarios.base_scenario import Scenario


class LanguageAdaptiveScenario(Scenario):
    def __init__(self):
        super().__init__(gene_length=12)  # tamaño de la frase
        self.charset = string.ascii_uppercase + " "

    def random_genes(self):
        return [random.choice(self.charset) for _ in range(self.gene_length)]

    def evaluate(self, genes):
        sentence = ''.join(genes)

        non_alpha_penalty = sum(1 for c in sentence if c not in self.charset)
        words = sentence.split(" ")
        word_bonus = sum(1 for word in words if len(word) >= 3)
        structure_bonus = 1 if sentence[0] in string.ascii_uppercase and " " in sentence else 0

        fitness = word_bonus + structure_bonus - non_alpha_penalty
        return max(fitness, 0)
