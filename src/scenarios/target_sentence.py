import random
import string
from scenarios.base_scenario import Scenario


class TargetSentenceScenario(Scenario):
    def __init__(self):
        self.target_sentence = "HELLO WORLD"
        super().__init__(gene_length=len(self.target_sentence))
        self.charset = string.ascii_uppercase + " "

    def random_genes(self):
        return [random.choice(self.charset) for _ in range(self.gene_length)]

    def evaluate(self, genes):
        matches = sum(1 for gene, target in zip(genes, self.target_sentence) if gene == target)
        fitness = matches / self.gene_length
        return fitness
