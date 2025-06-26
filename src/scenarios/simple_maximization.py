import random
from .base_scenario import Scenario


class SimpleMaximizationScenario(Scenario):
    def __init__(self):
        super().__init__(gene_length=1)
        self.gene_range = (0, 3)

    def random_genes(self):
        return [random.uniform(*self.gene_range)]

    def evaluate(self, genes):
        x = genes[0]
        fitness = x * (x ** 2 - 3 * x + 2)
        return fitness
