import random
from .base_scenario import Scenario


class TargetSearchScenario(Scenario):
    def __init__(self):
        super().__init__(gene_length=1)
        self.gene_range = (0, 100)
        self.target = random.uniform(*self.gene_range)

    def random_genes(self):
        return [random.uniform(*self.gene_range)]

    def evaluate(self, genes):
        x = genes[0]
        fitness = 1 / (1 + abs(x - self.target))
        return fitness
