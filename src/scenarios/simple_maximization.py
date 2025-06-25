import random
from models.individual import Individual

class SimpleMaximizationScenario:
    def __init__(self):
        self.gene_count = 1
        self.gene_range = (0, 3)

    def create_individual(self):
        genes = [random.uniform(*self.gene_range) for _ in range(self.gene_count)]
        return Individual(genes)

    def evaluate(self, individual):
        x = individual.genes[0]
        fitness = x * (x ** 2 - 3 * x + 2)
        individual.fitness = fitness
        return fitness
