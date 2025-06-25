import random
import copy
import string


class EvolutionOperators:
    def __init__(self):
        self.charset = string.ascii_uppercase + " "

    def select(self, population, tournament_size=3):
        selected = []
        for _ in range(len(population)):
            tournament = random.sample(population, tournament_size)
            winner = max(tournament, key=lambda ind: ind.fitness)
            selected.append(copy.deepcopy(winner))
        return selected

    def crossover(self, parent1, parent2):
        child_genes = []

        for g1, g2 in zip(parent1.genes, parent2.genes):
            if isinstance(g1, str):
                # Mejora: favorecer conservación de genes correctos
                if random.random() < 0.9:
                    gene = g1 if random.random() < 0.5 else g2
                else:
                    gene = random.choice(self.charset)
            else:
                alpha = 0.5
                gene = (1 - alpha) * g1 + alpha * g2

            child_genes.append(gene)

        return child_genes

    def mutate(self, individual, mutation_rate=0.05, mutation_sigma=0.1):
        for i in range(len(individual.genes)):
            if random.random() < mutation_rate:
                if isinstance(individual.genes[i], str):
                    individual.genes[i] = random.choice(self.charset)
                else:
                    individual.genes[i] += random.gauss(0, mutation_sigma)
