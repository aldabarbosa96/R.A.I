import random
import copy


class EvolutionOperators:
    def __init__(self):
        pass

    def select(self, population, tournament_size=3):
        selected = []
        for _ in range(len(population)):
            tournament = random.sample(population, tournament_size)
            winner = max(tournament, key=lambda ind: ind.fitness)
            selected.append(copy.deepcopy(winner))
        return selected

    def crossover(self, parent1, parent2, alpha=0.5):
        child_genes = []
        for g1, g2 in zip(parent1.genes, parent2.genes):
            gene = (1 - alpha) * g1 + alpha * g2
            child_genes.append(gene)
        return child_genes

    def mutate(self, individual, mutation_rate=0.2, mutation_sigma=0.1):
        for i in range(len(individual.genes)):
            if random.random() < mutation_rate:
                individual.genes[i] += random.gauss(0, mutation_sigma)
