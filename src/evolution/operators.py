import random
import copy
import string


class EvolutionOperators:
    def __init__(self):
        self.charset = string.ascii_uppercase + " "

    # -------------------------------------------------- #
    # Selección por torneo
    # -------------------------------------------------- #
    def select(self, population, tournament_size=3):
        selected = []
        for _ in range(len(population)):
            tournament = random.sample(population, tournament_size)
            winner = max(tournament, key=lambda ind: ind.fitness)
            selected.append(copy.deepcopy(winner))
        return selected

    # -------------------------------------------------- #
    # Crossover uniforme / aritmético
    # -------------------------------------------------- #
    def crossover(self, parent1, parent2):
        child_genes = []

        for g1, g2 in zip(parent1.genes, parent2.genes):
            if isinstance(g1, str):
                # Favorecemos conservación de genes correctos
                if random.random() < 0.9:
                    gene = g1 if random.random() < 0.5 else g2
                else:
                    gene = random.choice(self.charset)
            else:
                alpha = 0.5
                gene = (1 - alpha) * g1 + alpha * g2

            child_genes.append(gene)

        return child_genes

    # -------------------------------------------------- #
    # Mutación
    # -------------------------------------------------- #
    def mutate(
            self,
            individual,
            mutation_rate: float = 0.05,
            mutation_sigma: float = 0.1,
            incorrect_positions=None,
    ):
        """
        Si `incorrect_positions` es una lista/iterable de índices, se mutan
        **sólo** esas posiciones.  Si es None, se muta todo el genoma.
        """
        positions = (
            incorrect_positions if incorrect_positions is not None else range(len(individual.genes))
        )

        for i in positions:
            if random.random() < mutation_rate:
                if isinstance(individual.genes[i], str):
                    individual.genes[i] = random.choice(self.charset)
                else:
                    individual.genes[i] += random.gauss(0, mutation_sigma)
