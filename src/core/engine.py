import copy

from evolution.operators import EvolutionOperators
from models.individual import Individual


class EvolutionEngine:
    def __init__(self, scenario, population_size=100, generations=200):
        self.scenario = scenario
        self.population_size = population_size
        self.generations = generations
        self.operators = EvolutionOperators()

    def initialize_population(self):
        self.population = [Individual(self.scenario.random_genes()) for _ in range(self.population_size)]

    def evaluate_population(self):
        for individual in self.population:
            individual.fitness = self.scenario.evaluate(individual.genes)

    def report_best(self, generation):
        best = max(self.population, key=lambda ind: ind.fitness)
        print(f"Generación {generation} - Mejor individuo: {best}")

    def run(self):
        self.initialize_population()
        self.evaluate_population()
        self.report_best(0)

        for gen in range(1, self.generations + 1):
            best_previous = max(self.population, key=lambda ind: ind.fitness)
            selected = self.operators.select(self.population)

            next_generation = []

            for i in range(0, len(selected), 2):
                if i + 1 < len(selected):
                    parent1 = selected[i]
                    parent2 = selected[i + 1]
                    child_genes = self.operators.crossover(parent1, parent2)
                    child = Individual(child_genes)
                    self.operators.mutate(child)
                    next_generation.append(child)
                else:
                    next_generation.append(selected[i])

            # Preservamos el mejor
            if len(next_generation) > 0:
                next_generation[0] = copy.deepcopy(best_previous)

            # Si por cualquier motivo quedó corta la población
            while len(next_generation) < self.population_size:
                next_generation.append(copy.deepcopy(best_previous))

            self.population = next_generation
            self.evaluate_population()
            self.report_best(gen)
