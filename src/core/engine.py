import random
from models.individual import Individual
from evolution.operators import EvolutionOperators


class EvolutionEngine:
    def __init__(self, scenario, population_size=20, generations=20):
        self.scenario = scenario
        self.population_size = population_size
        self.generations = generations
        self.population = []
        self.operators = EvolutionOperators()

    def initialize_population(self):
        self.population = [self.scenario.create_individual() for _ in range(self.population_size)]

    def evaluate_population(self):
        for individual in self.population:
            self.scenario.evaluate(individual)

    def run(self):
        self.initialize_population()
        self.evaluate_population()
        self.report_best(0)

        for gen in range(1, self.generations + 1):
            selected = self.operators.select(self.population)
            next_generation = []

            i = 0
            while len(next_generation) < self.population_size:
                parent1 = selected[i % len(selected)]
                parent2 = selected[(i + 1) % len(selected)]
                child_genes = self.operators.crossover(parent1, parent2)
                child = Individual(child_genes)
                self.operators.mutate(child)
                next_generation.append(child)
                i += 2

            self.population = next_generation
            self.evaluate_population()
            self.report_best(gen)

    def report_best(self, generation):
        best = max(self.population, key=lambda ind: ind.fitness)
        print(f"Generación {generation} - Mejor individuo: {best}")
