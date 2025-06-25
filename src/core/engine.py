import copy
from evolution.operators import EvolutionOperators
from models.individual import Individual


class EvolutionEngine:
    """
    Motor evolutivo genérico con:
    - Parada por estancamiento (`stagnation_patience`)
    - Parada por alcanzar `scenario.max_fitness` (si se conoce)
    - Métricas de promedio y diversidad
    """

    def __init__(
            self,
            scenario,
            population_size: int = 100,
            generations: int = 200,
            stagnation_patience: int = 20,
    ):
        self.scenario = scenario
        self.population_size = population_size
        self.generations = generations
        self.stagnation_patience = stagnation_patience
        self.operators = EvolutionOperators()

        # Internos
        self.population = []
        self._best_fitness_so_far = None
        self._no_improve_counter = 0
        self._optimal_fitness = scenario.max_fitness  # Puede ser None

    # ------------------------------------------------------------------ #
    # Inicialización y evaluación
    # ------------------------------------------------------------------ #
    def initialize_population(self):
        self.population = [
            Individual(self.scenario.random_genes())
            for _ in range(self.population_size)
        ]

    def evaluate_population(self):
        for individual in self.population:
            individual.fitness = self.scenario.evaluate(individual.genes)

    # ------------------------------------------------------------------ #
    # Estancamiento
    # ------------------------------------------------------------------ #
    def _update_stagnation(self):
        current_best = max(self.population, key=lambda ind: ind.fitness).fitness
        if self._best_fitness_so_far is None or current_best > self._best_fitness_so_far:
            self._best_fitness_so_far = current_best
            self._no_improve_counter = 0
        else:
            self._no_improve_counter += 1

    # ------------------------------------------------------------------ #
    # Informes
    # ------------------------------------------------------------------ #
    def _population_diversity(self):
        return len({tuple(ind.genes) for ind in self.population})

    def report_best(self, generation: int):
        best = max(self.population, key=lambda ind: ind.fitness)
        avg_fitness = sum(ind.fitness for ind in self.population) / len(self.population)
        diversity = self._population_diversity()
        print(
            f"Generación {generation:<3} "
            f"- Mejor: {best.genes} (fit={best.fitness}) "
            f"- Promedio fit={avg_fitness:.2f} "
            f"- Diversidad={diversity}"
        )

    # ------------------------------------------------------------------ #
    # Bucle principal
    # ------------------------------------------------------------------ #
    def run(self):
        self.initialize_population()
        self.evaluate_population()
        self._update_stagnation()
        self.report_best(0)

        for gen in range(1, self.generations + 1):
            best_previous = max(self.population, key=lambda ind: ind.fitness)
            selected = self.operators.select(self.population)

            # Reproducción
            next_generation = []
            for i in range(0, len(selected), 2):
                if i + 1 < len(selected):
                    parent1, parent2 = selected[i], selected[i + 1]
                    child_genes = self.operators.crossover(parent1, parent2)
                    child = Individual(child_genes)
                    self.operators.mutate(child)
                    next_generation.append(child)
                else:
                    next_generation.append(selected[i])

            # Elitismo
            if next_generation:
                next_generation[0] = copy.deepcopy(best_previous)

            # Rellenar población si hace falta
            while len(next_generation) < self.population_size:
                next_generation.append(copy.deepcopy(best_previous))

            self.population = next_generation
            self.evaluate_population()
            self._update_stagnation()
            self.report_best(gen)

            # --- Condición de parada por fitness óptima ---
            if (
                    self._optimal_fitness is not None
                    and self._best_fitness_so_far >= self._optimal_fitness
            ):
                print(f"✅ Fitness óptima ({self._optimal_fitness}) alcanzada en la generación {gen}.")
                break

            # --- Condición de parada por estancamiento ---
            if self._no_improve_counter >= self.stagnation_patience:
                print(
                    f"🛑 Sin mejora en {self.stagnation_patience} generaciones. "
                    f"Parando en la generación {gen}."
                )
                break
