import copy
from typing import List, Tuple

from ..evolution.operators import EvolutionOperators
from ..models.individual import Individual
from ..utils.run_logger import RunLogger


class EvolutionEngine:
    """
    Motor evolutivo (versión con *micro-mutación*):

    • Mutación adaptativa + simulated annealing.
    • Mutación focalizada en genes incorrectos (si el escenario expone
      `incorrect_positions()`).
    • **Nuevo:** cuando un individuo tiene ≤ 3 genes incorrectos,
      se aplica una *micro-mutación* agresiva (μ = 0.20) sólo en esos
      genes, acelerando la recta final.
    """

    # -------------------------------------------------- #
    # Constructor
    # -------------------------------------------------- #
    def __init__(
            self,
            scenario,
            population_size: int = 400,
            generations: int = 3000,
            stagnation_patience: int = 100,
            # --- mutación adaptativa ----------------------- #
            base_mutation_rate: float = 0.05,
            low_diversity_factor: float = 0.20,
            high_diversity_factor: float = 0.50,
            mutation_boost: float = 2.0,
            mutation_cut: float = 0.5,
            # --- simulated annealing ---------------------- #
            anneal_factor: float = 0.998,
            # --- micro-mutación --------------------------- #
            micro_threshold: int = 3,
            micro_mutation_rate: float = 0.20,
    ):
        self.scenario = scenario
        self.population_size = population_size
        self.generations = generations
        self.stagnation_patience = stagnation_patience

        self.operators = EvolutionOperators()
        self.population: List[Individual] = []
        self._best_fitness_so_far: float | None = None
        self._no_improve_counter: int = 0
        self._optimal_fitness = scenario.max_fitness

        # Adaptative mutation
        self.base_mutation_rate = base_mutation_rate
        self.low_diversity_threshold = int(population_size * low_diversity_factor)
        self.high_diversity_threshold = int(population_size * high_diversity_factor)
        self.mutation_boost = mutation_boost
        self.mutation_cut = mutation_cut
        self.anneal_factor = anneal_factor

        # Micro-mutation
        self.micro_threshold = micro_threshold
        self.micro_mutation_rate = micro_mutation_rate

        # Logger
        self.logger = RunLogger(scenario_name=scenario.__class__.__name__)

    # -------------------------------------------------- #
    # Población: creación y evaluación
    # -------------------------------------------------- #
    def initialize_population(self):
        self.population = [
            Individual(self.scenario.random_genes())
            for _ in range(self.population_size)
        ]

    def evaluate_population(self):
        for ind in self.population:
            ind.fitness = self.scenario.evaluate(ind.genes)

    # -------------------------------------------------- #
    # Métricas y utilidades
    # -------------------------------------------------- #
    def _population_diversity(self) -> int:
        return len({tuple(ind.genes) for ind in self.population})

    def _collect_metrics(self) -> Tuple[Individual, float, int]:
        best = max(self.population, key=lambda ind: ind.fitness)
        avg = sum(ind.fitness for ind in self.population) / len(self.population)
        div = self._population_diversity()
        return best, avg, div

    def _print_report(
            self,
            gen: int,
            best: Individual,
            avg: float,
            div: int,
            μ: float,
    ):
        # Construir representación de los genes
        if all(isinstance(g, str) for g in best.genes):
            genes_str = ''.join(best.genes)
        else:
            # Si son números, convertir la lista entera a string
            genes_str = str(best.genes)

        print(
            f"Gen {gen:<4} "
            f"- Mejor: {genes_str} (fit={best.fitness:.3f}) "
            f"- Avg {avg:.2f} "
            f"- Div {div:<3} "
            f"- μ {μ:.3f}"
        )

    # -------------------------------------------------- #
    # Estancamiento y mutación adaptativa
    # -------------------------------------------------- #
    def _update_stagnation(self):
        current_best = max(self.population, key=lambda ind: ind.fitness).fitness
        if self._best_fitness_so_far is None or current_best > self._best_fitness_so_far:
            self._best_fitness_so_far = current_best
            self._no_improve_counter = 0
        else:
            self._no_improve_counter += 1

    def _adaptive_mu(self, diversity: int) -> float:
        if diversity < self.low_diversity_threshold:
            return self.base_mutation_rate * self.mutation_boost
        if diversity > self.high_diversity_threshold:
            return self.base_mutation_rate * self.mutation_cut
        return self.base_mutation_rate

    # -------------------------------------------------- #
    # Bucle principal
    # -------------------------------------------------- #
    def run(self):
        # --- Generación 0 ------------------------------------------- #
        self.initialize_population()
        self.evaluate_population()

        best, avg, div = self._collect_metrics()
        μ = self._adaptive_mu(div)
        self.logger.log(0, best.fitness, avg, div)
        self._print_report(0, best, avg, div, μ)
        self._update_stagnation()

        # --- Generaciones siguientes -------------------------------- #
        for gen in range(1, self.generations + 1):
            best_prev = max(self.population, key=lambda ind: ind.fitness)
            selected = self.operators.select(self.population)

            next_generation: List[Individual] = []
            for i in range(0, len(selected), 2):
                if i + 1 < len(selected):
                    child_genes = self.operators.crossover(selected[i], selected[i + 1])
                    next_generation.append(Individual(child_genes))
                else:
                    next_generation.append(selected[i])

            # ---- Mutación ------------------------------------------ #
            diversity_prev = self._population_diversity()
            μ = self._adaptive_mu(diversity_prev) * (self.anneal_factor ** gen)

            for ind in next_generation:
                incorrect = None
                if hasattr(self.scenario, "incorrect_positions"):
                    incorrect = self.scenario.incorrect_positions(ind.genes)

                if incorrect and len(incorrect) <= self.micro_threshold:
                    # micro-mutación agresiva sólo en estos genes
                    self.operators.mutate(
                        ind,
                        mutation_rate=self.micro_mutation_rate,
                        incorrect_positions=incorrect,
                    )
                else:
                    # mutación adaptativa estándar
                    self.operators.mutate(
                        ind,
                        mutation_rate=μ,
                        incorrect_positions=incorrect,
                    )

            # ---- Elitismo & relleno ------------------------------- #
            if next_generation:
                next_generation[0] = copy.deepcopy(best_prev)
            while len(next_generation) < self.population_size:
                next_generation.append(copy.deepcopy(best_prev))

            self.population = next_generation

            # ---- Evaluación & métricas ---------------------------- #
            self.evaluate_population()
            best, avg, div = self._collect_metrics()
            self.logger.log(gen, best.fitness, avg, div)
            self._print_report(gen, best, avg, div, μ)
            self._update_stagnation()

            # ---- Condiciones de parada --------------------------- #
            if self._optimal_fitness and self._best_fitness_so_far >= self._optimal_fitness:
                print(f"✅ Fitness óptima ({self._optimal_fitness}) alcanzada en la generación {gen}.")
                break

            if self._no_improve_counter >= self.stagnation_patience:
                print(f"🛑 Sin mejora en {self.stagnation_patience} generaciones. Parando en la generación {gen}.")
                break

        # ---- Guardar métricas ------------------------------------ #
        csv_path = self.logger.save()
        print(f"📄 Métricas guardadas en '{csv_path}'.")
