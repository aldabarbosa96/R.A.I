from abc import ABC, abstractmethod


class Scenario(ABC):
    """
    Clase base para todos los escenarios evolutivos.
    """

    def __init__(self, gene_length: int):
        self.gene_length = gene_length

    # ------------------------------------------------------------------ #
    # Métodos que cada escenario debe implementar
    # ------------------------------------------------------------------ #
    @abstractmethod
    def random_genes(self):
        """Devuelve una lista de genes aleatorios para un individuo nuevo."""
        pass

    @abstractmethod
    def evaluate(self, genes):
        """Devuelve la fitness de un conjunto de genes."""
        pass

    # ------------------------------------------------------------------ #
    # Opcional: fitness máxima conocida
    # ------------------------------------------------------------------ #
    @property
    def max_fitness(self):
        """
        Retorna la fitness máxima teórica si se conoce.
        Devuelve None cuando no es trivial conocerla.
        """
        return None
