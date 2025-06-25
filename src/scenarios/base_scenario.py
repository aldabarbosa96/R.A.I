from abc import ABC, abstractmethod


class Scenario(ABC):
    def __init__(self, gene_length):
        self.gene_length = gene_length

    @abstractmethod
    def random_genes(self):
        pass

    @abstractmethod
    def evaluate(self, genes):
        pass
