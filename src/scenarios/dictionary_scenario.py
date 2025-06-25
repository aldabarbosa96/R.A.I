import random
from scenarios.base_scenario import Scenario


class DictionaryScenario(Scenario):
    """
    Evoluciona palabras hasta coincidir con la palabra objetivo.
    Fitness = nº de caracteres correctos en posición correcta (máximo = longitud palabra).
    """

    def __init__(self, dictionary_file: str, target_word: str):
        self.target_word = target_word.upper()
        super().__init__(gene_length=len(self.target_word))
        self.dictionary = self._load_filtered_dictionary(dictionary_file)

    # ------------------------------------------------------------------ #
    # Propiedad de fitness máxima
    # ------------------------------------------------------------------ #
    @property
    def max_fitness(self):
        return self.gene_length  # 1 punto por carácter correcto

    # ------------------------------------------------------------------ #
    # Carga y filtrado de diccionario
    # ------------------------------------------------------------------ #
    def _load_filtered_dictionary(self, file_path: str):
        with open(file_path, "r") as file:
            words = [line.strip().upper() for line in file if line.strip()]

        words = [
            w for w in words
            if w != self.target_word and len(w) == self.gene_length
        ]

        if not words:
            raise ValueError(
                f"No hay palabras de longitud {self.gene_length} distintas de '{self.target_word}'"
            )
        return words

    # ------------------------------------------------------------------ #
    # API obligatoria del escenario
    # ------------------------------------------------------------------ #
    def random_genes(self):
        return list(random.choice(self.dictionary))

    def evaluate(self, genes):
        return sum(1 for g, t in zip(genes, self.target_word) if g == t)
