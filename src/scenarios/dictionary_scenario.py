import random
from scenarios.base_scenario import Scenario


class DictionaryScenario(Scenario):
    def __init__(self, dictionary_file, target_word):
        super().__init__(gene_length=len(target_word))
        self.dictionary = self.load_dictionary(dictionary_file)
        self.target_word = target_word.upper()

    def load_dictionary(self, file_path):
        with open(file_path, "r") as file:
            words = [line.strip().upper() for line in file if line.strip()]
        return words

    def random_genes(self):
        word = random.choice(self.dictionary)
        return list(word)

    def evaluate(self, genes):
        matches = sum(1 for a, b in zip(genes, self.target_word) if a == b)
        return matches
