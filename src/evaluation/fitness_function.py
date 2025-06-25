class FitnessFunction:
    def evaluate(self, individual):
        x = individual.genes[0]
        fitness = x * (x ** 2 - 3 * x + 2)
        individual.fitness = fitness
        return fitness
