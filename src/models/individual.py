class Individual:
    def __init__(self, genes):
        self.genes = genes or [0.0]
        self.fitness = None

    def __repr__(self):
        return f"Individual(genes={self.genes}, fitness={self.fitness})"
