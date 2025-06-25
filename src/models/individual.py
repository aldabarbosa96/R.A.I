class Individual:
    def __init__(self, genes=None):
        self.genes = genes or []
        self.fitness = None

    def __repr__(self):
        genes_str = ''.join(self.genes) if all(isinstance(g, str) for g in self.genes) else str(self.genes)
        return f"Individual(genes={genes_str}, fitness={self.fitness})"
