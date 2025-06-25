from core.engine import EvolutionEngine
from scenarios.simple_maximization import SimpleMaximizationScenario

if __name__ == "__main__":
    scenario = SimpleMaximizationScenario()
    engine = EvolutionEngine(scenario=scenario)
    engine.run()
