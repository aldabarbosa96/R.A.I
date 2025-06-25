from core.engine import EvolutionEngine
from scenarios.scenarios_manager import ScenarioManager

if __name__ == "__main__":
    scenario_name = "dictionary_scenario"  # Activamos el nuevo escenario
    scenario = ScenarioManager.get_scenario(scenario_name)
    engine = EvolutionEngine(scenario=scenario, population_size=100, generations=200)
    engine.run()
