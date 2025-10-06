# Bots package with registry system
from typing import Dict, Type, List
from .base_bot import BaseBot

class BotRegistry:
    """Registry for managing different bot types and versions"""
    
    def __init__(self):
        self.bots: Dict[str, Dict[str, Type[BaseBot]]] = {}
    
    def register_bot(self, bot_type: str, version: str, bot_class: Type[BaseBot]):
        """Register a bot class with a specific type and version"""
        if bot_type not in self.bots:
            self.bots[bot_type] = {}
        self.bots[bot_type][version] = bot_class
    
    def get_bot(self, bot_type: str, version: str) -> Type[BaseBot]:
        """Get a bot class by type and version"""
        if bot_type not in self.bots or version not in self.bots[bot_type]:
            raise ValueError(f"Bot {bot_type} version {version} not found")
        return self.bots[bot_type][version]
    
    def list_bot_types(self) -> List[str]:
        """List all available bot types"""
        return list(self.bots.keys())
    
    def list_versions(self, bot_type: str) -> List[str]:
        """List all versions for a specific bot type"""
        if bot_type not in self.bots:
            return []
        return list(self.bots[bot_type].keys())
    
    def create_bot(self, bot_type: str, version: str, **kwargs) -> BaseBot:
        """Create a bot instance by type and version"""
        bot_class = self.get_bot(bot_type, version)
        return bot_class(**kwargs)

registry = BotRegistry()

from .simple_bot import SimpleBot
from .expectimax_bot import ExpectimaxBot
from .neural_bot import NeuralBotV1, NeuralBotV2, NeuralBotV3
from .heuristic_bot import HeuristicBotV1, HeuristicBotV2, HeuristicBotV3
from .ev_bot import EVBot, EVBotV2
from .probability_bot import ProbabilityBot, ProbabilityBotV2

# Register all bots
registry.register_bot("Simple", "1.0", SimpleBot)
registry.register_bot("Expectimax", "1.0", ExpectimaxBot)
registry.register_bot("Expectimax", "2.0", ExpectimaxBot)
registry.register_bot("Neural", "1.0", NeuralBotV1)
registry.register_bot("Neural", "2.0", NeuralBotV2)
registry.register_bot("Neural", "3.0", NeuralBotV3)
registry.register_bot("Heuristic", "1.0", HeuristicBotV1)
registry.register_bot("Heuristic", "2.0", HeuristicBotV2)
registry.register_bot("Heuristic", "3.0", HeuristicBotV3)
registry.register_bot("EV", "1.0", EVBot)
registry.register_bot("EV", "2.0", EVBotV2)
registry.register_bot("Probability", "1.0", ProbabilityBot)
registry.register_bot("Probability", "2.0", ProbabilityBotV2)

def get_registry() -> BotRegistry:
    return registry 