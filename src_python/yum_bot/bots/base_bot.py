"""
Base bot class that all Yum bots inherit from
"""

from abc import ABC, abstractmethod
from typing import List, Tuple
from game_logic.state import GameState
from game_logic.scoring import score_category, CATEGORIES

class BaseBot(ABC):
    """Abstract base class for all Yum bots"""
    
    def __init__(self, name: str):
        self.name = name
    
    @abstractmethod
    def best_move(self, state: GameState) -> Tuple[List[int], str, int]:
        """
        Determine the best move for the current game state
        
        Args:
            state: Current game state
            
        Returns:
            Tuple of (dice_to_keep, category_to_assign, expected_score)
            - dice_to_keep: List of dice values to keep (empty list if no rerolls left)
            - category_to_assign: Category name to assign to (None if rerolls left)
            - expected_score: Expected score for this move
        """
        pass
    
    def get_name(self) -> str:
        """Get the bot's name"""
        return self.name
    
    def _check_constraints(self, dice: List[int], category: str, used_categories: dict) -> bool:
        """Check if a category assignment would violate constraints"""
        if category == "Low Score" and "High Score" in used_categories:
            if score_category(dice, category) >= used_categories["High Score"]:
                return False
        elif category == "High Score" and "Low Score" in used_categories:
            if score_category(dice, category) <= used_categories["Low Score"]:
                return False
        return True
    
    def _get_available_categories(self, used_categories: dict) -> List[str]:
        """Get list of available categories"""
        return [cat for cat in CATEGORIES if cat not in used_categories]
    
    def _get_valid_categories(self, dice: List[int], used_categories: dict) -> List[str]:
        """Get list of valid categories that don't violate constraints"""
        available = self._get_available_categories(used_categories)
        valid = []
        for cat in available:
            if self._check_constraints(dice, cat, used_categories):
                valid.append(cat)
        return valid 