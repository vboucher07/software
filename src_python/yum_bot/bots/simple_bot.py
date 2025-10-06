"""
Simple greedy bot that always picks the highest scoring available category
"""

from typing import List, Tuple
from game_logic.state import GameState
from game_logic.scoring import score_category
from .base_bot import BaseBot

class SimpleBot(BaseBot):
    """Simple greedy bot that always picks the highest scoring category"""
    
    def __init__(self):
        super().__init__("SimpleBot")
    
    def best_move(self, state: GameState) -> Tuple[List[int], str, int]:
        """Always keep all dice and pick highest scoring category"""
        if state.rolls_left > 0:
            # Keep all dice for reroll
            return (state.dice, None, 0)
        else:
            # Pick highest scoring available category
            best_score = -1
            best_category = None
            
            for cat in self._get_valid_categories(state.dice, state.used_categories):
                score = score_category(state.dice, cat)
                if score > best_score:
                    best_score = score
                    best_category = cat
            
            return (state.dice, best_category, best_score) 