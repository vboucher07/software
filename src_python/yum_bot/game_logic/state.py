from typing import List, Dict, Tuple
from collections import Counter
from .scoring import CATEGORIES

class GameState:
    def __init__(self, dice: List[int], rolls_left: int, used_categories: Dict[str, int]):
        self.dice = sorted(dice)  # always sort to avoid permutation noise
        self.rolls_left = rolls_left
        self.used_categories = used_categories.copy()  # category name -> score
        self.total_score = sum(used_categories.values())  # Initialize total score

    def dice_counts(self):
        return Counter(self.dice)

    def available_categories(self):
        return [cat for cat in CATEGORIES if cat not in self.used_categories]

    def clone_with_new_dice(self, new_dice: List[int], rolls_left: int):
        return GameState(new_dice, rolls_left, self.used_categories)

    def apply_category(self, category: str, score: int):
        self.used_categories[category] = score
        self.total_score += score

    def clone(self):
        return GameState(self.dice, self.rolls_left, self.used_categories)