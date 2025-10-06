"""
Heuristic-based bots for Yum game
"""

import random
from typing import List, Tuple, Dict
from game_logic.state import GameState
from game_logic.scoring import score_category, CATEGORIES, get_upper_section_bonus
from .base_bot import BaseBot
from collections import Counter

class HeuristicBotV1(BaseBot):
    """
    Heuristic Bot Version 1.0
    
    Basic heuristic strategy:
    - Keep 3+ of a kind for Yum potential
    - Keep pairs for Full House potential
    - Keep high-value dice for upper section
    - Pick highest scoring category
    """
    
    def __init__(self):
        super().__init__("HeuristicBotV1")
    
    def best_move(self, state: GameState) -> Tuple[List[int], str, int]:
        """Basic heuristic strategy"""
        if state.rolls_left > 0:
            keep_dice = self._decide_what_to_keep(state.dice)
            return (keep_dice, None, 0)
        else:
            return self._pick_best_category(state)
    
    def _decide_what_to_keep(self, dice: List[int]) -> List[int]:
        """Decide which dice to keep based on patterns"""
        counts = Counter(dice)
        keep_dice = []
        
        # Keep 3+ of a kind for Yum potential
        for number, count in counts.items():
            if count >= 3:
                keep_dice.extend([number] * count)
        
        # Keep pairs for Full House potential
        if not keep_dice:
            for number, count in counts.items():
                if count >= 2:
                    keep_dice.extend([number] * count)
                    break
        
        # Keep high-value dice for upper section
        if not keep_dice:
            for die in dice:
                if die >= 4:  # Keep 4s, 5s, 6s
                    keep_dice.append(die)
        
        return keep_dice
    
    def _pick_best_category(self, state: GameState) -> Tuple[List[int], str, int]:
        """Pick the highest scoring available category"""
        best_score = -1
        best_category = None
        
        for cat in self._get_valid_categories(state.dice, state.used_categories):
            score = score_category(state.dice, cat)
            if score > best_score:
                best_score = score
                best_category = cat
        
        return (state.dice, best_category, best_score)

class HeuristicBotV2(HeuristicBotV1):
    """
    Heuristic Bot Version 2.0
    
    Enhanced strategy with:
    - Upper section bonus optimization
    - Game phase awareness
    - Strategic category prioritization
    """
    
    def __init__(self):
        super().__init__()
        self.name = "HeuristicBotV2"
    
    def _decide_what_to_keep(self, dice: List[int]) -> List[int]:
        """Enhanced dice keeping strategy"""
        counts = Counter(dice)
        keep_dice = []
        
        # Check for Yum (5 of a kind)
        for number, count in counts.items():
            if count >= 5:
                return [number] * 5
        
        # Check for 4 of a kind
        for number, count in counts.items():
            if count >= 4:
                return [number] * 4
        
        # Check for Full House
        three_kind = None
        two_kind = None
        for number, count in counts.items():
            if count >= 3:
                three_kind = number
            elif count >= 2:
                two_kind = number
        
        if three_kind and two_kind:
            return [three_kind] * 3 + [two_kind] * 2
        
        # Check for straights
        unique_dice = set(dice)
        if {1, 2, 3, 4, 5}.issubset(unique_dice):
            return [1, 2, 3, 4, 5]
        if {2, 3, 4, 5, 6}.issubset(unique_dice):
            return [2, 3, 4, 5, 6]
        
        # Keep 3 of a kind
        for number, count in counts.items():
            if count >= 3:
                return [number] * count
        
        # Keep pairs
        for number, count in counts.items():
            if count >= 2:
                return [number] * count
        
        # Keep high-value dice for upper section
        high_dice = [die for die in dice if die >= 4]
        if high_dice:
            return high_dice
        
        return []
    
    def _pick_best_category(self, state: GameState) -> Tuple[List[int], str, int]:
        """Enhanced category selection with strategic considerations"""
        game_phase = len(state.used_categories)
        upper_subtotal = sum(state.used_categories.get(f"{i}s", 0) for i in range(1, 7))
        
        best_score = -1
        best_category = None
        
        for cat in self._get_valid_categories(state.dice, state.used_categories):
            base_score = score_category(state.dice, cat)
            adjusted_score = self._adjust_score(state, cat, base_score, upper_subtotal, game_phase)
            
            if adjusted_score > best_score:
                best_score = adjusted_score
                best_category = cat
        
        if best_category:
            actual_score = score_category(state.dice, best_category)
            return (state.dice, best_category, actual_score)
        else:
            # Fallback
            for cat in self._get_available_categories(state.used_categories):
                return (state.dice, cat, score_category(state.dice, cat))
    
    def _adjust_score(self, state: GameState, category: str, base_score: int, upper_subtotal: int, game_phase: int) -> float:
        """Adjust category score based on strategic considerations"""
        adjusted_score = base_score
        
        # Boost high-value categories in early game
        if game_phase < 4:
            if category in ["Yum", "High Straight", "Full House"]:
                adjusted_score *= 1.5
            elif category in ["6s", "5s", "4s"]:
                adjusted_score *= 1.3
        
        # Boost upper section if close to bonus
        elif game_phase < 8:
            if category in ["6s", "5s", "4s"] and upper_subtotal >= 50:
                adjusted_score *= 1.4
            elif category in ["Yum", "High Straight", "Full House"]:
                adjusted_score *= 1.3
        
        # Strongly boost upper section in late game if close to bonus
        else:
            if category in ["6s", "5s", "4s"] and upper_subtotal >= 55:
                adjusted_score *= 2.0
            elif category in ["Yum", "High Straight", "Full House"]:
                adjusted_score *= 1.2
        
        return adjusted_score

class HeuristicBotV3(HeuristicBotV2):
    """
    Heuristic Bot Version 3.0
    
    Advanced strategy with:
    - Monte Carlo tree search for reroll decisions
    - Advanced pattern recognition
    - Dynamic strategy adaptation
    """
    
    def __init__(self):
        super().__init__()
        self.name = "HeuristicBotV3"
    
    def _decide_what_to_keep(self, dice: List[int]) -> List[int]:
        """Advanced dice keeping with Monte Carlo evaluation"""
        # First, try the basic heuristic
        basic_keep = super()._decide_what_to_keep(dice)
        
        # If we have a strong pattern, use it
        if len(basic_keep) >= 3:
            return basic_keep
        
        # Otherwise, use Monte Carlo to evaluate options
        keep_options = self._generate_keep_options(dice)
        best_keep = []
        best_value = -1
        
        for keep_dice in keep_options:
            value = self._evaluate_keep_option(dice, keep_dice)
            if value > best_value:
                best_value = value
                best_keep = keep_dice
        
        return best_keep
    
    def _generate_keep_options(self, dice: List[int]) -> List[List[int]]:
        """Generate reasonable keep options"""
        options = []
        counts = Counter(dice)
        
        # No keep
        options.append([])
        
        # Keep individual high-value dice
        for die in dice:
            if die >= 4:
                options.append([die])
        
        # Keep pairs
        for number, count in counts.items():
            if count >= 2:
                options.append([number] * count)
        
        # Keep 3+ of a kind
        for number, count in counts.items():
            if count >= 3:
                options.append([number] * count)
        
        return options[:10]  # Limit options
    
    def _evaluate_keep_option(self, dice: List[int], keep_dice: List[int]) -> float:
        """Evaluate a keep option using Monte Carlo sampling"""
        if not keep_dice:
            return 0.0
        
        total_value = 0
        samples = 50
        
        for _ in range(samples):
            # Simulate reroll
            remaining_dice = [d for d in dice if d not in keep_dice]
            reroll_dice = [random.randint(1, 6) for _ in range(len(remaining_dice))]
            new_dice = keep_dice + reroll_dice
            new_dice.sort()
            
            # Evaluate the new dice
            value = self._evaluate_dice_combination(new_dice)
            total_value += value
        
        return total_value / samples
    
    def _evaluate_dice_combination(self, dice: List[int]) -> float:
        """Evaluate the value of a dice combination"""
        counts = Counter(dice)
        value = 0
        
        # Check for Yum
        if any(count >= 5 for count in counts.values()):
            value += 30
        
        # Check for Full House
        elif any(count >= 3 for count in counts.values()) and sum(1 for count in counts.values() if count >= 2) >= 2:
            value += 25
        
        # Check for straights
        unique_dice = set(dice)
        if {1, 2, 3, 4, 5}.issubset(unique_dice):
            value += 15
        elif {2, 3, 4, 5, 6}.issubset(unique_dice):
            value += 20
        
        # Upper section value
        for number, count in counts.items():
            value += number * count * 0.5  # Reduced weight for upper section
        
        return value 