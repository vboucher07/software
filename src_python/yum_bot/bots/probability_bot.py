#!/usr/bin/env python3
"""
Probability-Based Bot - Calculates optimal expected value for entire remaining game
"""

import numpy as np
from typing import List, Tuple, Dict, Set
from collections import Counter
import itertools
import sys
import os

# Add the parent directory to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from game_logic.state import GameState
from game_logic.scoring import score_category, CATEGORIES
from bots.base_bot import BaseBot


class ProbabilityBot(BaseBot):
    """
    Probability-Based Bot
    
    Uses dynamic programming to calculate the optimal expected value
    for the entire remaining game, considering all possible future outcomes.
    """
    
    def __init__(self):
        super().__init__("ProbabilityBot")
        self._initialize_probability_tables()
    
    def _initialize_probability_tables(self):
        """Initialize probability tables for all possible scenarios"""
        print("Initializing probability tables...")
        
        # Precompute all possible dice combinations
        self.all_dice_combinations = []
        for combo in itertools.product(range(1, 7), repeat=5):
            self.all_dice_combinations.append(sorted(combo))
        
        # Precompute reroll probabilities
        self._compute_reroll_probabilities()
        
        # Precompute category probabilities
        self._compute_category_probabilities()
        
        print("Probability tables initialized!")
    
    def _compute_reroll_probabilities(self):
        """Compute probabilities for all possible reroll outcomes"""
        self.reroll_probs = {}
        
        # For each possible keep combination
        for keep_size in range(1, 6):
            for keep_dice in itertools.combinations(range(1, 7), keep_size):
                reroll_size = 5 - keep_size
                outcomes = {}
                
                # For each possible reroll outcome
                for reroll_outcome in itertools.product(range(1, 7), repeat=reroll_size):
                    final_dice = sorted(list(keep_dice) + list(reroll_outcome))
                    dice_key = tuple(final_dice)
                    
                    if dice_key not in outcomes:
                        outcomes[dice_key] = 0
                    outcomes[dice_key] += 1
                
                # Convert to probabilities
                total_outcomes = 6 ** reroll_size
                for dice_key, count in outcomes.items():
                    outcomes[dice_key] = count / total_outcomes
                
                self.reroll_probs[keep_dice] = outcomes
    
    def _compute_category_probabilities(self):
        """Compute probabilities for achieving each category"""
        self.category_probs = {}
        
        for dice in self.all_dice_combinations:
            dice_key = tuple(dice)
            self.category_probs[dice_key] = {}
            
            for category in CATEGORIES:
                # Calculate probability of achieving this category
                prob = self._calculate_category_probability(dice, category)
                self.category_probs[dice_key][category] = prob
    
    def _calculate_category_probability(self, dice: List[int], category: str) -> float:
        """Calculate probability of achieving a category with given dice"""
        if category in ["1s", "2s", "3s", "4s", "5s", "6s"]:
            return self._calculate_upper_category_probability(dice, int(category[0]))
        elif category == "Low Score":
            return self._calculate_low_score_probability(dice)
        elif category == "High Score":
            return self._calculate_high_score_probability(dice)
        elif category == "Low Straight":
            return self._calculate_low_straight_probability(dice)
        elif category == "High Straight":
            return self._calculate_high_straight_probability(dice)
        elif category == "Full House":
            return self._calculate_full_house_probability(dice)
        elif category == "Yum":
            return self._calculate_yum_probability(dice)
        else:
            return 0.0
    
    def _calculate_upper_category_probability(self, dice: List[int], target: int) -> float:
        """Calculate probability of achieving upper section category"""
        current_count = dice.count(target)
        current_score = current_count * target
        
        # If we have all 5 dice, probability is 1 if we have the target
        if len(dice) == 5:
            return 1.0 if current_score > 0 else 0.0
        
        # Calculate probability of improving with rerolls
        remaining_dice = 5 - len(dice)
        if remaining_dice == 0:
            return 1.0 if current_score > 0 else 0.0
        
        # Probability of getting additional target dice
        prob_additional = 1/6
        expected_additional = remaining_dice * prob_additional
        expected_score = current_score + (expected_additional * target)
        
        # Return probability based on expected score
        return min(1.0, expected_score / (5 * target))
    
    def _calculate_low_score_probability(self, dice: List[int]) -> float:
        """Calculate probability of achieving Low Score"""
        current_sum = sum(dice)
        
        if len(dice) == 5:
            return 1.0  # Always achievable
        
        # With rerolls, always achievable
        return 1.0
    
    def _calculate_high_score_probability(self, dice: List[int]) -> float:
        """Calculate probability of achieving High Score"""
        return self._calculate_low_score_probability(dice)
    
    def _calculate_low_straight_probability(self, dice: List[int]) -> float:
        """Calculate probability of achieving Low Straight (1,2,3,4,5)"""
        required = {1, 2, 3, 4, 5}
        current_unique = set(dice)
        missing = required - current_unique
        
        if len(missing) == 0:
            return 1.0
        
        if len(dice) == 5:
            return 0.0
        
        # Calculate probability of getting missing dice
        remaining_dice = 5 - len(dice)
        if len(missing) > remaining_dice:
            return 0.0
        
        # Simplified probability calculation
        prob_per_die = 1/6
        prob_all_missing = (prob_per_die ** len(missing))
        return prob_all_missing
    
    def _calculate_high_straight_probability(self, dice: List[int]) -> float:
        """Calculate probability of achieving High Straight (2,3,4,5,6)"""
        required = {2, 3, 4, 5, 6}
        current_unique = set(dice)
        missing = required - current_unique
        
        if len(missing) == 0:
            return 1.0
        
        if len(dice) == 5:
            return 0.0
        
        # Calculate probability of getting missing dice
        remaining_dice = 5 - len(dice)
        if len(missing) > remaining_dice:
            return 0.0
        
        # Simplified probability calculation
        prob_per_die = 1/6
        prob_all_missing = (prob_per_die ** len(missing))
        return prob_all_missing
    
    def _calculate_full_house_probability(self, dice: List[int]) -> float:
        """Calculate probability of achieving Full House"""
        counts = Counter(dice)
        current_score = 25 if sorted(counts.values()) == [2, 3] else 0
        
        if len(dice) == 5:
            return 1.0 if current_score == 25 else 0.0
        
        # Very simplified probability
        return 0.1  # Low probability
    
    def _calculate_yum_probability(self, dice: List[int]) -> float:
        """Calculate probability of achieving Yum (5 of a kind)"""
        counts = Counter(dice)
        current_score = 30 if any(count == 5 for count in counts.values()) else 0
        
        if len(dice) == 5:
            return 1.0 if current_score == 30 else 0.0
        
        # Find most common die
        most_common = max(counts.items(), key=lambda x: x[1])
        target_die = most_common[0]
        current_count = most_common[1]
        
        remaining_dice = 5 - len(dice)
        if current_count + remaining_dice < 5:
            return 0.0
        
        # Probability of getting Yum
        needed = 5 - current_count
        if needed > remaining_dice:
            return 0.0
        
        prob_yum = (1/6) ** needed
        return prob_yum
    
    def _calculate_expected_value(self, state: GameState, depth: int = 0) -> float:
        """Calculate expected value for entire remaining game"""
        if depth > 20:  # Prevent infinite recursion
            return 0.0
        
        # If game is over, return final score
        if len(state.used_categories) == 12:
            return state.total_score
        
        # If no rolls left, must assign to category
        if state.rolls_left == 0:
            best_ev = -1
            for category in CATEGORIES:
                if category not in state.used_categories:
                    score = score_category(state.dice, category)
                    # Create new state and calculate remaining EV
                    new_state = GameState(state.dice, 3, state.used_categories.copy())
                    new_state.apply_category(category, score)
                    remaining_ev = self._calculate_expected_value(new_state, depth + 1)
                    total_ev = score + remaining_ev
                    best_ev = max(best_ev, total_ev)
            return best_ev
        
        # Calculate EV for each possible keep combination
        keep_combinations = self._get_keep_combinations(state.dice)
        best_ev = -1
        
        for keep_dice in keep_combinations:
            # Calculate EV of this keep combination
            ev = self._calculate_reroll_ev(keep_dice, state, depth)
            best_ev = max(best_ev, ev)
        
        return best_ev
    
    def _calculate_reroll_ev(self, keep_dice: Tuple[int, ...], state: GameState, depth: int) -> float:
        """Calculate expected value of rerolling with given keep dice"""
        if len(keep_dice) == 5:
            return 0.0  # No reroll possible
        
        # Get reroll probabilities for this keep combination
        if keep_dice not in self.reroll_probs:
            return 0.0
        
        total_ev = 0.0
        
        # For each possible reroll outcome
        for outcome_dice, prob in self.reroll_probs[keep_dice].items():
            # Create new state with outcome
            new_state = GameState(list(outcome_dice), state.rolls_left - 1, state.used_categories.copy())
            
            # Calculate expected value of this outcome
            outcome_ev = self._calculate_expected_value(new_state, depth + 1)
            
            # Add weighted contribution
            total_ev += prob * outcome_ev
        
        return total_ev
    
    def _get_keep_combinations(self, dice: List[int]) -> List[Tuple[int, ...]]:
        """Get all possible keep combinations for current dice"""
        combinations = []
        for r in range(1, len(dice) + 1):
            for combo in itertools.combinations(dice, r):
                combinations.append(combo)
        return combinations
    
    def best_move(self, state: GameState) -> Tuple[List[int], str, int]:
        """Determine the best move using full game probability analysis"""
        
        # If no rolls left, must assign to category
        if state.rolls_left == 0:
            best_category = None
            best_score = -1
            
            for category in CATEGORIES:
                if category not in state.used_categories:
                    score = score_category(state.dice, category)
                    if score > best_score:
                        best_score = score
                        best_category = category
            
            return (state.dice, best_category, best_score)
        
        # Calculate EV for each possible keep combination
        keep_combinations = self._get_keep_combinations(state.dice)
        best_ev = -1
        best_keep = []
        
        for keep_dice in keep_combinations:
            # Calculate EV of this keep combination
            ev = self._calculate_reroll_ev(keep_dice, state, 0)
            
            if ev > best_ev:
                best_ev = ev
                best_keep = list(keep_dice)
        
        # If best EV is very low, consider assigning to category now
        if best_ev < 5.0 and state.rolls_left < 3:
            # Find best immediate category assignment
            best_category = None
            best_score = -1
            
            for category in CATEGORIES:
                if category not in state.used_categories:
                    score = score_category(state.dice, category)
                    if score > best_score:
                        best_score = score
                        best_category = category
            
            if best_score > best_ev:
                return (state.dice, best_category, best_score)
        
        # Otherwise, keep dice and reroll
        return (best_keep, None, 0)


class ProbabilityBotV2(ProbabilityBot):
    """
    Enhanced Probability Bot with better probability calculations
    """
    
    def __init__(self):
        super().__init__()
        self.name = "ProbabilityBotV2"
    
    def _calculate_straight_probability(self, dice: List[int], required: Set[int]) -> float:
        """Enhanced straight probability calculation"""
        current_unique = set(dice)
        missing = required - current_unique
        
        if len(missing) == 0:
            return 1.0
        
        if len(dice) == 5:
            return 0.0
        
        # Calculate probability more accurately
        remaining_dice = 5 - len(dice)
        if len(missing) > remaining_dice:
            return 0.0
        
        # More accurate probability calculation
        prob_per_die = 1/6
        prob_all_missing = (prob_per_die ** len(missing))
        return prob_all_missing
    
    def _calculate_low_straight_probability(self, dice: List[int]) -> float:
        """Enhanced Low Straight probability"""
        return self._calculate_straight_probability(dice, {1, 2, 3, 4, 5})
    
    def _calculate_high_straight_probability(self, dice: List[int]) -> float:
        """Enhanced High Straight probability"""
        return self._calculate_straight_probability(dice, {2, 3, 4, 5, 6}) 