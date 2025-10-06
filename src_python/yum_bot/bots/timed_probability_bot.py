#!/usr/bin/env python3
"""
Time-Limited Probability Bot - Tests different thinking times
"""

import numpy as np
from typing import List, Tuple, Dict, Set
from collections import Counter
import itertools
import time
import sys
import os

# Add the parent directory to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from game_logic.state import GameState
from game_logic.scoring import score_category, CATEGORIES
from bots.base_bot import BaseBot


class TimedProbabilityBot(BaseBot):
    """
    Time-Limited Probability Bot
    
    Uses dynamic programming with a time limit to calculate optimal moves.
    Tests different thinking times to find the sweet spot.
    """
    
    def __init__(self, time_limit_ms: int = 100):
        super().__init__(f"TimedProbabilityBot_{time_limit_ms}ms")
        self.time_limit_ms = time_limit_ms
        self._initialize_probability_tables()
    
    def _initialize_probability_tables(self):
        """Initialize probability tables for all possible scenarios"""
        print(f"Initializing probability tables for {self.time_limit_ms}ms time limit...")
        
        # Precompute all possible dice combinations
        self.all_dice_combinations = []
        for combo in itertools.product(range(1, 7), repeat=5):
            self.all_dice_combinations.append(sorted(combo))
        
        # Precompute reroll probabilities
        self._compute_reroll_probabilities()
        
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
    
    def _calculate_expected_value(self, state: GameState, depth: int = 0, start_time: float = None) -> float:
        """Calculate expected value with time limit"""
        if start_time is None:
            start_time = time.time()
        
        # Check time limit
        elapsed_ms = (time.time() - start_time) * 1000
        if elapsed_ms > self.time_limit_ms:
            return self._quick_heuristic(state)  # Fallback to heuristic
        
        if depth > 10:  # Prevent infinite recursion
            return self._quick_heuristic(state)
        
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
                    remaining_ev = self._calculate_expected_value(new_state, depth + 1, start_time)
                    total_ev = score + remaining_ev
                    best_ev = max(best_ev, total_ev)
            return best_ev
        
        # Calculate EV for each possible keep combination
        keep_combinations = self._get_keep_combinations(state.dice)
        best_ev = -1
        
        for keep_dice in keep_combinations:
            # Check time limit before each calculation
            elapsed_ms = (time.time() - start_time) * 1000
            if elapsed_ms > self.time_limit_ms:
                break
            
            # Calculate EV of this keep combination
            ev = self._calculate_reroll_ev(keep_dice, state, depth, start_time)
            best_ev = max(best_ev, ev)
        
        return best_ev if best_ev > -1 else self._quick_heuristic(state)
    
    def _calculate_reroll_ev(self, keep_dice: Tuple[int, ...], state: GameState, depth: int, start_time: float) -> float:
        """Calculate expected value of rerolling with given keep dice"""
        if len(keep_dice) == 5:
            return 0.0  # No reroll possible
        
        # Get reroll probabilities for this keep combination
        if keep_dice not in self.reroll_probs:
            return 0.0
        
        total_ev = 0.0
        outcomes_considered = 0
        
        # For each possible reroll outcome
        for outcome_dice, prob in self.reroll_probs[keep_dice].items():
            # Check time limit
            elapsed_ms = (time.time() - start_time) * 1000
            if elapsed_ms > self.time_limit_ms:
                break
            
            # Create new state with outcome
            new_state = GameState(list(outcome_dice), state.rolls_left - 1, state.used_categories.copy())
            
            # Calculate expected value of this outcome
            outcome_ev = self._calculate_expected_value(new_state, depth + 1, start_time)
            
            # Add weighted contribution
            total_ev += prob * outcome_ev
            outcomes_considered += 1
        
        # If we didn't consider all outcomes due to time limit, adjust
        if outcomes_considered < len(self.reroll_probs[keep_dice]):
            # Use heuristic for remaining probability
            remaining_prob = sum(prob for _, prob in list(self.reroll_probs[keep_dice].items())[outcomes_considered:])
            heuristic_ev = self._quick_heuristic(state) * remaining_prob
            total_ev += heuristic_ev
        
        return total_ev
    
    def _quick_heuristic(self, state: GameState) -> float:
        """Quick heuristic for when time runs out"""
        # Simple heuristic: sum of current dice + expected value of remaining categories
        current_score = sum(state.dice)
        remaining_categories = 12 - len(state.used_categories)
        expected_per_category = 10  # Rough estimate
        return current_score + (remaining_categories * expected_per_category)
    
    def _get_keep_combinations(self, dice: List[int]) -> List[Tuple[int, ...]]:
        """Get all possible keep combinations for current dice"""
        combinations = []
        for r in range(1, len(dice) + 1):
            for combo in itertools.combinations(dice, r):
                combinations.append(combo)
        return combinations
    
    def best_move(self, state: GameState) -> Tuple[List[int], str, int]:
        """Determine the best move using time-limited probability analysis"""
        
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
        
        # Start timing
        start_time = time.time()
        
        # Calculate EV for each possible keep combination
        keep_combinations = self._get_keep_combinations(state.dice)
        best_ev = -1
        best_keep = []
        
        for keep_dice in keep_combinations:
            # Check time limit
            elapsed_ms = (time.time() - start_time) * 1000
            if elapsed_ms > self.time_limit_ms:
                break
            
            # Calculate EV of this keep combination
            ev = self._calculate_reroll_ev(keep_dice, state, 0, start_time)
            
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


class TimedProbabilityBotV2(TimedProbabilityBot):
    """
    Enhanced Time-Limited Probability Bot with better heuristics
    """
    
    def __init__(self, time_limit_ms: int = 100):
        super().__init__(time_limit_ms)
        self.name = f"TimedProbabilityBotV2_{time_limit_ms}ms"
    
    def _quick_heuristic(self, state: GameState) -> float:
        """Enhanced quick heuristic"""
        # More sophisticated heuristic
        current_score = sum(state.dice)
        
        # Calculate potential for each remaining category
        remaining_categories = [cat for cat in CATEGORIES if cat not in state.used_categories]
        total_potential = 0
        
        for category in remaining_categories:
            if category in ["1s", "2s", "3s", "4s", "5s", "6s"]:
                target = int(category[0])
                current_count = state.dice.count(target)
                potential = current_count * target + (5 - current_count) * target * 0.17  # Expected value
                total_potential += potential
            elif category in ["Low Score", "High Score"]:
                total_potential += 17.5  # Average of 5 dice
            elif category == "Low Straight":
                total_potential += 15 * 0.1  # Low probability
            elif category == "High Straight":
                total_potential += 20 * 0.1  # Low probability
            elif category == "Full House":
                total_potential += 25 * 0.05  # Very low probability
            elif category == "Yum":
                total_potential += 30 * 0.01  # Very low probability
        
        return current_score + total_potential 