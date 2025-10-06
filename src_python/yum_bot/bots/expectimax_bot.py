"""
Expectimax bot with advanced strategic planning
"""

from typing import List, Tuple, Dict
from game_logic.state import GameState
from game_logic.scoring import score_category, CATEGORIES, get_upper_section_bonus
from .base_bot import BaseBot
from collections import defaultdict, Counter
import itertools
import random

class ExpectimaxBot(BaseBot):
    def __init__(self, depth=2):
        super().__init__("ExpectimaxBot")
        self.depth = depth
        self.memo = {}
        self.category_values = {
            "Yum": 30,
            "High Straight": 20,
            "Full House": 25,
            "Low Straight": 15,
            "High Score": 25,
            "Low Score": 15,
            "6s": 18,
            "5s": 15,
            "4s": 12,
            "3s": 9,
            "2s": 6,
            "1s": 3
        }

    def best_move(self, state: GameState) -> Tuple[List[int], str, int]:
        best_score = float('-inf')
        best_keep = []
        best_category = None

        current_dice = state.dice

        # If we still have rolls, try rerolling options
        if state.rolls_left > 0:
            # Use smart pruning to reduce search space
            keep_combinations = self._get_smart_combinations(current_dice)
            
            for keep_indices in keep_combinations:
                kept_dice = [current_dice[i] for i in keep_indices]
                expected_score = self._quick_expectimax(state, kept_dice, state.rolls_left)
                if expected_score > best_score:
                    best_score = expected_score
                    best_keep = kept_dice
            return (best_keep, None, int(max(best_score, 0)))

        # Otherwise, assign to best category (final move)
        else:
            # Use advanced category selection
            best_category, best_score = self._select_best_category(state, current_dice)
            return (current_dice, best_category, int(max(best_score, 0)))

    def _select_best_category(self, state: GameState, dice: List[int]) -> Tuple[str, int]:
        """Advanced category selection with strategic considerations"""
        best_score = float('-inf')
        best_category = None
        
        # Calculate current upper section total
        upper_subtotal = sum(state.used_categories.get(f"{i}s", 0) for i in range(1, 7))
        remaining_upper = [f"{i}s" for i in range(1, 7) if f"{i}s" not in state.used_categories]
        
        for cat in CATEGORIES:
            if cat not in state.used_categories:
                # Check Low Score vs High Score constraint
                if cat == "Low Score" and "High Score" in state.used_categories:
                    if score_category(dice, cat) >= state.used_categories["High Score"]:
                        continue
                elif cat == "High Score" and "Low Score" in state.used_categories:
                    if score_category(dice, cat) <= state.used_categories["Low Score"]:
                        continue
                
                actual_score = score_category(dice, cat)
                
                # Strategic adjustments for selection (not for final score)
                adjusted_score = self._adjust_category_score(state, cat, actual_score, dice, upper_subtotal, remaining_upper)
                
                if adjusted_score > best_score:
                    best_score = adjusted_score
                    best_category = cat
                    best_actual_score = actual_score  # Store the actual score
        
        # If no valid category found, pick the first available one
        if best_category is None:
            for cat in CATEGORIES:
                if cat not in state.used_categories:
                    best_category = cat
                    best_actual_score = score_category(dice, cat)
                    break
        
        return best_category, best_actual_score

    def _adjust_category_score(self, state: GameState, category: str, base_score: int, dice: List[int], upper_subtotal: int, remaining_upper: List[str]) -> float:
        """Adjust category score based on strategic considerations"""
        adjusted_score = base_score
        game_phase = len(state.used_categories)
        
        # Early game (turns 1-4): Focus on high-value categories and upper section
        if game_phase < 4:
            if category in ["Yum", "High Straight", "Full House"]:
                adjusted_score *= 1.5  # Boost high-value categories
            elif category in ["6s", "5s", "4s"]:
                adjusted_score *= 1.3  # Boost upper section
            elif category in ["3s", "2s", "1s"]:
                adjusted_score *= 0.8  # Reduce lower upper section
        
        # Mid game (turns 5-8): Balance between categories
        elif game_phase < 8:
            if category in ["Yum", "High Straight", "Full House"]:
                adjusted_score *= 1.3
            elif category in remaining_upper:
                # Boost upper section if close to bonus
                if upper_subtotal >= 50:
                    adjusted_score *= 1.4
                elif upper_subtotal >= 40:
                    adjusted_score *= 1.2
        
        # Late game (turns 9-12): Focus on remaining opportunities
        else:
            if category in ["Yum", "High Straight", "Full House"]:
                adjusted_score *= 1.2
            elif category in remaining_upper:
                # Strongly boost upper section if close to bonus
                if upper_subtotal >= 55:
                    adjusted_score *= 2.0
                elif upper_subtotal >= 45:
                    adjusted_score *= 1.5
        
        # Penalty for wasting high-value dice on low categories
        dice_sum = sum(dice)
        if category in ["Low Score", "High Score"]:
            if dice_sum <= 15 and category == "High Score":
                adjusted_score *= 0.5  # Penalty for using low dice for High Score
            elif dice_sum >= 20 and category == "Low Score":
                adjusted_score *= 0.5  # Penalty for using high dice for Low Score
        
        return adjusted_score

    def _quick_expectimax(self, state: GameState, kept_dice: List[int], rolls_left: int) -> float:
        """Fast expectimax with smart sampling"""
        if rolls_left == 0:
            return self._evaluate_final_rolls_smart(state, kept_dice)
        
        # Sample intelligently based on what we're keeping
        sample_size = min(100, 6 ** (5 - len(kept_dice)))
        total_score = 0
        
        for _ in range(sample_size):
            # Generate random roll for remaining dice
            remaining_dice = [random.randint(1, 6) for _ in range(5 - len(kept_dice))]
            full_dice = kept_dice + remaining_dice
            full_dice.sort()
            
            # Find best category for this roll using advanced selection
            best_category, best_score = self._select_best_category(state, full_dice)
            total_score += best_score
        
        return total_score / sample_size if sample_size > 0 else 0

    def _evaluate_final_rolls_smart(self, state: GameState, kept_dice: List[int]) -> float:
        """Smart evaluation for final rolls"""
        # Sample intelligently
        sample_size = min(50, 6 ** (5 - len(kept_dice)))
        total = 0
        
        for _ in range(sample_size):
            remaining_dice = [random.randint(1, 6) for _ in range(5 - len(kept_dice))]
            full_dice = kept_dice + remaining_dice
            full_dice.sort()
            
            # Use advanced category selection
            best_category, best_score = self._select_best_category(state, full_dice)
            total += best_score
        
        return total / sample_size if sample_size > 0 else 0

    def _get_smart_combinations(self, dice: List[int]) -> List[List[int]]:
        """Get smart combinations to keep based on dice patterns"""
        combinations = []
        counts = Counter(dice)
        
        # Always consider keeping nothing
        combinations.append([])
        
        # Keep high-value combinations
        for number, count in counts.items():
            if count >= 2:
                # Keep 3+ of a kind
                if count >= 3:
                    indices = [i for i, die in enumerate(dice) if die == number]
                    combinations.append(indices)
                # Keep pairs for potential full house
                elif count == 2:
                    indices = [i for i, die in enumerate(dice) if die == number]
                    combinations.append(indices)
        
        # Keep straights
        unique_dice = set(dice)
        if len(unique_dice) >= 4:
            # Check for low straight potential
            if {1, 2, 3, 4}.issubset(unique_dice):
                indices = [i for i, die in enumerate(dice) if die in [1, 2, 3, 4]]
                combinations.append(indices)
            # Check for high straight potential
            if {2, 3, 4, 5}.issubset(unique_dice):
                indices = [i for i, die in enumerate(dice) if die in [2, 3, 4, 5]]
                combinations.append(indices)
            if {3, 4, 5, 6}.issubset(unique_dice):
                indices = [i for i, die in enumerate(dice) if die in [3, 4, 5, 6]]
                combinations.append(indices)
        
        # Keep individual high-value dice for upper section
        for i, die in enumerate(dice):
            if die >= 4:  # Keep 4s, 5s, 6s
                combinations.append([i])
        
        # Remove duplicates and limit total combinations
        unique_combinations = []
        seen = set()
        for combo in combinations:
            combo_tuple = tuple(sorted(combo))
            if combo_tuple not in seen:
                seen.add(combo_tuple)
                unique_combinations.append(list(combo))
        
        # Limit to top 15 combinations
        return unique_combinations[:15] 