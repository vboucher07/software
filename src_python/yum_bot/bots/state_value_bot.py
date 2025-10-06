#!/usr/bin/env python3
"""
State Value Bot for Yum Game

This bot uses a trained neural network to estimate the value of game states,
then makes deterministic decisions for categories and dice kept based on heuristics
and the estimated state values.
"""

import numpy as np
import random
from typing import List, Tuple, Dict, Optional
from game_logic.state import GameState
from game_logic.scoring import score_category, CATEGORIES, get_upper_section_bonus
from .base_bot import BaseBot


class StateValueBot(BaseBot):
    """
    State Value Bot
    
    Uses a trained neural network to estimate state values and makes
    deterministic decisions for categories and dice kept.
    """
    
    def __init__(self, model_path: str = None, hidden_size: int = 128):
        super().__init__("StateValueBot")
        self.hidden_size = hidden_size
        self.weights = None
        self.biases = None
        
        if model_path:
            self.load_model(model_path)
        else:
            self._initialize_network()
    
    def _initialize_network(self):
        """Initialize neural network weights"""
        input_size = 50
        output_size = 1
        
        self.weights = {
            'hidden': np.random.randn(input_size, self.hidden_size) * np.sqrt(2.0 / input_size),
            'output': np.random.randn(self.hidden_size, output_size) * np.sqrt(2.0 / self.hidden_size)
        }
        
        self.biases = {
            'hidden': np.zeros(self.hidden_size),
            'output': np.zeros(output_size)
        }
    
    def load_model(self, model_path: str):
        """Load a trained model"""
        try:
            model_data = np.load(model_path, allow_pickle=True).item()
            self.weights = model_data['weights']
            self.biases = model_data['biases']
            self.hidden_size = model_data['hidden_size']
            print(f"Loaded state value model from {model_path}")
        except Exception as e:
            print(f"Failed to load model from {model_path}: {e}")
            print("Initializing with random weights...")
            self._initialize_network()
    
    def _extract_features(self, state: GameState) -> np.ndarray:
        """Extract features from game state"""
        features = []
        
        # Dice features (25 features)
        dice_counts = [0] * 6
        for die in state.dice:
            dice_counts[die - 1] += 1
        features.extend(dice_counts)  # 6 features
        
        # Dice combinations (19 features)
        features.append(1 if any(count >= 5 for count in dice_counts) else 0)  # Yum
        features.append(1 if any(count >= 4 for count in dice_counts) else 0)  # 4 of a kind
        features.append(1 if any(count >= 3 for count in dice_counts) else 0)  # 3 of a kind
        features.append(1 if sum(1 for count in dice_counts if count >= 2) >= 2 else 0)  # 2 pairs
        features.append(1 if any(count >= 3 for count in dice_counts) and sum(1 for count in dice_counts if count >= 2) >= 2 else 0)  # Full house
        
        # Straight features
        unique_dice = set(state.dice)
        features.append(1 if {1, 2, 3, 4, 5}.issubset(unique_dice) else 0)  # Low straight
        features.append(1 if {2, 3, 4, 5, 6}.issubset(unique_dice) else 0)  # High straight
        features.append(1 if {1, 2, 3, 4}.issubset(unique_dice) else 0)  # Low straight potential
        features.append(1 if {2, 3, 4, 5}.issubset(unique_dice) else 0)  # High straight potential
        features.append(1 if {3, 4, 5, 6}.issubset(unique_dice) else 0)  # High straight potential 2
        
        # Sum features
        dice_sum = sum(state.dice)
        features.append(dice_sum / 30.0)  # Normalized sum
        features.append(1 if dice_sum <= 15 else 0)  # Low sum
        features.append(1 if dice_sum >= 20 else 0)  # High sum
        
        # Game state features (15 features)
        features.append(state.rolls_left / 3.0)  # Normalized rolls left
        features.append(len(state.used_categories) / 12.0)  # Normalized turns passed
        
        # Upper section features
        upper_subtotal = sum(state.used_categories.get(f"{i}s", 0) for i in range(1, 7))
        features.append(upper_subtotal / 63.0)  # Normalized upper section progress
        features.append(1 if upper_subtotal >= 63 else 0)  # Bonus achieved
        features.append(1 if upper_subtotal >= 50 else 0)  # Close to bonus
        
        # Category availability (12 features)
        for cat in CATEGORIES:
            features.append(1 if cat not in state.used_categories else 0)
        
        # Constraint features
        low_score = state.used_categories.get("Low Score", 0)
        high_score = state.used_categories.get("High Score", 0)
        features.append(1 if low_score > 0 else 0)  # Low Score used
        features.append(1 if high_score > 0 else 0)  # High Score used
        features.append(1 if low_score > 0 and high_score > 0 else 0)  # Both used
        
        # Pad to exactly 50 features
        while len(features) < 50:
            features.append(0.0)
        
        return np.array(features, dtype=np.float32)
    
    def _predict_state_value(self, features: np.ndarray) -> float:
        """Predict the value of a state using the neural network"""
        # Hidden layer with ReLU activation
        hidden = np.maximum(0, np.dot(features, self.weights['hidden']) + self.biases['hidden'])
        
        # Output layer (linear activation for regression)
        output = np.dot(hidden, self.weights['output']) + self.biases['output']
        
        return output[0]
    
    def _evaluate_keep_decision(self, state: GameState, keep_dice: List[int]) -> float:
        """Evaluate the value of keeping certain dice"""
        if not keep_dice:
            # No dice kept - evaluate current state
            return self._predict_state_value(self._extract_features(state))
        
        # Create hypothetical state with kept dice
        hypothetical_state = GameState(keep_dice, state.rolls_left - 1, state.used_categories.copy())
        hypothetical_state.total_score = state.total_score
        
        # Extract features and predict value
        features = self._extract_features(hypothetical_state)
        value = self._predict_state_value(features)
        
        # Bonus for keeping high-value dice
        dice_bonus = sum(keep_dice) / 30.0
        value += dice_bonus * 0.1
        
        return value
    
    def _get_keep_combinations(self, dice: List[int]) -> List[List[int]]:
        """Get all possible keep combinations for given dice"""
        combinations = []
        
        # No keep
        combinations.append([])
        
        # Keep individual dice
        for i in range(len(dice)):
            combinations.append([dice[i]])
        
        # Keep pairs
        for i in range(len(dice)):
            for j in range(i + 1, len(dice)):
                combinations.append([dice[i], dice[j]])
        
        # Keep triples
        for i in range(len(dice)):
            for j in range(i + 1, len(dice)):
                for k in range(j + 1, len(dice)):
                    combinations.append([dice[i], dice[j], dice[k]])
        
        # Keep quads
        for i in range(len(dice)):
            for j in range(i + 1, len(dice)):
                for k in range(j + 1, len(dice)):
                    for l in range(k + 1, len(dice)):
                        combinations.append([dice[i], dice[j], dice[k], dice[l]])
        
        # Keep all
        combinations.append(dice.copy())
        
        return combinations
    
    def _find_best_keep_decision(self, state: GameState) -> List[int]:
        """Find the best dice to keep using the neural network"""
        if state.rolls_left == 0:
            return state.dice  # No rolls left, must keep all
        
        # Get all possible keep combinations
        keep_combinations = self._get_keep_combinations(state.dice)
        
        best_value = float('-inf')
        best_keep = []
        
        for keep_dice in keep_combinations:
            value = self._evaluate_keep_decision(state, keep_dice)
            if value > best_value:
                best_value = value
                best_keep = keep_dice
        
        return best_keep
    
    def _find_best_category(self, dice: List[int], used_categories: Dict[str, int]) -> Tuple[str, int]:
        """Find the best category to score using heuristics and neural network"""
        available_categories = [cat for cat in CATEGORIES if cat not in used_categories]
        
        if not available_categories:
            return None, 0
        
        # Calculate scores for all available categories
        category_scores = {}
        for category in available_categories:
            score = score_category(dice, category)
            category_scores[category] = score
        
        # Use heuristics to prioritize categories
        best_category = None
        best_score = -1
        
        # Priority 1: High-scoring categories
        for category, score in category_scores.items():
            if score > best_score:
                best_score = score
                best_category = category
        
        # Priority 2: Upper section bonus considerations
        upper_subtotal = sum(used_categories.get(f"{i}s", 0) for i in range(1, 7))
        if upper_subtotal < 63:
            # Try to get upper section bonus
            for i in range(1, 7):
                cat = f"{i}s"
                if cat in available_categories:
                    score = category_scores[cat]
                    if score > 0:
                        best_category = cat
                        best_score = score
                        break
        
        # Priority 3: Strategic category selection
        if best_score == 0:
            # If no good scoring options, pick the least bad
            for category in available_categories:
                if category not in ["Low Score", "High Score"]:  # Avoid these early
                    best_category = category
                    best_score = 0
                    break
        
        # Fallback
        if best_category is None:
            best_category = available_categories[0]
            best_score = category_scores[best_category]
        
        return best_category, best_score
    
    def best_move(self, state: GameState) -> Tuple[List[int], str, int]:
        """Determine the best move using the state value network and heuristics"""
        if state.rolls_left > 0:
            # Decide which dice to keep
            keep_dice = self._find_best_keep_decision(state)
            return (keep_dice, None, 0)
        else:
            # Decide which category to score
            best_category, best_score = self._find_best_category(state.dice, state.used_categories)
            return (state.dice, best_category, best_score)
    
    def _evaluate_game_state(self, state: GameState) -> float:
        """Evaluate the overall game state using the neural network"""
        features = self._extract_features(state)
        return self._predict_state_value(features)


class StateValueBotV2(StateValueBot):
    """
    Enhanced State Value Bot with more sophisticated decision making
    """
    
    def __init__(self, model_path: str = None, hidden_size: int = 128):
        super().__init__(model_path, hidden_size)
        self.name = "StateValueBotV2"
    
    def _evaluate_keep_decision(self, state: GameState, keep_dice: List[int]) -> float:
        """Enhanced evaluation of keep decisions"""
        if not keep_dice:
            return self._predict_state_value(self._extract_features(state))
        
        # Create hypothetical state
        hypothetical_state = GameState(keep_dice, state.rolls_left - 1, state.used_categories.copy())
        hypothetical_state.total_score = state.total_score
        
        # Base value from neural network
        features = self._extract_features(hypothetical_state)
        base_value = self._predict_state_value(features)
        
        # Enhanced bonuses
        dice_bonus = sum(keep_dice) / 30.0
        pattern_bonus = self._calculate_pattern_bonus(keep_dice)
        strategic_bonus = self._calculate_strategic_bonus(state, keep_dice)
        
        total_value = base_value + dice_bonus * 0.1 + pattern_bonus * 0.2 + strategic_bonus * 0.15
        
        return total_value
    
    def _calculate_pattern_bonus(self, dice: List[int]) -> float:
        """Calculate bonus for keeping dice that form patterns"""
        if not dice:
            return 0.0
        
        counts = {}
        for die in dice:
            counts[die] = counts.get(die, 0) + 1
        
        # Bonus for pairs, triples, etc.
        max_count = max(counts.values())
        if max_count >= 3:
            return 0.3  # Good pattern
        elif max_count >= 2:
            return 0.2  # Decent pattern
        else:
            return 0.0
    
    def _calculate_strategic_bonus(self, state: GameState, keep_dice: List[int]) -> float:
        """Calculate strategic bonus based on game state"""
        bonus = 0.0
        
        # Bonus for keeping dice that could help with upper section bonus
        upper_subtotal = sum(state.used_categories.get(f"{i}s", 0) for i in range(1, 7))
        if upper_subtotal < 63:
            for die in keep_dice:
                if f"{die}s" not in state.used_categories:
                    bonus += 0.1
        
        # Bonus for keeping dice that could form straights
        unique_dice = set(keep_dice)
        if {1, 2, 3, 4, 5}.issubset(unique_dice) or {2, 3, 4, 5, 6}.issubset(unique_dice):
            bonus += 0.2
        
        return bonus


def main():
    """Test the State Value Bot"""
    print("Testing State Value Bot")
    print("=" * 30)
    
    # Create bot (will use random weights if no model provided)
    bot = StateValueBot()
    
    # Test with a sample game state
    test_dice = [1, 1, 1, 2, 3]
    test_used_categories = {"1s": 3, "2s": 4}
    test_state = GameState(test_dice, 2, test_used_categories)
    
    print(f"Test state: dice={test_dice}, rolls_left=2, used={test_used_categories}")
    
    # Get best move
    keep_dice, category, score = bot.best_move(test_state)
    print(f"Best move: keep={keep_dice}, category={category}, score={score}")
    
    # Evaluate state
    state_value = bot._evaluate_game_state(test_state)
    print(f"State value: {state_value:.4f}")


if __name__ == "__main__":
    main()
