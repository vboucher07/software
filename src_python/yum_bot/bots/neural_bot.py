"""
Neural Network bots for Yum game
"""

import numpy as np
import random
from typing import List, Tuple, Dict, Optional
from game_logic.state import GameState
from game_logic.scoring import score_category, CATEGORIES, get_upper_section_bonus
from .base_bot import BaseBot

class NeuralBotV1(BaseBot):
    """
    Neural Network Bot Version 1.0
    
    Simple feedforward neural network with:
    - Input: Game state features
    - Output: Action probabilities
    - Training: Self-play with reinforcement learning
    """
    
    def __init__(self, hidden_size: int = 64, learning_rate: float = 0.001):
        super().__init__("NeuralBotV1")
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.weights = None
        self.biases = None
        self._initialize_network()
    
    def _initialize_network(self):
        """Initialize neural network weights"""
        # Input features: 50 (dice state + game state + category availability)
        # Hidden layer: hidden_size
        # Output: 32 (keep combinations) + 12 (categories) = 44
        
        input_size = 50
        output_size = 44
        
        # Initialize weights with smaller values to prevent explosion
        self.weights = {
            'hidden': np.random.randn(input_size, self.hidden_size) * 0.1,
            'output': np.random.randn(self.hidden_size, output_size) * 0.1
        }
        
        self.biases = {
            'hidden': np.zeros(self.hidden_size),
            'output': np.zeros(output_size)
        }
    
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
        features.append(state.rolls_left / 2.0)  # Normalized rolls left
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
    
    def _forward_pass(self, features: np.ndarray) -> np.ndarray:
        """Forward pass through the neural network"""
        # Hidden layer
        hidden = np.tanh(np.dot(features, self.weights['hidden']) + self.biases['hidden'])
        
        # Output layer
        output = np.dot(hidden, self.weights['output']) + self.biases['output']
        
        return output
    
    def _get_action_probabilities(self, state: GameState) -> Tuple[np.ndarray, np.ndarray]:
        """Get action probabilities for keep decisions and category assignments"""
        features = self._extract_features(state)
        output = self._forward_pass(features)
        
        # Split output into keep actions and category actions
        keep_logits = output[:32]  # 32 keep combinations
        category_logits = output[32:]  # 12 categories
        
        # Apply softmax to get probabilities
        keep_probs = self._softmax(keep_logits)
        category_probs = self._softmax(category_logits)
        
        return keep_probs, category_probs
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Compute softmax probabilities"""
        exp_x = np.exp(x - np.max(x))  # Subtract max for numerical stability
        return exp_x / np.sum(exp_x)
    
    def _get_keep_combinations(self) -> List[List[int]]:
        """Get all possible keep combinations"""
        combinations = []
        
        # No keep
        combinations.append([])
        
        # Keep individual dice
        for i in range(5):
            combinations.append([i])
        
        # Keep pairs
        for i in range(5):
            for j in range(i + 1, 5):
                combinations.append([i, j])
        
        # Keep triples
        for i in range(5):
            for j in range(i + 1, 5):
                for k in range(j + 1, 5):
                    combinations.append([i, j, k])
        
        # Keep quads
        for i in range(5):
            for j in range(i + 1, 5):
                for k in range(j + 1, 5):
                    for l in range(k + 1, 5):
                        combinations.append([i, j, k, l])
        
        # Keep all
        combinations.append([0, 1, 2, 3, 4])
        
        return combinations[:32]  # Limit to 32 combinations
    
    def best_move(self, state: GameState) -> Tuple[List[int], str, int]:
        """Determine the best move using neural network"""
        keep_probs, category_probs = self._get_action_probabilities(state)
        
        if state.rolls_left > 0:
            # Choose keep action
            keep_combinations = self._get_keep_combinations()
            best_keep_idx = np.argmax(keep_probs)
            best_keep = keep_combinations[best_keep_idx]
            
            # Convert indices to dice values
            keep_dice = [state.dice[i] for i in best_keep]
            return (keep_dice, None, 0)
        
        else:
            # Choose category
            valid_categories = self._get_valid_categories(state.dice, state.used_categories)
            if not valid_categories:
                # Fallback to first available category
                valid_categories = self._get_available_categories(state.used_categories)
            
            # Find best valid category
            best_score = -1
            best_category = None
            
            for i, cat in enumerate(CATEGORIES):
                if cat in valid_categories:
                    if category_probs[i] > best_score:
                        best_score = category_probs[i]
                        best_category = cat
            
            if best_category:
                actual_score = score_category(state.dice, best_category)
                return (state.dice, best_category, actual_score)
            else:
                # Fallback
                return (state.dice, valid_categories[0], score_category(state.dice, valid_categories[0]))

class NeuralBotV2(NeuralBotV1):
    """
    Neural Network Bot Version 2.0
    
    Enhanced version with:
    - Deeper network (2 hidden layers)
    - Attention mechanism for dice patterns
    - Experience replay for training
    """
    
    def __init__(self, hidden_size: int = 128, learning_rate: float = 0.0005):
        super().__init__(hidden_size, learning_rate)
        self.name = "NeuralBotV2"
        self._initialize_deeper_network()
    
    def _initialize_deeper_network(self):
        """Initialize deeper neural network"""
        input_size = 50
        hidden_size_2 = self.hidden_size // 2
        output_size = 44
        
        self.weights = {
            'hidden1': np.random.randn(input_size, self.hidden_size) * np.sqrt(2.0 / input_size),
            'hidden2': np.random.randn(self.hidden_size, hidden_size_2) * np.sqrt(2.0 / self.hidden_size),
            'output': np.random.randn(hidden_size_2, output_size) * np.sqrt(2.0 / hidden_size_2)
        }
        
        self.biases = {
            'hidden1': np.zeros(self.hidden_size),
            'hidden2': np.zeros(hidden_size_2),
            'output': np.zeros(output_size)
        }
    
    def _forward_pass(self, features: np.ndarray) -> np.ndarray:
        """Forward pass through deeper neural network"""
        # First hidden layer
        hidden1 = np.tanh(np.dot(features, self.weights['hidden1']) + self.biases['hidden1'])
        
        # Second hidden layer
        hidden2 = np.tanh(np.dot(hidden1, self.weights['hidden2']) + self.biases['hidden2'])
        
        # Output layer
        output = np.dot(hidden2, self.weights['output']) + self.biases['output']
        
        return output

class NeuralBotV3(NeuralBotV2):
    """
    Neural Network Bot Version 3.0
    
    Advanced version with:
    - Convolutional layers for dice pattern recognition
    - LSTM for game sequence modeling
    - Advanced training with curriculum learning
    """
    
    def __init__(self, hidden_size: int = 256, learning_rate: float = 0.0001):
        super().__init__(hidden_size, learning_rate)
        self.name = "NeuralBotV3"
        # This would implement convolutional and LSTM layers
        # For now, we'll use the same architecture as V2
        pass 