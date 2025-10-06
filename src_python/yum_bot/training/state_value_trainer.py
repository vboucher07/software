#!/usr/bin/env python3
"""
State Value Neural Network Trainer for Yum Bot

This trainer creates a neural network that estimates the value of game states,
then uses deterministic decision-making for categories and dice kept.
"""

import numpy as np
import random
from collections import deque
from typing import List, Tuple, Dict, Any
import time
import sys
import os

# Add the parent directory to the path so we can import our modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from game_logic.state import GameState
from game_logic.scoring import score_category, CATEGORIES, get_upper_section_bonus


class StateValueDataset:
    """Dataset for training state value estimation"""
    
    def __init__(self, max_size: int = 50000):
        self.states = deque(maxlen=max_size)
        self.values = deque(maxlen=max_size)
        self.metadata = deque(maxlen=max_size)
    
    def add_state(self, state_features: np.ndarray, value: float, metadata: Dict = None):
        """Add a state-value pair to the dataset"""
        self.states.append(state_features)
        self.values.append(value)
        if metadata:
            self.metadata.append(metadata)
        else:
            self.metadata.append({})
    
    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray]:
        """Sample a batch of state-value pairs"""
        if len(self.states) < batch_size:
            return np.array(self.states), np.array(self.values)
        
        indices = random.sample(range(len(self.states)), batch_size)
        batch_states = np.array([self.states[i] for i in indices])
        batch_values = np.array([self.values[i] for i in indices])
        return batch_states, batch_values
    
    def __len__(self):
        return len(self.states)


class StateValueNeuralNetwork:
    """Neural network for estimating game state values"""
    
    def __init__(self, hidden_size: int = 128):
        self.hidden_size = hidden_size
        self.weights = None
        self.biases = None
        self._initialize_network()
    
    def _initialize_network(self):
        """Initialize neural network weights"""
        # Input features: 50 (dice state + game state + category availability)
        # Hidden layer: hidden_size
        # Output: 1 (state value)
        
        input_size = 50
        output_size = 1
        
        # Initialize weights with Xavier/Glorot initialization
        self.weights = {
            'hidden': np.random.randn(input_size, self.hidden_size) * np.sqrt(2.0 / input_size),
            'output': np.random.randn(self.hidden_size, output_size) * np.sqrt(2.0 / self.hidden_size)
        }
        
        self.biases = {
            'hidden': np.zeros(self.hidden_size),
            'output': np.zeros(output_size)
        }
    
    def forward(self, features: np.ndarray) -> np.ndarray:
        """Forward pass through the neural network"""
        # Handle both single features and batches
        if features.ndim == 1:
            features = features.reshape(1, -1)
        
        # Hidden layer with ReLU activation
        hidden = np.maximum(0, np.dot(features, self.weights['hidden']) + self.biases['hidden'])
        
        # Output layer (linear activation for regression)
        output = np.dot(hidden, self.weights['output']) + self.biases['output']
        
        return output
    
    def predict(self, features: np.ndarray) -> float:
        """Predict the value of a state"""
        output = self.forward(features)
        return output.flatten()[0]


class StateValueTrainer:
    """Trains a neural network to estimate state values"""
    
    def __init__(self, hidden_size: int = 128):
        self.network = StateValueNeuralNetwork(hidden_size)
        self.dataset = StateValueDataset()
        self.training_stats = {
            'episodes': [],
            'losses': [],
            'dataset_size': []
        }
    
    def extract_features(self, state: GameState) -> np.ndarray:
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
    
    def generate_dataset(self, num_games: int = 1000):
        """Generate training dataset by playing games and recording state values"""
        print(f"Generating dataset from {num_games} games...")
        
        for game in range(num_games):
            if game % 100 == 0 and game > 0:
                print(f"  Completed {game}/{num_games} games")
            
            # Play a complete game and record all states
            game_states = self._play_complete_game()
            
            # Add all states to dataset with their final game values
            final_score = game_states[-1]['score']
            for state_info in game_states:
                features = state_info['features']
                # Value is the final score minus what we've already scored
                current_score = state_info['score']
                remaining_potential = final_score - current_score
                # Normalize to reasonable range
                value = (final_score + remaining_potential) / 100.0
                
                metadata = {
                    'game_id': game,
                    'turn': state_info['turn'],
                    'final_score': final_score,
                    'current_score': current_score
                }
                
                self.dataset.add_state(features, value, metadata)
        
        print(f"Dataset generation complete! Total states: {len(self.dataset)}")
    
    def _play_complete_game(self) -> List[Dict]:
        """Play a complete game and return all states encountered"""
        states = []
        
        # Start with random dice
        dice = [random.randint(1, 6) for _ in range(5)]
        used_categories = {}
        total_score = 0
        
        # Play all 12 turns
        for turn in range(12):
            # Create game state
            state = GameState(dice, 3, used_categories.copy())
            state.total_score = total_score
            
            # Extract features
            features = self.extract_features(state)
            
            # Record state
            states.append({
                'features': features,
                'score': total_score,
                'turn': turn,
                'dice': dice.copy(),
                'used_categories': used_categories.copy()
            })
            
            # Simulate optimal play for this turn
            best_category, best_score = self._find_best_category(dice, used_categories)
            
            # Apply the move
            used_categories[best_category] = best_score
            total_score += best_score
            
            # Roll new dice for next turn
            dice = [random.randint(1, 6) for _ in range(5)]
        
        return states
    
    def _find_best_category(self, dice: List[int], used_categories: Dict[str, int]) -> Tuple[str, int]:
        """Find the best category to score with given dice"""
        best_score = -1
        best_category = None
        
        for category in CATEGORIES:
            if category not in used_categories:
                score = score_category(dice, category)
                if score > best_score:
                    best_score = score
                    best_category = category
        
        return best_category, best_score
    
    def train_step(self, batch_size: int = 32, learning_rate: float = 0.001):
        """Train on a batch of state-value pairs"""
        if len(self.dataset) < batch_size:
            return 0.0
        
        # Sample batch
        states, targets = self.dataset.sample(batch_size)
        
        # Forward pass
        predictions = self.network.forward(states)
        
        # Calculate loss (MSE)
        loss = np.mean((predictions - targets) ** 2)
        
        # Backpropagation (simplified)
        self._backward_pass(states, predictions, targets, learning_rate)
        
        return loss
    
    def _backward_pass(self, states: np.ndarray, predictions: np.ndarray, 
                      targets: np.ndarray, learning_rate: float):
        """Simplified backpropagation for the neural network"""
        batch_size = states.shape[0]
        
        # Ensure targets has the right shape
        if targets.ndim == 1:
            targets = targets.reshape(-1, 1)
        if predictions.ndim == 1:
            predictions = predictions.reshape(-1, 1)
        
        # Gradient of loss with respect to predictions
        grad_predictions = 2 * (predictions - targets) / batch_size
        
        # Gradient clipping
        grad_predictions = np.clip(grad_predictions, -1.0, 1.0)
        
        # Forward pass to get intermediate values
        hidden_input = np.dot(states, self.network.weights['hidden']) + self.network.biases['hidden']
        hidden_output = np.maximum(0, hidden_input)  # ReLU
        
        # Gradients for output layer
        grad_output_weights = np.dot(hidden_output.T, grad_predictions)
        grad_output_bias = np.sum(grad_predictions, axis=0)
        
        # Gradients for hidden layer
        grad_hidden = np.dot(grad_predictions, self.network.weights['output'].T)
        grad_hidden = grad_hidden * (hidden_input > 0)  # ReLU gradient
        
        grad_hidden_weights = np.dot(states.T, grad_hidden)
        grad_hidden_bias = np.sum(grad_hidden, axis=0)
        
        # Update weights with gradient clipping
        self.network.weights['output'] -= learning_rate * np.clip(grad_output_weights, -0.5, 0.5)
        self.network.biases['output'] -= learning_rate * np.clip(grad_output_bias, -0.5, 0.5)
        
        self.network.weights['hidden'] -= learning_rate * np.clip(grad_hidden_weights, -0.5, 0.5)
        self.network.biases['hidden'] -= learning_rate * np.clip(grad_hidden_bias, -0.5, 0.5)
    
    def train(self, num_episodes: int = 100, batch_size: int = 32, learning_rate: float = 0.001):
        """Train the neural network"""
        print(f"Starting training for State Value Neural Network")
        print(f"Episodes: {num_episodes}, Batch size: {batch_size}")
        
        for episode in range(num_episodes):
            start_time = time.time()
            
            # Train on multiple batches
            total_loss = 0.0
            num_batches = max(1, len(self.dataset) // batch_size)
            
            for _ in range(num_batches):
                loss = self.train_step(batch_size, learning_rate)
                total_loss += loss
            
            avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
            
            # Store stats
            self.training_stats['episodes'].append(episode)
            self.training_stats['losses'].append(avg_loss)
            self.training_stats['dataset_size'].append(len(self.dataset))
            
            episode_time = time.time() - start_time
            
            # Print progress every 10 episodes or at the end
            if episode % 10 == 0 or episode == num_episodes - 1:
                print(f"Episode {episode}: Loss = {avg_loss:.6f}, "
                      f"Dataset size = {len(self.dataset)}, "
                      f"Time = {episode_time:.2f}s")
            
            # Early stopping if loss is very low
            if episode > 20 and avg_loss < 0.001:
                print(f"Stopping early - loss very low: {avg_loss:.6f}")
                break
        
        print(f"Training complete! Final loss: {self.training_stats['losses'][-1]:.6f}")
        return self.training_stats
    
    def save_model(self, filename: str):
        """Save the trained model"""
        model_data = {
            'weights': self.network.weights,
            'biases': self.network.biases,
            'hidden_size': self.network.hidden_size
        }
        np.save(filename, model_data)
        print(f"Model saved to {filename}")
    
    def load_model(self, filename: str):
        """Load a trained model"""
        model_data = np.load(filename, allow_pickle=True).item()
        self.network.weights = model_data['weights']
        self.network.biases = model_data['biases']
        self.network.hidden_size = model_data['hidden_size']
        print(f"Model loaded from {filename}")


def main():
    """Main training function"""
    print("State Value Neural Network Trainer for Yum Bot")
    print("=" * 60)
    
    # Create trainer
    trainer = StateValueTrainer(hidden_size=128)
    
    # Generate dataset
    print("\n1. Generating training dataset...")
    trainer.generate_dataset(num_games=2000)
    
    # Train the network
    print("\n2. Training neural network...")
    stats = trainer.train(num_episodes=200, batch_size=64, learning_rate=0.001)
    
    # Save the model
    print("\n3. Saving trained model...")
    trainer.save_model("trained_models/state_value_network.npy")
    
    print("\nTraining complete!")
    print(f"Final loss: {stats['losses'][-1]:.6f}")
    print(f"Dataset size: {stats['dataset_size'][-1]}")


if __name__ == "__main__":
    main()
