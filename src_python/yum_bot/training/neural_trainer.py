#!/usr/bin/env python3
"""
Neural Network Training for Yum Bot
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

from bots.neural_bot import NeuralBotV1, NeuralBotV2, NeuralBotV3
from game_logic.state import GameState
from game_logic.scoring import score_category


class ExperienceBuffer:
    """Simple experience replay buffer"""
    
    def __init__(self, max_size: int = 10000):
        self.buffer = deque(maxlen=max_size)
    
    def add_experience(self, state_features: np.ndarray, action: int, 
                      reward: float, next_state_features: np.ndarray, done: bool):
        self.buffer.append((state_features, action, reward, next_state_features, done))
    
    def sample(self, batch_size: int) -> List[Tuple]:
        if len(self.buffer) < batch_size:
            return list(self.buffer)
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)


class NeuralTrainer:
    """Trains a neural network bot"""
    
    def __init__(self, bot_class, hidden_size: int = 64):
        self.bot = bot_class(hidden_size=hidden_size)
        self.experience_buffer = ExperienceBuffer(max_size=5000)
        self.training_stats = {
            'episodes': [],
            'avg_rewards': [],
            'best_rewards': [],
            'losses': []
        }
    
    def collect_experience(self, num_games: int = 100) -> List[float]:
        """Collect experience by playing games"""
        rewards = []
        print(f"Collecting experience from {num_games} games...")
        
        for game in range(num_games):
            if game % 100 == 0 and game > 0:
                print(f"  Completed {game}/{num_games} games")
            
            game_experiences = self._play_game_with_tracking()
            rewards.append(game_experiences['final_score'])
            
            # Add experiences to buffer
            for exp in game_experiences['experiences']:
                self.experience_buffer.add_experience(*exp)
        
        return rewards
    
    def _play_game_with_tracking(self) -> Dict[str, Any]:
        """Play a single game and track all experiences"""
        # Initialize with starting dice and empty used categories
        initial_dice = [random.randint(1, 6) for _ in range(5)]
        state = GameState(initial_dice, 3, {})
        experiences = []
        total_score = 0
        
        # Play all 12 turns
        for turn in range(12):
            # Get current state features
            features = self.bot._extract_features(state)
            
            # Get bot's action
            keep_dice, category, expected_score = self.bot.best_move(state)
            action = self._action_to_index(keep_dice, category)
            
            # Simulate the move
            old_score = state.total_score
            state = self._simulate_move(state, keep_dice, category)
            new_score = state.total_score
            
            # Calculate reward (immediate score gain)
            reward = new_score - old_score
            
            # Get next state features
            next_features = self.bot._extract_features(state)
            
            # Store experience
            done = (turn == 11)  # Last turn
            experiences.append((features, action, reward, next_features, done))
            
            total_score = new_score
        
        return {
            'final_score': total_score,
            'experiences': experiences
        }
    
    def _simulate_move(self, state: GameState, keep_dice: List[int], category: str) -> GameState:
        """Simulate a move without full reroll logic"""
        # Handle None category (when bot wants to keep dice)
        if category is None:
            category = "1s"  # Default category
        
        # Create new state with proper initialization
        new_state = GameState(keep_dice.copy(), 0, state.used_categories.copy())
        new_state.total_score = state.total_score
        
        # Score the category
        score = score_category(new_state.dice, category)
        new_state.used_categories[category] = score
        new_state.total_score += score
        
        # Roll new dice for next turn
        new_state.dice = [random.randint(1, 6) for _ in range(5)]
        new_state.rolls_left = 3
        
        return new_state
    
    def _action_to_index(self, keep_dice: List[int], category: str) -> int:
        """Convert bot action to a single index"""
        # Handle None category (when bot wants to keep dice)
        if category is None:
            category = "1s"  # Default category
        
        # Map category to index (32-43, since 0-31 are for keep actions)
        category_index = {
            '1s': 32, '2s': 33, '3s': 34, '4s': 35, '5s': 36, '6s': 37,
            'Low Score': 38, 'High Score': 39, 'Low Straight': 40, 
            'High Straight': 41, 'Full House': 42, 'Yum': 43
        }[category]
        
        # If we have keep_dice, use keep action (0-31)
        if keep_dice and len(keep_dice) < 5:
            # Simple mapping: convert dice values to a number
            keep_index = 0
            for i, die in enumerate(keep_dice):
                keep_index += die * (6 ** i)
            keep_index = keep_index % 32  # Ensure it fits in 32
            return keep_index
        else:
            # Use category action
            return category_index
    
    def _backward_pass(self, features: np.ndarray, target: float, action: int, learning_rate: float = 0.0001):
        """Perform backpropagation to update weights (for single hidden layer)"""
        # Forward pass to get intermediate values
        hidden_input = np.dot(features, self.bot.weights['hidden']) + self.bot.biases['hidden']
        hidden_output = np.tanh(hidden_input)
        output_input = np.dot(hidden_output, self.bot.weights['output']) + self.bot.biases['output']
        
        # Calculate error
        error = target - output_input[action]
        
        # Clip error to prevent explosion
        error = np.clip(error, -5.0, 5.0)
        
        # Backpropagate through output layer
        output_grad = np.zeros_like(output_input)
        output_grad[action] = error
        
        # Update output weights and bias with gradient clipping
        weight_update = np.outer(hidden_output, output_grad)
        weight_update = np.clip(weight_update, -0.5, 0.5)
        self.bot.weights['output'] -= learning_rate * weight_update
        
        bias_update = output_grad
        bias_update = np.clip(bias_update, -0.5, 0.5)
        self.bot.biases['output'] -= learning_rate * bias_update
        
        # Backpropagate through hidden layer
        hidden_grad = np.dot(self.bot.weights['output'], output_grad) * (1 - np.tanh(hidden_input)**2)
        hidden_grad = np.clip(hidden_grad, -0.5, 0.5)
        
        # Update hidden weights and bias with gradient clipping
        weight_update = np.outer(features, hidden_grad)
        weight_update = np.clip(weight_update, -0.5, 0.5)
        self.bot.weights['hidden'] -= learning_rate * weight_update
        
        bias_update = hidden_grad
        bias_update = np.clip(bias_update, -0.5, 0.5)
        self.bot.biases['hidden'] -= learning_rate * bias_update
    
    def train_step(self, batch_size: int = 32, learning_rate: float = 0.0001):
        """Train on a batch of experiences"""
        if len(self.experience_buffer) < batch_size:
            return 0.0
        
        batch = self.experience_buffer.sample(batch_size)
        total_loss = 0.0
        
        for features, action, reward, next_features, done in batch:
            # Scale reward to be smaller
            reward = reward / 10.0  # Scale down rewards
            
            # Calculate target Q-value
            if done:
                target_q = reward
            else:
                # Get next state Q-values
                next_hidden = np.tanh(np.dot(next_features, self.bot.weights['hidden']) + self.bot.biases['hidden'])
                next_q_values = np.dot(next_hidden, self.bot.weights['output']) + self.bot.biases['output']
                max_next_q = np.max(next_q_values)
                target_q = reward + 0.9 * max_next_q  # gamma = 0.9
            
            # Clip target to prevent explosion
            target_q = np.clip(target_q, -10.0, 10.0)
            
            # Perform backpropagation
            self._backward_pass(features, target_q, action, learning_rate)
            
            # Calculate loss
            hidden = np.tanh(np.dot(features, self.bot.weights['hidden']) + self.bot.biases['hidden'])
            q_values = np.dot(hidden, self.bot.weights['output']) + self.bot.biases['output']
            current_q = q_values[action]
            loss = (target_q - current_q) ** 2
            total_loss += np.clip(loss, 0.0, 100.0)  # Clip loss to prevent explosion
        
        return total_loss / batch_size
    
    def _check_weights_stability(self):
        """Check if weights are stable and reinitialize if needed"""
        max_weight = max(
            np.max(np.abs(self.bot.weights['hidden'])),
            np.max(np.abs(self.bot.weights['output'])),
            np.max(np.abs(self.bot.biases['hidden'])),
            np.max(np.abs(self.bot.biases['output']))
        )
        
        if max_weight > 100.0:
            print(f"  Warning: Weights too large ({max_weight:.2f}), reinitializing...")
            self.bot._initialize_network()
            return True
        return False
    
    def train(self, num_episodes: int = 100, games_per_episode: int = 50) -> Dict[str, List]:
        """Train the neural network"""
        print(f"Starting training for {self.bot.__class__.__name__}")
        print(f"Episodes: {num_episodes}, Games per episode: {games_per_episode}")
        
        for episode in range(num_episodes):
            start_time = time.time()
            
            # Collect experience
            rewards = self.collect_experience(games_per_episode)
            
            # Train on collected experience
            total_loss = 0.0
            num_batches = max(1, len(self.experience_buffer) // 32)
            
            for _ in range(num_batches):
                loss = self.train_step(batch_size=32, learning_rate=0.001)
                total_loss += loss
            
            avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
            avg_reward = np.mean(rewards)
            best_reward = np.max(rewards)
            
            # Check for numerical instability
            if avg_loss > 1000.0 or np.isnan(avg_loss) or np.isinf(avg_loss):
                print(f"  Warning: Loss too high ({avg_loss:.2f}), reinitializing network...")
                self.bot._initialize_network()
                avg_loss = 0.0
            
            # Check weight stability
            self._check_weights_stability()
            
            # Store stats
            self.training_stats['episodes'].append(episode)
            self.training_stats['avg_rewards'].append(avg_reward)
            self.training_stats['best_rewards'].append(best_reward)
            self.training_stats['losses'].append(avg_loss)
            
            episode_time = time.time() - start_time
            
            # Print progress every 5 episodes or at the end
            if episode % 5 == 0 or episode == num_episodes - 1:
                print(f"Episode {episode}: Avg Reward = {avg_reward:.2f}, "
                      f"Best = {best_reward:.2f}, Loss = {avg_loss:.4f}, "
                      f"Time = {episode_time:.2f}s")
            
            # Early stopping if performance is getting worse or loss is exploding
            if episode > 10 and (avg_reward < 40 or avg_loss > 10000.0):
                print(f"Stopping early - performance too low: {avg_reward:.2f} or loss too high: {avg_loss:.2f}")
                break
        
        print(f"Training complete! Final avg reward: {self.training_stats['avg_rewards'][-1]:.2f}")
        return self.training_stats
    
    def save_model(self, filename: str):
        """Save the trained model"""
        model_data = {
            'weights': self.bot.weights,
            'biases': self.bot.biases,
            'hidden_size': self.bot.hidden_size
        }
        np.save(filename, model_data)
        print(f"Model saved to {filename}")
    
    def load_model(self, filename: str):
        """Load a trained model"""
        model_data = np.load(filename, allow_pickle=True).item()
        self.bot.weights = model_data['weights']
        self.bot.biases = model_data['biases']
        self.bot.hidden_size = model_data['hidden_size']
        print(f"Model loaded from {filename}")


class TrainingManager:
    """Manages training for multiple neural bot types"""
    
    def __init__(self):
        self.trainers = {}
    
    def create_trainer(self, bot_type: str, hidden_size: int = 64):
        """Create a trainer for a specific bot type"""
        bot_classes = {
            'NeuralBotV1': NeuralBotV1,
            'NeuralBotV2': NeuralBotV2,
            'NeuralBotV3': NeuralBotV3
        }
        
        if bot_type not in bot_classes:
            raise ValueError(f"Unknown bot type: {bot_type}")
        
        trainer = NeuralTrainer(bot_classes[bot_type], hidden_size)
        self.trainers[bot_type] = trainer
        return trainer
    
    def train_bot(self, bot_type: str, num_episodes: int = 100, 
                  games_per_episode: int = 50, hidden_size: int = 64):
        """Train a specific bot type"""
        trainer = self.create_trainer(bot_type, hidden_size)
        return trainer.train(num_episodes, games_per_episode) 