"""
Visual Training UI for Neural Network bots
"""

import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import threading
import time
from typing import Dict, List, Optional

from training.neural_trainer import NeuralTrainer, TrainingManager
from bots.neural_bot import NeuralBotV1, NeuralBotV2, NeuralBotV3
from testing.game_simulator import GameSimulator

class TrainingUI:
    """Visual UI for training neural network bots"""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Yum Bot Neural Network Training")
        self.root.geometry("1200x800")
        
        self.training_manager = TrainingManager()
        self.current_trainer = None
        self.training_thread = None
        self.is_training = False
        
        self.setup_ui()
        
    def setup_ui(self):
        """Setup the UI components"""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(2, weight=1)
        
        # Control panel
        self.setup_control_panel(main_frame)
        
        # Progress panel
        self.setup_progress_panel(main_frame)
        
        # Training plots
        self.setup_training_plots(main_frame)
        
    def setup_control_panel(self, parent):
        """Setup the control panel"""
        control_frame = ttk.LabelFrame(parent, text="Training Controls", padding="10")
        control_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Bot selection
        ttk.Label(control_frame, text="Bot Type:").grid(row=0, column=0, sticky=tk.W, padx=(0, 5))
        self.bot_var = tk.StringVar(value="NeuralBotV1")
        bot_combo = ttk.Combobox(control_frame, textvariable=self.bot_var, 
                                 values=["NeuralBotV1", "NeuralBotV2", "NeuralBotV3"], 
                                 state="readonly", width=15)
        bot_combo.grid(row=0, column=1, sticky=tk.W, padx=(0, 20))
        
        # Training parameters
        ttk.Label(control_frame, text="Episodes:").grid(row=0, column=2, sticky=tk.W, padx=(20, 5))
        self.episodes_var = tk.StringVar(value="100")
        episodes_entry = ttk.Entry(control_frame, textvariable=self.episodes_var, width=10)
        episodes_entry.grid(row=0, column=3, sticky=tk.W, padx=(0, 20))
        
        ttk.Label(control_frame, text="Games/Episode:").grid(row=0, column=4, sticky=tk.W, padx=(20, 5))
        self.games_var = tk.StringVar(value="10")
        games_entry = ttk.Entry(control_frame, textvariable=self.games_var, width=10)
        games_entry.grid(row=0, column=5, sticky=tk.W, padx=(0, 20))
        
        # Buttons
        self.start_button = ttk.Button(control_frame, text="Start Training", command=self.start_training)
        self.start_button.grid(row=0, column=6, padx=(20, 10))
        
        self.stop_button = ttk.Button(control_frame, text="Stop Training", command=self.stop_training, state="disabled")
        self.stop_button.grid(row=0, column=7, padx=(0, 10))
        
        self.test_button = ttk.Button(control_frame, text="Test Bot", command=self.test_bot)
        self.test_button.grid(row=0, column=8, padx=(0, 10))
        
    def setup_progress_panel(self, parent):
        """Setup the progress panel"""
        progress_frame = ttk.LabelFrame(parent, text="Training Progress", padding="10")
        progress_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(progress_frame, variable=self.progress_var, 
                                           maximum=100, length=400)
        self.progress_bar.grid(row=0, column=0, columnspan=4, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Status labels
        self.status_var = tk.StringVar(value="Ready to train")
        status_label = ttk.Label(progress_frame, textvariable=self.status_var, font=("Arial", 10, "bold"))
        status_label.grid(row=1, column=0, columnspan=4, sticky=tk.W)
        
        # Stats frame
        stats_frame = ttk.Frame(progress_frame)
        stats_frame.grid(row=2, column=0, columnspan=4, sticky=(tk.W, tk.E), pady=(10, 0))
        
        # Episode info
        self.episode_var = tk.StringVar(value="Episode: 0")
        ttk.Label(stats_frame, textvariable=self.episode_var).grid(row=0, column=0, sticky=tk.W, padx=(0, 20))
        
        # Average reward
        self.avg_reward_var = tk.StringVar(value="Avg Reward: 0.00")
        ttk.Label(stats_frame, textvariable=self.avg_reward_var).grid(row=0, column=1, sticky=tk.W, padx=(0, 20))
        
        # Best reward
        self.best_reward_var = tk.StringVar(value="Best Reward: 0.00")
        ttk.Label(stats_frame, textvariable=self.best_reward_var).grid(row=0, column=2, sticky=tk.W, padx=(0, 20))
        
        # Loss
        self.loss_var = tk.StringVar(value="Loss: 0.0000")
        ttk.Label(stats_frame, textvariable=self.loss_var).grid(row=0, column=3, sticky=tk.W)
        
    def setup_training_plots(self, parent):
        """Setup the training plots"""
        plots_frame = ttk.LabelFrame(parent, text="Training Plots", padding="10")
        plots_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Create matplotlib figure
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(12, 4))
        self.canvas = FigureCanvasTkAgg(self.fig, plots_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Setup plots
        self.setup_reward_plot()
        self.setup_loss_plot()
        
    def setup_reward_plot(self):
        """Setup the reward plot"""
        self.ax1.set_title("Training Reward")
        self.ax1.set_xlabel("Episode")
        self.ax1.set_ylabel("Average Reward")
        self.ax1.grid(True, alpha=0.3)
        
    def setup_loss_plot(self):
        """Setup the loss plot"""
        self.ax2.set_title("Training Loss")
        self.ax2.set_xlabel("Episode")
        self.ax2.set_ylabel("Loss")
        self.ax2.grid(True, alpha=0.3)
        
    def start_training(self):
        """Start the training process"""
        if self.is_training:
            return
            
        try:
            episodes = int(self.episodes_var.get())
            games_per_episode = int(self.games_var.get())
            bot_type = self.bot_var.get()
        except ValueError:
            messagebox.showerror("Error", "Please enter valid numbers for episodes and games")
            return
        
        # Create trainer
        config = self.training_manager.training_configs.get(bot_type, {})
        self.current_trainer = self.training_manager.create_trainer(bot_type, config)
        
        # Start training in separate thread
        self.is_training = True
        self.start_button.config(state="disabled")
        self.stop_button.config(state="normal")
        
        self.training_thread = threading.Thread(
            target=self._training_loop,
            args=(episodes, games_per_episode)
        )
        self.training_thread.daemon = True
        self.training_thread.start()
        
    def _training_loop(self, episodes: int, games_per_episode: int):
        """Training loop in separate thread"""
        try:
            for episode in range(episodes):
                if not self.is_training:
                    break
                
                # Collect experience
                rewards = self.current_trainer.collect_experience(games_per_episode)
                avg_reward = np.mean(rewards)
                best_reward = np.max(rewards)
                
                # Update stats
                self.current_trainer.training_stats['episodes'] += 1
                self.current_trainer.training_stats['total_reward'] += avg_reward
                self.current_trainer.training_stats['avg_reward'] = (
                    self.current_trainer.training_stats['total_reward'] / 
                    self.current_trainer.training_stats['episodes']
                )
                self.current_trainer.training_stats['best_reward'] = max(
                    self.current_trainer.training_stats['best_reward'], best_reward
                )
                self.current_trainer.training_stats['reward_history'].append(avg_reward)
                
                # Train step
                loss = self.current_trainer.train_step()
                
                # Update UI (in main thread)
                self.root.after(0, self._update_ui, episode, avg_reward, best_reward, loss)
                
                # Small delay to prevent UI freezing
                time.sleep(0.01)
                
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Training Error", str(e)))
        finally:
            self.root.after(0, self._training_finished)
    
    def _update_ui(self, episode: int, avg_reward: float, best_reward: float, loss: float):
        """Update UI elements"""
        # Update progress
        progress = (episode + 1) / int(self.episodes_var.get()) * 100
        self.progress_var.set(progress)
        
        # Update status
        self.status_var.set(f"Training... Episode {episode + 1}")
        
        # Update stats
        self.episode_var.set(f"Episode: {episode + 1}")
        self.avg_reward_var.set(f"Avg Reward: {avg_reward:.2f}")
        self.best_reward_var.set(f"Best Reward: {best_reward:.2f}")
        self.loss_var.set(f"Loss: {loss:.4f}")
        
        # Update plots
        self._update_plots()
        
    def _update_plots(self):
        """Update the training plots"""
        if not self.current_trainer:
            return
            
        stats = self.current_trainer.training_stats
        
        # Clear plots
        self.ax1.clear()
        self.ax2.clear()
        
        # Plot reward history
        if stats['reward_history']:
            self.ax1.plot(stats['reward_history'], 'b-', alpha=0.7)
            self.ax1.set_title("Training Reward")
            self.ax1.set_xlabel("Episode")
            self.ax1.set_ylabel("Average Reward")
            self.ax1.grid(True, alpha=0.3)
        
        # Plot loss history
        if stats['loss_history']:
            self.ax2.plot(stats['loss_history'], 'r-', alpha=0.7)
            self.ax2.set_title("Training Loss")
            self.ax2.set_xlabel("Episode")
            self.ax2.set_ylabel("Loss")
            self.ax2.grid(True, alpha=0.3)
        
        # Redraw canvas
        self.canvas.draw()
        
    def stop_training(self):
        """Stop the training process"""
        self.is_training = False
        self.start_button.config(state="normal")
        self.stop_button.config(state="disabled")
        self.status_var.set("Training stopped")
        
    def _training_finished(self):
        """Called when training finishes"""
        self.is_training = False
        self.start_button.config(state="normal")
        self.stop_button.config(state="disabled")
        self.status_var.set("Training completed")
        
        # Save model
        if self.current_trainer:
            filename = f"trained_models/{self.current_trainer.bot.get_name().lower()}_trained.pkl"
            import os
            os.makedirs("trained_models", exist_ok=True)
            self.current_trainer.save_model(filename)
            
        messagebox.showinfo("Training Complete", "Training has finished!")
        
    def test_bot(self):
        """Test the current bot"""
        if not self.current_trainer:
            messagebox.showwarning("No Bot", "Please start training first")
            return
            
        # Run a quick test
        simulator = GameSimulator(self.current_trainer.bot)
        scores = []
        
        for i in range(10):
            score, categories = simulator.play_game(verbose=False)
            scores.append(score)
            
        avg_score = np.mean(scores)
        best_score = np.max(scores)
        
        messagebox.showinfo("Test Results", 
                          f"Test Results (10 games):\n"
                          f"Average Score: {avg_score:.2f}\n"
                          f"Best Score: {best_score:.2f}")
        
    def run(self):
        """Run the UI"""
        self.root.mainloop()

if __name__ == "__main__":
    ui = TrainingUI()
    ui.run() 