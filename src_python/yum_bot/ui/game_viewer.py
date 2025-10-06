"""
Visual Game Viewer for watching bots play Yum
"""

import tkinter as tk
from tkinter import ttk, messagebox
import random
import time
import threading
from typing import Dict, List, Optional

from bots import get_registry
from testing.game_simulator import GameSimulator
from game_logic.state import GameState
from game_logic.scoring import score_category, CATEGORIES

class GameViewer:
    """Visual game viewer for watching bots play"""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Yum Bot Game Viewer")
        self.root.geometry("1000x700")
        
        self.registry = get_registry()
        self.current_bot = None
        self.simulator = None
        self.is_playing = False
        self.play_thread = None
        self.game_history = []
        
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
        main_frame.rowconfigure(1, weight=1)
        
        # Control panel
        self.setup_control_panel(main_frame)
        
        # Game display
        self.setup_game_display(main_frame)
        
    def setup_control_panel(self, parent):
        """Setup the control panel"""
        control_frame = ttk.LabelFrame(parent, text="Game Controls", padding="10")
        control_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Bot selection
        ttk.Label(control_frame, text="Bot Type:").grid(row=0, column=0, sticky=tk.W, padx=(0, 5))
        self.bot_type_var = tk.StringVar(value="Expectimax")
        bot_type_combo = ttk.Combobox(control_frame, textvariable=self.bot_type_var, 
                                      values=self.registry.list_bot_types(), 
                                      state="readonly", width=15)
        bot_type_combo.grid(row=0, column=1, sticky=tk.W, padx=(0, 20))
        bot_type_combo.bind('<<ComboboxSelected>>', self.on_bot_type_change)
        
        # Version selection
        ttk.Label(control_frame, text="Version:").grid(row=0, column=2, sticky=tk.W, padx=(20, 5))
        self.version_var = tk.StringVar(value="latest")
        self.version_combo = ttk.Combobox(control_frame, textvariable=self.version_var, 
                                         state="readonly", width=10)
        self.version_combo.grid(row=0, column=3, sticky=tk.W, padx=(0, 20))
        
        # Update version list
        self.on_bot_type_change()
        
        # Speed control
        ttk.Label(control_frame, text="Speed:").grid(row=0, column=4, sticky=tk.W, padx=(20, 5))
        self.speed_var = tk.DoubleVar(value=1.0)
        speed_scale = ttk.Scale(control_frame, from_=0.1, to=5.0, variable=self.speed_var, 
                               orient=tk.HORIZONTAL, length=100)
        speed_scale.grid(row=0, column=5, sticky=tk.W, padx=(0, 20))
        
        # Buttons
        self.play_button = ttk.Button(control_frame, text="Play Game", command=self.play_game)
        self.play_button.grid(row=0, column=6, padx=(20, 10))
        
        self.stop_button = ttk.Button(control_frame, text="Stop", command=self.stop_game, state="disabled")
        self.stop_button.grid(row=0, column=7, padx=(0, 10))
        
        self.step_button = ttk.Button(control_frame, text="Step", command=self.step_game)
        self.step_button.grid(row=0, column=8, padx=(0, 10))
        
        self.new_game_button = ttk.Button(control_frame, text="New Game", command=self.new_game)
        self.new_game_button.grid(row=0, column=9, padx=(0, 10))
        
    def setup_game_display(self, parent):
        """Setup the game display"""
        game_frame = ttk.LabelFrame(parent, text="Game Display", padding="10")
        game_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Game info
        info_frame = ttk.Frame(game_frame)
        info_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        self.turn_var = tk.StringVar(value="Turn: 1/12")
        ttk.Label(info_frame, textvariable=self.turn_var, font=("Arial", 12, "bold")).grid(row=0, column=0, sticky=tk.W, padx=(0, 20))
        
        self.score_var = tk.StringVar(value="Score: 0")
        ttk.Label(info_frame, textvariable=self.score_var, font=("Arial", 12, "bold")).grid(row=0, column=1, sticky=tk.W, padx=(0, 20))
        
        self.rolls_var = tk.StringVar(value="Rolls Left: 2")
        ttk.Label(info_frame, textvariable=self.rolls_var, font=("Arial", 12, "bold")).grid(row=0, column=2, sticky=tk.W)
        
        # Dice display
        dice_frame = ttk.LabelFrame(game_frame, text="Dice", padding="10")
        dice_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))
        
        self.dice_labels = []
        for i in range(5):
            label = ttk.Label(dice_frame, text="?", font=("Arial", 24, "bold"), 
                             width=3, relief="raised", borderwidth=2)
            label.grid(row=0, column=i, padx=5)
            self.dice_labels.append(label)
        
        # Game state
        state_frame = ttk.LabelFrame(game_frame, text="Game State", padding="10")
        state_frame.grid(row=1, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Used categories
        used_frame = ttk.Frame(state_frame)
        used_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        ttk.Label(used_frame, text="Used Categories:", font=("Arial", 10, "bold")).grid(row=0, column=0, sticky=tk.W, pady=(0, 5))
        
        self.used_categories_text = tk.Text(used_frame, height=8, width=40, font=("Courier", 9))
        self.used_categories_text.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Scrollbar for used categories
        used_scrollbar = ttk.Scrollbar(used_frame, orient=tk.VERTICAL, command=self.used_categories_text.yview)
        used_scrollbar.grid(row=1, column=1, sticky=(tk.N, tk.S))
        self.used_categories_text.configure(yscrollcommand=used_scrollbar.set)
        
        # Action log
        log_frame = ttk.LabelFrame(game_frame, text="Action Log", padding="10")
        log_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(10, 0))
        
        self.action_log_text = tk.Text(log_frame, height=6, width=80, font=("Courier", 9))
        self.action_log_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Scrollbar for action log
        log_scrollbar = ttk.Scrollbar(log_frame, orient=tk.VERTICAL, command=self.action_log_text.yview)
        log_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.action_log_text.configure(yscrollcommand=log_scrollbar.set)
        
    def on_bot_type_change(self, event=None):
        """Handle bot type change"""
        bot_type = self.bot_type_var.get()
        versions = self.registry.list_versions(bot_type)
        self.version_combo['values'] = versions
        if versions:
            self.version_combo.set(versions[-1])  # Set to latest
        
    def create_bot(self):
        """Create the selected bot"""
        try:
            bot_type = self.bot_type_var.get()
            version = self.version_var.get()
            
            # Handle neural bot parameters
            if bot_type == "Neural":
                if version == "1.0":
                    self.current_bot = self.registry.create_bot(bot_type, version, hidden_size=64)
                elif version == "2.0":
                    self.current_bot = self.registry.create_bot(bot_type, version, hidden_size=128)
                else:  # v3.0
                    self.current_bot = self.registry.create_bot(bot_type, version, hidden_size=256)
            else:
                self.current_bot = self.registry.create_bot(bot_type, version)
                
            self.simulator = GameSimulator(self.current_bot)
            self.log_action(f"Created {bot_type} v{version}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to create bot: {e}")
            
    def new_game(self):
        """Start a new game"""
        if not self.current_bot:
            self.create_bot()
            
        if not self.current_bot:
            return
            
        # Reset game state
        self.game_history = []
        self.current_turn = 0
        self.current_score = 0
        self.used_categories = {}
        
        # Clear display
        self.action_log_text.delete(1.0, tk.END)
        self.used_categories_text.delete(1.0, tk.END)
        
        # Update display
        self.update_display()
        self.log_action("New game started")
        
    def play_game(self):
        """Play a complete game"""
        if self.is_playing:
            return
            
        if not self.current_bot:
            self.create_bot()
            
        if not self.current_bot:
            return
            
        self.is_playing = True
        self.play_button.config(state="disabled")
        self.stop_button.config(state="normal")
        
        # Start game in separate thread
        self.play_thread = threading.Thread(target=self._play_game_thread)
        self.play_thread.daemon = True
        self.play_thread.start()
        
    def _play_game_thread(self):
        """Play game in separate thread"""
        try:
            # Simulate complete game
            for turn in range(12):
                if not self.is_playing:
                    break
                    
                # Roll initial dice
                dice = [random.randint(1, 6) for _ in range(5)]
                dice.sort()
                rolls_left = 2
                
                self.root.after(0, self._update_dice, dice)
                self.root.after(0, self.log_action, f"Turn {turn + 1}: Initial dice {dice}")
                
                # Make reroll decisions
                while rolls_left > 0 and self.is_playing:
                    self.root.after(0, self._update_rolls, rolls_left)
                    
                    # Simulate bot decision
                    state = GameState(dice, rolls_left, self.used_categories)
                    keep_dice, category, expected_score = self.current_bot.best_move(state)
                    
                    if keep_dice:
                        self.root.after(0, self.log_action, f"  Keeps: {keep_dice}")
                        
                        # Reroll the dice we're not keeping
                        kept_indices = []
                        for i, die in enumerate(dice):
                            if die in keep_dice:
                                kept_indices.append(i)
                        
                        # Reroll the rest
                        for i in range(5):
                            if i not in kept_indices:
                                dice[i] = random.randint(1, 6)
                        
                        dice.sort()
                        rolls_left -= 1
                        
                        self.root.after(0, self._update_dice, dice)
                        self.root.after(0, self.log_action, f"  After reroll: {dice}")
                    
                    # Delay based on speed
                    time.sleep(1.0 / self.speed_var.get())
                
                # Final move - assign to category
                if self.is_playing:
                    state = GameState(dice, 0, self.used_categories)
                    final_dice, category, score = self.current_bot.best_move(state)
                    
                    if category:
                        self.used_categories[category] = score
                        self.current_score += score
                        
                        self.root.after(0, self.log_action, f"  Assigned to {category}: {score} points")
                        self.root.after(0, self._update_score, self.current_score)
                        self.root.after(0, self._update_used_categories)
                    
                    # Delay between turns
                    time.sleep(0.5 / self.speed_var.get())
            
            # Add bonus
            from game_logic.scoring import get_upper_section_bonus
            bonus = get_upper_section_bonus(self.used_categories)
            self.current_score += bonus
            
            self.root.after(0, self.log_action, f"Final score: {self.current_score} (including {bonus} bonus)")
            self.root.after(0, self._game_finished)
            
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Game Error", str(e)))
        finally:
            self.root.after(0, self._game_finished)
    
    def _update_dice(self, dice):
        """Update dice display"""
        for i, die in enumerate(dice):
            self.dice_labels[i].config(text=str(die))
    
    def _update_rolls(self, rolls):
        """Update rolls left display"""
        self.rolls_var.set(f"Rolls Left: {rolls}")
    
    def _update_score(self, score):
        """Update score display"""
        self.score_var.set(f"Score: {score}")
    
    def _update_used_categories(self):
        """Update used categories display"""
        self.used_categories_text.delete(1.0, tk.END)
        for cat, score in self.used_categories.items():
            self.used_categories_text.insert(tk.END, f"{cat}: {score}\n")
    
    def _game_finished(self):
        """Called when game finishes"""
        self.is_playing = False
        self.play_button.config(state="normal")
        self.stop_button.config(state="disabled")
    
    def stop_game(self):
        """Stop the current game"""
        self.is_playing = False
        self.play_button.config(state="normal")
        self.stop_button.config(state="disabled")
        self.log_action("Game stopped")
    
    def step_game(self):
        """Step through one turn"""
        if not self.current_bot:
            self.create_bot()
            
        if not self.current_bot:
            return
            
        # Simulate one turn
        dice = [random.randint(1, 6) for _ in range(5)]
        dice.sort()
        
        self._update_dice(dice)
        self.log_action(f"Step: Dice {dice}")
        
        # Show bot's decision
        state = GameState(dice, 2, self.used_categories)
        keep_dice, category, expected_score = self.current_bot.best_move(state)
        
        if keep_dice:
            self.log_action(f"  Bot would keep: {keep_dice}")
        if category:
            self.log_action(f"  Bot would assign to: {category}")
    
    def log_action(self, message):
        """Add message to action log"""
        self.action_log_text.insert(tk.END, f"{message}\n")
        self.action_log_text.see(tk.END)
    
    def update_display(self):
        """Update the game display"""
        self.turn_var.set(f"Turn: {self.current_turn}/12")
        self.score_var.set(f"Score: {self.current_score}")
        self.rolls_var.set("Rolls Left: 2")
    
    def run(self):
        """Run the UI"""
        self.root.mainloop()

if __name__ == "__main__":
    viewer = GameViewer()
    viewer.run() 