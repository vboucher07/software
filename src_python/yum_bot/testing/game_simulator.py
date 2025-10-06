"""
Game simulator for testing Yum bots
"""

import random
import time
from typing import List, Dict, Tuple
from game_logic.state import GameState
from game_logic.scoring import get_upper_section_bonus
from bots.base_bot import BaseBot

class GameSimulator:
    """Simulates a complete Yum game for testing bots"""
    
    def __init__(self, bot: BaseBot):
        self.bot = bot
    
    def play_game(self, verbose: bool = False) -> Tuple[int, Dict[str, int]]:
        """
        Play a complete game and return final score and category breakdown
        
        Args:
            verbose: Whether to print detailed game information
            
        Returns:
            Tuple of (final_score, category_breakdown)
        """
        used_categories = {}
        total_score = 0
        
        for turn in range(12):
            if verbose:
                print(f"\n--- Turn {turn + 1} ---")
            
            # Roll initial dice
            dice = [random.randint(1, 6) for _ in range(5)]
            dice.sort()
            rolls_left = 2
            
            if verbose:
                print(f"Initial dice: {dice}")
                print(f"Used categories: {used_categories}")
            
            # Make reroll decisions
            while rolls_left > 0:
                if verbose:
                    print(f"  Rolls left: {rolls_left}")
                
                state = GameState(dice, rolls_left, used_categories)
                keep_dice, category, expected_score = self.bot.best_move(state)
                
                if verbose:
                    print(f"  Bot keeps: {keep_dice}")
                    print(f"  Expected score: {expected_score}")
                
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
                
                if verbose:
                    print(f"  After reroll: {dice}")
            
            # Final move - assign to category
            if verbose:
                print("  Final assignment phase...")
            
            state = GameState(dice, 0, used_categories)
            final_dice, category, score = self.bot.best_move(state)
            
            if category:
                used_categories[category] = score
                total_score += score
                
                if verbose:
                    print(f"  Assigned to {category}: {score} points")
                    print(f"  Total score so far: {total_score}")
        
        # Add upper section bonus
        bonus = get_upper_section_bonus(used_categories)
        total_score += bonus
        
        if verbose:
            print(f"\n=== GAME COMPLETE ===")
            print(f"Final score: {total_score} (including {bonus} bonus)")
            print(f"Used categories: {used_categories}")
        
        return total_score, used_categories

class BotTester:
    """Framework for testing multiple bots"""
    
    def __init__(self):
        self.results = {}
    
    def test_bot(self, bot: BaseBot, num_games: int = 50, verbose: bool = False) -> Dict:
        """
        Test a bot over multiple games
        
        Args:
            bot: The bot to test
            num_games: Number of games to play
            verbose: Whether to print detailed information
            
        Returns:
            Dictionary with test results
        """
        print(f"Testing {bot.get_name()} over {num_games} games...")
        
        simulator = GameSimulator(bot)
        scores = []
        start_time = time.time()
        
        for game in range(num_games):
            if verbose and game % 10 == 0:
                print(f"Progress: {game}/{num_games} games...")
            
            score, categories = simulator.play_game(verbose=False)
            scores.append(score)
        
        end_time = time.time()
        
        # Calculate statistics
        avg_score = sum(scores) / len(scores)
        min_score = min(scores)
        max_score = max(scores)
        scores.sort()
        median_score = scores[len(scores) // 2]
        
        # Score distribution
        ranges = [(0, 100), (100, 120), (120, 140), (140, 160), (160, 180), (180, 200)]
        distribution = {}
        for low, high in ranges:
            count = sum(1 for score in scores if low <= score < high)
            percentage = (count / len(scores)) * 100
            distribution[f"{low}-{high}"] = {"count": count, "percentage": percentage}
        
        results = {
            "bot_name": bot.get_name(),
            "num_games": num_games,
            "average_score": avg_score,
            "median_score": median_score,
            "min_score": min_score,
            "max_score": max_score,
            "total_time": end_time - start_time,
            "avg_time_per_game": (end_time - start_time) / num_games,
            "score_distribution": distribution,
            "all_scores": scores
        }
        
        self.results[bot.get_name()] = results
        
        # Print results
        print(f"\nResults for {bot.get_name()}:")
        print(f"  Average score: {avg_score:.2f}")
        print(f"  Median score: {median_score}")
        print(f"  Minimum score: {min_score}")
        print(f"  Maximum score: {max_score}")
        print(f"  Total time: {end_time - start_time:.2f} seconds")
        print(f"  Average time per game: {(end_time - start_time) / num_games:.2f} seconds")
        
        print(f"\nScore distribution:")
        for range_name, data in distribution.items():
            print(f"  {range_name}: {data['count']} games ({data['percentage']:.1f}%)")
        
        return results
    
    def compare_bots(self, bots: List[BaseBot], num_games: int = 50) -> Dict:
        """
        Compare multiple bots
        
        Args:
            bots: List of bots to compare
            num_games: Number of games per bot
            
        Returns:
            Dictionary with comparison results
        """
        print(f"Comparing {len(bots)} bots over {num_games} games each...")
        print("=" * 60)
        
        for bot in bots:
            self.test_bot(bot, num_games)
            print()
        
        # Print comparison table
        print("COMPARISON TABLE:")
        print("=" * 60)
        print(f"{'Bot Name':<20} {'Avg Score':<10} {'Median':<10} {'Min':<8} {'Max':<8} {'Time/Game':<12}")
        print("-" * 60)
        
        for bot_name, results in self.results.items():
            print(f"{bot_name:<20} {results['average_score']:<10.2f} {results['median_score']:<10} {results['min_score']:<8} {results['max_score']:<8} {results['avg_time_per_game']:<12.3f}")
        
        return self.results 