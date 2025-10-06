import random
from collections import Counter, defaultdict
import itertools

class BoggleDiceGenerator:
    def __init__(self):
        # English letter frequencies (approximate percentages)
        self.letter_frequencies = {
            'E': 12.7, 'T': 9.1, 'A': 8.2, 'O': 7.5, 'I': 7.0, 'N': 6.7, 'S': 6.3,
            'H': 6.1, 'R': 6.0, 'D': 4.3, 'L': 4.0, 'C': 2.8, 'U': 2.8, 'M': 2.4,
            'W': 2.4, 'F': 2.2, 'G': 2.0, 'Y': 2.0, 'P': 1.9, 'B': 1.3, 'V': 1.0,
            'K': 0.8, 'J': 0.15, 'X': 0.15, 'Q': 0.10, 'Z': 0.07
        }
        
        # Common letter combinations that work well together on dice faces
        self.good_combinations = [
            'AEIOU', 'AEILR', 'AENST', 'AORST', 'EILNR', 'EHNST',
            'BCDMN', 'CFGHL', 'DGLMN', 'FGHKL', 'JKQVX', 'BPWYZ'
        ]
        
        # Special combinations
        self.special_faces = ['QU', 'TH', 'IN', 'ER', 'AN', '*']  # * = blank
    
    def calculate_target_distribution(self, num_dice=64, faces_per_die=6):
        """Calculate target number of each letter based on frequency"""
        total_faces = num_dice * faces_per_die
        
        # Reserve some faces for special combinations
        special_faces_count = 8  # QU, TH, blanks, etc.
        available_faces = total_faces - special_faces_count
        
        distribution = {}
        total_freq = sum(self.letter_frequencies.values())
        
        for letter, freq in self.letter_frequencies.items():
            target_count = int((freq / total_freq) * available_faces)
            distribution[letter] = max(1, target_count)  # At least 1 of each
        
        # Add special combinations
        distribution['QU'] = 2
        distribution['TH'] = 2
        distribution['*'] = 4  # Blank faces
        
        return distribution
    
    def generate_unique_dice_designs(self, num_unique=16):
        """Generate a smaller set of unique dice designs that can be repeated"""
        # Predefined dice designs based on standard Boggle and optimal letter combinations
        base_dice = [
            ['A', 'A', 'E', 'E', 'G', 'N'],  # Vowel-heavy with common consonants
            ['E', 'L', 'R', 'T', 'T', 'Y'],  # Common letters
            ['A', 'O', 'O', 'T', 'T', 'W'],  # Vowel + common endings
            ['A', 'B', 'B', 'J', 'O', 'O'],  # Mixed frequency
            ['E', 'H', 'R', 'T', 'V', 'W'],  # Good consonant mix
            ['C', 'I', 'M', 'O', 'T', 'U'],  # Vowel + consonant balance
            ['D', 'I', 'S', 'T', 'T', 'Y'],  # Common endings
            ['E', 'I', 'O', 'S', 'S', 'T'],  # High frequency letters
            ['D', 'E', 'L', 'R', 'V', 'Y'],  # Mixed consonants
            ['A', 'C', 'H', 'O', 'P', 'S'],  # Balanced mix
            ['H', 'I', 'M', 'N', 'QU', '*'], # Special faces (QU, blank)
            ['B', 'I', 'F', 'O', 'R', 'X'],  # Less common letters
            ['D', 'E', 'N', 'O', 'W', 'S'],  # Common word parts
            ['E', 'E', 'F', 'H', 'I', 'Y'],  # Double E design
            ['A', 'D', 'E', 'N', 'N', 'S'],  # Common combinations
            ['K', 'L', 'P', 'U', 'Z', '*'],  # Rare letters + blank
        ]
        
        # If we need more unique designs, generate additional ones
        while len(base_dice) < num_unique:
            # Create balanced dice with mix of vowels and consonants
            vowels = ['A', 'E', 'I', 'O', 'U']
            consonants = ['B', 'C', 'D', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'R', 'S', 'T', 'V', 'W', 'Y', 'Z']
            
            die_faces = []
            # Add 1-3 vowels per die
            num_vowels = random.choice([1, 2, 2, 3])  # Weighted toward 2
            die_faces.extend(random.choices(vowels, k=num_vowels))
            
            # Fill rest with consonants
            remaining = 6 - num_vowels
            die_faces.extend(random.choices(consonants, k=remaining))
            
            base_dice.append(die_faces)
        
        return base_dice[:num_unique]
    
    def generate_dice_set(self, num_dice=64, num_unique_designs=16):
        """Generate a set of dice by repeating unique designs"""
        unique_designs = self.generate_unique_dice_designs(num_unique_designs)
        
        # Calculate how many of each design we need
        dice_per_design = num_dice // num_unique_designs
        remainder = num_dice % num_unique_designs
        
        dice_set = []
        for i, design in enumerate(unique_designs):
            # Add the base number for each design
            for _ in range(dice_per_design):
                dice_set.append(design.copy())
            
            # Add one extra for the remainder
            if i < remainder:
                dice_set.append(design.copy())
        
        random.shuffle(dice_set)  # Shuffle the final set
        return dice_set
    
    def optimize_dice_balance(self, dice, iterations=1000):
        """Optimize dice by swapping faces to improve balance"""
        best_dice = [die.copy() for die in dice]
        best_score = self.evaluate_dice_set(dice)
        
        for _ in range(iterations):
            # Make a copy to modify
            current_dice = [die.copy() for die in best_dice]
            
            # Random swap between two dice
            die1_idx = random.randint(0, len(current_dice) - 1)
            die2_idx = random.randint(0, len(current_dice) - 1)
            
            if die1_idx != die2_idx:
                face1_idx = random.randint(0, 5)
                face2_idx = random.randint(0, 5)
                
                # Swap faces
                current_dice[die1_idx][face1_idx], current_dice[die2_idx][face2_idx] = \
                    current_dice[die2_idx][face2_idx], current_dice[die1_idx][face1_idx]
                
                # Evaluate new configuration
                score = self.evaluate_dice_set(current_dice)
                if score > best_score:
                    best_dice = current_dice
                    best_score = score
        
        return best_dice
    
    def evaluate_dice_set(self, dice):
        """Evaluate the quality of a dice set"""
        score = 0
        
        # Count letter frequencies
        letter_counts = Counter()
        for die in dice:
            for face in die:
                letter_counts[face] += 1
        
        # Score based on vowel distribution (want good vowel spread)
        vowels = ['A', 'E', 'I', 'O', 'U']
        vowel_count = sum(letter_counts[v] for v in vowels)
        target_vowels = len(dice) * 6 * 0.25  # ~25% vowels
        score += 100 - abs(vowel_count - target_vowels)
        
        # Score based on variety (penalize too many of any letter)
        for count in letter_counts.values():
            if count > 15:  # Too many of one letter
                score -= (count - 15) * 5
        
        # Bonus for having special combinations
        if letter_counts['QU'] >= 2:
            score += 20
        if letter_counts['*'] >= 3:  # Blank faces
            score += 15
        
        return score
    
    def print_dice_analysis(self, dice):
        """Print analysis of the dice set"""
        letter_counts = Counter()
        for die in dice:
            for face in die:
                letter_counts[face] += 1
        
        # Count unique dice designs
        unique_designs = []
        design_counts = Counter()
        for die in dice:
            sorted_die = tuple(sorted(die))
            if sorted_die not in unique_designs:
                unique_designs.append(sorted_die)
            design_counts[sorted_die] += 1
        
        print(f"=== DICE SET ANALYSIS ===")
        print(f"Total dice: {len(dice)}")
        print(f"Unique dice designs: {len(unique_designs)}")
        print(f"Total faces: {len(dice) * 6}")
        print(f"Unique letters/combinations: {len(letter_counts)}")
        
        print(f"\n=== DICE DESIGN COUNTS ===")
        for i, (design, count) in enumerate(design_counts.most_common()):
            print(f"Design {i+1:>2}: {' '.join(f'{face}' for face in sorted(design))} x{count}")
        
        print(f"\n=== LETTER DISTRIBUTION ===")
        for letter, count in sorted(letter_counts.items()):
            percentage = (count / (len(dice) * 6)) * 100
            print(f"{letter:>2}: {count:>2} ({percentage:>4.1f}%)")
        
        vowels = ['A', 'E', 'I', 'O', 'U']
        vowel_count = sum(letter_counts[v] for v in vowels if v in letter_counts)
        print(f"\nVowels (A,E,I,O,U): {vowel_count} ({vowel_count/(len(dice)*6)*100:.1f}%)")
        
        print(f"\n=== SAMPLE DICE (showing unique designs) ===")
        shown_designs = set()
        design_num = 1
        for die in dice:
            sorted_die = tuple(sorted(die))
            if sorted_die not in shown_designs:
                count = design_counts[sorted_die]
                print(f"Design {design_num:>2}: {' '.join(f'{face:>2}' for face in die)} (used {count}x)")
                shown_designs.add(sorted_die)
                design_num += 1
    
    def export_dice_set(self, dice, filename="boggle_dice.txt"):
        """Export dice set to a file"""
        # Count unique designs
        design_counts = Counter()
        for die in dice:
            sorted_die = tuple(sorted(die))
            design_counts[sorted_die] += 1
        
        with open(filename, 'w') as f:
            f.write("8x8 Boggle Dice Set\n")
            f.write("===================\n\n")
            
            f.write("UNIQUE DICE DESIGNS (to manufacture):\n")
            f.write("-" * 40 + "\n")
            for i, (design, count) in enumerate(design_counts.most_common(), 1):
                f.write(f"Design {i:>2}: {' | '.join(design)} (make {count} dice)\n")
            
            f.write(f"\nTOTAL: {len(design_counts)} unique designs, {sum(design_counts.values())} total dice\n")
            
            f.write("\n" + "="*50 + "\n")
            f.write("COMPLETE DICE LIST:\n")
            f.write("-" * 20 + "\n")
            for i, die in enumerate(dice):
                f.write(f"Die {i+1:>2}: {' | '.join(die)}\n")
                
        print(f"Dice set exported to {filename}")
        print(f"You only need to manufacture {len(design_counts)} unique dice designs!")

def main():
    # Generate optimized dice set
    generator = BoggleDiceGenerator()
    print("Generating initial dice set...")
    
    dice = generator.generate_dice_set(64)
    print("Optimizing dice balance...")
    
    optimized_dice = generator.optimize_dice_balance(dice, iterations=2000)
    
    # Analyze and display results
    generator.print_dice_analysis(optimized_dice)
    
    # Export to file
    generator.export_dice_set(optimized_dice)
    
    print(f"\n=== RECOMMENDATIONS ===")
    print("• Print each die on a separate piece of paper")
    print("• Consider using different colors for dice with more vowels")
    print("• QU counts as one letter when forming words")
    print("• * (blank) can be any letter you choose")
    print("• TH, IN, ER, AN are single units if included")

if __name__ == "__main__":
    main()