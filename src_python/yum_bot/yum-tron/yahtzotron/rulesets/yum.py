"""
Yum game rules adapted from Yahtzee

Yum is a variant of Yahtzee with different scoring rules:
- Upper section: 1s, 2s, 3s, 4s, 5s, 6s (sum of dice * number)
- Low Score: sum of all dice
- High Score: sum of all dice  
- Low Straight: 15 points for 1-2-3-4-5
- High Straight: 20 points for 2-3-4-5-6
- Full House: 25 points for 3 of one + 2 of another
- Yum: 30 points for 5 of a kind
- Upper section bonus: 25 points if upper section >= 63
"""

from .base import make_category, Ruleset


def _sum_of_all_dice(roll):
    return sum(die_value * num_dice for die_value, num_dice in enumerate(roll, 1))


def _yum_bonus(roll, filled_categories, scores):
    """Yum bonus: 30 points for 5 of a kind"""
    if 5 in roll:
        return 30
    return 0


@make_category(counts_towards_bonus=True)
def ones(roll, filled_categories, scores):
    return 1 * roll[0]


@make_category(counts_towards_bonus=True)
def twos(roll, filled_categories, scores):
    return 2 * roll[1]


@make_category(counts_towards_bonus=True)
def threes(roll, filled_categories, scores):
    return 3 * roll[2]


@make_category(counts_towards_bonus=True)
def fours(roll, filled_categories, scores):
    return 4 * roll[3]


@make_category(counts_towards_bonus=True)
def fives(roll, filled_categories, scores):
    return 5 * roll[4]


@make_category(counts_towards_bonus=True)
def sixes(roll, filled_categories, scores):
    return 6 * roll[5]


@make_category
def low_score(roll, filled_categories, scores):
    """Low Score: sum of all dice"""
    return _sum_of_all_dice(roll)


@make_category
def high_score(roll, filled_categories, scores):
    """High Score: sum of all dice"""
    return _sum_of_all_dice(roll)


@make_category
def low_straight(roll, filled_categories, scores):
    """Low Straight: 15 points for 1-2-3-4-5"""
    # Check if we have 1,2,3,4,5
    if roll[0] >= 1 and roll[1] >= 1 and roll[2] >= 1 and roll[3] >= 1 and roll[4] >= 1:
        return 15
    return 0


@make_category
def high_straight(roll, filled_categories, scores):
    """High Straight: 20 points for 2-3-4-5-6"""
    # Check if we have 2,3,4,5,6
    if roll[1] >= 1 and roll[2] >= 1 and roll[3] >= 1 and roll[4] >= 1 and roll[5] >= 1:
        return 20
    return 0


@make_category
def full_house(roll, filled_categories, scores):
    """Full House: 25 points for 3 of one + 2 of another"""
    # Check if we have exactly 3 of one value and 2 of another
    has_three = any(count >= 3 for count in roll)
    has_two = sum(1 for count in roll if count >= 2) >= 2
    
    if has_three and has_two:
        return 25
    return 0


@make_category
def yum(roll, filled_categories, scores):
    """Yum: 30 points for 5 of a kind"""
    if 5 in roll:
        return 30
    return 0


# Create the Yum ruleset
yum_rules = Ruleset(
    ruleset_name="yum",
    num_dice=5,
    categories=(
        ones,
        twos,
        threes,
        fours,
        fives,
        sixes,
        low_score,
        high_score,
        low_straight,
        high_straight,
        full_house,
        yum,
    ),
    bonus_cutoff=63,
    bonus_score=25,  # Yum uses 25 points instead of 35
)
