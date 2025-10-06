from typing import List
from collections import Counter

CATEGORIES = ["1s", "2s", "3s", "4s", "5s", "6s",
              "Low Score", "High Score", "Low Straight", "High Straight",
              "Full House", "Yum"]

def score_category(dice: List[int], category: str) -> int:
    counts = Counter(dice)
    total = sum(dice)

    if category in ["1s", "2s", "3s", "4s", "5s", "6s"]:
        num = int(category[0])
        return counts[num] * num
    elif category == "Low Score":
        return total
    elif category == "High Score":
        return total
    elif category == "Low Straight":
        return 15 if set([1,2,3,4,5]).issubset(dice) else 0
    elif category == "High Straight":
        return 20 if set([2,3,4,5,6]).issubset(dice) else 0
    elif category == "Full House":
        return 25 if sorted(counts.values()) == [2, 3] else 0
    elif category == "Yum":
        return 30 if any(v == 5 for v in counts.values()) else 0
    else:
        raise ValueError(f"Unknown category: {category}")

def get_upper_section_bonus(used_categories: dict) -> int:
    """Calculate the 25-point bonus for upper section subtotal >= 63"""
    upper_subtotal = 0
    for i in range(1, 7):
        cat = f"{i}s"
        if cat in used_categories:
            upper_subtotal += used_categories[cat]
    
    return 25 if upper_subtotal >= 63 else 0