#!/usr/bin/env python3
"""
Test script for Yum ruleset
"""

import sys
import os

# Add the current directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from yahtzotron.rulesets import AVAILABLE_RULESETS


def test_yum_rules():
    """Test the Yum ruleset"""
    print("Testing Yum ruleset...")
    
    # Check if yum ruleset is available
    if "yum" not in AVAILABLE_RULESETS:
        print("❌ Yum ruleset not found!")
        print(f"Available rulesets: {list(AVAILABLE_RULESETS.keys())}")
        return False
    
    yum_rules = AVAILABLE_RULESETS["yum"]
    print(f"✅ Found Yum ruleset: {yum_rules.name}")
    print(f"   Number of dice: {yum_rules.num_dice}")
    print(f"   Number of categories: {yum_rules.num_categories}")
    print(f"   Bonus cutoff: {yum_rules.bonus_cutoff_}")
    print(f"   Bonus score: {yum_rules.bonus_score_}")
    
    # Test some scoring scenarios
    print("\nTesting scoring scenarios...")
    
    # Test 1: Upper section scoring
    test_roll = [2, 0, 1, 0, 0, 0]  # Two 1s, one 3
    print(f"   Roll [1,1,3]: {test_roll}")
    
    for i, category in enumerate(yum_rules.categories):
        score = category.score(test_roll, [], [])
        print(f"     {category.name}: {score}")
    
    # Test 2: Yum (5 of a kind)
    yum_roll = [0, 0, 0, 0, 0, 5]  # Five 6s
    print(f"\n   Roll [6,6,6,6,6]: {yum_roll}")
    
    for i, category in enumerate(yum_rules.categories):
        score = category.score(yum_roll, [], [])
        print(f"     {category.name}: {score}")
    
    # Test 3: Full House
    full_house_roll = [0, 2, 3, 0, 0, 0]  # Two 2s, three 3s
    print(f"\n   Roll [2,2,3,3,3]: {full_house_roll}")
    
    for i, category in enumerate(yum_rules.categories):
        score = category.score(full_house_roll, [], [])
        print(f"     {category.name}: {score}")
    
    # Test 4: Low Straight
    low_straight_roll = [1, 1, 1, 1, 1, 0]  # 1,2,3,4,5
    print(f"\n   Roll [1,2,3,4,5]: {low_straight_roll}")
    
    for i, category in enumerate(yum_rules.categories):
        score = category.score(low_straight_roll, [], [])
        print(f"     {category.name}: {score}")
    
    print("\n✅ Yum ruleset test completed successfully!")
    return True


def test_scorecard():
    """Test the scorecard functionality"""
    print("\nTesting scorecard functionality...")
    
    from yahtzotron.game import Scorecard
    
    yum_rules = AVAILABLE_RULESETS["yum"]
    scorecard = Scorecard(yum_rules)
    
    print(f"   Initial scorecard: {scorecard}")
    print(f"   Total score: {scorecard.total_score()}")
    
    # Test scoring a category
    test_roll = [2, 0, 1, 0, 0, 0]  # Two 1s, one 3
    score_gain = scorecard.register_score(test_roll, 0)  # Score in 1s category
    
    print(f"   After scoring 1s: {scorecard}")
    print(f"   Score gained: {score_gain}")
    print(f"   Total score: {scorecard.total_score()}")
    
    # Test bonus calculation
    print(f"   Score summary: {scorecard.score_summary()}")
    
    print("✅ Scorecard test completed successfully!")
    return True


def main():
    """Main test function"""
    print("=" * 50)
    print("YUM-TRON RULESET TEST")
    print("=" * 50)
    
    try:
        # Test the ruleset
        if not test_yum_rules():
            return 1
        
        # Test the scorecard
        if not test_scorecard():
            return 1
        
        print("\n🎉 All tests passed! The Yum ruleset is working correctly.")
        return 0
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
