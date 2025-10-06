package com.yumbot.app

import kotlin.math.max

/**
 * Yum game rules adapted from the Python implementation
 * 
 * Yum is a variant of Yahtzee with different scoring rules:
 * - Upper section: 1s, 2s, 3s, 4s, 5s, 6s (sum of dice * number)
 * - Low Score: sum of all dice
 * - High Score: sum of all dice  
 * - Low Straight: 15 points for 1-2-3-4-5
 * - High Straight: 20 points for 2-3-4-5-6
 * - Full House: 25 points for 3 of one + 2 of another
 * - Yum: 30 points for 5 of a kind
 * - Upper section bonus: 25 points if upper section >= 63
 */

enum class YumCategory(val displayName: String, val countsTowardsBonus: Boolean = false) {
    ONES("Ones", true),
    TWOS("Twos", true),
    THREES("Threes", true),
    FOURS("Fours", true),
    FIVES("Fives", true),
    SIXES("Sixes", true),
    LOW_SCORE("Low Score"),
    HIGH_SCORE("High Score"),
    LOW_STRAIGHT("Low Straight"),
    HIGH_STRAIGHT("High Straight"),
    FULL_HOUSE("Full House"),
    YUM("Yum");
    
    companion object {
        fun fromIndex(index: Int): YumCategory = values()[index]
        fun getNumCategories(): Int = values().size
    }
}

class YumRuleset {
    companion object {
        const val NUM_DICE = 5
        const val NUM_ROUNDS = 12 // One round per category
        const val BONUS_CUTOFF = 63
        const val BONUS_SCORE = 25
        
        /**
         * Calculate the score for a given roll in a specific category
         */
        fun calculateScore(
            roll: IntArray, 
            category: YumCategory, 
            filledCategories: BooleanArray, 
            scores: IntArray
        ): Int {
            // Convert roll (dice values) to counts array [count of 1s, count of 2s, ..., count of 6s]
            val counts = IntArray(6) { 0 }
            for (die in roll) {
                if (die in 1..6) {
                    counts[die - 1]++
                }
            }
            
            return when (category) {
                YumCategory.ONES -> 1 * counts[0]
                YumCategory.TWOS -> 2 * counts[1]
                YumCategory.THREES -> 3 * counts[2]
                YumCategory.FOURS -> 4 * counts[3]
                YumCategory.FIVES -> 5 * counts[4]
                YumCategory.SIXES -> 6 * counts[5]
                YumCategory.LOW_SCORE -> roll.sum()
                YumCategory.HIGH_SCORE -> roll.sum()
                YumCategory.LOW_STRAIGHT -> {
                    // Check if we have 1,2,3,4,5
                    if (counts[0] >= 1 && counts[1] >= 1 && counts[2] >= 1 && 
                        counts[3] >= 1 && counts[4] >= 1) 15 else 0
                }
                YumCategory.HIGH_STRAIGHT -> {
                    // Check if we have 2,3,4,5,6
                    if (counts[1] >= 1 && counts[2] >= 1 && counts[3] >= 1 && 
                        counts[4] >= 1 && counts[5] >= 1) 20 else 0
                }
                YumCategory.FULL_HOUSE -> {
                    // Check if we have exactly 3 of one value and 2 of another
                    val hasThree = counts.any { it >= 3 }
                    val hasTwo = counts.count { it >= 2 } >= 2
                    if (hasThree && hasTwo) 25 else 0
                }
                YumCategory.YUM -> {
                    // Check for 5 of a kind
                    if (counts.any { it == 5 }) 30 else 0
                }
            }
        }
        
        /**
         * Calculate the total score including bonus
         */
        fun calculateTotalScore(scores: IntArray): Int {
            var total = scores.sum()
            
            // Calculate upper section bonus
            val upperScore = YumCategory.values()
                .filter { it.countsTowardsBonus }
                .sumOf { scores[it.ordinal] }
                
            if (upperScore >= BONUS_CUTOFF) {
                total += BONUS_SCORE
            }
            
            return total
        }
        
        /**
         * Get the upper section score and bonus information
         */
        fun getScoreSummary(scores: IntArray): Pair<Int, Int> {
            val upperScore = YumCategory.values()
                .filter { it.countsTowardsBonus }
                .sumOf { scores[it.ordinal] }
                
            val bonus = if (upperScore >= BONUS_CUTOFF) BONUS_SCORE else 0
            
            return Pair(upperScore, bonus)
        }
        
        /**
         * Check if the game is complete (all categories filled)
         */
        fun isGameComplete(filledCategories: BooleanArray): Boolean {
            return filledCategories.all { it }
        }
        
        /**
         * Get available categories for scoring
         */
        fun getAvailableCategories(filledCategories: BooleanArray): List<YumCategory> {
            return YumCategory.values().filterIndexed { index, _ -> 
                !filledCategories[index] 
            }
        }
    }
}
