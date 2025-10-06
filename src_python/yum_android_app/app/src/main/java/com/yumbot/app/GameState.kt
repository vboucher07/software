package com.yumbot.app

import kotlin.random.Random

/**
 * Represents the current state of a Yum game
 */
data class GameState(
    val scorecard: Scorecard,
    val currentDice: IntArray = IntArray(YumRuleset.NUM_DICE) { 0 },
    val diceToKeep: BooleanArray = BooleanArray(YumRuleset.NUM_DICE) { false },
    val rollsLeft: Int = 3,
    val isGameComplete: Boolean = false,
    val isPlayerTurn: Boolean = true
) {
    fun copyWithArrays(
        scorecard: Scorecard = this.scorecard,
        currentDice: IntArray = this.currentDice.clone(),
        diceToKeep: BooleanArray = this.diceToKeep.clone(),
        rollsLeft: Int = this.rollsLeft,
        isGameComplete: Boolean = this.isGameComplete,
        isPlayerTurn: Boolean = this.isPlayerTurn
    ): GameState {
        return GameState(scorecard, currentDice, diceToKeep, rollsLeft, isGameComplete, isPlayerTurn)
    }
    
    /**
     * Roll the dice that are not kept
     */
    fun rollDice(): GameState {
        if (rollsLeft <= 0) return this
        
        val newDice = currentDice.clone()
        for (i in newDice.indices) {
            if (!diceToKeep[i]) {
                newDice[i] = Random.nextInt(1, 7)
            }
        }
        
        return copyWithArrays(
            currentDice = newDice,
            rollsLeft = rollsLeft - 1
        )
    }
    
    /**
     * Set specific dice values (for manual input mode)
     */
    fun setDiceValues(diceValues: IntArray): GameState {
        require(diceValues.size == YumRuleset.NUM_DICE) { 
            "Must provide exactly ${YumRuleset.NUM_DICE} dice values" 
        }
        require(diceValues.all { it in 1..6 }) { 
            "All dice values must be between 1 and 6" 
        }
        
        return copyWithArrays(currentDice = diceValues.clone())
    }
    
    /**
     * Toggle whether a die is kept
     */
    fun toggleKeepDie(dieIndex: Int): GameState {
        require(dieIndex in 0 until YumRuleset.NUM_DICE) { 
            "Die index must be between 0 and ${YumRuleset.NUM_DICE - 1}" 
        }
        
        val newKeep = diceToKeep.clone()
        newKeep[dieIndex] = !newKeep[dieIndex]
        
        return copyWithArrays(diceToKeep = newKeep)
    }
    
    /**
     * Score in a category and advance to next turn
     */
    fun scoreInCategory(category: YumCategory): GameState {
        if (scorecard.isCategoryFilled(category)) {
            return this // Can't score in already filled category
        }
        
        val newScorecard = scorecard.scoreCategory(currentDice, category)
        val gameComplete = YumRuleset.isGameComplete(newScorecard.filledCategories)
        
        return copyWithArrays(
            scorecard = newScorecard,
            currentDice = IntArray(YumRuleset.NUM_DICE) { 0 },
            diceToKeep = BooleanArray(YumRuleset.NUM_DICE) { false },
            rollsLeft = 3,
            isGameComplete = gameComplete
        )
    }
    
    /**
     * Get dice counts for neural network input
     */
    fun getDiceCounts(): IntArray {
        val counts = IntArray(6) { 0 }
        for (die in currentDice) {
            if (die in 1..6) {
                counts[die - 1]++
            }
        }
        return counts
    }
    
    /**
     * Check if current dice contain any values (i.e., have been rolled)
     */
    fun hasRolledDice(): Boolean {
        return currentDice.any { it > 0 }
    }
}

/**
 * Represents a player's scorecard
 */
data class Scorecard(
    val scores: IntArray = IntArray(YumCategory.getNumCategories()) { 0 },
    val filledCategories: BooleanArray = BooleanArray(YumCategory.getNumCategories()) { false }
) {
    
    fun copy(): Scorecard {
        return Scorecard(scores.clone(), filledCategories.clone())
    }
    
    /**
     * Score a category and return a new scorecard
     */
    fun scoreCategory(dice: IntArray, category: YumCategory): Scorecard {
        if (isCategoryFilled(category)) {
            return this // Already filled
        }
        
        val newScores = scores.clone()
        val newFilled = filledCategories.clone()
        
        val score = YumRuleset.calculateScore(dice, category, filledCategories, scores)
        newScores[category.ordinal] = score
        newFilled[category.ordinal] = true
        
        return Scorecard(newScores, newFilled)
    }
    
    /**
     * Check if a category is already filled
     */
    fun isCategoryFilled(category: YumCategory): Boolean {
        return filledCategories[category.ordinal]
    }
    
    /**
     * Get the score for a specific category
     */
    fun getCategoryScore(category: YumCategory): Int {
        return scores[category.ordinal]
    }
    
    /**
     * Get total score including bonuses
     */
    fun getTotalScore(): Int {
        return YumRuleset.calculateTotalScore(scores)
    }
    
    /**
     * Get upper section score and bonus
     */
    fun getScoreSummary(): Pair<Int, Int> {
        return YumRuleset.getScoreSummary(scores)
    }
    
    /**
     * Get available categories for scoring
     */
    fun getAvailableCategories(): List<YumCategory> {
        return YumRuleset.getAvailableCategories(filledCategories)
    }
    
    /**
     * Convert to array format for neural network input
     */
    fun toArray(): FloatArray {
        val (upperScore, bonus) = getScoreSummary()
        
        // Return: [filled_flags..., upper_score, lower_score]
        val result = FloatArray(YumCategory.getNumCategories() + 2)
        
        // Copy filled flags
        for (i in filledCategories.indices) {
            result[i] = if (filledCategories[i]) 1.0f else 0.0f
        }
        
        // Add upper and lower scores
        result[YumCategory.getNumCategories()] = upperScore.toFloat()
        result[YumCategory.getNumCategories() + 1] = (getTotalScore() - upperScore - bonus).toFloat()
        
        return result
    }
    
    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (javaClass != other?.javaClass) return false
        
        other as Scorecard
        
        if (!scores.contentEquals(other.scores)) return false
        if (!filledCategories.contentEquals(other.filledCategories)) return false
        
        return true
    }
    
    override fun hashCode(): Int {
        var result = scores.contentHashCode()
        result = 31 * result + filledCategories.contentHashCode()
        return result
    }
}
