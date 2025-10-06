package com.yumbot.app

import kotlin.math.max

/**
 * Simple heuristic-based Yum bot that provides suggestions
 * This is a simplified version that doesn't require neural networks
 */
class YumBot {
    
    data class BotSuggestion(
        val suggestedCategory: YumCategory?,
        val suggestedKeepDice: BooleanArray?,
        val explanation: String,
        val confidence: Float // 0.0 to 1.0
    )
    
    /**
     * Get bot suggestion for the current game state
     */
    fun getSuggestion(gameState: GameState): BotSuggestion {
        // If no rolls left, must choose a category
        if (gameState.rollsLeft == 0) {
            return suggestCategory(gameState)
        }
        
        // If we have rolls left, suggest which dice to keep
        return suggestDiceToKeep(gameState)
    }
    
    /**
     * Suggest which category to score in
     */
    private fun suggestCategory(gameState: GameState): BotSuggestion {
        val availableCategories = gameState.scorecard.getAvailableCategories()
        
        if (availableCategories.isEmpty()) {
            return BotSuggestion(null, null, "No categories available", 0.0f)
        }
        
        // Calculate potential score for each available category
        val categoryScores = availableCategories.map { category ->
            val score = YumRuleset.calculateScore(
                gameState.currentDice, 
                category, 
                gameState.scorecard.filledCategories, 
                gameState.scorecard.scores
            )
            category to score
        }
        
        // Find the best category
        val bestCategory = categoryScores.maxByOrNull { it.second }?.first
            ?: availableCategories.first()
        
        val bestScore = categoryScores.find { it.first == bestCategory }?.second ?: 0
        
        val explanation = when {
            bestScore == 0 -> "No good options available. Take ${bestCategory.displayName} for 0 points."
            bestScore >= 20 -> "Excellent! Score ${bestCategory.displayName} for $bestScore points."
            bestScore >= 10 -> "Good choice. Score ${bestCategory.displayName} for $bestScore points."
            else -> "Best available option. Score ${bestCategory.displayName} for $bestScore points."
        }
        
        val confidence = when {
            bestScore >= 20 -> 0.9f
            bestScore >= 10 -> 0.7f
            bestScore > 0 -> 0.5f
            else -> 0.3f
        }
        
        return BotSuggestion(bestCategory, null, explanation, confidence)
    }
    
    /**
     * Suggest which dice to keep for the next roll
     */
    private fun suggestDiceToKeep(gameState: GameState): BotSuggestion {
        val dice = gameState.currentDice
        val availableCategories = gameState.scorecard.getAvailableCategories()
        
        // Analyze current dice for patterns
        val diceCounts = gameState.getDiceCounts()
        val keepSuggestion = BooleanArray(YumRuleset.NUM_DICE) { false }
        
        // Strategy: Look for the most promising pattern
        val strategy = determineBestStrategy(dice, diceCounts, availableCategories)
        
        // Apply the strategy to determine which dice to keep
        when (strategy.type) {
            KeepStrategy.YUM_OR_FOUR_OF_KIND -> {
                // Keep all dice of the most frequent value
                val targetValue = strategy.targetValue
                for (i in dice.indices) {
                    if (dice[i] == targetValue) {
                        keepSuggestion[i] = true
                    }
                }
            }
            KeepStrategy.FULL_HOUSE -> {
                // Keep the pair and three of a kind
                val values = diceCounts.mapIndexed { index, count -> (index + 1) to count }
                    .filter { it.second >= 2 }
                    .sortedByDescending { it.second }
                
                if (values.size >= 2) {
                    val keepValues = setOf(values[0].first, values[1].first)
                    for (i in dice.indices) {
                        if (dice[i] in keepValues) {
                            keepSuggestion[i] = true
                        }
                    }
                }
            }
            KeepStrategy.STRAIGHT -> {
                // Keep dice that contribute to the straight
                val targetStraight = strategy.targetValue
                val straightValues = if (targetStraight == 1) {
                    setOf(1, 2, 3, 4, 5) // Low straight
                } else {
                    setOf(2, 3, 4, 5, 6) // High straight
                }
                
                for (i in dice.indices) {
                    if (dice[i] in straightValues) {
                        keepSuggestion[i] = true
                    }
                }
            }
            KeepStrategy.UPPER_SECTION -> {
                // Keep all dice of the target value
                val targetValue = strategy.targetValue
                for (i in dice.indices) {
                    if (dice[i] == targetValue) {
                        keepSuggestion[i] = true
                    }
                }
            }
            KeepStrategy.HIGH_VALUES -> {
                // Keep high value dice for sum categories
                for (i in dice.indices) {
                    if (dice[i] >= 4) {
                        keepSuggestion[i] = true
                    }
                }
            }
        }
        
        return BotSuggestion(null, keepSuggestion, strategy.explanation, strategy.confidence)
    }
    
    private data class StrategyInfo(
        val type: KeepStrategy,
        val targetValue: Int,
        val explanation: String,
        val confidence: Float
    )
    
    private enum class KeepStrategy {
        YUM_OR_FOUR_OF_KIND,
        FULL_HOUSE,
        STRAIGHT,
        UPPER_SECTION,
        HIGH_VALUES
    }
    
    /**
     * Determine the best strategy based on current dice and available categories
     */
    private fun determineBestStrategy(
        dice: IntArray, 
        diceCounts: IntArray, 
        availableCategories: List<YumCategory>
    ): StrategyInfo {
        
        // Check for existing patterns
        val maxCount = diceCounts.maxOrNull() ?: 0
        val maxCountValue = diceCounts.indexOfFirst { it == maxCount } + 1
        
        // Look for Yum (5 of a kind) or 4 of a kind
        if (maxCount >= 4 && YumCategory.YUM in availableCategories) {
            return StrategyInfo(
                KeepStrategy.YUM_OR_FOUR_OF_KIND,
                maxCountValue,
                "Keep all ${maxCountValue}s and go for Yum!",
                0.9f
            )
        }
        
        if (maxCount == 3) {
            val pairs = diceCounts.count { it >= 2 }
            if (pairs >= 2 && YumCategory.FULL_HOUSE in availableCategories) {
                return StrategyInfo(
                    KeepStrategy.FULL_HOUSE,
                    maxCountValue,
                    "Keep the pairs and three-of-a-kind for Full House",
                    0.8f
                )
            }
        }
        
        // Check for straight potential
        val sortedDice = dice.sorted()
        val uniqueDice = dice.toSet()
        
        if (checkStraightPotential(uniqueDice, true) && YumCategory.LOW_STRAIGHT in availableCategories) {
            return StrategyInfo(
                KeepStrategy.STRAIGHT,
                1, // Low straight
                "Keep dice for Low Straight (1-2-3-4-5)",
                0.7f
            )
        }
        
        if (checkStraightPotential(uniqueDice, false) && YumCategory.HIGH_STRAIGHT in availableCategories) {
            return StrategyInfo(
                KeepStrategy.STRAIGHT,
                2, // High straight
                "Keep dice for High Straight (2-3-4-5-6)",
                0.7f
            )
        }
        
        // Check for upper section opportunities
        if (maxCount >= 2) {
            val upperCategory = when (maxCountValue) {
                1 -> YumCategory.ONES
                2 -> YumCategory.TWOS
                3 -> YumCategory.THREES
                4 -> YumCategory.FOURS
                5 -> YumCategory.FIVES
                6 -> YumCategory.SIXES
                else -> null
            }
            
            if (upperCategory != null && upperCategory in availableCategories) {
                return StrategyInfo(
                    KeepStrategy.UPPER_SECTION,
                    maxCountValue,
                    "Keep ${maxCountValue}s for ${upperCategory.displayName}",
                    0.6f
                )
            }
        }
        
        // Default: keep high values for sum categories
        return StrategyInfo(
            KeepStrategy.HIGH_VALUES,
            0,
            "Keep high value dice (4, 5, 6) for sum categories",
            0.4f
        )
    }
    
    /**
     * Check if dice have potential for a straight
     */
    private fun checkStraightPotential(uniqueDice: Set<Int>, isLowStraight: Boolean): Boolean {
        val targetSet = if (isLowStraight) setOf(1, 2, 3, 4, 5) else setOf(2, 3, 4, 5, 6)
        val intersection = uniqueDice.intersect(targetSet)
        return intersection.size >= 3 // Need at least 3 dice for potential
    }
}
