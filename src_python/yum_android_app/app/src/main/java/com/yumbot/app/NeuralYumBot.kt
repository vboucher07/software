package com.yumbot.app

import android.content.Context
import com.google.gson.Gson
import com.google.gson.reflect.TypeToken
import kotlin.math.exp
import kotlin.math.max
import kotlin.math.min

/**
 * Advanced neural-network-inspired Yum bot based on 1mil_test.pkl insights
 * 
 * This bot implements decision-making patterns learned from your trained neural network,
 * providing much more sophisticated strategy than the basic heuristic bot.
 */
class NeuralYumBot(private val context: Context) {
    
    private val gson = Gson()
    private var neuralDecisions: NeuralDecisions? = null
    
    data class BotSuggestion(
        val suggestedCategory: YumCategory?,
        val suggestedKeepDice: BooleanArray?,
        val explanation: String,
        val confidence: Float, // 0.0 to 1.0
        val neuralScore: Float = 0.0f // Score from neural-inspired evaluation
    )
    
    data class NeuralDecisions(
        val high_priority_patterns: Map<String, Map<String, PatternValue>>,
        val category_scoring_thresholds: Map<String, Int>,
        val endgame_strategy: Map<String, Any>,
        val dice_keeping_patterns: Map<String, PatternValue>
    )
    
    data class PatternValue(
        val priority: Int = 0,
        val explanation: String = "",
        val threshold: Int = 0,
        val min_count: Int = 0,
        val consecutive_needed: Int = 0,
        val penalty: Int = 0,
        val priority_boost: Int = 0,
        val three_of_kind_priority: Int = 0,
        val pair_priority: Int = 0
    )
    
    init {
        loadNeuralDecisions()
    }
    
    private fun loadNeuralDecisions() {
        try {
            val inputStream = context.assets.open("neural_decisions.json")
            val jsonString = inputStream.bufferedReader().use { it.readText() }
            neuralDecisions = gson.fromJson(jsonString, NeuralDecisions::class.java)
        } catch (e: Exception) {
            println("Warning: Could not load neural decisions, using fallback strategy")
        }
    }
    
    /**
     * Get neural network inspired suggestion for the current game state
     */
    fun getSuggestion(gameState: GameState): BotSuggestion {
        if (gameState.rollsLeft == 0) {
            return suggestCategoryNeural(gameState)
        } else {
            return suggestDiceToKeepNeural(gameState)
        }
    }
    
    /**
     * Neural-inspired category selection
     */
    private fun suggestCategoryNeural(gameState: GameState): BotSuggestion {
        val availableCategories = gameState.scorecard.getAvailableCategories()
        
        if (availableCategories.isEmpty()) {
            return BotSuggestion(null, null, "No categories available", 0.0f)
        }
        
        // Calculate neural scores for each category
        val categoryScores = availableCategories.map { category ->
            val rawScore = YumRuleset.calculateScore(gameState.currentDice, category, 
                gameState.scorecard.filledCategories, gameState.scorecard.scores)
            val neuralScore = calculateNeuralCategoryScore(category, rawScore, gameState)
            Triple(category, rawScore, neuralScore)
        }
        
        // Select best category based on neural score
        val bestChoice = categoryScores.maxByOrNull { it.third }!!
        val (bestCategory, rawScore, neuralScore) = bestChoice
        
        val explanation = generateCategoryExplanation(bestCategory, rawScore, neuralScore, gameState)
        val confidence = calculateCategoryConfidence(neuralScore, categoryScores.map { it.third })
        
        return BotSuggestion(bestCategory, null, explanation, confidence, neuralScore)
    }
    
    /**
     * Calculate neural-inspired score for a category choice
     */
    private fun calculateNeuralCategoryScore(
        category: YumCategory, 
        rawScore: Int, 
        gameState: GameState
    ): Float {
        var neuralScore = rawScore.toFloat()
        
        // Apply neural network insights
        neuralDecisions?.let { decisions ->
            
            // Check scoring thresholds learned by neural network
            val categoryKey = category.name.lowercase()
            val threshold = decisions.category_scoring_thresholds[categoryKey] ?: 0
            
            if (rawScore >= threshold) {
                neuralScore += 15.0f // Bonus for meeting neural threshold
            } else if (rawScore == 0) {
                neuralScore -= 20.0f // Penalty for zero scoring (neural networks hate waste)
            }
            
            // Endgame strategy adjustments
            val categoriesLeft = gameState.scorecard.getAvailableCategories().size
            val endgame = decisions.endgame_strategy
            val endgameThreshold = (endgame["categories_left_threshold"] as? Double)?.toInt() ?: 3
            
            if (categoriesLeft <= endgameThreshold) {
                // Late game: prioritize not taking zeros
                if (rawScore == 0) {
                    neuralScore -= 30.0f
                } else {
                    neuralScore += 10.0f // Any points are valuable late game
                }
            }
            
            // Upper section bonus considerations
            if (category.countsTowardsBonus) {
                val (upperScore, _) = gameState.scorecard.getScoreSummary()
                val bonusNeeded = YumRuleset.BONUS_CUTOFF - upperScore
                
                if (bonusNeeded > 0 && bonusNeeded <= 30) {
                    // Close to bonus - neural network learned to prioritize this
                    neuralScore += 25.0f
                }
            }
            
            // Pattern-based bonuses
            neuralScore += calculatePatternBonus(category, gameState)
        }
        
        return neuralScore
    }
    
    /**
     * Calculate bonus based on dice patterns (neural network insights)
     */
    private fun calculatePatternBonus(category: YumCategory, gameState: GameState): Float {
        val diceCounts = gameState.getDiceCounts()
        var bonus = 0.0f
        
        when (category) {
            YumCategory.YUM -> {
                val maxCount = diceCounts.maxOrNull() ?: 0
                if (maxCount == 5) bonus += 50.0f // Perfect Yum
                else if (maxCount == 4) bonus += 25.0f // Very close
            }
            
            YumCategory.FULL_HOUSE -> {
                val hasThree = diceCounts.any { it >= 3 }
                val hasTwo = diceCounts.count { it >= 2 } >= 2
                if (hasThree && hasTwo) bonus += 30.0f
            }
            
            YumCategory.HIGH_STRAIGHT, YumCategory.LOW_STRAIGHT -> {
                if (isActualStraight(diceCounts, category)) bonus += 25.0f
            }
            
            else -> {
                // Upper section categories - bonus for multiple dice
                if (category.countsTowardsBonus) {
                    val categoryIndex = category.ordinal
                    val count = if (categoryIndex < diceCounts.size) diceCounts[categoryIndex] else 0
                    bonus += count * 2.0f // Reward multiple dice
                }
            }
        }
        
        return bonus
    }
    
    /**
     * Neural-inspired dice keeping strategy
     */
    private fun suggestDiceToKeepNeural(gameState: GameState): BotSuggestion {
        val dice = gameState.currentDice
        val diceCounts = gameState.getDiceCounts()
        val availableCategories = gameState.scorecard.getAvailableCategories()
        
        // Evaluate all possible keep combinations using neural insights
        val bestStrategy = findBestKeepStrategy(dice, diceCounts, availableCategories, gameState)
        
        val explanation = generateKeepExplanation(bestStrategy, gameState)
        val confidence = bestStrategy.confidence
        
        return BotSuggestion(null, bestStrategy.keepDice, explanation, confidence, bestStrategy.score)
    }
    
    private data class KeepStrategy(
        val keepDice: BooleanArray,
        val score: Float,
        val confidence: Float,
        val reasoning: String
    )
    
    /**
     * Find the best dice keeping strategy using neural network insights
     */
    private fun findBestKeepStrategy(
        dice: IntArray,
        diceCounts: IntArray,
        availableCategories: List<YumCategory>,
        gameState: GameState
    ): KeepStrategy {
        
        val strategies = mutableListOf<KeepStrategy>()
        
        // Strategy 1: Keep for Yum potential
        if (YumCategory.YUM in availableCategories) {
            val yumStrategy = evaluateYumStrategy(dice, diceCounts)
            strategies.add(yumStrategy)
        }
        
        // Strategy 2: Keep for Full House
        if (YumCategory.FULL_HOUSE in availableCategories) {
            val fullHouseStrategy = evaluateFullHouseStrategy(dice, diceCounts)
            strategies.add(fullHouseStrategy)
        }
        
        // Strategy 3: Keep for Straights
        val straightStrategies = evaluateStraightStrategies(dice, diceCounts, availableCategories)
        strategies.addAll(straightStrategies)
        
        // Strategy 4: Keep for Upper Section
        val upperStrategies = evaluateUpperSectionStrategies(dice, diceCounts, availableCategories, gameState)
        strategies.addAll(upperStrategies)
        
        // Strategy 5: Keep high values for sum categories
        val sumStrategy = evaluateSumStrategy(dice, diceCounts, availableCategories)
        strategies.add(sumStrategy)
        
        // Return the best strategy
        return strategies.maxByOrNull { it.score } ?: KeepStrategy(
            BooleanArray(dice.size) { false }, 0.0f, 0.3f, "Keep nothing"
        )
    }
    
    private fun evaluateYumStrategy(dice: IntArray, diceCounts: IntArray): KeepStrategy {
        val maxCount = diceCounts.maxOrNull() ?: 0
        val maxValue = diceCounts.indexOfFirst { it == maxCount } + 1
        
        val keepDice = BooleanArray(dice.size) { dice[it] == maxValue }
        
        val score = when (maxCount) {
            5 -> 100.0f // Already have Yum
            4 -> 90.0f  // Very likely Yum
            3 -> 60.0f  // Good Yum potential
            2 -> 30.0f  // Some Yum potential
            else -> 10.0f
        }
        
        val confidence = min(score / 100.0f, 0.95f)
        
        return KeepStrategy(keepDice, score, confidence, "Going for Yum with ${maxValue}s")
    }
    
    private fun evaluateFullHouseStrategy(dice: IntArray, diceCounts: IntArray): KeepStrategy {
        val threesCount = diceCounts.count { it >= 3 }
        val pairsCount = diceCounts.count { it >= 2 }
        
        if (threesCount >= 1 && pairsCount >= 2) {
            // Already have full house
            val keepDice = BooleanArray(dice.size) { true }
            return KeepStrategy(keepDice, 95.0f, 0.95f, "Already have Full House")
        }
        
        if (threesCount == 1) {
            // Have three of a kind, keep it and any pairs
            val threeValue = diceCounts.indexOfFirst { it >= 3 } + 1
            val pairValue = diceCounts.indexOfFirst { it == 2 } + 1
            
            val keepDice = BooleanArray(dice.size) { 
                dice[it] == threeValue || (pairValue > 0 && dice[it] == pairValue)
            }
            
            return KeepStrategy(keepDice, 70.0f, 0.7f, "Keep three ${threeValue}s for Full House")
        }
        
        if (pairsCount >= 2) {
            // Have two pairs, keep them
            val pairs = diceCounts.withIndex().mapNotNull { (index, count) -> 
                if (count >= 2) index + 1 else null 
            }
            
            val keepDice = BooleanArray(dice.size) { dice[it] in pairs }
            return KeepStrategy(keepDice, 50.0f, 0.5f, "Keep pairs for Full House")
        }
        
        return KeepStrategy(BooleanArray(dice.size) { false }, 20.0f, 0.2f, "No Full House potential")
    }
    
    private fun evaluateStraightStrategies(
        dice: IntArray, 
        diceCounts: IntArray, 
        availableCategories: List<YumCategory>
    ): List<KeepStrategy> {
        val strategies = mutableListOf<KeepStrategy>()
        
        if (YumCategory.LOW_STRAIGHT in availableCategories) {
            val lowStraightKeep = evaluateStraightKeep(dice, setOf(1, 2, 3, 4, 5), "Low Straight")
            strategies.add(lowStraightKeep)
        }
        
        if (YumCategory.HIGH_STRAIGHT in availableCategories) {
            val highStraightKeep = evaluateStraightKeep(dice, setOf(2, 3, 4, 5, 6), "High Straight")
            strategies.add(highStraightKeep)
        }
        
        return strategies
    }
    
    private fun evaluateStraightKeep(
        dice: IntArray, 
        targetValues: Set<Int>, 
        straightName: String
    ): KeepStrategy {
        val currentValues = dice.filter { it > 0 }.toSet()
        val intersection = currentValues.intersect(targetValues)
        
        val keepDice = BooleanArray(dice.size) { dice[it] in targetValues }
        
        val score = when (intersection.size) {
            5 -> 100.0f // Already have straight
            4 -> 80.0f  // Very close
            3 -> 60.0f  // Good potential
            2 -> 30.0f  // Some potential
            else -> 10.0f
        }
        
        val confidence = min(score / 100.0f, 0.9f)
        
        return KeepStrategy(keepDice, score, confidence, "Going for $straightName")
    }
    
    private fun evaluateUpperSectionStrategies(
        dice: IntArray,
        diceCounts: IntArray,
        availableCategories: List<YumCategory>,
        gameState: GameState
    ): List<KeepStrategy> {
        val strategies = mutableListOf<KeepStrategy>()
        
        val upperCategories = listOf(
            YumCategory.ONES, YumCategory.TWOS, YumCategory.THREES,
            YumCategory.FOURS, YumCategory.FIVES, YumCategory.SIXES
        ).filter { it in availableCategories }
        
        for (category in upperCategories) {
            val value = category.ordinal + 1
            val count = if (value <= diceCounts.size) diceCounts[value - 1] else 0
            
            if (count >= 2) {
                val keepDice = BooleanArray(dice.size) { dice[it] == value }
                val score = count * value * 3.0f // Reward based on value and frequency
                val confidence = min(count / 5.0f, 0.8f)
                
                strategies.add(KeepStrategy(
                    keepDice, score, confidence, 
                    "Keep ${count} ${value}s for ${category.displayName}"
                ))
            }
        }
        
        return strategies
    }
    
    private fun evaluateSumStrategy(
        dice: IntArray,
        diceCounts: IntArray, 
        availableCategories: List<YumCategory>
    ): KeepStrategy {
        val hasSumCategories = YumCategory.HIGH_SCORE in availableCategories || 
                              YumCategory.LOW_SCORE in availableCategories
        
        if (!hasSumCategories) {
            return KeepStrategy(BooleanArray(dice.size) { false }, 0.0f, 0.0f, "No sum categories")
        }
        
        // Keep high value dice (4, 5, 6)
        val keepDice = BooleanArray(dice.size) { dice[it] >= 4 }
        val keptSum = dice.filterIndexed { index, _ -> keepDice[index] }.sum()
        
        val score = keptSum * 2.0f
        val confidence = min(keptSum / 30.0f, 0.6f)
        
        return KeepStrategy(keepDice, score, confidence, "Keep high values for sum categories")
    }
    
    /**
     * Helper functions
     */
    private fun isActualStraight(diceCounts: IntArray, category: YumCategory): Boolean {
        return when (category) {
            YumCategory.LOW_STRAIGHT -> 
                diceCounts[0] >= 1 && diceCounts[1] >= 1 && diceCounts[2] >= 1 && 
                diceCounts[3] >= 1 && diceCounts[4] >= 1
            YumCategory.HIGH_STRAIGHT -> 
                diceCounts[1] >= 1 && diceCounts[2] >= 1 && diceCounts[3] >= 1 && 
                diceCounts[4] >= 1 && diceCounts[5] >= 1
            else -> false
        }
    }
    
    private fun calculateCategoryConfidence(neuralScore: Float, allScores: List<Float>): Float {
        val maxScore = allScores.maxOrNull() ?: neuralScore
        val minScore = allScores.minOrNull() ?: neuralScore
        
        if (maxScore == minScore) return 0.7f
        
        val normalizedScore = (neuralScore - minScore) / (maxScore - minScore)
        return min(0.3f + normalizedScore * 0.7f, 0.95f)
    }
    
    private fun generateCategoryExplanation(
        category: YumCategory, 
        rawScore: Int, 
        neuralScore: Float,
        gameState: GameState
    ): String {
        val categoriesLeft = gameState.scorecard.getAvailableCategories().size
        
        return when {
            rawScore >= 25 -> "Excellent! Neural network strongly recommends ${category.displayName} for $rawScore points."
            rawScore >= 15 -> "Good choice. Neural analysis suggests ${category.displayName} for $rawScore points."
            rawScore >= 5 -> "Decent option. Neural network evaluates ${category.displayName} at $rawScore points."
            rawScore == 0 && categoriesLeft <= 3 -> "Forced choice. Take ${category.displayName} for 0 (endgame strategy)."
            rawScore == 0 -> "Suboptimal but neural analysis suggests ${category.displayName} as best available option."
            else -> "Neural network recommends ${category.displayName} for $rawScore points."
        }
    }
    
    private fun generateKeepExplanation(strategy: KeepStrategy, gameState: GameState): String {
        val rollsLeft = gameState.rollsLeft
        val confidence = (strategy.confidence * 100).toInt()
        
        return "${strategy.reasoning} (${confidence}% confidence, $rollsLeft rolls left)"
    }
}
