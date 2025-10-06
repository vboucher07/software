package com.yumbot.app

import android.os.Bundle
import android.view.View
import android.widget.*
import androidx.appcompat.app.AppCompatActivity
import androidx.recyclerview.widget.LinearLayoutManager
import androidx.recyclerview.widget.RecyclerView
import com.google.android.material.button.MaterialButton
import kotlinx.coroutines.*
import kotlin.random.Random

/**
 * Activity for running game simulations and displaying statistics
 */
class SimulationActivity : AppCompatActivity() {
    
    private lateinit var numGamesInput: EditText
    private lateinit var numPlayersInput: EditText
    private lateinit var botTypeSpinner: Spinner
    private lateinit var startSimulationButton: MaterialButton
    private lateinit var progressBar: ProgressBar
    private lateinit var progressText: TextView
    private lateinit var resultsContainer: LinearLayout
    private lateinit var resultsRecyclerView: RecyclerView
    
    private lateinit var yumBot: YumBot
    private lateinit var neuralBot: NeuralYumBot  
    private lateinit var realNeuralBot: RealNeuralBot
    
    private var simulationJob: Job? = null
    
    data class SimulationResult(
        val gameNumber: Int,
        val playerScores: List<Int>,
        val winner: Int,
        val averageScore: Float
    )
    
    data class SimulationStats(
        val totalGames: Int,
        val averageScore: Float,
        val minScore: Int,
        val maxScore: Int,
        val standardDeviation: Float,
        val gamesWon: Int,
        val winRate: Float,
        val results: List<SimulationResult>
    )
    
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_simulation)
        
        initializeViews()
        setupBots()
        setupSpinner()
        setupListeners()
    }
    
    private fun initializeViews() {
        numGamesInput = findViewById(R.id.numGamesInput)
        numPlayersInput = findViewById(R.id.numPlayersInput)
        botTypeSpinner = findViewById(R.id.botTypeSpinner)
        startSimulationButton = findViewById(R.id.startSimulationButton)
        progressBar = findViewById(R.id.progressBar)
        progressText = findViewById(R.id.progressText)
        resultsContainer = findViewById(R.id.resultsContainer)
        resultsRecyclerView = findViewById(R.id.resultsRecyclerView)
        
        // Set default values
        numGamesInput.setText("100")
        numPlayersInput.setText("1")
    }
    
    private fun setupBots() {
        yumBot = YumBot()
        neuralBot = NeuralYumBot(this)
        realNeuralBot = RealNeuralBot(this)
    }
    
    private fun setupSpinner() {
        val botTypes = arrayOf(
            "Real Neural Bot (1mil_test.pkl - 176 avg)",
            "Neural Bot (heuristic)",
            "Basic Bot"
        )
        
        val adapter = ArrayAdapter(this, android.R.layout.simple_spinner_item, botTypes)
        adapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item)
        botTypeSpinner.adapter = adapter
    }
    
    private fun setupListeners() {
        startSimulationButton.setOnClickListener {
            if (simulationJob?.isActive == true) {
                stopSimulation()
            } else {
                startSimulation()
            }
        }
    }
    
    private fun startSimulation() {
        val numGames = numGamesInput.text.toString().toIntOrNull() ?: 100
        val numPlayers = numPlayersInput.text.toString().toIntOrNull() ?: 1
        val botType = botTypeSpinner.selectedItemPosition
        
        if (numGames <= 0 || numGames > 10000) {
            Toast.makeText(this, "Number of games must be between 1 and 10,000", Toast.LENGTH_SHORT).show()
            return
        }
        
        if (numPlayers <= 0 || numPlayers > 8) {
            Toast.makeText(this, "Number of players must be between 1 and 8", Toast.LENGTH_SHORT).show()
            return
        }
        
        // Update UI for simulation start
        startSimulationButton.text = "Stop Simulation"
        progressBar.visibility = View.VISIBLE
        progressText.visibility = View.VISIBLE
        resultsContainer.visibility = View.GONE
        
        // Start simulation in background
        simulationJob = CoroutineScope(Dispatchers.Main).launch {
            try {
                val stats = runSimulation(numGames, numPlayers, botType)
                displayResults(stats)
            } catch (e: Exception) {
                Toast.makeText(this@SimulationActivity, "Simulation error: ${e.message}", Toast.LENGTH_LONG).show()
            } finally {
                stopSimulation()
            }
        }
    }
    
    private fun stopSimulation() {
        simulationJob?.cancel()
        startSimulationButton.text = "Start Simulation"
        progressBar.visibility = View.GONE
        progressText.visibility = View.GONE
    }
    
    private suspend fun runSimulation(numGames: Int, numPlayers: Int, botType: Int): SimulationStats {
        return withContext(Dispatchers.Default) {
            val results = mutableListOf<SimulationResult>()
            val allScores = mutableListOf<Int>()
            var gamesWon = 0
            
            for (gameNum in 1..numGames) {
                // Update progress on main thread
                withContext(Dispatchers.Main) {
                    progressBar.progress = (gameNum * 100) / numGames
                    progressText.text = "Game $gameNum of $numGames"
                }
                
                // Simulate one game
                val gameResult = simulateGame(numPlayers, botType)
                results.add(gameResult)
                
                // Track player 1's score (the bot we're testing)
                val botScore = gameResult.playerScores[0]
                allScores.add(botScore)
                
                if (gameResult.winner == 0) { // Player 1 won
                    gamesWon++
                }
                
                // Allow cancellation
                if (!isActive) break
            }
            
            // Calculate statistics
            val avgScore = allScores.average().toFloat()
            val minScore = allScores.minOrNull() ?: 0
            val maxScore = allScores.maxOrNull() ?: 0
            
            val variance = allScores.map { (it - avgScore) * (it - avgScore) }.average()
            val stdDev = kotlin.math.sqrt(variance).toFloat()
            
            val winRate = if (numPlayers > 1) (gamesWon.toFloat() / results.size) * 100 else 0f
            
            SimulationStats(
                totalGames = results.size,
                averageScore = avgScore,
                minScore = minScore,
                maxScore = maxScore,
                standardDeviation = stdDev,
                gamesWon = gamesWon,
                winRate = winRate,
                results = results
            )
        }
    }
    
    private fun simulateGame(numPlayers: Int, botType: Int): SimulationResult {
        val playerScores = mutableListOf<Int>()
        
        // Simulate each player
        for (playerIndex in 0 until numPlayers) {
            val scorecard = Scorecard()
            var gameState = GameState(scorecard)
            
            // Play until all categories are filled
            while (!gameState.isGameComplete) {
                // Roll dice (simulate 3 rolls with optimal keeping)
                gameState = simulatePlayerTurn(gameState, botType, playerIndex)
            }
            
            playerScores.add(gameState.scorecard.getTotalScore())
        }
        
        // Find winner (highest score)
        val maxScore = playerScores.maxOrNull() ?: 0
        val winner = playerScores.indexOfFirst { it == maxScore }
        
        return SimulationResult(
            gameNumber = 0, // Will be set by caller
            playerScores = playerScores,
            winner = winner,
            averageScore = playerScores.average().toFloat()
        )
    }
    
    private fun simulatePlayerTurn(gameState: GameState, botType: Int, playerIndex: Int): GameState {
        var currentState = gameState
        
        // First roll
        currentState = currentState.rollDice()
        
        // Use bot to decide what to keep for remaining rolls
        for (rollsLeft in 2 downTo 1) {
            if (rollsLeft > 0) {
                // Get bot suggestion for dice keeping
                val suggestion = getBotSuggestion(currentState, botType)
                
                if (suggestion.suggestedKeepDice != null) {
                    // Apply the bot's keep suggestion
                    val newKeep = suggestion.suggestedKeepDice.clone()
                    currentState = currentState.copyWithArrays(diceToKeep = newKeep)
                }
                
                // Roll again
                currentState = currentState.rollDice()
            }
        }
        
        // Choose category to score
        val categorySuggestion = getBotSuggestion(currentState, botType)
        val category = categorySuggestion.suggestedCategory ?: YumCategory.ONES // Fallback
        
        // Score the category  
        val score = YumRuleset.calculateScore(currentState.currentDice, category, 
            currentState.scorecard.filledCategories, currentState.scorecard.scores)
        val newScorecard = currentState.scorecard.scoreCategory(currentState.currentDice, category)
        val scoredState = currentState.copyWithArrays(scorecard = newScorecard)
        
        return scoredState
    }
    
    private fun getBotSuggestion(gameState: GameState, botType: Int): YumBot.BotSuggestion {
        return when (botType) {
            0 -> { // Real Neural Bot
                val suggestion = realNeuralBot.getSuggestion(gameState)
                YumBot.BotSuggestion(
                    suggestion.suggestedCategory,
                    suggestion.suggestedKeepDice,
                    suggestion.explanation,
                    suggestion.confidence
                )
            }
            1 -> { // Neural Bot (heuristic)
                val suggestion = neuralBot.getSuggestion(gameState)
                YumBot.BotSuggestion(
                    suggestion.suggestedCategory,
                    suggestion.suggestedKeepDice,
                    suggestion.explanation,
                    suggestion.confidence
                )
            }
            else -> { // Basic Bot
                yumBot.getSuggestion(gameState)
            }
        }
    }
    
    private fun displayResults(stats: SimulationStats) {
        resultsContainer.visibility = View.VISIBLE
        
        // Clear previous results
        resultsContainer.removeAllViews()
        
        // Create summary text
        val summaryText = """
            🎯 Simulation Results
            
            📊 Performance Statistics:
            • Total Games: ${stats.totalGames}
            • Average Score: ${String.format("%.1f", stats.averageScore)}
            • Min Score: ${stats.minScore}
            • Max Score: ${stats.maxScore}
            • Standard Deviation: ${String.format("%.1f", stats.standardDeviation)}
            
            ${if (stats.winRate > 0) "🏆 Competitive Stats:\n• Games Won: ${stats.gamesWon}\n• Win Rate: ${String.format("%.1f", stats.winRate)}%" else ""}
            
            📈 Score Distribution:
            ${getScoreDistribution(stats.results.map { it.playerScores[0] })}
        """.trimIndent()
        
        val summaryTextView = TextView(this).apply {
            text = summaryText
            textSize = 14f
            setPadding(16, 16, 16, 16)
        }
        
        resultsContainer.addView(summaryTextView)
        
        // Show expected vs actual for neural bot
        if (botTypeSpinner.selectedItemPosition == 0) { // Real Neural Bot
            val expectedScore = 176f
            val performance = (stats.averageScore / expectedScore) * 100
            val comparisonText = """
                🧠 Neural Network Analysis:
                • Expected Score: ${expectedScore}
                • Actual Score: ${String.format("%.1f", stats.averageScore)}
                • Performance: ${String.format("%.1f", performance)}% of training performance
                
                ${when {
                    performance >= 95 -> "✅ EXCELLENT - Bot performing at expected level!"
                    performance >= 85 -> "✅ GOOD - Bot performing well"
                    performance >= 70 -> "⚠️ FAIR - Below expectations but reasonable"
                    else -> "❌ POOR - Significantly underperforming"
                }}
            """.trimIndent()
            
            val comparisonTextView = TextView(this).apply {
                text = comparisonText
                textSize = 14f
                setPadding(16, 8, 16, 16)
                setBackgroundColor(0x11000000) // Light background
            }
            
            resultsContainer.addView(comparisonTextView)
        }
    }
    
    private fun getScoreDistribution(scores: List<Int>): String {
        val ranges = listOf(
            0..50 to "Very Low (0-50)",
            51..100 to "Low (51-100)", 
            101..150 to "Average (101-150)",
            151..200 to "Good (151-200)",
            201..250 to "Excellent (201-250)",
            251..Int.MAX_VALUE to "Outstanding (251+)"
        )
        
        val distribution = ranges.map { (range, label) ->
            val count = scores.count { it in range }
            val percentage = (count.toFloat() / scores.size) * 100
            "$label: $count (${String.format("%.1f", percentage)}%)"
        }.filter { !it.contains(": 0 (") } // Only show non-zero ranges
        
        return distribution.joinToString("\n")
    }
}
