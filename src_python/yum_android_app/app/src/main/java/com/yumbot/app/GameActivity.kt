package com.yumbot.app

import android.os.Bundle
import android.view.View
import android.widget.EditText
import android.widget.LinearLayout
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.recyclerview.widget.LinearLayoutManager
import androidx.recyclerview.widget.RecyclerView
import com.google.android.material.button.MaterialButton

class GameActivity : AppCompatActivity() {
    
    private lateinit var gameState: GameState
    private lateinit var yumBot: YumBot
    private lateinit var neuralBot: NeuralYumBot
    private lateinit var realNeuralBot: RealNeuralBot
    private var withBot: Boolean = false
    private var useNeuralBot: Boolean = true // Use the advanced neural bot by default
    private var useRealNeural: Boolean = true // Use actual 1mil_test.pkl weights
    private var isManualInputMode: Boolean = false
    
    // UI Components
    private lateinit var statusText: TextView
    private lateinit var diceContainer: LinearLayout
    private lateinit var diceInputSection: LinearLayout
    private lateinit var rollDiceButton: MaterialButton
    private lateinit var manualInputButton: MaterialButton
    private lateinit var submitDiceButton: MaterialButton
    private lateinit var botSuggestionSection: LinearLayout
    private lateinit var botSuggestionText: TextView
    private lateinit var upperSectionRecyclerView: RecyclerView
    private lateinit var lowerSectionRecyclerView: RecyclerView
    private lateinit var totalScoreText: TextView
    private lateinit var newGameButton: MaterialButton
    private lateinit var toggleBotButton: MaterialButton
    
    // Dice input fields
    private lateinit var diceInputs: List<EditText>
    
    // Adapters
    private lateinit var upperSectionAdapter: ScorecardAdapter
    private lateinit var lowerSectionAdapter: ScorecardAdapter
    
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_game)
        
        withBot = intent.getBooleanExtra("WITH_BOT", false)
        
        initializeComponents()
        setupRecyclerViews()
        setupButtons()
        startNewGame()
    }
    
    private fun initializeComponents() {
        statusText = findViewById(R.id.statusText)
        diceContainer = findViewById(R.id.diceContainer)
        diceInputSection = findViewById(R.id.diceInputSection)
        rollDiceButton = findViewById(R.id.rollDiceButton)
        manualInputButton = findViewById(R.id.manualInputButton)
        submitDiceButton = findViewById(R.id.submitDiceButton)
        botSuggestionSection = findViewById(R.id.botSuggestionSection)
        botSuggestionText = findViewById(R.id.botSuggestionText)
        upperSectionRecyclerView = findViewById(R.id.upperSectionRecyclerView)
        lowerSectionRecyclerView = findViewById(R.id.lowerSectionRecyclerView)
        totalScoreText = findViewById(R.id.totalScoreText)
        newGameButton = findViewById(R.id.newGameButton)
        toggleBotButton = findViewById(R.id.toggleBotButton)
        
        diceInputs = listOf(
            findViewById(R.id.dice1Input),
            findViewById(R.id.dice2Input),
            findViewById(R.id.dice3Input),
            findViewById(R.id.dice4Input),
            findViewById(R.id.dice5Input)
        )
        
        yumBot = YumBot()
        neuralBot = NeuralYumBot(this)
        realNeuralBot = RealNeuralBot(this)
    }
    
    private fun setupRecyclerViews() {
        // Upper section categories
        val upperCategories = listOf(
            YumCategory.ONES, YumCategory.TWOS, YumCategory.THREES,
            YumCategory.FOURS, YumCategory.FIVES, YumCategory.SIXES
        )
        
        // Lower section categories
        val lowerCategories = listOf(
            YumCategory.LOW_SCORE, YumCategory.HIGH_SCORE, YumCategory.LOW_STRAIGHT,
            YumCategory.HIGH_STRAIGHT, YumCategory.FULL_HOUSE, YumCategory.YUM
        )
        
        upperSectionAdapter = ScorecardAdapter(upperCategories) { category ->
            scoreInCategory(category)
        }
        
        lowerSectionAdapter = ScorecardAdapter(lowerCategories) { category ->
            scoreInCategory(category)
        }
        
        upperSectionRecyclerView.apply {
            layoutManager = LinearLayoutManager(this@GameActivity)
            adapter = upperSectionAdapter
        }
        
        lowerSectionRecyclerView.apply {
            layoutManager = LinearLayoutManager(this@GameActivity)
            adapter = lowerSectionAdapter
        }
    }
    
    private fun setupButtons() {
        rollDiceButton.setOnClickListener {
            rollDice()
        }
        
        manualInputButton.setOnClickListener {
            toggleManualInputMode()
        }
        
        submitDiceButton.setOnClickListener {
            submitManualDice()
        }
        
        newGameButton.setOnClickListener {
            startNewGame()
        }
        
        toggleBotButton.setOnClickListener {
            toggleBot()
        }
    }
    
    private fun startNewGame() {
        gameState = GameState(Scorecard())
        isManualInputMode = false
        updateUI()
        
        if (withBot) {
            botSuggestionSection.visibility = View.VISIBLE
            updateBotToggleButton()
        }
        
        Toast.makeText(this, "New game started!", Toast.LENGTH_SHORT).show()
    }
    
    private fun rollDice() {
        if (gameState.rollsLeft <= 0) {
            Toast.makeText(this, "No rolls left! Please choose a category to score.", Toast.LENGTH_SHORT).show()
            return
        }
        
        gameState = gameState.rollDice()
        updateUI()
        updateBotSuggestion()
    }
    
    private fun toggleManualInputMode() {
        isManualInputMode = !isManualInputMode
        
        if (isManualInputMode) {
            diceInputSection.visibility = View.VISIBLE
            rollDiceButton.visibility = View.GONE
            manualInputButton.text = "Cancel Manual Input"
        } else {
            diceInputSection.visibility = View.GONE
            rollDiceButton.visibility = View.VISIBLE
            manualInputButton.text = "Manual Input"
            clearDiceInputs()
        }
    }
    
    private fun submitManualDice() {
        try {
            val diceValues = IntArray(YumRuleset.NUM_DICE)
            
            for (i in diceInputs.indices) {
                val input = diceInputs[i].text.toString()
                if (input.isEmpty()) {
                    Toast.makeText(this, "Please enter all dice values", Toast.LENGTH_SHORT).show()
                    return
                }
                
                val value = input.toInt()
                if (value !in 1..6) {
                    Toast.makeText(this, "Dice values must be between 1 and 6", Toast.LENGTH_SHORT).show()
                    return
                }
                
                diceValues[i] = value
            }
            
            gameState = gameState.setDiceValues(diceValues).copyWithArrays(rollsLeft = 0)
            toggleManualInputMode()
            updateUI()
            updateBotSuggestion()
            
        } catch (e: NumberFormatException) {
            Toast.makeText(this, "Please enter valid numbers", Toast.LENGTH_SHORT).show()
        }
    }
    
    private fun scoreInCategory(category: YumCategory) {
        if (!gameState.hasRolledDice()) {
            Toast.makeText(this, "Please roll dice first!", Toast.LENGTH_SHORT).show()
            return
        }
        
        if (gameState.scorecard.isCategoryFilled(category)) {
            Toast.makeText(this, "Category already scored!", Toast.LENGTH_SHORT).show()
            return
        }
        
        gameState = gameState.scoreInCategory(category)
        updateUI()
        
        if (gameState.isGameComplete) {
            showGameComplete()
        } else {
            updateBotSuggestion()
        }
    }
    
    private fun updateUI() {
        updateStatusText()
        updateDiceDisplay()
        updateScorecard()
        updateButtons()
    }
    
    private fun updateStatusText() {
        when {
            gameState.isGameComplete -> {
                statusText.text = getString(R.string.game_over)
            }
            gameState.rollsLeft == 0 -> {
                statusText.text = "Choose a category to score"
            }
            gameState.rollsLeft == 3 -> {
                statusText.text = "Roll the dice to start your turn"
            }
            else -> {
                statusText.text = "Rolls left: ${gameState.rollsLeft}"
            }
        }
    }
    
    private fun updateDiceDisplay() {
        diceContainer.removeAllViews()
        
        for (i in 0 until YumRuleset.NUM_DICE) {
            val diceView = DiceView(this)
            diceView.setValue(gameState.currentDice[i])
            diceView.setKeep(gameState.diceToKeep[i])
            diceView.setOnClickListener {
                if (gameState.rollsLeft > 0 && gameState.hasRolledDice()) {
                    gameState = gameState.toggleKeepDie(i)
                    updateDiceDisplay()
                }
            }
            diceContainer.addView(diceView)
        }
    }
    
    private fun updateScorecard() {
        upperSectionAdapter.updateScorecard(gameState.scorecard)
        lowerSectionAdapter.updateScorecard(gameState.scorecard)
        
        totalScoreText.text = "Total: ${gameState.scorecard.getTotalScore()}"
    }
    
    private fun updateButtons() {
        rollDiceButton.isEnabled = gameState.rollsLeft > 0 && !gameState.isGameComplete
        manualInputButton.isEnabled = !gameState.isGameComplete
    }
    
    private fun updateBotSuggestion() {
        if (!withBot || gameState.isGameComplete) {
            botSuggestionSection.visibility = View.GONE
            return
        }
        
        // Use the best available bot based on settings
        val suggestion = when {
            useRealNeural -> {
                val realSuggestion = realNeuralBot.getSuggestion(gameState)
                // Convert to common interface
                YumBot.BotSuggestion(
                    realSuggestion.suggestedCategory,
                    realSuggestion.suggestedKeepDice,
                    realSuggestion.explanation,
                    realSuggestion.confidence
                )
            }
            useNeuralBot -> {
                val neuralSuggestion = neuralBot.getSuggestion(gameState)
                // Convert to common interface
                YumBot.BotSuggestion(
                    neuralSuggestion.suggestedCategory,
                    neuralSuggestion.suggestedKeepDice,
                    neuralSuggestion.explanation,
                    neuralSuggestion.confidence
                )
            }
            else -> {
                yumBot.getSuggestion(gameState)
            }
        }
        
        val botType = when {
            useRealNeural -> "Real Neural Bot (176 avg)"
            useNeuralBot -> "Neural Bot"
            else -> "Basic Bot"
        }
        val confidenceText = "(${(suggestion.confidence * 100).toInt()}% confidence)"
        
        val suggestionText = when {
            suggestion.suggestedCategory != null -> {
                "$botType suggests: Score in ${suggestion.suggestedCategory.displayName}\n${suggestion.explanation}\n$confidenceText"
            }
            suggestion.suggestedKeepDice != null -> {
                "$botType suggests: ${suggestion.explanation}\n$confidenceText"
            }
            else -> "$botType: ${suggestion.explanation}\n$confidenceText"
        }
        
        botSuggestionText.text = suggestionText
        botSuggestionSection.visibility = View.VISIBLE
    }
    
    private fun showGameComplete() {
        val finalScore = gameState.scorecard.getTotalScore()
        Toast.makeText(
            this,
            "Game Complete! Final Score: $finalScore",
            Toast.LENGTH_LONG
        ).show()
    }
    
    private fun clearDiceInputs() {
        diceInputs.forEach { it.text.clear() }
    }
    
    private fun toggleBot() {
        when {
            useRealNeural -> {
                useRealNeural = false
                useNeuralBot = true
            }
            useNeuralBot -> {
                useNeuralBot = false
                useRealNeural = false
            }
            else -> {
                useRealNeural = true
                useNeuralBot = false
            }
        }
        
        updateBotToggleButton()
        updateBotSuggestion() // Refresh suggestion with new bot
        
        val botName = when {
            useRealNeural -> "Real Neural Bot (1mil_test.pkl)"
            useNeuralBot -> "Neural Bot (heuristic)"
            else -> "Basic Bot"
        }
        Toast.makeText(this, "Switched to $botName", Toast.LENGTH_SHORT).show()
    }
    
    private fun updateBotToggleButton() {
        if (withBot) {
            val buttonText = when {
                useRealNeural -> "Using: Real Neural Bot (1mil_test.pkl - 176 avg)"
                useNeuralBot -> "Using: Neural Bot (heuristic)"
                else -> "Using: Basic Bot"
            }
            toggleBotButton.text = buttonText
            toggleBotButton.visibility = View.VISIBLE
        } else {
            toggleBotButton.visibility = View.GONE
        }
    }
}
