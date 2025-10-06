package com.yumbot.app

import android.content.Intent
import android.os.Bundle
import androidx.appcompat.app.AppCompatActivity
import com.google.android.material.button.MaterialButton

class MainActivity : AppCompatActivity() {
    
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        
        setupButtons()
    }
    
    private fun setupButtons() {
        val playWithBotButton: MaterialButton = findViewById(R.id.playWithBotButton)
        val playSoloButton: MaterialButton = findViewById(R.id.playSoloButton)
        val multiplayerButton: MaterialButton = findViewById(R.id.multiplayerButton)
        val simulationButton: MaterialButton = findViewById(R.id.simulationButton)
        
        playWithBotButton.setOnClickListener {
            startGame(true) // With bot assistance
        }
        
        playSoloButton.setOnClickListener {
            startGame(false) // Solo play
        }
        
        multiplayerButton.setOnClickListener {
            startMultiplayerSetup()
        }
        
        simulationButton.setOnClickListener {
            startSimulation()
        }
    }
    
    private fun startGame(withBot: Boolean, numPlayers: Int = 1) {
        val intent = Intent(this, GameActivity::class.java).apply {
            putExtra("WITH_BOT", withBot)
            putExtra("NUM_PLAYERS", numPlayers)
        }
        startActivity(intent)
    }
    
    private fun startMultiplayerSetup() {
        // Show dialog to select number of players
        val playerOptions = arrayOf("2 Players", "3 Players", "4 Players", "5 Players", "6 Players")
        
        androidx.appcompat.app.AlertDialog.Builder(this)
            .setTitle("Select Number of Players")
            .setItems(playerOptions) { _, which ->
                val numPlayers = which + 2 // 2-6 players
                startGame(withBot = false, numPlayers = numPlayers)
            }
            .setNegativeButton("Cancel", null)
            .show()
    }
    
    private fun startSimulation() {
        val intent = Intent(this, SimulationActivity::class.java)
        startActivity(intent)
    }
}
