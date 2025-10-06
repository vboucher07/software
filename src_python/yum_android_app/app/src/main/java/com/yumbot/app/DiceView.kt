package com.yumbot.app

import android.content.Context
import android.view.LayoutInflater
import android.widget.FrameLayout
import android.widget.TextView

class DiceView(context: Context) : FrameLayout(context) {
    
    private val diceValue: TextView
    private val keepIndicator: TextView
    private var isKept = false
    private var onKeepToggleListener: ((Boolean) -> Unit)? = null
    
    init {
        LayoutInflater.from(context).inflate(R.layout.item_dice, this, true)
        diceValue = findViewById(R.id.diceValue)
        keepIndicator = findViewById(R.id.keepIndicator)
        
        // Improve touch detection
        isClickable = true
        isFocusable = true
        
        // Add padding for larger touch area
        setPadding(16, 16, 16, 16)
        
        // Make entire view clickable with better feedback
        setOnClickListener {
            toggleKeep()
        }
        
        // Add visual pressed state
        background = context.getDrawable(R.drawable.dice_background)
    }
    
    fun setValue(value: Int) {
        if (value > 0) {
            diceValue.text = value.toString()
            visibility = VISIBLE
        } else {
            diceValue.text = ""
            visibility = INVISIBLE
        }
    }
    
    fun setKeep(keep: Boolean) {
        isKept = keep
        keepIndicator.visibility = if (keep) VISIBLE else GONE
        isSelected = keep
        
        // Better visual feedback for kept dice
        if (keep) {
            alpha = 1.0f
            scaleX = 1.1f
            scaleY = 1.1f
            // Add a border or background color change
            setBackgroundColor(context.getColor(R.color.dice_selected))
        } else {
            alpha = 0.8f
            scaleX = 1.0f
            scaleY = 1.0f
            setBackgroundColor(context.getColor(R.color.dice_background))
        }
    }
    
    fun isKept(): Boolean = isKept
    
    fun setOnKeepToggleListener(listener: (Boolean) -> Unit) {
        onKeepToggleListener = listener
    }
    
    private fun toggleKeep() {
        // Only allow toggling if dice has a value and is not zero
        val currentValue = diceValue.text.toString()
        if (currentValue.isNotEmpty() && currentValue != "0") {
            setKeep(!isKept)
            
            // Add haptic feedback for better user experience
            performHapticFeedback(android.view.HapticFeedbackConstants.VIRTUAL_KEY)
            
            // Animate the toggle for better visual feedback
            animate()
                .scaleX(if (isKept) 1.1f else 1.0f)
                .scaleY(if (isKept) 1.1f else 1.0f)
                .setDuration(150)
                .start()
            
            onKeepToggleListener?.invoke(isKept)
        }
    }
}
