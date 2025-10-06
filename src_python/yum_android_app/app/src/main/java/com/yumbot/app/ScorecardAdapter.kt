package com.yumbot.app

import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.TextView
import androidx.recyclerview.widget.RecyclerView

class ScorecardAdapter(
    private val categories: List<YumCategory>,
    private val onCategoryClick: (YumCategory) -> Unit
) : RecyclerView.Adapter<ScorecardAdapter.ScorecardViewHolder>() {
    
    private var scorecard: Scorecard = Scorecard()
    
    class ScorecardViewHolder(itemView: View) : RecyclerView.ViewHolder(itemView) {
        val categoryName: TextView = itemView.findViewById(R.id.categoryName)
        val categoryScore: TextView = itemView.findViewById(R.id.categoryScore)
    }
    
    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): ScorecardViewHolder {
        val view = LayoutInflater.from(parent.context)
            .inflate(R.layout.item_scorecard_category, parent, false)
        return ScorecardViewHolder(view)
    }
    
    override fun onBindViewHolder(holder: ScorecardViewHolder, position: Int) {
        val category = categories[position]
        
        holder.categoryName.text = category.displayName
        
        if (scorecard.isCategoryFilled(category)) {
            holder.categoryScore.text = scorecard.getCategoryScore(category).toString()
            holder.itemView.setBackgroundColor(
                holder.itemView.context.getColor(R.color.category_filled)
            )
            holder.itemView.isClickable = false
        } else {
            holder.categoryScore.text = "-"
            holder.itemView.setBackgroundColor(
                holder.itemView.context.getColor(R.color.category_available)
            )
            holder.itemView.isClickable = true
            holder.itemView.setOnClickListener {
                onCategoryClick(category)
            }
        }
    }
    
    override fun getItemCount(): Int = categories.size
    
    fun updateScorecard(newScorecard: Scorecard) {
        scorecard = newScorecard
        notifyDataSetChanged()
    }
}
