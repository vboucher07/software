// ===== Simulation Parameters =====

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ColorVisualization {
    Diet,
    Speed,
    Size,
    Sense,
    Attack,
    Defense,
    IdealTemp,
    Panic,
    Energy,
}

pub struct SimParams {
    pub world_size: f32,
    pub plant_spawn_rate: f32,
    pub plant_energy: f32,
    pub reproduction_threshold: f32,
    pub reproduction_cost: f32,
    pub reproduction_cost_sexual: f32, // Cost per parent for sexual reproduction
    pub max_age: f32,
    pub speed_multiplier: f32,
    pub sim_speed: i32, // 1-5x speed
    pub velocity_damping: f32, // 0.0-1.0, how much creatures slow down near food
    pub energy_drain_multiplier: f32, // 0.1-5.0, global energy drain rate
    pub color_mode: ColorVisualization, // What trait to visualize
    
    // Diet efficiency multipliers
    pub herbivore_efficiency: f32, // 0.1-2.0, lower = more efficient
    pub carnivore_efficiency: f32, // 0.1-2.0, lower = more efficient
    pub omnivore_efficiency: f32,  // 0.1-2.0, lower = more efficient
    
    // Digestion times
    pub herbivore_plant_digestion: f32,  // Seconds to digest plants
    pub carnivore_meat_digestion: f32,   // Seconds to digest meat
    pub omnivore_plant_digestion: f32,   // Seconds to digest plants
    pub omnivore_meat_digestion: f32,    // Seconds to digest meat
    
    // Plant clustering
    pub plant_cluster_strength: f32, // 0.0-1.0, higher = more clustering
    
    // Energy transfer efficiency
    pub meat_energy_efficiency: f32, // 0.0-1.0, how much prey energy is transferred
    pub offspring_starting_energy: f32, // Energy newborns start with
    
    // Neural network settings
    pub use_neural_networks: bool, // Toggle between deterministic and neural behavior
    pub neural_mutation_rate: f32, // Rate of neural network mutations
}

// Get temperature at a position - smooth gradient from cold (top) to hot (bottom)
pub fn get_temperature_at_position(_x: f32, y: f32, world_size: f32) -> f32 {
    let normalized_y = y / world_size; // 0.0 to 1.0
    
    // Smooth gradient: 0°C at top to 100°C at bottom
    normalized_y * 100.0
}

impl Default for SimParams {
    fn default() -> Self {
        Self {
            world_size: 1600.0,
            plant_spawn_rate: 7.5,
            plant_energy: 100.0,
            reproduction_threshold: 120.0,
            reproduction_cost: 60.0,
            reproduction_cost_sexual: 30.0, // Half cost per parent
            max_age: 110.0,
            speed_multiplier: 1.0,
            sim_speed: 1,
            velocity_damping: 0.95,
            energy_drain_multiplier: 0.5,
            color_mode: ColorVisualization::Diet,
            
            // Diet efficiency
            herbivore_efficiency: 0.7,
            carnivore_efficiency: 1.2,
            omnivore_efficiency: 1.0,
            
            // Digestion times
            herbivore_plant_digestion: 0.3,
            carnivore_meat_digestion: 2.0,
            omnivore_plant_digestion: 1.2,
            omnivore_meat_digestion: 4.0,
            
            // Plant clustering
            plant_cluster_strength: 0.5,
            
            // Energy transfer
            meat_energy_efficiency: 0.4, // Only 40% of prey energy transferred
            offspring_starting_energy: 60.0, // Start with less energy
            
            // Neural network settings
            use_neural_networks: false, // Start with deterministic behavior
            neural_mutation_rate: 0.1, // 10% chance to mutate neural weights
        }
    }
}

pub const INITIAL_CREATURES: usize = 50;
pub const INITIAL_PLANTS: usize = 150;
pub const MAX_POPULATION: usize = 2000; // Hard cap to prevent crashes

