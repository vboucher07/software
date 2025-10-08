use ::rand::Rng;

// ===== Components =====

// Lightweight neural network for creature behavior
#[derive(Clone, Debug)]
pub struct NeuralNetwork {
    // Simple 2-layer network: 8 inputs -> 4 hidden -> 2 outputs
    // Inputs: energy, nearest_food_dist, nearest_food_dir_x, nearest_food_dir_y, 
    //         nearest_threat_dist, nearest_threat_dir_x, nearest_threat_dir_y, temperature
    // Outputs: velocity_x, velocity_y
    pub weights_input_hidden: [f32; 32], // 8 inputs * 4 hidden
    pub weights_hidden_output: [f32; 8],  // 4 hidden * 2 outputs
    pub biases_hidden: [f32; 4],
    pub biases_output: [f32; 2],
}

impl NeuralNetwork {
    pub fn random() -> Self {
        let mut rng = ::rand::rng();
        
        Self {
            weights_input_hidden: [(); 32].map(|_| rng.random_range(-1.0..1.0)),
            weights_hidden_output: [(); 8].map(|_| rng.random_range(-1.0..1.0)),
            biases_hidden: [(); 4].map(|_| rng.random_range(-0.5..0.5)),
            biases_output: [(); 2].map(|_| rng.random_range(-0.5..0.5)),
        }
    }
    
    pub fn mutate(&self, mutation_rate: f32) -> Self {
        let mut rng = ::rand::rng();
        
        Self {
            weights_input_hidden: self.weights_input_hidden.map(|w| {
                if rng.random_range(0.0..1.0) < mutation_rate {
                    w + rng.random_range(-0.2..0.2)
                } else {
                    w
                }
            }),
            weights_hidden_output: self.weights_hidden_output.map(|w| {
                if rng.random_range(0.0..1.0) < mutation_rate {
                    w + rng.random_range(-0.2..0.2)
                } else {
                    w
                }
            }),
            biases_hidden: self.biases_hidden.map(|b| {
                if rng.random_range(0.0..1.0) < mutation_rate {
                    b + rng.random_range(-0.1..0.1)
                } else {
                    b
                }
            }),
            biases_output: self.biases_output.map(|b| {
                if rng.random_range(0.0..1.0) < mutation_rate {
                    b + rng.random_range(-0.1..0.1)
                } else {
                    b
                }
            }),
        }
    }
    
    #[allow(dead_code)]
    pub fn forward(&self, inputs: &[f32; 8]) -> (f32, f32) {
        // Normalize inputs to [-1, 1] range
        let normalized_inputs = [
            inputs[0] / 100.0,                    // energy (0-100)
            inputs[1] / 500.0,                    // nearest_food_dist (0-500)
            inputs[2],                            // nearest_food_dir_x (-1 to 1)
            inputs[3],                            // nearest_food_dir_y (-1 to 1)
            inputs[4] / 500.0,                   // nearest_threat_dist (0-500)
            inputs[5],                            // nearest_threat_dir_x (-1 to 1)
            inputs[6],                            // nearest_threat_dir_y (-1 to 1)
            inputs[7] / 50.0 - 1.0,              // temperature (0-100) -> (-1 to 1)
        ];
        
        // Hidden layer
        let mut hidden = [0.0; 4];
        for i in 0..4 {
            hidden[i] = self.biases_hidden[i];
            for j in 0..8 {
                hidden[i] += normalized_inputs[j] * self.weights_input_hidden[i * 8 + j];
            }
            hidden[i] = hidden[i].tanh(); // Activation function
        }
        
        // Output layer
        let mut outputs = [0.0; 2];
        for i in 0..2 {
            outputs[i] = self.biases_output[i];
            for j in 0..4 {
                outputs[i] += hidden[j] * self.weights_hidden_output[i * 4 + j];
            }
            outputs[i] = outputs[i].tanh(); // Output in [-1, 1] range
        }
        
        (outputs[0], outputs[1])
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Color {
    pub r: f32,
    pub g: f32,
    pub b: f32,
    pub a: f32,
}

impl Color {
    pub fn new(r: f32, g: f32, b: f32, a: f32) -> Self {
        Self { r, g, b, a }
    }
}

// ===== Components =====

#[derive(Clone, Copy, Debug)]
pub struct Position {
    pub x: f32,
    pub y: f32,
}

#[derive(Clone, Copy, Debug)]
pub struct Velocity {
    pub x: f32,
    pub y: f32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Diet {
    Herbivore,
    Carnivore,
    Omnivore,
}

impl Diet {
    pub fn mutate(&self) -> Self {
        let mut rng = ::rand::rng();
        let mutation_chance = rng.random_range(0.0..1.0);
        
        // 5% chance to mutate diet
        if mutation_chance < 0.05 {
            match rng.random_range(0..3) {
                0 => Diet::Herbivore,
                1 => Diet::Carnivore,
                _ => Diet::Omnivore,
            }
        } else {
            *self
        }
    }
}

#[derive(Clone, Debug)]
pub struct Genome {
    pub size: f32,          // 5.0 - 20.0
    pub speed: f32,         // 50.0 - 200.0
    pub sense_range: f32,   // 50.0 - 300.0
    pub color_hue: f32,     // 0.0 - 360.0 for variety
    pub diet: Diet,         // Herbivore, Carnivore, or Omnivore
    pub attack: f32,        // 0.0 - 10.0 offensive capability
    pub defense: f32,       // 0.0 - 10.0 defensive capability
    pub hunger_threshold: f32, // 50.0 - 150.0 energy level to seek food aggressively
    pub ideal_temperature: f32, // 0.0 - 100.0 temperature preference
    pub panic: f32,         // 0.0 - 10.0 how strongly to flee from predators
    pub hermaphrodite: bool, // If false, needs partner for reproduction
    #[allow(dead_code)]
    pub neural_network: NeuralNetwork, // Neural network for behavior
}

impl Genome {
    pub fn random() -> Self {
        let mut rng = ::rand::rng();
        Self {
            size: rng.random_range(8.0..12.0),
            speed: rng.random_range(80.0..120.0),
            sense_range: rng.random_range(100.0..150.0),
            color_hue: rng.random_range(0.0..360.0),
            diet: Diet::Herbivore, // Start as herbivores
            attack: rng.random_range(0.0..2.0),
            defense: rng.random_range(0.0..2.0),
            hunger_threshold: rng.random_range(80.0..100.0),
            ideal_temperature: rng.random_range(30.0..70.0), // Start with temperate preference
            panic: rng.random_range(3.0..7.0),
            hermaphrodite: true, // Start as hermaphrodites (asexual)
            neural_network: NeuralNetwork::random(),
        }
    }

    pub fn mutate(&self, neural_mutation_rate: f32) -> Self {
        let mut rng = ::rand::rng();
        Self {
            size: (self.size + rng.random_range(-1.0..1.0)).clamp(5.0, 20.0),
            speed: (self.speed + rng.random_range(-10.0..10.0)).clamp(40.0, 200.0),
            sense_range: (self.sense_range + rng.random_range(-15.0..15.0)).clamp(40.0, 300.0),
            color_hue: (self.color_hue + rng.random_range(-20.0..20.0)) % 360.0,
            diet: self.diet.mutate(),
            attack: (self.attack + rng.random_range(-0.5..0.5)).clamp(0.0, 10.0),
            defense: (self.defense + rng.random_range(-0.5..0.5)).clamp(0.0, 10.0),
            hunger_threshold: (self.hunger_threshold + rng.random_range(-5.0..5.0)).clamp(50.0, 150.0),
            ideal_temperature: (self.ideal_temperature + rng.random_range(-3.0..3.0)).clamp(0.0, 100.0),
            panic: (self.panic + rng.random_range(-0.5..0.5)).clamp(0.0, 10.0),
            hermaphrodite: if rng.random_range(0.0..1.0) < 0.03 { !self.hermaphrodite } else { self.hermaphrodite },
            neural_network: self.neural_network.mutate(neural_mutation_rate),
        }
    }

    pub fn get_color(&self, energy: f32, color_mode: crate::params::ColorVisualization) -> Color {
        use crate::params::ColorVisualization;
        
        let brightness = (energy / 100.0).clamp(0.3, 1.0);
        
        // Determine hue based on selected visualization mode
        let h = match color_mode {
            ColorVisualization::Diet => {
                // Color based on diet
                let base_h = match self.diet {
                    Diet::Herbivore => 120.0, // Green
                    Diet::Carnivore => 0.0,   // Red
                    Diet::Omnivore => 45.0,   // Orange
                };
                base_h + (self.color_hue * 0.15) // Add variation
            },
            ColorVisualization::Speed => {
                // 40-200 -> 240 (blue) to 0 (red)
                240.0 - ((self.speed - 40.0) / 160.0) * 240.0
            },
            ColorVisualization::Size => {
                // 5-20 -> 120 (green) to 300 (magenta)
                120.0 + ((self.size - 5.0) / 15.0) * 180.0
            },
            ColorVisualization::Sense => {
                // 40-300 -> 60 (yellow) to 180 (cyan)
                60.0 + ((self.sense_range - 40.0) / 260.0) * 120.0
            },
            ColorVisualization::Attack => {
                // 0-10 -> 120 (green) to 0 (red)
                120.0 - (self.attack / 10.0) * 120.0
            },
            ColorVisualization::Defense => {
                // 0-10 -> 180 (cyan) to 270 (blue)
                180.0 + (self.defense / 10.0) * 90.0
            },
            ColorVisualization::IdealTemp => {
                // 0-100 -> 240 (blue/cold) to 0 (red/hot)
                240.0 - (self.ideal_temperature / 100.0) * 240.0
            },
            ColorVisualization::Panic => {
                // 0-10 -> 300 (magenta/calm) to 60 (yellow/panicky)
                300.0 - (self.panic / 10.0) * 240.0
            },
            ColorVisualization::Energy => {
                // Use energy-based coloring
                return self.get_color(energy, ColorVisualization::Diet);
            },
        };
        
        let s = 0.7;
        let v = brightness;

        let c = v * s;
        let x = c * (1.0 - ((h / 60.0) % 2.0 - 1.0).abs());
        let m = v - c;

        let (r, g, b) = match h {
            h if h < 60.0 => (c, x, 0.0),
            h if h < 120.0 => (x, c, 0.0),
            h if h < 180.0 => (0.0, c, x),
            h if h < 240.0 => (0.0, x, c),
            h if h < 300.0 => (x, 0.0, c),
            _ => (c, 0.0, x),
        };

        Color::new(r + m, g + m, b + m, 1.0)
    }

    pub fn energy_drain(&self, params: &crate::params::SimParams) -> f32 {
        // Base energy cost
        let base_drain = self.size * 0.3 
            + self.speed * 0.01 
            + self.sense_range * 0.005
            + self.attack * 0.15
            + self.defense * 0.10;
        
        // Apply diet efficiency multiplier from params
        let diet_multiplier = match self.diet {
            Diet::Herbivore => params.herbivore_efficiency,
            Diet::Omnivore => params.omnivore_efficiency,
            Diet::Carnivore => params.carnivore_efficiency,
        };
        
        base_drain * diet_multiplier
    }
}

#[derive(Clone, Debug)]
pub struct Creature {
    pub energy: f32,
    pub age: f32,
    pub reproduction_cooldown: f32, // Time until can reproduce again
    pub digestion_cooldown: f32, // Time slowed down while digesting
    pub childhood_remaining: f32, // Time until creature can reproduce (childhood period)
    pub neural_network: NeuralNetwork,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PlantType {
    Basic,    // Normal energy
    Nutritious, // High energy but rare
    Toxic,    // Gives less energy, slightly harmful
}

#[derive(Clone, Copy, Debug)]
pub struct Plant {
    pub energy: f32,
    pub plant_type: PlantType,
}

#[derive(Clone, Copy, Debug)]
pub struct Obstacle {
    pub radius: f32,
}

