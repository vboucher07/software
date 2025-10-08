use hecs::{Entity, World};
use ::rand::Rng;

use crate::components::*;
use crate::params::SimParams;
use crate::stats::SimStats;

// ===== Helper Functions =====

pub fn distance_wrapped(p1: Position, p2: Position, world_size: f32) -> f32 {
    let dx = (p1.x - p2.x).abs().min(world_size - (p1.x - p2.x).abs());
    let dy = (p1.y - p2.y).abs().min(world_size - (p1.y - p2.y).abs());
    (dx * dx + dy * dy).sqrt()
}

// ===== Spawn Functions =====

pub fn spawn_creature(world: &mut World, pos: Position, genome: Genome, starting_energy: f32) {
    let neural_network = genome.neural_network.clone();
    world.spawn((
        pos,
        Velocity { x: 0.0, y: 0.0 },
        genome,
        Creature {
            energy: starting_energy,
            age: 0.0,
            reproduction_cooldown: 0.0,
            digestion_cooldown: 0.0,
            childhood_remaining: 10.0, // 10 seconds of childhood
            neural_network,
        },
    ));
}

// Helper function to check if position is valid (not in obstacles)
fn is_position_valid(pos: (f32, f32), obstacles: &[(Position, f32)], buffer: f32) -> bool {
    for &(obs_pos, obs_radius) in obstacles {
        let dx = pos.0 - obs_pos.x;
        let dy = pos.1 - obs_pos.y;
        let dist = (dx * dx + dy * dy).sqrt();
        if dist < obs_radius + buffer {
            return false;
        }
    }
    true
}

// Clean up plants that are inside obstacles
#[allow(dead_code)]
pub fn cleanup_plants_in_obstacles(world: &mut World) {
    use crate::components::Obstacle;
    
    // Collect obstacles
    let obstacles: Vec<(Position, f32)> = world
        .query::<(&Position, &Obstacle)>()
        .iter()
        .map(|(_, (pos, obs))| (*pos, obs.radius))
        .collect();
    
    if obstacles.is_empty() {
        return;
    }
    
    // Find plants to remove
    let mut plants_to_remove = Vec::new();
    for (plant_id, (plant_pos, _)) in world.query::<(&Position, &Plant)>().iter() {
        if !is_position_valid((plant_pos.x, plant_pos.y), &obstacles, 5.0) {
            plants_to_remove.push(plant_id);
        }
    }
    
    // Remove plants inside obstacles
    for plant_id in plants_to_remove {
        let _ = world.despawn(plant_id);
    }
}

pub fn spawn_plant(world: &mut World, params: &SimParams) {
    use crate::components::Obstacle;
    let mut rng = ::rand::rng();
    
    // Collect obstacles for collision checking
    let obstacles: Vec<(Position, f32)> = world
        .query::<(&Position, &Obstacle)>()
        .iter()
        .map(|(_, (pos, obs))| (*pos, obs.radius))
        .collect();
    
    // Try to find a valid position (up to 50 attempts for clustering)
    let (pos_x, pos_y) = 'find_position: loop {
        for _ in 0..50 {
            let candidate = if params.plant_cluster_strength > 0.0 && rng.random_range(0.0..1.0) < params.plant_cluster_strength {
                // Cluster near existing plants
                let existing_plants: Vec<Position> = world
                    .query::<&Position>()
                    .with::<&Plant>()
                    .iter()
                    .map(|(_, pos)| *pos)
                    .collect();
                
                if !existing_plants.is_empty() {
                    let parent_plant = existing_plants[rng.random_range(0..existing_plants.len())];
                    let offset_x = rng.random_range(-80.0..80.0);
                    let offset_y = rng.random_range(-80.0..80.0);
                    (
                        (parent_plant.x + offset_x).clamp(0.0, params.world_size),
                        (parent_plant.y + offset_y).clamp(0.0, params.world_size),
                    )
                } else {
                    (rng.random_range(0.0..params.world_size), rng.random_range(0.0..params.world_size))
                }
            } else {
                // Random spawn with temperate zone bias
                let zone_roll = rng.random_range(0.0..5.0);
                let pos_y = if zone_roll < 1.0 {
                    rng.random_range(0.0..params.world_size / 3.0)
                } else if zone_roll < 4.0 {
                    rng.random_range(params.world_size / 3.0..params.world_size * 2.0 / 3.0)
                } else {
                    rng.random_range(params.world_size * 2.0 / 3.0..params.world_size)
                };
                (rng.random_range(0.0..params.world_size), pos_y)
            };
            
            // Check if position collides with any obstacle
            if is_position_valid(candidate, &obstacles, 15.0) { // Increased buffer
                break 'find_position candidate;
            }
        }
        
        // If all attempts failed, just use a random position (fallback)
        break (rng.random_range(0.0..params.world_size), rng.random_range(0.0..params.world_size));
    };
    
    // Get temperature at this position
    let temp = crate::params::get_temperature_at_position(pos_x, pos_y, params.world_size);
    
    // Different plant types prefer different temperatures
    let plant_type = if temp < 35.0 {
        // Cold zone - mostly Toxic plants (hardy, cold-resistant)
        if rng.random_range(0.0..1.0) < 0.70 {
            crate::components::PlantType::Toxic
        } else if rng.random_range(0.0..1.0) < 0.20 {
            crate::components::PlantType::Nutritious
        } else {
            crate::components::PlantType::Basic
        }
    } else if temp < 65.0 {
        // Temperate zone - mostly Basic plants, some Nutritious
        if rng.random_range(0.0..1.0) < 0.70 {
            crate::components::PlantType::Basic
        } else if rng.random_range(0.0..1.0) < 0.20 {
            crate::components::PlantType::Nutritious
        } else {
            crate::components::PlantType::Toxic
        }
    } else {
        // Hot zone - mostly Nutritious (tropical abundance), some Toxic
        if rng.random_range(0.0..1.0) < 0.60 {
            crate::components::PlantType::Nutritious
        } else if rng.random_range(0.0..1.0) < 0.30 {
            crate::components::PlantType::Basic
        } else {
            crate::components::PlantType::Toxic
        }
    };
    
    let energy = match plant_type {
        crate::components::PlantType::Nutritious => params.plant_energy * 2.0,
        crate::components::PlantType::Toxic => params.plant_energy * 0.5,
        crate::components::PlantType::Basic => params.plant_energy,
    };
    
    world.spawn((
        Position {
            x: pos_x,
            y: pos_y,
        },
        Plant {
            energy,
            plant_type,
        },
    ));
}

// ===== Systems =====

// Neural network-based movement system
fn neural_movement_system(world: &mut World, params: &SimParams, dt: f32) {
    use crate::components::{Obstacle, Plant};
    
    // Collect all entities for neural network inputs
    let creatures: Vec<(Entity, Position, Velocity, Genome, Creature)> = world
        .query::<(&Position, &mut Velocity, &Genome, &Creature)>()
        .iter()
        .map(|(id, (pos, vel, genome, creature))| (id, *pos, *vel, genome.clone(), creature.clone()))
        .collect();
    
    let plants: Vec<(Position, Plant)> = world
        .query::<(&Position, &Plant)>()
        .iter()
        .map(|(_, (pos, plant))| (*pos, *plant))
        .collect();
    
    let obstacles: Vec<(Position, f32)> = world
        .query::<(&Position, &Obstacle)>()
        .iter()
        .map(|(_, (pos, obs))| (*pos, obs.radius))
        .collect();
    
    for (creature_id, pos, mut vel, genome, creature) in &creatures {
        // Prepare neural network inputs
        let mut nearest_food_dist = 500.0;
        let mut nearest_food_dir_x = 0.0;
        let mut nearest_food_dir_y = 0.0;
        let mut nearest_threat_dist = 500.0;
        let mut nearest_threat_dir_x = 0.0;
        let mut nearest_threat_dir_y = 0.0;
        
        // Find nearest food
        for (plant_pos, _plant) in &plants {
            let dx = plant_pos.x - pos.x;
            let dy = plant_pos.y - pos.y;
            let dist = (dx * dx + dy * dy).sqrt();
            
            if dist < nearest_food_dist {
                nearest_food_dist = dist;
                nearest_food_dir_x = if dist > 0.0 { dx / dist } else { 0.0 };
                nearest_food_dir_y = if dist > 0.0 { dy / dist } else { 0.0 };
            }
        }
        
        // Find nearest threat (larger creatures of different diet)
        for (other_id, other_pos, _other_vel, other_genome, _other_creature) in &creatures {
            if other_id == creature_id {
                continue;
            }
            
            let dx = other_pos.x - pos.x;
            let dy = other_pos.y - pos.y;
            let dist = (dx * dx + dy * dy).sqrt();
            
            // Check if this is a threat (different diet and larger)
            let is_threat = match (genome.diet, other_genome.diet) {
                (crate::components::Diet::Herbivore, crate::components::Diet::Carnivore) => other_genome.size > genome.size * 0.8,
                (crate::components::Diet::Carnivore, crate::components::Diet::Herbivore) => false, // Carnivores don't flee from herbivores
                (crate::components::Diet::Omnivore, crate::components::Diet::Carnivore) => other_genome.size > genome.size * 0.8,
                (crate::components::Diet::Carnivore, crate::components::Diet::Omnivore) => other_genome.size > genome.size * 0.8,
                _ => false,
            };
            
            if is_threat && dist < nearest_threat_dist {
                nearest_threat_dist = dist;
                nearest_threat_dir_x = if dist > 0.0 { dx / dist } else { 0.0 };
                nearest_threat_dir_y = if dist > 0.0 { dy / dist } else { 0.0 };
            }
        }
        
        // Get temperature at position
        let temperature = crate::params::get_temperature_at_position(pos.x, pos.y, params.world_size);
        
        // Prepare neural network inputs
        let inputs = [
            creature.energy,
            nearest_food_dist,
            nearest_food_dir_x,
            nearest_food_dir_y,
            nearest_threat_dist,
            nearest_threat_dir_x,
            nearest_threat_dir_y,
            temperature,
        ];
        
        // Get neural network output
        let (neural_vel_x, neural_vel_y) = creature.neural_network.forward(&inputs);
        
        // Scale neural output by creature speed
        let speed_multiplier = genome.speed * params.speed_multiplier;
        vel.x = neural_vel_x * speed_multiplier;
        vel.y = neural_vel_y * speed_multiplier;
        
        // Apply velocity damping
        vel.x *= params.velocity_damping;
        vel.y *= params.velocity_damping;
        
        // Update position
        let mut new_pos = *pos;
        new_pos.x += vel.x * dt;
        new_pos.y += vel.y * dt;
        
        // Handle obstacle collisions
        for (obs_pos, obs_radius) in &obstacles {
            let dx = new_pos.x - obs_pos.x;
            let dy = new_pos.y - obs_pos.y;
            let dist = (dx * dx + dy * dy).sqrt();
            
            if dist < obs_radius + genome.size {
                // Bounce off obstacle
                let push_dist = obs_radius + genome.size - dist;
                let push_x = if dist > 0.0 { dx / dist } else { 1.0 };
                let push_y = if dist > 0.0 { dy / dist } else { 0.0 };
                
                new_pos.x += push_x * push_dist;
                new_pos.y += push_y * push_dist;
                
                // Reverse velocity
                vel.x *= -0.5;
                vel.y *= -0.5;
            }
        }
        
        // Handle world boundaries (bounce)
        if new_pos.x < genome.size {
            new_pos.x = genome.size;
            vel.x = vel.x.abs() * 0.5;
        } else if new_pos.x > params.world_size - genome.size {
            new_pos.x = params.world_size - genome.size;
            vel.x = -vel.x.abs() * 0.5;
        }
        
        if new_pos.y < genome.size {
            new_pos.y = genome.size;
            vel.y = vel.y.abs() * 0.5;
        } else if new_pos.y > params.world_size - genome.size {
            new_pos.y = params.world_size - genome.size;
            vel.y = -vel.y.abs() * 0.5;
        }
        
        // Update position and velocity in world
        if let Ok(mut pos_comp) = world.get::<&mut Position>(*creature_id) {
            *pos_comp = new_pos;
        }
        if let Ok(mut vel_comp) = world.get::<&mut Velocity>(*creature_id) {
            *vel_comp = vel;
        }
        
        // Update creature age and cooldowns
        if let Ok(mut creature_comp) = world.get::<&mut Creature>(*creature_id) {
            creature_comp.age += dt;
            creature_comp.reproduction_cooldown = (creature_comp.reproduction_cooldown - dt).max(0.0);
            creature_comp.digestion_cooldown = (creature_comp.digestion_cooldown - dt).max(0.0);
            creature_comp.childhood_remaining = (creature_comp.childhood_remaining - dt).max(0.0);
            
            // Energy drain with temperature penalty
            let local_temp = crate::params::get_temperature_at_position(new_pos.x, new_pos.y, params.world_size);
            let temp_diff = (genome.ideal_temperature - local_temp).abs();
            let temp_penalty = temp_diff * 0.01; // Small penalty for temperature mismatch
            
            let energy_drain = genome.energy_drain(params) * params.energy_drain_multiplier + temp_penalty;
            creature_comp.energy = (creature_comp.energy - energy_drain * dt).max(0.0);
        }
    }
}

// ===== Systems =====

pub fn movement_system(world: &mut World, dt: f32, params: &SimParams) {
    if params.use_neural_networks {
        neural_movement_system(world, params, dt);
        return;
    }
    
    // Original deterministic movement system
    use crate::components::{Diet, Obstacle};
    
    // Collect all plant positions
    let plant_positions: Vec<Position> = world
        .query::<&Position>()
        .with::<&Plant>()
        .iter()
        .map(|(_, pos)| *pos)
        .collect();
    
    // Collect obstacles for collision detection
    let obstacles: Vec<(Position, f32)> = world
        .query::<(&Position, &Obstacle)>()
        .iter()
        .map(|(_, (pos, obs))| (*pos, obs.radius))
        .collect();

    // Collect all creature positions, sizes, diet, and entities for predator targeting
    let creature_data: Vec<(Entity, Position, f32, f32, Diet)> = world
        .query::<(&Position, &Genome, &Creature)>()
        .iter()
        .map(|(id, (pos, genome, creature))| (id, *pos, genome.size, creature.energy, genome.diet))
        .collect();

    for (my_id, (pos, vel, genome, creature)) in
        world.query_mut::<(&mut Position, &mut Velocity, &Genome, &mut Creature)>()
    {
        let is_hungry = creature.energy < genome.hunger_threshold;
        let mut nearest_food: Option<(Position, f32)> = None;
        let mut nearest_threat: Option<(Position, f32)> = None;
        
        // Check for threats (predators that can eat me)
        for &(other_id, other_pos, other_size, _other_energy, other_diet) in &creature_data {
            if my_id == other_id {
                continue;
            }
            
            // Check if this creature is a threat to me
            let is_threat = match other_diet {
                Diet::Carnivore => other_size > genome.size * 0.8, // Carnivores that are bigger
                Diet::Omnivore => other_size > genome.size * 1.3, // Only large omnivores
                Diet::Herbivore => false, // Herbivores are never threats
            };
            
            if is_threat {
                let dist = distance_wrapped(*pos, other_pos, params.world_size);
                if dist < genome.sense_range {
                    if let Some((_, nearest_dist)) = nearest_threat {
                        if dist < nearest_dist {
                            nearest_threat = Some((other_pos, dist));
                        }
                    } else {
                        nearest_threat = Some((other_pos, dist));
                    }
                }
            }
        }

        // Herbivores and omnivores look for plants
        if genome.diet == Diet::Herbivore || genome.diet == Diet::Omnivore {
            for plant_pos in &plant_positions {
                let dist = distance_wrapped(*pos, *plant_pos, params.world_size);
                if dist < genome.sense_range {
                    if let Some((_, nearest_dist)) = nearest_food {
                        if dist < nearest_dist {
                            nearest_food = Some((*plant_pos, dist));
                        }
                    } else {
                        nearest_food = Some((*plant_pos, dist));
                    }
                }
            }
        }

        // Carnivores and omnivores look for prey
        if genome.diet == Diet::Carnivore || genome.diet == Diet::Omnivore {
            for &(other_id, other_pos, other_size, _other_energy, other_diet) in &creature_data {
                if my_id == other_id {
                    continue; // Don't target self
                }
                
                // Don't target creatures of the same diet (prevents carnivore clustering)
                if other_diet == genome.diet {
                    continue;
                }
                
                // Target creatures that are smaller OR weaker
                // More aggressive targeting for carnivores
                let is_valid_prey = if genome.diet == Diet::Carnivore {
                    // Carnivores mainly target herbivores that are smaller or equal size
                    (other_diet == Diet::Herbivore && other_size < genome.size * 1.3) ||
                    // Or very small omnivores
                    (other_diet == Diet::Omnivore && other_size < genome.size * 0.7)
                } else {
                    // Omnivores only hunt much smaller herbivores
                    other_diet == Diet::Herbivore && other_size < genome.size * 0.7
                };
                
                if is_valid_prey {
                    let dist = distance_wrapped(*pos, other_pos, params.world_size);
                    if dist < genome.sense_range {
                        // Prefer closer prey
                        if let Some((_, nearest_dist)) = nearest_food {
                            if dist < nearest_dist {
                                nearest_food = Some((other_pos, dist));
                            }
                        } else {
                            nearest_food = Some((other_pos, dist));
                        }
                    }
                }
            }
        }

        let mut rng = ::rand::rng();
        
        // PANIC MODE: Flee from threats (overrides food seeking)
        if let Some((threat_pos, _)) = nearest_threat {
            // Calculate direction AWAY from threat
            let dx = pos.x - threat_pos.x;
            let dy = pos.y - threat_pos.y;
            
            let dist = (dx * dx + dy * dy).sqrt();
            if dist > 0.0 {
                // Flee strongly based on panic level
                let flee_strength = 300.0 * (genome.panic / 5.0); // Higher panic = faster fleeing
                vel.x += (dx / dist) * flee_strength * dt;
                vel.y += (dy / dist) * flee_strength * dt;
            }
        } 
        // Normal behavior: Seek food or wander
        else if let Some((target_pos, _dist_to_food)) = nearest_food {
            // Calculate direction to target (no wrapping needed now with boundaries)
            let dx = target_pos.x - pos.x;
            let dy = target_pos.y - pos.y;

            let dist = (dx * dx + dy * dy).sqrt();
            if dist > 0.0 {
                // Apply velocity damping when close to food
                if dist < genome.size * 5.0 {
                    vel.x *= params.velocity_damping;
                    vel.y *= params.velocity_damping;
                }
                
                // Reduce steering strength when close to food
                let distance_factor = if dist < genome.size * 4.0 {
                    (dist / (genome.size * 4.0)).clamp(0.1, 1.0)
                } else {
                    1.0
                };
                
                // Steer more aggressively when hungry
                let hunger_factor = if is_hungry { 1.5 } else { 1.0 };
                let steer_strength = 150.0 * hunger_factor * distance_factor; // Reduced from 200
                vel.x += (dx / dist) * steer_strength * dt;
                vel.y += (dy / dist) * steer_strength * dt;
            }
        } else {
            // Random wandering when no food in range
            vel.x += rng.random_range(-30.0..30.0) * dt;
            vel.y += rng.random_range(-30.0..30.0) * dt;
        }

        // Limit to max speed (reduced while digesting)
        let speed = (vel.x * vel.x + vel.y * vel.y).sqrt();
        let digestion_factor = if creature.digestion_cooldown > 0.0 { 0.3 } else { 1.0 };
        let max_speed = genome.speed * params.speed_multiplier * digestion_factor;
        if speed > max_speed {
            vel.x = (vel.x / speed) * max_speed;
            vel.y = (vel.y / speed) * max_speed;
        }

        // Update position
        pos.x += vel.x * dt;
        pos.y += vel.y * dt;

        // Bounce off world boundaries
        if pos.x < 0.0 {
            pos.x = 0.0;
            vel.x = -vel.x * 0.5;
        } else if pos.x > params.world_size {
            pos.x = params.world_size;
            vel.x = -vel.x * 0.5;
        }
        
        if pos.y < 0.0 {
            pos.y = 0.0;
            vel.y = -vel.y * 0.5;
        } else if pos.y > params.world_size {
            pos.y = params.world_size;
            vel.y = -vel.y * 0.5;
        }
        
        // Bounce off obstacles
        for &(obs_pos, obs_radius) in &obstacles {
            let dx = pos.x - obs_pos.x;
            let dy = pos.y - obs_pos.y;
            let dist = (dx * dx + dy * dy).sqrt();
            
            if dist < obs_radius + genome.size {
                // Push creature out of obstacle
                if dist > 0.0 {
                    let push_dist = obs_radius + genome.size - dist;
                    pos.x += (dx / dist) * push_dist;
                    pos.y += (dy / dist) * push_dist;
                    
                    // Bounce velocity
                    let dot = vel.x * dx + vel.y * dy;
                    vel.x = (vel.x - 2.0 * dot * dx / (dist * dist)) * 0.5;
                    vel.y = (vel.y - 2.0 * dot * dy / (dist * dist)) * 0.5;
                }
            }
        }

        // Energy drain with multiplier and temperature adaptation
        let base_drain = genome.energy_drain(params) * dt * params.energy_drain_multiplier;
        
        // Temperature penalty - creatures far from their ideal temperature lose more energy
        let local_temp = crate::params::get_temperature_at_position(pos.x, pos.y, params.world_size);
        let temp_diff = (local_temp - genome.ideal_temperature).abs();
        let temp_penalty = 1.0 + (temp_diff / 30.0); // +33% energy loss for every 10 degrees off
        
        creature.energy -= base_drain * temp_penalty;
        creature.age += dt;
        
        // Reduce cooldowns
        if creature.reproduction_cooldown > 0.0 {
            creature.reproduction_cooldown -= dt;
        }
        if creature.digestion_cooldown > 0.0 {
            creature.digestion_cooldown -= dt;
        }
    }
}

pub fn eating_system(world: &mut World, params: &SimParams) {
    use crate::components::{Diet, PlantType};
    
    let mut plants_to_remove = Vec::new();
    let mut creatures_to_feed = Vec::new();
    let mut creatures_to_remove = Vec::new();
    let mut creatures_to_slow = Vec::new(); // Track which creatures are eating

    // Collect creature data for collision checks  
    let creature_data: Vec<(Entity, Position, f32, Diet)> = world
        .query::<(&Position, &Genome, &Creature)>()
        .iter()
        .map(|(id, (pos, genome, _))| (id, *pos, genome.size, genome.diet))
        .collect();

    // Plants eating by herbivores and omnivores
    for (plant_id, (plant_pos, plant)) in world.query::<(&Position, &Plant)>().iter() {
        for &(creature_id, creature_pos, size, diet) in &creature_data {
            // Only herbivores and omnivores eat plants
            if diet == Diet::Carnivore {
                continue;
            }
            
            let dist = distance_wrapped(creature_pos, *plant_pos, params.world_size);
            // Much larger eating radius to prevent death-by-circling
            if dist < size + 15.0 {
                plants_to_remove.push(plant_id);
                
                // Different plant types have different effects
                let energy_gain = match plant.plant_type {
                    PlantType::Basic => plant.energy,
                    PlantType::Nutritious => plant.energy, // Already has 2x energy
                    PlantType::Toxic => plant.energy * 0.8, // Reduced effectiveness
                };
                
                creatures_to_feed.push((creature_id, energy_gain));
                
                // Digestion time based on diet specialization
                let digestion_time = match diet {
                    Diet::Herbivore => params.herbivore_plant_digestion,
                    Diet::Omnivore => params.omnivore_plant_digestion,
                    Diet::Carnivore => 0.0, // N/A but won't happen
                };
                creatures_to_slow.push((creature_id, digestion_time));
                break;
            }
        }
    }

    // Creature-on-creature predation
    let predator_prey_data: Vec<(Entity, Position, f32, f32, f32, Diet)> = world
        .query::<(&Position, &Genome, &Creature)>()
        .iter()
        .map(|(id, (pos, genome, _creature))| {
            (id, *pos, genome.size, genome.attack, genome.defense, genome.diet)
        })
        .collect();

    for &(predator_id, pred_pos, pred_size, pred_attack, _, pred_diet) in &predator_prey_data {
        // Only carnivores and omnivores can hunt
        if pred_diet == Diet::Herbivore {
            continue;
        }

        for &(prey_id, prey_pos, prey_size, _, prey_defense, prey_diet) in &predator_prey_data {
            if predator_id == prey_id {
                continue;
            }
            
            // Don't attack creatures of the same diet
            if pred_diet == prey_diet {
                continue;
            }

            let dist = distance_wrapped(pred_pos, prey_pos, params.world_size);
            
            // Check if predator is close enough to attack (larger radius for hunting)
            if dist < pred_size + prey_size + 10.0 {
                // Combat resolution: attack vs defense with size factor
                let attack_power = pred_attack + (pred_size * 0.5);
                let defense_power = prey_defense + (prey_size * 0.3);
                
                // Predator needs to be stronger to successfully hunt
                if attack_power > defense_power * 1.2 {
                    creatures_to_remove.push(prey_id);
                    
                    // Predator gains energy based on prey size - VERY inefficient
                    // This forces carnivores to depend on healthy herbivore populations
                    let energy_gain = prey_size * 8.0 * params.meat_energy_efficiency;
                    creatures_to_feed.push((predator_id, energy_gain));
                    
                    // Digestion time based on diet specialization
                    let digestion_time = match pred_diet {
                        Diet::Carnivore => params.carnivore_meat_digestion,
                        Diet::Omnivore => params.omnivore_meat_digestion,
                        Diet::Herbivore => 0.0, // N/A but won't happen
                    };
                    creatures_to_slow.push((predator_id, digestion_time));
                    break;
                }
            }
        }
    }

    // Apply energy gains and digestion slowdown
    for (creature_id, energy_gain) in creatures_to_feed {
        if let Ok(mut creature) = world.get::<&mut Creature>(creature_id) {
            creature.energy += energy_gain;
        }
    }
    
    for (creature_id, digestion_time) in creatures_to_slow {
        if let Ok(mut creature) = world.get::<&mut Creature>(creature_id) {
            creature.digestion_cooldown = digestion_time;
        }
    }

    // Remove eaten entities
    for plant_id in plants_to_remove {
        let _ = world.despawn(plant_id);
    }
    
    for creature_id in creatures_to_remove {
        let _ = world.despawn(creature_id);
    }
}

pub fn reproduction_system(world: &mut World, params: &SimParams, stats: &mut SimStats) {
    // Check current population - hard cap to prevent crashes
    let current_population = world.query::<&Creature>().iter().count();
    if current_population >= crate::params::MAX_POPULATION {
        return; // Don't reproduce if at cap
    }
    
    let mut offspring = Vec::new();
    let mut hermaphrodite_reproducers = Vec::new();
    let mut sexual_pairs = Vec::new();
    
    // Collect all creatures ready to reproduce
    let ready_creatures: Vec<(Entity, Position, Genome, bool)> = world
        .query::<(&Position, &Genome, &Creature)>()
        .iter()
        .filter(|(_, (_, genome, creature))| {
            let well_fed = creature.energy > genome.hunger_threshold * 1.2;
            let can_afford = creature.energy > params.reproduction_threshold;
            let cooldown_ready = creature.reproduction_cooldown <= 0.0;
            let not_a_child = creature.childhood_remaining <= 0.0; // Must be adult
            well_fed && can_afford && cooldown_ready && not_a_child
        })
        .map(|(id, (pos, genome, _))| (id, *pos, genome.clone(), genome.hermaphrodite))
        .collect();

    for (id, pos, genome, is_hermaphrodite) in &ready_creatures {
        // Stop if we've already queued enough offspring
        if current_population + offspring.len() >= crate::params::MAX_POPULATION {
            break;
        }
        
        if *is_hermaphrodite {
            // Asexual reproduction - can reproduce alone
            let mut child_genome = genome.mutate(params.neural_mutation_rate);
            child_genome.neural_network = genome.neural_network.mutate(params.neural_mutation_rate);
            offspring.push((*pos, child_genome));
            hermaphrodite_reproducers.push(*id);
        } else {
            // Sexual reproduction - need to find a partner
            let mate_range = 50.0; // Must be within 50 units
            
            for (other_id, other_pos, _other_genome, other_is_hermaphrodite) in &ready_creatures {
                if id == other_id || *other_is_hermaphrodite {
                    continue; // Skip self and hermaphrodites
                }
                
                let dist = distance_wrapped(*pos, *other_pos, params.world_size);
                if dist < mate_range {
                    // Found a mate! Both contribute to offspring
                    let mut child_genome = genome.mutate(params.neural_mutation_rate); // Could blend genes in future
                    child_genome.neural_network = genome.neural_network.mutate(params.neural_mutation_rate);
                    offspring.push((*pos, child_genome));
                    sexual_pairs.push((*id, *other_id));
                    break;
                }
            }
        }
    }

    // Spawn offspring
    for (mut pos, genome) in offspring {
        let mut rng = ::rand::rng();
        pos.x += rng.random_range(-20.0..20.0);
        pos.y += rng.random_range(-20.0..20.0);
        spawn_creature(world, pos, genome, params.offspring_starting_energy);
        stats.record_birth();
    }

    // Deduct reproduction cost and set cooldown for hermaphrodites
    for reproducer_id in hermaphrodite_reproducers {
        if let Ok(mut creature) = world.get::<&mut Creature>(reproducer_id) {
            creature.energy -= params.reproduction_cost;
            creature.reproduction_cooldown = 5.0;
        }
    }
    
    // Deduct cost from each sexual parent
    for (parent1_id, parent2_id) in sexual_pairs {
        if let Ok(mut creature) = world.get::<&mut Creature>(parent1_id) {
            creature.energy -= params.reproduction_cost_sexual;
            creature.reproduction_cooldown = 5.0;
        }
        if let Ok(mut creature) = world.get::<&mut Creature>(parent2_id) {
            creature.energy -= params.reproduction_cost_sexual;
            creature.reproduction_cooldown = 5.0;
        }
    }
}

pub fn death_system(world: &mut World, params: &SimParams, stats: &mut SimStats) {
    let mut to_remove = Vec::new();

    for (id, creature) in world.query::<&Creature>().iter() {
        if creature.energy <= 0.0 || creature.age > params.max_age {
            to_remove.push(id);
        }
    }

    for id in to_remove {
        let _ = world.despawn(id);
        stats.record_death();
    }
}

