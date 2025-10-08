use eframe::egui;
use egui::{Color32, Pos2, Rect, Stroke, Vec2};
use egui_plot::{Line, Plot, PlotPoints};
use hecs::World;

use crate::components::*;
use crate::params::SimParams;
use crate::stats::SimStats;

// Camera state for the simulation view
pub struct Camera {
    pub x: f32,
    pub y: f32,
    pub zoom: f32,
}

impl Default for Camera {
    fn default() -> Self {
        Self {
            x: 800.0,
            y: 800.0,
            zoom: 1.0,
        }
    }
}

// Convert our Color to egui Color32
fn to_egui_color(color: crate::components::Color) -> Color32 {
    Color32::from_rgba_unmultiplied(
        (color.r * 255.0) as u8,
        (color.g * 255.0) as u8,
        (color.b * 255.0) as u8,
        (color.a * 255.0) as u8,
    )
}

pub fn render_simulation_view(
    ui: &mut egui::Ui,
    world: &World,
    camera: &mut Camera,
    params: &SimParams,
    drawing_obstacle: &mut bool,
    obstacle_start: &mut Option<crate::components::Position>,
) -> Option<(crate::components::Position, f32)> {
    let available_rect = ui.available_rect_before_wrap();
    let response = ui.allocate_rect(available_rect, egui::Sense::click_and_drag());
    
    let mut new_obstacle = None;

    // Helper to convert screen to world coordinates
    let screen_to_world = |screen_pos: Pos2| -> crate::components::Position {
        crate::components::Position {
            x: camera.x + (screen_pos.x - available_rect.left()) / camera.zoom,
            y: camera.y + (screen_pos.y - available_rect.top()) / camera.zoom,
        }
    };
    
    // Handle obstacle drawing (hold Ctrl and drag)
    let ctrl_held = ui.input(|i| i.modifiers.ctrl);
    
    if ctrl_held && response.drag_started() {
        *drawing_obstacle = true;
        if let Some(pos) = response.interact_pointer_pos() {
            *obstacle_start = Some(screen_to_world(pos));
        }
    }
    
    if *drawing_obstacle && response.drag_stopped() {
        if let (Some(start), Some(end_screen)) = (*obstacle_start, response.interact_pointer_pos()) {
            let end = screen_to_world(end_screen);
            let dx = end.x - start.x;
            let dy = end.y - start.y;
            let radius = (dx * dx + dy * dy).sqrt();
            new_obstacle = Some((start, radius));
        }
        *drawing_obstacle = false;
        *obstacle_start = None;
    }
    
    // Handle camera dragging (only when not drawing obstacles)
    if !ctrl_held && response.dragged() {
        let delta = response.drag_delta();
        camera.x -= delta.x / camera.zoom;
        camera.y -= delta.y / camera.zoom;
    }

    // Handle zoom with scroll
    if response.hovered() {
        let scroll_delta = ui.input(|i| i.smooth_scroll_delta.y);
        if scroll_delta != 0.0 {
            let zoom_factor = 1.0 + scroll_delta * 0.001;
            camera.zoom = (camera.zoom * zoom_factor).clamp(0.2, 3.0);
        }
    }

    // Allow camera to show padding around world
    let view_width = available_rect.width() / camera.zoom;
    let view_height = available_rect.height() / camera.zoom;
    let padding = 200.0; // Allow 200 units of padding on each side
    
    // Handle case where viewport is larger than world (when zoomed out or large window)
    let world_with_padding = params.world_size + padding * 2.0;
    
    if view_width < world_with_padding {
        camera.x = camera.x.clamp(-padding, params.world_size + padding - view_width);
    } else {
        // Center horizontally when viewport is larger than world
        camera.x = (params.world_size - view_width) / 2.0;
    }
    
    if view_height < world_with_padding {
        camera.y = camera.y.clamp(-padding, params.world_size + padding - view_height);
    } else {
        // Center vertically when viewport is larger than world
        camera.y = (params.world_size - view_height) / 2.0;
    }

    let painter = ui.painter_at(available_rect);

    // Draw dark background
    painter.rect_filled(available_rect, 0.0, Color32::from_rgb(15, 15, 25));

    // Helper to convert world coordinates to screen coordinates
    let world_to_screen = |world_x: f32, world_y: f32| -> Pos2 {
        Pos2::new(
            available_rect.left() + (world_x - camera.x) * camera.zoom,
            available_rect.top() + (world_y - camera.y) * camera.zoom,
        )
    };

    // Draw simulation world boundary
    let world_top_left = world_to_screen(0.0, 0.0);
    let world_bottom_right = world_to_screen(params.world_size, params.world_size);
    let world_rect = Rect::from_two_pos(world_top_left, world_bottom_right);
    
    // Fill world area with slightly lighter background
    painter.rect_filled(world_rect, 0.0, Color32::from_rgb(20, 20, 30));
    
    // Draw smooth temperature gradient (cold blue at top to hot red at bottom)
    let gradient_steps = 20;
    for i in 0..gradient_steps {
        let y_start = (i as f32 / gradient_steps as f32) * params.world_size;
        let y_end = ((i + 1) as f32 / gradient_steps as f32) * params.world_size;
        
        let temp_ratio = (i as f32 + 0.5) / gradient_steps as f32;
        
        // Interpolate from blue (cold) to red (hot)
        let blue_intensity = ((1.0 - temp_ratio) * 50.0) as u8;
        let red_intensity = (temp_ratio * 50.0) as u8;
        let green_intensity = 40u8;
        
        let gradient_color = Color32::from_rgba_unmultiplied(
            20 + red_intensity,
            20 + green_intensity,
            30 + blue_intensity,
            40,
        );
        
        let band_top_left = world_to_screen(0.0, y_start);
        let band_bottom_right = world_to_screen(params.world_size, y_end);
        let band_rect = Rect::from_two_pos(band_top_left, band_bottom_right);
        painter.rect_filled(band_rect, 0.0, gradient_color);
    }
    
    // Draw border around world
    painter.rect_stroke(world_rect, 0.0, Stroke::new(3.0, Color32::from_rgb(60, 60, 80)));
    
    // Draw obstacles as gray circles
    for (_id, (pos, obstacle)) in world.query::<(&crate::components::Position, &crate::components::Obstacle)>().iter() {
        let screen_pos = world_to_screen(pos.x, pos.y);
        let radius = obstacle.radius * camera.zoom;
        painter.circle_filled(screen_pos, radius, Color32::from_rgb(80, 80, 80));
        painter.circle_stroke(screen_pos, radius, Stroke::new(2.0, Color32::from_rgb(60, 60, 60)));
    }
    
    // Draw obstacle preview while drawing
    if *drawing_obstacle {
        if let (Some(start), Some(current_screen)) = (*obstacle_start, response.interact_pointer_pos()) {
            let screen_start = world_to_screen(start.x, start.y);
            let dx = current_screen.x - screen_start.x;
            let dy = current_screen.y - screen_start.y;
            let preview_radius = (dx * dx + dy * dy).sqrt();
            painter.circle_stroke(screen_start, preview_radius, Stroke::new(2.0, Color32::from_rgba_unmultiplied(200, 200, 200, 150)));
        }
    }

    // Draw plants as squares with different colors based on type
    for (_id, (pos, plant)) in world.query::<(&Position, &Plant)>().iter() {
        let screen_pos = world_to_screen(pos.x, pos.y);
        let size = 6.0 * camera.zoom;
        let rect = Rect::from_center_size(screen_pos, Vec2::splat(size));
        
        // Different colors for different plant types
        let color = match plant.plant_type {
            crate::components::PlantType::Basic => Color32::from_rgb(50, 205, 50), // Lime green
            crate::components::PlantType::Nutritious => Color32::from_rgb(100, 255, 100), // Bright green
            crate::components::PlantType::Toxic => Color32::from_rgb(139, 195, 74), // Yellow-green
        };
        
        painter.rect_filled(rect, 0.0, color);
    }

    // Draw creatures as triangles pointing in movement direction
    for (_id, (pos, vel, genome, creature)) in world.query::<(&Position, &Velocity, &Genome, &Creature)>().iter() {
        let screen_pos = world_to_screen(pos.x, pos.y);
        let size = genome.size * camera.zoom;
        let color = genome.get_color(creature.energy, params.color_mode);
        let egui_color = to_egui_color(color);

        // Calculate direction from velocity
        let speed = (vel.x * vel.x + vel.y * vel.y).sqrt();
        let (dir_x, dir_y) = if speed > 0.1 {
            (vel.x / speed, vel.y / speed)
        } else {
            (1.0, 0.0) // Default direction if not moving
        };

        // Create triangle points pointing in direction of movement
        let angle = dir_y.atan2(dir_x);
        let triangle_points = [
            // Front point
            Pos2::new(
                screen_pos.x + (angle.cos() * size * 1.5),
                screen_pos.y + (angle.sin() * size * 1.5),
            ),
            // Back left
            Pos2::new(
                screen_pos.x + ((angle + 2.5).cos() * size),
                screen_pos.y + ((angle + 2.5).sin() * size),
            ),
            // Back right
            Pos2::new(
                screen_pos.x + ((angle - 2.5).cos() * size),
                screen_pos.y + ((angle - 2.5).sin() * size),
            ),
        ];

        // Draw the triangle body
        painter.add(egui::Shape::convex_polygon(
            triangle_points.to_vec(),
            egui_color,
            Stroke::new(1.0, Color32::from_rgb(0, 0, 0)),
        ));

        // Draw sense range at higher zoom levels
        if camera.zoom > 0.5 {
            let sense_radius = genome.sense_range * camera.zoom;
            let mut sense_color = egui_color;
            sense_color = Color32::from_rgba_unmultiplied(
                sense_color.r(),
                sense_color.g(),
                sense_color.b(),
                20,
            );
            painter.circle_stroke(screen_pos, sense_radius, Stroke::new(1.0, sense_color));
        }
    }
    
    new_obstacle
}

pub fn render_left_panel(
    ctx: &egui::Context,
    world: &World,
    params: &mut SimParams,
    paused: &mut bool,
    restart_requested: &mut bool,
) {
    egui::SidePanel::left("left_panel")
        .resizable(false)
        .default_width(280.0)
        .show(ctx, |ui| {
            ui.vertical_centered(|ui| {
                ui.heading("PRIMORDIAL SOUP");
            });
            ui.add_space(10.0);

            let creature_count = world.query::<&Creature>().iter().count();
            let plant_count = world.query::<&Plant>().iter().count();

            ui.label(
                egui::RichText::new(format!("Creatures: {}", creature_count))
                    .color(Color32::YELLOW)
                    .size(18.0),
            );
            ui.label(
                egui::RichText::new(format!("Plants: {}", plant_count))
                    .color(Color32::GREEN)
                    .size(18.0),
            );
            ui.add_space(15.0);

            ui.separator();
            ui.heading("Controls");
            ui.add_space(5.0);

            // Pause button
            let pause_text = if *paused { "▶ Resume" } else { "⏸ Pause" };
            if ui.button(pause_text).clicked() {
                *paused = !*paused;
            }

            // Restart button
            if ui.button("🔄 Restart").clicked() {
                *restart_requested = true;
            }

            ui.add_space(5.0);

            // Simulation speed
            ui.horizontal(|ui| {
                ui.label("Speed:");
                for speed in 1..=5 {
                    if ui
                        .selectable_label(params.sim_speed == speed, format!("{}x", speed))
                        .clicked()
                    {
                        params.sim_speed = speed;
                    }
                }
            });

            ui.add_space(15.0);
            ui.separator();
            ui.heading("Parameters");
            ui.add_space(5.0);

            ui.label("Plant spawn rate (per second):");
            ui.add(
                egui::Slider::new(&mut params.plant_spawn_rate, 0.0..=1000.0)
                    .step_by(1.0)
                    .fixed_decimals(0),
            );

            ui.label("Plant energy:");
            ui.add(egui::Slider::new(&mut params.plant_energy, 0.0..=200.0).step_by(5.0));

            ui.label("Reproduction threshold:");
            ui.add(
                egui::Slider::new(&mut params.reproduction_threshold, 0.0..=500.0).step_by(10.0),
            );

            ui.label("Reproduction cost (asexual):");
            ui.add(egui::Slider::new(&mut params.reproduction_cost, 0.0..=400.0).step_by(10.0));
            
            ui.label("Reproduction cost (sexual, per parent):");
            ui.add(egui::Slider::new(&mut params.reproduction_cost_sexual, 0.0..=200.0).step_by(5.0));

            ui.add_space(10.0);
            
            ui.collapsing("🔧 Advanced Parameters", |ui| {
                ui.separator();
                ui.heading("Physics");
                ui.add_space(5.0);

                ui.label("Velocity damping near food:");
                ui.add(
                    egui::Slider::new(&mut params.velocity_damping, 0.0..=1.0)
                        .step_by(0.01)
                        .fixed_decimals(2),
                );

                ui.label("Energy drain rate:");
                ui.add(
                    egui::Slider::new(&mut params.energy_drain_multiplier, 0.1..=5.0)
                        .step_by(0.1)
                        .fixed_decimals(1),
                );
                
                ui.add_space(10.0);
                ui.separator();
                ui.heading("Diet Efficiency");
                ui.label(egui::RichText::new("Lower = less energy drain").size(11.0).color(egui::Color32::GRAY));
                ui.add_space(5.0);
                
                ui.label("Herbivore efficiency:");
                ui.add(egui::Slider::new(&mut params.herbivore_efficiency, 0.1..=2.0).step_by(0.05).fixed_decimals(2));
                
                ui.label("Carnivore efficiency:");
                ui.add(egui::Slider::new(&mut params.carnivore_efficiency, 0.1..=2.0).step_by(0.05).fixed_decimals(2));
                
                ui.label("Omnivore efficiency:");
                ui.add(egui::Slider::new(&mut params.omnivore_efficiency, 0.1..=2.0).step_by(0.05).fixed_decimals(2));
                
                ui.add_space(10.0);
                ui.separator();
                ui.heading("Digestion Times");
                ui.label(egui::RichText::new("Time slowed to 30% speed").size(11.0).color(egui::Color32::GRAY));
                ui.add_space(5.0);
                
                ui.label("Herbivore (plants):");
                ui.add(egui::Slider::new(&mut params.herbivore_plant_digestion, 0.0..=3.0).step_by(0.1).suffix("s"));
                
                ui.label("Carnivore (meat):");
                ui.add(egui::Slider::new(&mut params.carnivore_meat_digestion, 0.0..=5.0).step_by(0.1).suffix("s"));
                
                ui.label("Omnivore (plants):");
                ui.add(egui::Slider::new(&mut params.omnivore_plant_digestion, 0.0..=5.0).step_by(0.1).suffix("s"));
                
                ui.label("Omnivore (meat):");
                ui.add(egui::Slider::new(&mut params.omnivore_meat_digestion, 0.0..=8.0).step_by(0.1).suffix("s"));
                
                ui.add_space(10.0);
                ui.separator();
                ui.heading("Energy Transfer");
                ui.add_space(5.0);
                
                ui.label("Meat energy efficiency:");
                ui.add(egui::Slider::new(&mut params.meat_energy_efficiency, 0.0..=1.0).step_by(0.05).fixed_decimals(2));
                ui.label(egui::RichText::new("Lower = carnivores need more prey").size(11.0).color(egui::Color32::GRAY));
                
                ui.add_space(5.0);
                
                ui.label("Offspring starting energy:");
                ui.add(egui::Slider::new(&mut params.offspring_starting_energy, 10.0..=100.0).step_by(5.0).fixed_decimals(0));
                ui.label(egui::RichText::new("Lower = harder for newborns").size(11.0).color(egui::Color32::GRAY));
                
                ui.add_space(10.0);
                ui.separator();
                ui.heading("Environment");
                ui.add_space(5.0);
                
                ui.label("Plant clustering:");
                ui.add(egui::Slider::new(&mut params.plant_cluster_strength, 0.0..=1.0).step_by(0.05).fixed_decimals(2));
                ui.label(egui::RichText::new("Higher = more forests").size(11.0).color(egui::Color32::GRAY));
                
                ui.add_space(10.0);
                ui.separator();
                ui.heading("Neural Networks");
                ui.add_space(5.0);
                
                ui.checkbox(&mut params.use_neural_networks, "Enable Neural Networks");
                ui.label(egui::RichText::new("Toggle between deterministic and neural behavior").size(11.0).color(egui::Color32::GRAY));
                
                ui.add_space(5.0);
                
                ui.label("Neural mutation rate:");
                ui.add(egui::Slider::new(&mut params.neural_mutation_rate, 0.0..=0.5).step_by(0.01).fixed_decimals(2));
                ui.label(egui::RichText::new("Higher = more neural variation").size(11.0).color(egui::Color32::GRAY));
            });

    ui.add_space(10.0);

            ui.add_space(10.0);
            ui.separator();
            ui.heading("Visualization");
            ui.add_space(5.0);
            
            ui.label("Color creatures by:");
            egui::ComboBox::from_id_salt("color_mode")
                .selected_text(format!("{:?}", params.color_mode))
                .show_ui(ui, |ui| {
                    ui.selectable_value(&mut params.color_mode, crate::params::ColorVisualization::Diet, "Diet");
                    ui.selectable_value(&mut params.color_mode, crate::params::ColorVisualization::Speed, "Speed");
                    ui.selectable_value(&mut params.color_mode, crate::params::ColorVisualization::Size, "Size");
                    ui.selectable_value(&mut params.color_mode, crate::params::ColorVisualization::Sense, "Sense Range");
                    ui.selectable_value(&mut params.color_mode, crate::params::ColorVisualization::Attack, "Attack");
                    ui.selectable_value(&mut params.color_mode, crate::params::ColorVisualization::Defense, "Defense");
                    ui.selectable_value(&mut params.color_mode, crate::params::ColorVisualization::IdealTemp, "Ideal Temperature");
                    ui.selectable_value(&mut params.color_mode, crate::params::ColorVisualization::Panic, "Panic Level");
                    ui.selectable_value(&mut params.color_mode, crate::params::ColorVisualization::Energy, "Energy");
                });

            ui.collapsing("⌨️ Keyboard Shortcuts", |ui| {
                ui.label("Arrow Keys: Pan camera");
                ui.label("Scroll: Zoom in/out");
                ui.label("Space: Pause/Resume");
                ui.label("R: Restart");
                ui.label("1-5: Change speed");
                ui.label("Ctrl+Drag: Draw obstacle");
            });
        });
}

pub fn render_right_panel(ctx: &egui::Context, world: &World, stats: &SimStats) {
    egui::SidePanel::right("right_panel")
        .resizable(false)
        .default_width(300.0)
        .show(ctx, |ui| {
            ui.vertical_centered(|ui| {
                ui.heading("STATISTICS");
            });
            ui.add_space(10.0);

            // Generation counter
            ui.label(
                egui::RichText::new(format!("Generation: {}", stats.generation))
                    .color(Color32::YELLOW)
                    .size(18.0),
            );

            ui.add_space(5.0);

            // Birth/Death rates
            ui.horizontal(|ui| {
                ui.label(
                    egui::RichText::new(format!("Births/s: {}", stats.births_this_second * 2))
                        .color(Color32::GREEN),
                );
            });
            ui.horizontal(|ui| {
                ui.label(
                    egui::RichText::new(format!("Deaths/s: {}", stats.deaths_this_second * 2))
                        .color(Color32::RED),
                );
            });

    ui.add_space(10.0);
    ui.separator();

    // Calculate average traits and diet counts first
    let mut total_speed = 0.0;
    let mut total_size = 0.0;
    let mut total_sense = 0.0;
    let mut total_energy = 0.0;
    let mut total_attack = 0.0;
    let mut total_defense = 0.0;
    let mut total_ideal_temp = 0.0;
    let mut total_hunger_threshold = 0.0;
    let mut total_panic = 0.0;
    let mut count = 0;
    let mut herbivore_count = 0;
    let mut carnivore_count = 0;
    let mut omnivore_count = 0;
    let mut hermaphrodite_count = 0;

    for (_id, (genome, creature)) in world.query::<(&Genome, &Creature)>().iter() {
        total_speed += genome.speed;
        total_size += genome.size;
        total_sense += genome.sense_range;
        total_energy += creature.energy;
        total_attack += genome.attack;
        total_defense += genome.defense;
        total_ideal_temp += genome.ideal_temperature;
        total_hunger_threshold += genome.hunger_threshold;
        total_panic += genome.panic;
        count += 1;
        
        if genome.hermaphrodite {
            hermaphrodite_count += 1;
        }
        
        match genome.diet {
            crate::components::Diet::Herbivore => herbivore_count += 1,
            crate::components::Diet::Carnivore => carnivore_count += 1,
            crate::components::Diet::Omnivore => omnivore_count += 1,
        }
    }
    
    // Diet distribution
    ui.heading("Diet Distribution");
    ui.add_space(5.0);
    
    if count > 0 {
        ui.horizontal(|ui| {
            ui.label("🌿 Herbivores:");
            ui.label(
                egui::RichText::new(format!("{}", herbivore_count))
                    .color(egui::Color32::from_rgb(100, 200, 100)),
            );
        });
        ui.horizontal(|ui| {
            ui.label("🥩 Carnivores:");
            ui.label(
                egui::RichText::new(format!("{}", carnivore_count))
                    .color(egui::Color32::from_rgb(255, 100, 100)),
            );
        });
        ui.horizontal(|ui| {
            ui.label("🍖 Omnivores:");
            ui.label(
                egui::RichText::new(format!("{}", omnivore_count))
                    .color(egui::Color32::from_rgb(255, 180, 100)),
            );
        });
        
        ui.add_space(5.0);
        
        ui.horizontal(|ui| {
            ui.label("♾️ Asexual:");
            ui.label(
                egui::RichText::new(format!("{}", hermaphrodite_count))
                    .color(egui::Color32::from_rgb(200, 200, 200)),
            );
        });
        ui.horizontal(|ui| {
            ui.label("♂️♀️ Sexual:");
            ui.label(
                egui::RichText::new(format!("{}", count - hermaphrodite_count))
                    .color(egui::Color32::from_rgb(255, 150, 200)),
            );
        });
        
        ui.add_space(10.0);
        ui.separator();
        
        // Average traits section
                let avg_speed = total_speed / count as f32;
                let avg_size = total_size / count as f32;
                let avg_sense = total_sense / count as f32;
                let avg_energy = total_energy / count as f32;
                let avg_attack = total_attack / count as f32;
                let avg_defense = total_defense / count as f32;
                let avg_ideal_temp = total_ideal_temp / count as f32;
                let avg_hunger = total_hunger_threshold / count as f32;
                let avg_panic = total_panic / count as f32;

                ui.heading("Average Traits");
                ui.add_space(5.0);

                // Speed bar
                ui.horizontal(|ui| {
                    ui.label("Speed:");
                    ui.add(
                        egui::ProgressBar::new(avg_speed / 200.0)
                            .text(format!("{:.1}", avg_speed))
                            .fill(Color32::from_rgb(100, 150, 255)),
                    );
                });

                // Size bar
                ui.horizontal(|ui| {
                    ui.label("Size:");
                    ui.add(
                        egui::ProgressBar::new(avg_size / 20.0)
                            .text(format!("{:.1}", avg_size))
                            .fill(Color32::from_rgb(255, 150, 100)),
                    );
                });

                // Sense bar
                ui.horizontal(|ui| {
                    ui.label("Sense:");
                    ui.add(
                        egui::ProgressBar::new(avg_sense / 300.0)
                            .text(format!("{:.1}", avg_sense))
                            .fill(Color32::from_rgb(255, 255, 100)),
                    );
                });
                
                // Attack bar
                ui.horizontal(|ui| {
                    ui.label("Attack:");
                    ui.add(
                        egui::ProgressBar::new(avg_attack / 10.0)
                            .text(format!("{:.1}", avg_attack))
                            .fill(Color32::from_rgb(255, 100, 100)),
                    );
                });
                
                // Defense bar
                ui.horizontal(|ui| {
                    ui.label("Defense:");
                    ui.add(
                        egui::ProgressBar::new(avg_defense / 10.0)
                            .text(format!("{:.1}", avg_defense))
                            .fill(Color32::from_rgb(150, 150, 255)),
                    );
                });
                
                // Ideal Temperature bar
                ui.horizontal(|ui| {
                    ui.label("Ideal Temp:");
                    ui.add(
                        egui::ProgressBar::new(avg_ideal_temp / 100.0)
                            .text(format!("{:.1}°", avg_ideal_temp))
                            .fill(Color32::from_rgb(255, 180, 50)),
                    );
                });
                
                // Hunger threshold bar
                ui.horizontal(|ui| {
                    ui.label("Hunger:");
                    ui.add(
                        egui::ProgressBar::new(avg_hunger / 150.0)
                            .text(format!("{:.1}", avg_hunger))
                            .fill(Color32::from_rgb(200, 100, 200)),
                    );
                });
                
                // Panic bar
                ui.horizontal(|ui| {
                    ui.label("Panic:");
                    ui.add(
                        egui::ProgressBar::new(avg_panic / 10.0)
                            .text(format!("{:.1}", avg_panic))
                            .fill(Color32::from_rgb(255, 200, 100)),
                    );
                });

                ui.add_space(10.0);

                ui.label(
                    egui::RichText::new(format!("Average Energy: {:.1}", avg_energy))
                        .size(14.0),
                );
            }

            ui.add_space(15.0);
            ui.separator();

            // Population history graph
            if stats.population_history.len() > 1 {
                ui.heading("Population History");
                ui.add_space(5.0);

                let points: PlotPoints = stats
                    .population_history
                    .iter()
                    .enumerate()
                    .map(|(i, &pop)| [i as f64, pop as f64])
                    .collect();

                let line = Line::new(points)
                    .color(Color32::from_rgb(100, 200, 255))
                    .width(2.0);

                Plot::new("population_plot")
                    .height(150.0)
                    .show_axes([false, true])
                    .show_grid(false)
                    .show_background(true)
                    .allow_zoom(false)
                    .allow_drag(false)
                    .allow_boxed_zoom(false)
                    .allow_scroll(false)
                    .show(ui, |plot_ui| {
                        plot_ui.line(line);
                    });
            }
        });
}
