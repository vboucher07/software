mod components;
mod params;
mod stats;
mod systems;
mod render;

use eframe::egui;
use hecs::World;
use ::rand::Rng;

use components::*;
use params::*;
use stats::SimStats;
use systems::*;
use render::*;

struct EvolutionApp {
    world: World,
    params: SimParams,
    stats: SimStats,
    camera: Camera,
    plant_spawn_timer: f32,
    rate_reset_timer: f32,
    paused: bool,
    last_frame_time: std::time::Instant,
    restart_requested: bool,
    drawing_obstacle: bool,
    obstacle_start: Option<Position>,
}

impl Default for EvolutionApp {
    fn default() -> Self {
        let mut world = World::new();
        let params = SimParams::default();

        // Initial spawn
        for _ in 0..INITIAL_CREATURES {
            let mut rng = ::rand::rng();
            spawn_creature(
                &mut world,
                Position {
                    x: rng.random_range(0.0..params.world_size),
                    y: rng.random_range(0.0..params.world_size),
                },
                Genome::random(),
                100.0, // Initial creatures start with full energy
            );
        }

        for _ in 0..INITIAL_PLANTS {
            spawn_plant(&mut world, &params);
        }

        // Set camera to show whole world centered
        let mut camera = Camera::default();
        camera.x = 0.0;
        camera.y = 0.0;
        camera.zoom = 0.4; // Zoom out to see the whole world

        Self {
            world,
            params,
            stats: SimStats::new(),
            camera,
            plant_spawn_timer: 0.0,
            rate_reset_timer: 0.0,
            paused: false,
            last_frame_time: std::time::Instant::now(),
            restart_requested: false,
            drawing_obstacle: false,
            obstacle_start: None,
        }
    }
}

impl EvolutionApp {
    fn restart(&mut self) {
        self.world.clear();
        self.stats = SimStats::new();
        self.plant_spawn_timer = 0.0;
        self.rate_reset_timer = 0.0;
        // Don't reset camera - preserve user's view

            for _ in 0..INITIAL_CREATURES {
                let mut rng = ::rand::rng();
                spawn_creature(
                    &mut self.world,
                    Position {
                        x: rng.random_range(0.0..self.params.world_size),
                        y: rng.random_range(0.0..self.params.world_size),
                    },
                    Genome::random(),
                    100.0, // Initial creatures start with full energy
                );
            }

        for _ in 0..INITIAL_PLANTS {
            spawn_plant(&mut self.world, &self.params);
        }
    }

    fn handle_input(&mut self, ctx: &egui::Context) {
        // Keyboard shortcuts
        ctx.input(|i| {
            if i.key_pressed(egui::Key::Space) {
                self.paused = !self.paused;
            }
            if i.key_pressed(egui::Key::R) {
                self.restart_requested = true;
            }
            for (key, speed) in [
                (egui::Key::Num1, 1),
                (egui::Key::Num2, 2),
                (egui::Key::Num3, 3),
                (egui::Key::Num4, 4),
                (egui::Key::Num5, 5),
            ] {
                if i.key_pressed(key) {
                    self.params.sim_speed = speed;
                }
            }

            // Camera panning with arrow keys
            let pan_speed = 10.0 / self.camera.zoom;
            if i.key_down(egui::Key::ArrowLeft) {
                self.camera.x -= pan_speed;
            }
            if i.key_down(egui::Key::ArrowRight) {
                self.camera.x += pan_speed;
            }
            if i.key_down(egui::Key::ArrowUp) {
                self.camera.y -= pan_speed;
            }
            if i.key_down(egui::Key::ArrowDown) {
                self.camera.y += pan_speed;
            }
        });
    }

    fn update_simulation(&mut self, dt: f32) {
        if self.paused {
            return;
        }

        for _ in 0..self.params.sim_speed {
            movement_system(&mut self.world, dt, &self.params);
            eating_system(&mut self.world, &self.params);
            reproduction_system(&mut self.world, &self.params, &mut self.stats);
            death_system(&mut self.world, &self.params, &mut self.stats);

            self.plant_spawn_timer += dt;
            if self.plant_spawn_timer > 1.0 / self.params.plant_spawn_rate {
                spawn_plant(&mut self.world, &self.params);
                self.plant_spawn_timer = 0.0;
            }
        }

        // Update statistics
        let creature_count = self.world.query::<&Creature>().iter().count();
        self.stats.update(creature_count, dt);

        // Reset birth/death rates every half second
        self.rate_reset_timer += dt;
        if self.rate_reset_timer >= 0.5 {
            self.stats.reset_rates();
            self.rate_reset_timer = 0.0;
        }
    }
}

impl eframe::App for EvolutionApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Calculate delta time
        let now = std::time::Instant::now();
        let dt = (now - self.last_frame_time).as_secs_f32().min(0.05);
        self.last_frame_time = now;

        // Handle input
        self.handle_input(ctx);

        // Handle restart
        if self.restart_requested {
            self.restart();
            self.restart_requested = false;
        }

        // Update simulation
        self.update_simulation(dt);

        // Render UI panels
        render_left_panel(
            ctx,
            &self.world,
            &mut self.params,
            &mut self.paused,
            &mut self.restart_requested,
        );
        render_right_panel(ctx, &self.world, &self.stats);

        // Render simulation view in the center
        let new_obstacle = egui::CentralPanel::default().show(ctx, |ui| {
            render_simulation_view(ui, &self.world, &mut self.camera, &self.params, &mut self.drawing_obstacle, &mut self.obstacle_start)
        }).inner;
        
        // Add new obstacle if one was drawn
        if let Some((pos, radius)) = new_obstacle {
            if radius > 5.0 { // Minimum size
                self.world.spawn((pos, components::Obstacle { radius }));
            }
        }

        // Request continuous repainting for smooth animation
        ctx.request_repaint();
    }
}

fn main() -> eframe::Result<()> {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1600.0, 900.0])
            .with_title("Primordial Soup - Evolution Simulation"),
        ..Default::default()
    };

    eframe::run_native(
        "Primordial Soup",
        options,
        Box::new(|_cc| Ok(Box::new(EvolutionApp::default()))),
    )
}
