// ===== Statistics Tracking =====

pub struct SimStats {
    pub population_history: Vec<usize>,
    pub max_history_length: usize,
    pub births_this_second: usize,
    pub deaths_this_second: usize,
    pub generation: usize,
    pub time_since_last_stat: f32,
}

impl SimStats {
    pub fn new() -> Self {
        Self {
            population_history: Vec::new(),
            max_history_length: 200,
            births_this_second: 0,
            deaths_this_second: 0,
            generation: 1,
            time_since_last_stat: 0.0,
        }
    }

    pub fn update(&mut self, creature_count: usize, dt: f32) {
        self.time_since_last_stat += dt;
        if self.time_since_last_stat >= 0.5 {
            self.population_history.push(creature_count);
            if self.population_history.len() > self.max_history_length {
                self.population_history.remove(0);
            }
            self.time_since_last_stat = 0.0;
        }
    }

    pub fn record_birth(&mut self) {
        self.births_this_second += 1;
        self.generation += 1;
    }

    pub fn record_death(&mut self) {
        self.deaths_this_second += 1;
    }

    pub fn reset_rates(&mut self) {
        self.births_this_second = 0;
        self.deaths_this_second = 0;
    }
}

