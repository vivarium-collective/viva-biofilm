#[derive(Default)]
pub struct World {
    time: f64,
}

impl World {
    pub fn new() -> Self {
        World { time: 0.0 }
    }
    pub fn step(&mut self, dt: f64) {
        self.time += dt;
    }
    pub fn time(&self) -> f64 {
        self.time
    }
}
