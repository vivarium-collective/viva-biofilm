use rand::Rng;
use std::f64::consts::PI;

#[derive(Clone)]
pub struct Agent {
    pub x: f64,
    pub y: f64,
    pub mass: f64,
    pub species: u16,
}

impl Agent {
    /// Area-equivalent radius for a 2D coccoid: mass = π r² density.
    pub fn radius(&self, density: f64) -> f64 {
        (self.mass / (PI * density)).sqrt()
    }
}

pub fn grow(agents: &mut [Agent], rate_per_agent: &[f64], dt: f64) {
    for (a, &r) in agents.iter_mut().zip(rate_per_agent) {
        a.mass += r * dt;
        if a.mass < 0.0 {
            a.mass = 0.0;
        }
    }
}

pub fn divide(agents: &mut Vec<Agent>, division_mass: f64, rng: &mut impl Rng) {
    divide_with_density(agents, division_mass, 0.15, rng);
}

pub fn divide_with_density(
    agents: &mut Vec<Agent>,
    division_mass: f64,
    density: f64,
    rng: &mut impl Rng,
) {
    let n = agents.len();
    for i in 0..n {
        if agents[i].mass >= division_mass {
            let half = agents[i].mass / 2.0;
            agents[i].mass = half;
            let angle = rng.gen_range(0.0..(2.0 * PI));
            // place daughter one (post-split) radius away
            let r = (half / (PI * density)).sqrt();
            let daughter = Agent {
                x: agents[i].x + r * angle.cos(),
                y: (agents[i].y + r * angle.sin()).max(0.0),
                mass: half,
                species: agents[i].species,
            };
            agents.push(daughter);
        }
    }
}
