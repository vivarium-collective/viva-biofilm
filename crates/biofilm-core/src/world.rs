use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

use crate::agent::Agent;
use crate::grid::{Grid, SoluteField};
use crate::reaction::{Monod, Reaction};

/// A recorded request to spawn `n` agents randomly within a substratum-adjacent
/// band of height `band_height`, using an independent RNG stream derived from
/// `seed_offset` so multiple spawn groups don't collide. Actual placement is
/// deferred to `finalize(seed)` so that identical seeds reproduce identical runs.
struct SpawnSpec {
    n: usize,
    band_height: f64,
    seed_offset: u64,
}

pub struct World {
    grid: Grid,
    layer_thickness: f64,

    solutes: Vec<SoluteField>,
    solute_names: Vec<String>,
    diff_biofilm: Vec<f64>, // stored per Phase-A API contract; unused until Phase B

    reactions: Vec<Reaction>,

    density: f64,
    division_mass: f64,

    agents: Vec<Agent>,
    pending_spawns: Vec<SpawnSpec>,

    rng: ChaCha8Rng,
    seed: u64,

    time: f64,
}

impl Default for World {
    fn default() -> Self {
        World::new()
    }
}

impl World {
    pub fn new() -> Self {
        World {
            grid: Grid::new(1, 1, 1.0),
            layer_thickness: 0.0,
            solutes: Vec::new(),
            solute_names: Vec::new(),
            diff_biofilm: Vec::new(),
            reactions: Vec::new(),
            density: 0.15,
            division_mass: f64::INFINITY,
            agents: Vec::new(),
            pending_spawns: Vec::new(),
            rng: ChaCha8Rng::seed_from_u64(0),
            seed: 0,
            time: 0.0,
        }
    }

    // ---- Builders ----------------------------------------------------

    pub fn set_domain(&mut self, nx: usize, ny: usize, dx: f64, layer_thickness: f64) {
        self.grid = Grid::new(nx, ny, dx);
        self.layer_thickness = layer_thickness;
    }

    pub fn add_solute(
        &mut self,
        name: &str,
        init: f64,
        diff_liquid: f64,
        diff_biofilm: f64,
        bulk: f64,
    ) -> usize {
        let mut field = SoluteField::new(&self.grid, init, diff_liquid);
        field.bulk = bulk;
        self.solutes.push(field);
        self.solute_names.push(name.to_string());
        self.diff_biofilm.push(diff_biofilm);
        self.solutes.len() - 1
    }

    pub fn add_reaction(&mut self, mu_max: f64, monod_terms: Vec<(usize, f64)>, yields: Vec<(usize, f64)>) {
        self.reactions.push(Reaction {
            kinetics: Monod {
                mu_max,
                terms: monod_terms,
            },
            yield_per_solute: yields,
        });
    }

    pub fn set_species(&mut self, density: f64, division_mass: f64) {
        self.density = density;
        self.division_mass = division_mass;
    }

    /// Record intent to spawn `n` agents randomly placed within a band of
    /// height `band_height` above the substratum. Placement is deferred to
    /// `finalize(seed)` for reproducibility.
    pub fn spawn_agents(&mut self, n: usize, band_height: f64, seed_offset: u64) {
        self.pending_spawns.push(SpawnSpec {
            n,
            band_height,
            seed_offset,
        });
    }

    /// Seed the world's RNG and place all pending spawned agents. Each spawn
    /// group is placed with an independent RNG stream derived from
    /// `seed.wrapping_add(seed_offset)`, so identical `seed` always yields
    /// identical initial agent placement. The world's own `rng` field is
    /// seeded fresh from `seed` here and persists (advancing) across
    /// subsequent `step()` calls (e.g. for `divide_with_density`).
    pub fn finalize(&mut self, seed: u64) {
        self.seed = seed;
        self.rng = ChaCha8Rng::seed_from_u64(seed);

        let domain_x = self.grid.nx as f64 * self.grid.dx;
        for spec in self.pending_spawns.drain(..) {
            let mut local_rng = ChaCha8Rng::seed_from_u64(seed.wrapping_add(spec.seed_offset));
            for _ in 0..spec.n {
                let x = local_rng.gen_range(0.0..domain_x);
                let y = local_rng.gen_range(0.0..spec.band_height);
                self.agents.push(Agent {
                    x,
                    y,
                    mass: self.division_mass / 2.0,
                    species: 0,
                });
            }
        }
    }

    // ---- Simulation step -----------------------------------------------

    pub fn step(&mut self, dt: f64) {
        let ncell = self.grid.nx * self.grid.ny;
        // 1. accumulate per-cell sink for each solute (g/m³/day)
        let mut sinks: Vec<Vec<f64>> = self.solutes.iter().map(|_| vec![0.0; ncell]).collect();
        let cell_vol = self.grid.dx * self.grid.dx; // µm² (2D, unit depth)
        for a in &self.agents {
            let (i, j) = self.cell_of(a);
            let concs: Vec<f64> = self.solutes.iter().map(|f| f.conc[self.grid.idx(i, j)]).collect();
            for rxn in &self.reactions {
                let bio_rate = rxn.biomass_rate(a.mass, &concs); // pg/day
                for &(k, coeff) in &rxn.yield_per_solute {
                    // consumption of solute k: coeff (negative) * biomass rate, per cell volume
                    //
                    // UNIT NOTE (documented, not fixed — see TODO below): the
                    // internal unit system here is length=µm, mass=pg,
                    // time=day, and solute fields are carried in g/m³. This
                    // sink term is `-coeff * bio_rate / cell_vol`, i.e.
                    // (dimensionless yield coeff) * pg/day / µm² (`cell_vol`
                    // is a 2D µm² cell area, not a true µm³ volume), which
                    // works out to pg/µm²/day, not the g/m³/day the solute
                    // field expects. Converting pg/µm³ -> g/m³ requires a
                    // 1e6 factor (1 pg/µm³ = 1e6 g/m³, since 1 µm³ = 1e-18 m³
                    // and 1 pg = 1e-12 g), and there is currently a 2D-area
                    // vs 3D-volume mismatch on top of that (cell_vol should
                    // include the layer/unit depth if this is meant to model
                    // a true volumetric concentration).
                    // TODO(phase-b): reconcile this sink scaling (suspected
                    // ~1e6 gap, plus the area-vs-volume question) against the
                    // real iDynoMiCS-2 Java oracle once the biofilm-
                    // equivalence study lands (Phase A validates only the
                    // chemostat, which has no PDE/solute-sink path, so no
                    // current test exercises this number). Do not change the
                    // constant without a validating oracle comparison.
                    sinks[k][self.grid.idx(i, j)] += -coeff * bio_rate / cell_vol;
                }
            }
        }
        // 2. solve each solute field to steady state
        for (k, f) in self.solutes.iter_mut().enumerate() {
            crate::grid::solve_steady_state(f, &self.grid, &sinks[k], f.bulk, 1.4, 1e-7, 20_000);
        }
        // 3. grow agents at their local concentrations
        let rates: Vec<f64> = self
            .agents
            .iter()
            .map(|a| {
                let (i, j) = self.cell_of(a);
                let concs: Vec<f64> = self.solutes.iter().map(|f| f.conc[self.grid.idx(i, j)]).collect();
                self.reactions.iter().map(|r| r.biomass_rate(a.mass, &concs)).sum()
            })
            .collect();
        crate::agent::grow(&mut self.agents, &rates, dt);
        // 4. division, relaxation, detachment
        crate::agent::divide_with_density(&mut self.agents, self.division_mass, self.density, &mut self.rng);
        let domain_x = self.grid.nx as f64 * self.grid.dx;
        crate::relaxation::relax(&mut self.agents, self.density, domain_x, 30, 0.5);
        let max_h = self.grid.ny as f64 * self.grid.dx;
        crate::detachment::detach_above_height(&mut self.agents, max_h);
        self.time += dt;
    }

    /// Map an agent's continuous position to grid indices: x wraps
    /// (cyclic domain), y clamps to the top row.
    pub fn cell_of(&self, a: &Agent) -> (usize, usize) {
        let nx = self.grid.nx as isize;
        let ny = self.grid.ny;
        let dx = self.grid.dx;
        let mut i = (a.x / dx).floor() as isize % nx;
        if i < 0 {
            i += nx;
        }
        let j = (a.y / dx).floor();
        let j = if j < 0.0 { 0 } else { j as usize };
        let j = j.min(ny - 1);
        (i as usize, j)
    }

    // ---- Readback accessors ---------------------------------------------

    pub fn population(&self) -> usize {
        self.agents.len()
    }

    pub fn total_biomass(&self) -> f64 {
        self.agents.iter().map(|a| a.mass).sum()
    }

    pub fn biofilm_thickness(&self) -> f64 {
        self.agents.iter().map(|a| a.y).fold(0.0, f64::max)
    }

    pub fn solute_mean(&self, k: usize) -> f64 {
        let f = &self.solutes[k];
        f.conc.iter().sum::<f64>() / f.conc.len() as f64
    }

    pub fn agents(&self) -> &[Agent] {
        &self.agents
    }

    pub fn solute_field(&self, k: usize) -> &[f64] {
        &self.solutes[k].conc
    }

    pub fn time(&self) -> f64 {
        self.time
    }
}
