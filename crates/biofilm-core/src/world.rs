use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

use crate::agent::Agent;
use crate::grid::{Grid, SoluteField};
use crate::reaction::{Monod, Reaction};

// ---- Unit convention (Task 1: reconciled) ------------------------------
//
// Internal unit system, used consistently everywhere in this module:
//   length         µm
//   time           day
//   agent mass     pg
//   solute conc.   g/m³
//
// Two unit gaps existed in Phase A and are fixed here:
//
// 1. Diffusivity time base. The schema/`simple.xml` layer supplies solute
//    diffusivities in µm²/s (the natural unit for a diffusion coefficient),
//    but the quasi-steady solver (`grid::solve_steady_state`) is run once
//    per `step(dt)` with `dt` in days and combines D with a sink that is
//    itself per-day (`bio_rate` below is pg/day, since `mu_max` in
//    `add_reaction` is a per-day specific rate). For `D*laplacian(C) = sink`
//    to balance dimensionally, D must also be per-day. We convert once,
//    where the field is built (`add_solute`), by the fixed factor
//    `SECONDS_PER_DAY = 86400.0`: `diff_liquid[µm²/day] = diff_liquid[µm²/s]
//    * 86400.0`. Without this conversion D is ~86400x too small relative to
//    the day-scaled sink, so diffusive replenishment cannot keep up with
//    consumption and the field is driven to zero (over-depletion).
//
// 2. Sink concentration units. `bio_rate` (from `Reaction::biomass_rate`) is
//    in pg/day; dividing by `cell_vol` (a µm² grid-cell area, used as a
//    stand-in "volume" for this 2D model) gives pg/µm²/day, which the
//    original code accumulated directly into a sink meant to be consumed
//    against a g/m³ field — a ~1e6 unit mismatch (1 pg/µm³ = 1e6 g/m³,
//    since 1 µm³ = 1e-18 m³ and 1 pg = 1e-12 g). We apply the missing
//    conversion factor `PG_PER_UM3_TO_G_PER_M3` explicitly (see `step`
//    below) so the sink accumulated in g/m³/day is commensurate with the
//    solute field it acts on.
//
// With both gaps closed, the dominant balance `D*laplacian(C) ~ sink` sets
// a penetration depth `~ sqrt(D * C_bulk / sink)` that is now a real
// (finite, non-degenerate) fraction of the domain height for the reference
// `simple.xml`-equivalent parameters, producing a partial (not flat, not
// fully-depleted) substrate gradient as the biofilm grows. See
// `tests/gradient.rs` for the validating assertions.
const SECONDS_PER_DAY: f64 = 86_400.0;
/// pg/µm³ -> g/m³ (1 µm³ = 1e-18 m³, 1 pg = 1e-12 g -> 1 pg/µm³ = 1e6 g/m³).
const PG_PER_UM3_TO_G_PER_M3: f64 = 1.0e6;

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

    // ---- PDE solver knobs (configurable via `set_pde_params`) -----------
    //
    // Defaults tuned for the simulation/visualization regime: with the
    // persistent warm-started field (`SoluteField.conc` carries over
    // between `step()` calls), the steady-state solve only needs to relax
    // away from the previous tick's near-converged field, not from a cold
    // start every time. A loose tolerance (1e-4 vs. the old 1e-7) and a
    // small max_iter budget (2000 vs. the old 20000) preserve the steady-
    // state gradient SHAPE (only trailing digits differ) while cutting
    // iterations-per-tick sharply. omega=1.8 is close to the SOR-optimal
    // 2/(1+sin(pi/N)) for grids in the N~32-256 range and must stay < 2.0
    // to avoid divergence.
    pde_tol: f64,
    pde_max_iter: usize,
    pde_omega: f64,
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
            pde_tol: 1e-4,
            pde_max_iter: 2_000,
            pde_omega: 1.8,
        }
    }

    // ---- Builders ----------------------------------------------------

    pub fn set_domain(&mut self, nx: usize, ny: usize, dx: f64, layer_thickness: f64) {
        self.grid = Grid::new(nx, ny, dx);
        self.layer_thickness = layer_thickness;
    }

    /// `diff_liquid`/`diff_biofilm` are given in µm²/s at the schema layer
    /// (the natural unit for a diffusion coefficient). Converted once here
    /// to µm²/day (`* SECONDS_PER_DAY`) so the field's diffusivity is
    /// dimensionally consistent with the day-scaled solver and sink — see
    /// the unit-convention note at the top of this file.
    pub fn add_solute(
        &mut self,
        name: &str,
        init: f64,
        diff_liquid: f64,
        diff_biofilm: f64,
        bulk: f64,
    ) -> usize {
        let diff_liquid_per_day = diff_liquid * SECONDS_PER_DAY;
        let diff_biofilm_per_day = diff_biofilm * SECONDS_PER_DAY;
        let mut field = SoluteField::new(&self.grid, init, diff_liquid_per_day);
        field.bulk = bulk;
        self.solutes.push(field);
        self.solute_names.push(name.to_string());
        self.diff_biofilm.push(diff_biofilm_per_day);
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

    /// Configure the red-black SOR solver knobs used by `step()`'s
    /// reaction-diffusion solve: `tol` (residual convergence threshold),
    /// `max_iter` (iteration budget per solute per tick), and `omega`
    /// (over-relaxation factor; must stay < 2.0 to avoid SOR divergence).
    /// Safe to call any time before or after `finalize`; defaults (set in
    /// `World::new`) are the fast values tol=1e-4, max_iter=2000,
    /// omega=1.8.
    pub fn set_pde_params(&mut self, tol: f64, max_iter: usize, omega: f64) {
        self.pde_tol = tol;
        self.pde_max_iter = max_iter;
        self.pde_omega = omega;
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
                    // sink [g/m³/day] = -coeff * bio_rate[pg/day] / cell_vol[µm²]
                    //                   * PG_PER_UM3_TO_G_PER_M3
                    // (`cell_vol` is a 2D µm² cell area used as a
                    // volume stand-in for this 2D model; see the unit note
                    // at the top of this file for the pg/µm³ -> g/m³
                    // derivation of the 1e6 factor. cell_vol/area convention
                    // is unchanged from Phase A — only the previously-
                    // missing 1e6 factor is new.)
                    sinks[k][self.grid.idx(i, j)] +=
                        -coeff * bio_rate / cell_vol * PG_PER_UM3_TO_G_PER_M3;
                }
            }
        }
        // 2. solve each solute field to steady state
        for (k, f) in self.solutes.iter_mut().enumerate() {
            crate::grid::solve_steady_state(
                f,
                &self.grid,
                &sinks[k],
                f.bulk,
                self.pde_omega,
                self.pde_tol,
                self.pde_max_iter,
            );
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

    /// Mean concentration of the named solute's field over grid row `j`
    /// (averaged over all `i`). Resolves `name` to a solute index via
    /// `solute_names`; panics if the name is unknown.
    pub fn solute_row_mean(&self, name: &str, j: usize) -> f64 {
        let k = self
            .solute_names
            .iter()
            .position(|n| n == name)
            .unwrap_or_else(|| panic!("unknown solute name: {name}"));
        let f = &self.solutes[k];
        let mut sum = 0.0;
        for i in 0..self.grid.nx {
            sum += f.conc[self.grid.idx(i, j)];
        }
        sum / self.grid.nx as f64
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
