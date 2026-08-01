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

/// Safety cap on the per-macro-step sink<->solve Picard coupling loop (see
/// `step`'s stage 1+2 note below). Gentle kinetics converge in ~1 inner
/// iteration; this bounds the cost of stiff/pathological kinetics that
/// converge slowly.
const MAX_COUPLING_ITERS: usize = 20;
/// Convergence threshold (g/m³) for the sink<->solve Picard loop: the max
/// absolute per-cell concentration change between successive inner
/// iterations must drop below this before the loop is considered converged.
const COUPLING_TOL: f64 = 1.0e-3;
/// Under-relaxation factor for the sink<->solve Picard update: each inner
/// iterate blends `COUPLING_RELAXATION` of the freshly solved candidate
/// field with `1 - COUPLING_RELAXATION` of the pre-solve field, rather than
/// taking the full Picard step. Required for stability: under Table-K-stiff
/// kinetics (steep Monod response, small Ks relative to bulk) the undamped
/// map `C -> solve(sink(C))` has its own period-2 orbit (empirically
/// verified) rather than converging, which both prevents the loop from ever
/// reaching `COUPLING_TOL` and — because `MAX_COUPLING_ITERS` is even —
/// deterministically strands the field on the unconstrained/"replenished"
/// phase every step, defeating the whole point of the coupling fix. 0.5 is
/// a standard, conservative damping factor for Picard/successive-
/// substitution iteration on strongly nonlinear reactive feedback; it does
/// not change the fixed point, only the (now convergent) path to it.
const COUPLING_RELAXATION: f64 = 0.5;

/// Newborn agents placed by `spawn_distributed` start at
/// `min(division_mass / 2, DISTRIBUTED_SEED_MASS_CAP)`. `spawn_agents` (the
/// old single-species path) keeps its original, uncapped `division_mass / 2`
/// formula for exact backward compatibility. The cap exists because
/// `spawn_distributed` is also used with deliberately huge `division_mass`
/// values (e.g. in growth-only tests, to suppress division) where
/// `division_mass / 2` would seed a physically absurd starting mass and
/// swamp the reaction-diffusion sink; 0.1 pg matches the typical starting
/// mass used across the existing single-species tests (`division_mass=0.2`).
const DISTRIBUTED_SEED_MASS_CAP: f64 = 0.1;

/// A recorded request to spawn `n` agents of a given `species` within a
/// substratum-adjacent band of height `band_height`, using an independent RNG
/// stream derived from `seed_offset` so multiple spawn groups don't collide.
/// Actual placement is deferred to `finalize(seed)` so that identical seeds
/// reproduce identical runs. `distributed` selects alternating/equidistant x
/// placement (`spawn_distributed`) vs. uniform-random x placement
/// (`spawn_agents`, the original behavior).
struct SpawnSpec {
    species: usize,
    n: usize,
    band_height: f64,
    seed_offset: u64,
    distributed: bool,
}

/// A strategy/species' own growth kinetics + morphology: its Monod
/// reaction(s) (consumption/growth rates), its packing density (mass ->
/// area-equivalent radius), and its division threshold mass. `World::species`
/// holds one of these per strategy; each `Agent::species` indexes into it.
pub struct Species {
    pub density: f64,
    pub division_mass: f64,
    pub reactions: Vec<Reaction>,
}

pub struct World {
    grid: Grid,
    layer_thickness: f64,

    solutes: Vec<SoluteField>,
    solute_names: Vec<String>,
    diff_biofilm: Vec<f64>, // stored per Phase-A API contract; unused until Phase B

    /// Per-species/strategy kinetics + morphology. `species[0]` is
    /// pre-populated in `new()` with the pre-Task-1 global defaults
    /// (density=0.15, division_mass=INFINITY, no reactions) so the old
    /// single-species API (`set_species`/`add_reaction`/`spawn_agents`, which
    /// always tags agents with `species: 0`) keeps working unchanged: it
    /// simply reads/writes `species[0]` instead of standalone globals.
    species: Vec<Species>,

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
            species: vec![Species {
                density: 0.15,
                division_mass: f64::INFINITY,
                reactions: Vec::new(),
            }],
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

    /// Backward-compat shim: appends to `species[0]` (auto-created with the
    /// original global defaults if somehow missing — `new()` always
    /// pre-populates it, so this is a defensive no-op in practice).
    pub fn add_reaction(&mut self, mu_max: f64, monod_terms: Vec<(usize, f64)>, yields: Vec<(usize, f64)>) {
        if self.species.is_empty() {
            self.species.push(Species {
                density: 0.15,
                division_mass: f64::INFINITY,
                reactions: Vec::new(),
            });
        }
        self.species[0].reactions.push(Reaction {
            kinetics: Monod {
                mu_max,
                terms: monod_terms,
            },
            yield_per_solute: yields,
        });
    }

    /// Appends a reaction to an existing species (see `add_species`).
    pub fn add_reaction_for(
        &mut self,
        species_idx: usize,
        mu_max: f64,
        monod_terms: Vec<(usize, f64)>,
        yields: Vec<(usize, f64)>,
    ) {
        self.species[species_idx].reactions.push(Reaction {
            kinetics: Monod {
                mu_max,
                terms: monod_terms,
            },
            yield_per_solute: yields,
        });
    }

    /// Registers a new strategy/species (empty reactions; add its kinetics
    /// via `add_reaction_for`). Returns the species index, to be passed to
    /// `add_reaction_for` and `spawn_distributed`.
    pub fn add_species(&mut self, density: f64, division_mass: f64) -> usize {
        self.species.push(Species {
            density,
            division_mass,
            reactions: Vec::new(),
        });
        self.species.len() - 1
    }

    /// Set a solute's `bulk` (the top-boundary Dirichlet value used by
    /// `step`'s steady-state solve). Lets an external process perturb the
    /// environment at runtime, e.g. via `BiofilmProcess`'s
    /// `boundary_concentrations` input.
    pub fn set_bulk(&mut self, k: usize, value: f64) {
        self.solutes[k].bulk = value;
    }

    /// Backward-compat shim: if `species` is empty, pushes a new one;
    /// otherwise sets `species[0]`'s density/division_mass.
    pub fn set_species(&mut self, density: f64, division_mass: f64) {
        if self.species.is_empty() {
            self.species.push(Species {
                density,
                division_mass,
                reactions: Vec::new(),
            });
        } else {
            self.species[0].density = density;
            self.species[0].division_mass = division_mass;
        }
    }

    /// Configure the red-black SOR solver knobs used by `step()`'s
    /// reaction-diffusion solve: `tol` (residual convergence threshold),
    /// `max_iter` (iteration budget per solute per tick), and `omega`
    /// (over-relaxation factor; must stay < 2.0 to avoid SOR divergence).
    /// Safe to call any time before or after `finalize`; defaults (set in
    /// `World::new`) are the fast values tol=1e-4, max_iter=2000,
    /// omega=1.8.
    ///
    /// # Panics
    /// Panics with a descriptive message if `omega` is outside `[1.0, 2.0)`
    /// (SOR requires `1 <= omega < 2` for guaranteed convergence; `>= 2.0`
    /// drives the iteration to divergence, `< 1.0` is under-relaxation, not
    /// the intended over-relaxation regime), if `tol <= 0.0`, or if
    /// `max_iter == 0`. This is a safety net for direct Rust callers; the
    /// pyo3 binding validates first and raises a `PyValueError` instead of
    /// letting an invalid value reach here.
    pub fn set_pde_params(&mut self, tol: f64, max_iter: usize, omega: f64) {
        assert!(
            (1.0..2.0).contains(&omega),
            "set_pde_params: omega must be in [1.0, 2.0) for SOR convergence, got {omega}"
        );
        assert!(tol > 0.0, "set_pde_params: tol must be > 0.0, got {tol}");
        assert!(max_iter > 0, "set_pde_params: max_iter must be > 0, got {max_iter}");
        self.pde_tol = tol;
        self.pde_max_iter = max_iter;
        self.pde_omega = omega;
    }

    /// Record intent to spawn `n` species-0 agents randomly placed within a
    /// band of height `band_height` above the substratum. Placement is
    /// deferred to `finalize(seed)` for reproducibility. Unchanged from
    /// pre-Task-1 behavior (species is always 0; x is uniform-random, not
    /// distributed/equidistant — see `spawn_distributed` for that).
    pub fn spawn_agents(&mut self, n: usize, band_height: f64, seed_offset: u64) {
        self.pending_spawns.push(SpawnSpec {
            species: 0,
            n,
            band_height,
            seed_offset,
            distributed: false,
        });
    }

    /// Record intent to spawn `n` agents of `species_idx`, placed at
    /// alternating, roughly equidistant x-positions along the substratum
    /// band (rather than `spawn_agents`' uniform-random x) — the "distributed
    /// alternating" spawner used for multi-strategy competition seeding.
    /// `y` is still randomized within `band_height`. Placement is deferred to
    /// `finalize(seed)` for reproducibility; `seed_offset`'s parity also
    /// phase-shifts the x-lattice half a spacing so that two calls with
    /// even/odd `seed_offset` interleave (e.g. RS at offset 0, YS at offset 1
    /// land in an alternating checkerboard along x).
    pub fn spawn_distributed(&mut self, species_idx: usize, n: usize, band_height: f64, seed_offset: u64) {
        self.pending_spawns.push(SpawnSpec {
            species: species_idx,
            n,
            band_height,
            seed_offset,
            distributed: true,
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
            let division_mass = self.species[spec.species].division_mass;
            if spec.distributed {
                // Alternating/equidistant x: n equally-spaced slots across
                // [0, domain_x), phase-shifted by half a slot when
                // seed_offset is odd so a second distributed group
                // interleaves with the first instead of overlapping it.
                let spacing = domain_x / spec.n.max(1) as f64;
                let phase = if spec.seed_offset % 2 == 0 { 0.5 } else { 1.0 };
                let seed_mass = (division_mass / 2.0).min(DISTRIBUTED_SEED_MASS_CAP);
                for i in 0..spec.n {
                    let x = ((i as f64 + phase) * spacing).rem_euclid(domain_x);
                    let y = local_rng.gen_range(0.0..spec.band_height);
                    self.agents.push(Agent {
                        x,
                        y,
                        mass: seed_mass,
                        species: spec.species as u16,
                    });
                }
            } else {
                for _ in 0..spec.n {
                    let x = local_rng.gen_range(0.0..domain_x);
                    let y = local_rng.gen_range(0.0..spec.band_height);
                    self.agents.push(Agent {
                        x,
                        y,
                        mass: division_mass / 2.0,
                        species: spec.species as u16,
                    });
                }
            }
        }
    }

    // ---- Simulation step -----------------------------------------------

    /// Accumulate the per-cell, per-solute sink (g/m³/day) from the CURRENT
    /// solute field: each agent consumes/produces according to its OWN
    /// species' reactions, but every agent's contribution lands in the same
    /// shared sink array (one substrate field, however many strategies are
    /// competing for it). Iterates `self.agents` in order (deterministic).
    /// Called once per inner Picard iteration in `step` (see its comment).
    fn build_sinks(&self, ncell: usize) -> Vec<Vec<f64>> {
        let mut sinks: Vec<Vec<f64>> = self.solutes.iter().map(|_| vec![0.0; ncell]).collect();
        let cell_vol = self.grid.dx * self.grid.dx; // µm² (2D, unit depth)
        for a in &self.agents {
            let (i, j) = self.cell_of(a);
            let concs: Vec<f64> = self.solutes.iter().map(|f| f.conc[self.grid.idx(i, j)]).collect();
            for rxn in &self.species[a.species as usize].reactions {
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
        sinks
    }

    pub fn step(&mut self, dt: f64) {
        let ncell = self.grid.nx * self.grid.ny;
        // 1+2. Iterate sink<->solve to self-consistency (damped Picard fixed
        //    point) before growth reads the field. Each inner pass rebuilds
        //    the per-cell sink from the CURRENT solute field (same per-agent
        //    Monod sink math as before, factored into `build_sinks`), solves
        //    each solute to steady state (warm-started from the running
        //    field, as before), then blends the solved candidate back toward
        //    the pre-solve field by `COUPLING_RELAXATION`. Recomputing the
        //    sink once and solving once — reading last step's leftover field
        //    — lags sink and field one macro-step apart; under stiff
        //    kinetics that lag forms a period-2 oxygen limit cycle that
        //    defeats Monod substrate limitation (see the Task-1 coupling
        //    brief). Plain (undamped) Picard iteration does not fix this on
        //    its own: under Table-K-stiff kinetics the sink is a steep
        //    function of concentration (Ks small relative to bulk), so the
        //    undamped map `C -> solve(sink(C))` itself has a period-2 orbit
        //    (verified empirically: sink swings ~1e3 vs ~1e6 g/m^3/day every
        //    other inner iteration, never damping, exhausting
        //    MAX_COUPLING_ITERS every step) — it just moves the same
        //    oscillation one level in, and because MAX_COUPLING_ITERS is
        //    even the loop always exits on the "replenished" (unconstrained)
        //    phase, making the runaway *worse* than the original bug (every
        //    step gets an unconstrained growth pulse instead of every other
        //    step). Under-relaxing the update — standard for Picard/
        //    successive-substitution iteration on strongly nonlinear
        //    reactive feedback — damps that orbit so the iteration actually
        //    converges to the true self-consistent partial-depletion fixed
        //    point. `C_{k+1} = C_k + COUPLING_RELAXATION * (solve(sink(C_k))
        //    - C_k)`, capped at `MAX_COUPLING_ITERS`, converged when the max
        //    per-cell change drops below `COUPLING_TOL`. This does not
        //    change the fixed point itself (only the convergence path), so
        //    gentle kinetics (which already have a stable, fast-converging
        //    map) still converge in a handful of cheap inner iterations to
        //    the same physical answer.
        let mut prev_conc: Vec<Vec<f64>> = self.solutes.iter().map(|f| f.conc.clone()).collect();
        for _coupling_iter in 0..MAX_COUPLING_ITERS {
            let sinks = self.build_sinks(ncell);
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
                for c in 0..ncell {
                    f.conc[c] = prev_conc[k][c] + COUPLING_RELAXATION * (f.conc[c] - prev_conc[k][c]);
                }
            }
            let mut max_delta: f64 = 0.0;
            for (k, f) in self.solutes.iter().enumerate() {
                for c in 0..ncell {
                    let d = (f.conc[c] - prev_conc[k][c]).abs();
                    if d > max_delta {
                        max_delta = d;
                    }
                }
                prev_conc[k].copy_from_slice(&f.conc);
            }
            if max_delta < COUPLING_TOL {
                break;
            }
        }
        // 3. grow agents at their local concentrations, by their own species'
        //    reactions (this is where Rate- vs Yield-Strategist divergence
        //    comes from: same local concentration, different mu_max/yield).
        let rates: Vec<f64> = self
            .agents
            .iter()
            .map(|a| {
                let (i, j) = self.cell_of(a);
                let concs: Vec<f64> = self.solutes.iter().map(|f| f.conc[self.grid.idx(i, j)]).collect();
                self.species[a.species as usize]
                    .reactions
                    .iter()
                    .map(|r| r.biomass_rate(a.mass, &concs))
                    .sum()
            })
            .collect();
        crate::agent::grow(&mut self.agents, &rates, dt);
        // 4. division: per-species division_mass + placement density. Grouped
        //    by species (each species' agents run through the unmodified
        //    `divide_with_density` as a homogeneous sub-population) rather
        //    than changed in-place, so the single-species case (exactly one
        //    species, i.e. all agents in one group, same relative order) is
        //    byte-for-byte the same call as before Task 1.
        let mut divided: Vec<Agent> = Vec::with_capacity(self.agents.len());
        for (s_idx, sp) in self.species.iter().enumerate() {
            let mut group: Vec<Agent> = self
                .agents
                .iter()
                .filter(|a| a.species as usize == s_idx)
                .cloned()
                .collect();
            if !group.is_empty() {
                crate::agent::divide_with_density(&mut group, sp.division_mass, sp.density, &mut self.rng);
            }
            divided.append(&mut group);
        }
        self.agents = divided;
        // 5. relaxation (shoving) + detachment: unchanged algorithms, one
        //    representative packing density. Heterogeneous per-species
        //    shoving mechanics is out of scope for this phase — the Fig-5
        //    RS/YS strategies share the same packing density, so this is not
        //    a limitation for that study; a fully per-agent-density shove
        //    would need `relaxation::relax`'s signature to change, which
        //    Task 1 deliberately leaves untouched.
        let domain_x = self.grid.nx as f64 * self.grid.dx;
        let shove_density = self.species.first().map(|s| s.density).unwrap_or(0.15);
        crate::relaxation::relax(&mut self.agents, shove_density, domain_x, 30, 0.5);
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

    /// Count agents of a given species.
    pub fn population_of(&self, species_idx: usize) -> usize {
        self.agents.iter().filter(|a| a.species as usize == species_idx).count()
    }

    /// Sum the mass of agents of a given species.
    pub fn biomass_of(&self, species_idx: usize) -> f64 {
        self.agents.iter().filter(|a| a.species as usize == species_idx).map(|a| a.mass).sum()
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
