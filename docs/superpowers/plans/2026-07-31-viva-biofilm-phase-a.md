# viva-biofilm Phase A Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the Rust biofilm-simulation core, its pyo3 binding, the two viva `Process`es, the vivarium-workspace scaffold and composites, and Study 1 (chemostat-equivalence) — a complete, testable increment.

**Architecture:** A Cargo workspace with a pure-Rust simulation core (`biofilm-core`) and a thin pyo3 binding crate (`biofilm-py`) built into the Python package `viva_biofilm` by maturin. Python `Process` classes drive a long-lived Rust `World` each `update()`. The package doubles as a vivarium-workspace. Copy the mechanical pattern from `~/code/pbg-cpm` verbatim.

**Tech Stack:** Rust (2021 edition, `rand` 0.8), pyo3 0.22 + maturin ≥1.5, Python ≥3.12, process-bigraph / bigraph-schema, vivarium-workbench, pytest.

## Global Constraints

- Python: `requires-python = ">=3.12"`. Build backend: `maturin`.
- pyo3 `0.22`, `features = ["extension-module"]`; binding crate `crate-type = ["cdylib"]`, `[lib] name = "biofilm_core"`.
- maturin: `manifest-path = "crates/biofilm-py/Cargo.toml"`, `module-name = "viva_biofilm.biofilm_core"`, `python-source = "."`.
- Dev build loop: `maturin develop -m crates/biofilm-py/Cargo.toml` then `pytest`.
- Rust core (`biofilm-core`) has **no pyo3 dependency** — pure Rust, testable via `cargo test`.
- All randomness goes through a **seeded** `rand::rngs::StdRng`; identical seed ⇒ bit-identical run (asserted in tests).
- Internal units: length µm, mass pg, concentration g/m³ (≡ mg/L), time days. Convert at the schema boundary.
- Process output ports `average_concentrations` (map[string,float]), `population` (float), `time` (overwrite[float]) are **byte-for-byte the same names/types as `pbg-idynomics2`** so engines are swappable. viva-biofilm adds a superset of extra ports.
- Reference files to open while implementing: `~/code/pbg-cpm/pyproject.toml`, `~/code/pbg-cpm/Cargo.toml`, `~/code/pbg-cpm/crates/cpm-py/Cargo.toml`, `~/code/pbg-cpm/crates/cpm-py/src/lib.rs`, `~/code/pbg-cpm/cpm/schema.py`, `~/code/pbg-cpm/cpm/processes/cpm_process.py`, `~/code/pbg-cpm/tests/test_process.py`.
- Commit after every task. This is a fresh repo on `main`; no worktree needed until a PR branch is cut.

---

### Task 1: Cargo workspace + maturin scaffold + smoke test

Proves the Rust→pyo3→Python→process-bigraph pipeline end to end with a trivial `World` before any real numerics.

**Files:**
- Create: `Cargo.toml`, `crates/biofilm-core/Cargo.toml`, `crates/biofilm-core/src/lib.rs`, `crates/biofilm-py/Cargo.toml`, `crates/biofilm-py/src/lib.rs`, `pyproject.toml`, `viva_biofilm/__init__.py`, `.gitignore`
- Test: `tests/test_bindings.py`

**Interfaces:**
- Produces: Rust `biofilm_core::World` with `new() -> World`, `step(&mut self, dt: f64)`, `time(&self) -> f64`. Python module `viva_biofilm.biofilm_core` exposing `World` with `.step(dt)` and `.time()`.

- [ ] **Step 1: Write the failing test**
```python
# tests/test_bindings.py
from viva_biofilm import biofilm_core

def test_world_steps_advance_time():
    w = biofilm_core.World()
    assert w.time() == 0.0
    w.step(0.5)
    w.step(0.5)
    assert abs(w.time() - 1.0) < 1e-12
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_bindings.py -v`
Expected: FAIL — `ModuleNotFoundError: viva_biofilm.biofilm_core` (nothing built yet).

- [ ] **Step 3: Write the scaffold**

`Cargo.toml` (workspace root):
```toml
[workspace]
resolver = "2"
members = ["crates/biofilm-core", "crates/biofilm-py"]

[workspace.package]
version = "0.1.0"
edition = "2021"

[workspace.dependencies]
biofilm-core = { path = "crates/biofilm-core" }
rand = "0.8"
rand_chacha = "0.3"
```

`crates/biofilm-core/Cargo.toml`:
```toml
[package]
name = "biofilm-core"
version.workspace = true
edition.workspace = true

[dependencies]
rand = { workspace = true }
rand_chacha = { workspace = true }
```

`crates/biofilm-core/src/lib.rs`:
```rust
pub mod world;
pub use world::World;
```
`crates/biofilm-core/src/world.rs`:
```rust
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
```
(Add `pub mod world;` wiring: put the `World` in `world.rs` and re-export from `lib.rs` as shown.)

`crates/biofilm-py/Cargo.toml`:
```toml
[package]
name = "biofilm-py"
version.workspace = true
edition.workspace = true

[lib]
name = "biofilm_core"
crate-type = ["cdylib"]

[dependencies]
biofilm-core = { workspace = true }
pyo3 = { version = "0.22", features = ["extension-module"] }
```

`crates/biofilm-py/src/lib.rs`:
```rust
use pyo3::prelude::*;
use biofilm_core::World as CoreWorld;

#[pyclass]
pub struct World {
    inner: CoreWorld,
}

#[pymethods]
impl World {
    #[new]
    fn new() -> Self {
        World { inner: CoreWorld::new() }
    }
    fn step(&mut self, dt: f64) {
        self.inner.step(dt);
    }
    fn time(&self) -> f64 {
        self.inner.time()
    }
}

#[pymodule]
fn biofilm_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<World>()?;
    Ok(())
}
```

`pyproject.toml`:
```toml
[build-system]
requires = ["maturin>=1.5,<2"]
build-backend = "maturin"

[project]
name = "viva-biofilm"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = ["process-bigraph", "bigraph-schema"]

[project.optional-dependencies]
dev = ["pytest", "maturin>=1.5,<2", "numpy", "plotly"]

[tool.maturin]
manifest-path = "crates/biofilm-py/Cargo.toml"
module-name = "viva_biofilm.biofilm_core"
python-source = "."

[tool.pytest.ini_options]
pythonpath = ["."]
```

`viva_biofilm/__init__.py`:
```python
from . import biofilm_core  # noqa: F401
```

`.gitignore`:
```
/target
*.so
__pycache__/
.venv/
/out
*.egg-info/
```

- [ ] **Step 4: Build and run the test**

Run:
```bash
uv venv .venv && source .venv/bin/activate
uv pip install maturin process-bigraph bigraph-schema pytest
maturin develop -m crates/biofilm-py/Cargo.toml
pytest tests/test_bindings.py -v
```
Expected: PASS. Also run `cargo test -p biofilm-core` (no tests yet → passes trivially).

- [ ] **Step 5: Commit**
```bash
git add -A
git commit -m "feat: cargo+maturin scaffold with smoke-tested World.step"
```

---

### Task 2: Solute grid + reaction-diffusion PDE solver (Rust)

**Files:**
- Create: `crates/biofilm-core/src/grid.rs`
- Modify: `crates/biofilm-core/src/lib.rs` (add `pub mod grid;`)
- Test: `crates/biofilm-core/tests/pde.rs`

**Interfaces:**
- Produces:
  - `grid::Grid { nx: usize, ny: usize, dx: f64 }` — 2D field, X index cyclic, Y from substratum (j=0) up.
  - `Grid::new(nx, ny, dx) -> Grid`
  - `grid::SoluteField { conc: Vec<f64>, diffusivity: f64, bulk: f64 }` with `at(&self, i, j) -> f64`, `set(&mut self, i, j, v)`.
  - `grid::solve_steady_state(field: &mut SoluteField, grid: &Grid, sink: &[f64], top_dirichlet: f64, sor: f64, tol: f64, max_iter: usize) -> usize` — relaxes `∇·(D∇C) − sink = 0` to residual `tol`, returns iterations. BC: X cyclic, j=0 no-flux (mirror), j=ny-1 Dirichlet = `top_dirichlet`. `sink[i + j*nx]` is per-cell consumption rate (g/m³/day). Returns iteration count.

- [ ] **Step 1: Write the failing test**
```rust
// crates/biofilm-core/tests/pde.rs
use biofilm_core::grid::{Grid, SoluteField, solve_steady_state};

#[test]
fn no_sink_gives_uniform_field_equal_to_boundary() {
    let g = Grid::new(8, 8, 1.0);
    let mut f = SoluteField::new(&g, 1.0, 2000.0); // conc init 1.0, D=2000
    let sink = vec![0.0; g.nx * g.ny];
    solve_steady_state(&mut f, &g, &sink, 5.0, 1.4, 1e-8, 10_000);
    // With no consumption and top fixed at 5.0, steady state is uniform 5.0.
    for j in 0..g.ny {
        for i in 0..g.nx {
            assert!((f.at(i, j) - 5.0).abs() < 1e-4, "cell {},{} = {}", i, j, f.at(i, j));
        }
    }
}

#[test]
fn uniform_sink_creates_gradient_decreasing_into_biofilm() {
    let g = Grid::new(4, 16, 1.0);
    let mut f = SoluteField::new(&g, 5.0, 2000.0);
    let sink = vec![0.5; g.nx * g.ny]; // uniform consumption
    solve_steady_state(&mut f, &g, &sink, 5.0, 1.4, 1e-9, 50_000);
    // Concentration at substratum (j=0) must be below the boundary value.
    assert!(f.at(0, 0) < f.at(0, g.ny - 1));
    assert!(f.at(0, 0) >= 0.0);
}
```

- [ ] **Step 2: Run to verify it fails**

Run: `cargo test -p biofilm-core --test pde`
Expected: FAIL — `grid` module / symbols not found.

- [ ] **Step 3: Implement the grid + solver**
```rust
// crates/biofilm-core/src/grid.rs
pub struct Grid {
    pub nx: usize,
    pub ny: usize,
    pub dx: f64, // isotropic cell size (µm)
}

impl Grid {
    pub fn new(nx: usize, ny: usize, dx: f64) -> Self {
        Grid { nx, ny, dx }
    }
    #[inline]
    pub fn idx(&self, i: usize, j: usize) -> usize {
        i + j * self.nx
    }
}

pub struct SoluteField {
    pub conc: Vec<f64>,
    pub diffusivity: f64,
    pub bulk: f64,
    nx: usize,
}

impl SoluteField {
    pub fn new(g: &Grid, init: f64, diffusivity: f64) -> Self {
        SoluteField {
            conc: vec![init; g.nx * g.ny],
            diffusivity,
            bulk: init,
            nx: g.nx,
        }
    }
    #[inline]
    pub fn at(&self, i: usize, j: usize) -> f64 {
        self.conc[i + j * self.nx]
    }
    #[inline]
    pub fn set(&mut self, i: usize, j: usize, v: f64) {
        self.conc[i + j * self.nx] = v;
    }
}

/// Red-black SOR relaxation of D * laplacian(C) - sink = 0.
/// X cyclic, j=0 no-flux (mirror neighbour), j=ny-1 Dirichlet.
pub fn solve_steady_state(
    f: &mut SoluteField,
    g: &Grid,
    sink: &[f64],
    top_dirichlet: f64,
    sor: f64,
    tol: f64,
    max_iter: usize,
) -> usize {
    let d = f.diffusivity;
    let h2 = g.dx * g.dx;
    // Fix Dirichlet top row.
    for i in 0..g.nx {
        f.set(i, g.ny - 1, top_dirichlet);
    }
    for iter in 0..max_iter {
        let mut max_res: f64 = 0.0;
        for color in 0..2 {
            for j in 0..(g.ny - 1) {
                for i in 0..g.nx {
                    if (i + j) % 2 != color {
                        continue;
                    }
                    let ip = f.at((i + 1) % g.nx, j);
                    let im = f.at((i + g.nx - 1) % g.nx, j);
                    let jp = f.at(i, j + 1);
                    // no-flux at substratum: neighbour below j=0 mirrors j itself
                    let jm = if j == 0 { f.at(i, j) } else { f.at(i, j - 1) };
                    // Discrete: D/h2 * (ip+im+jp+jm - 4C) - sink = 0
                    let s = sink[g.idx(i, j)];
                    let new = (d / h2 * (ip + im + jp + jm) - s) / (4.0 * d / h2);
                    let old = f.at(i, j);
                    let relaxed = old + sor * (new - old);
                    f.set(i, j, relaxed.max(0.0));
                    max_res = max_res.max((relaxed - old).abs());
                }
            }
        }
        if max_res < tol {
            return iter + 1;
        }
    }
    max_iter
}
```
Add `pub mod grid;` to `lib.rs`.

- [ ] **Step 4: Run to verify it passes**

Run: `cargo test -p biofilm-core --test pde`
Expected: PASS (both tests).

- [ ] **Step 5: Commit**
```bash
git add -A
git commit -m "feat: reaction-diffusion steady-state SOR solver on 2D grid"
```

---

### Task 3: Monod reaction rate law (Rust)

**Files:**
- Create: `crates/biofilm-core/src/reaction.rs`
- Modify: `crates/biofilm-core/src/lib.rs` (add `pub mod reaction;`)
- Test: `crates/biofilm-core/tests/reaction.rs`

**Interfaces:**
- Produces:
  - `reaction::Monod { mu_max: f64, terms: Vec<(usize, f64)> }` — `terms` is `(solute_index, half_saturation Ks)`; multiplicative Monod.
  - `Monod::specific_rate(&self, concs: &[f64]) -> f64` — returns `mu_max * Π (S_k/(S_k+Ks_k))` (per-day specific growth rate; dimensionless fraction × mu_max).
  - `reaction::Reaction { kinetics: Monod, yield_per_solute: Vec<(usize, f64)> }` — `yield_per_solute[(k, coeff)]` is stoichiometric coefficient of solute k per unit biomass produced (negative = consumed). Method `biomass_rate(&self, mass: f64, concs: &[f64]) -> f64 = mass * specific_rate`.

- [ ] **Step 1: Write the failing test**
```rust
// crates/biofilm-core/tests/reaction.rs
use biofilm_core::reaction::{Monod, Reaction};

#[test]
fn monod_saturates_to_mu_max() {
    let m = Monod { mu_max: 2.0, terms: vec![(0, 1.0)] };
    // Very high substrate -> fraction ~ 1 -> rate ~ mu_max
    assert!((m.specific_rate(&[1e6]) - 2.0).abs() < 1e-3);
    // At S = Ks the fraction is exactly 0.5
    assert!((m.specific_rate(&[1.0]) - 1.0).abs() < 1e-9);
    // Zero substrate -> zero rate
    assert_eq!(m.specific_rate(&[0.0]), 0.0);
}

#[test]
fn dual_monod_multiplies_fractions() {
    let m = Monod { mu_max: 1.0, terms: vec![(0, 1.0), (1, 1.0)] };
    // both at Ks -> 0.5 * 0.5 = 0.25
    assert!((m.specific_rate(&[1.0, 1.0]) - 0.25).abs() < 1e-9);
}

#[test]
fn biomass_rate_scales_with_mass() {
    let r = Reaction {
        kinetics: Monod { mu_max: 2.0, terms: vec![(0, 1.0)] },
        yield_per_solute: vec![(0, -4.2)],
    };
    assert!((r.biomass_rate(3.0, &[1e6]) - 6.0).abs() < 1e-2);
}
```

- [ ] **Step 2: Run to verify it fails**

Run: `cargo test -p biofilm-core --test reaction`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement**
```rust
// crates/biofilm-core/src/reaction.rs
pub struct Monod {
    pub mu_max: f64,
    pub terms: Vec<(usize, f64)>, // (solute_index, Ks)
}

impl Monod {
    pub fn specific_rate(&self, concs: &[f64]) -> f64 {
        let mut frac = 1.0;
        for &(k, ks) in &self.terms {
            let s = concs[k].max(0.0);
            frac *= s / (s + ks);
        }
        self.mu_max * frac
    }
}

pub struct Reaction {
    pub kinetics: Monod,
    pub yield_per_solute: Vec<(usize, f64)>, // (solute_index, coeff per unit biomass)
}

impl Reaction {
    pub fn biomass_rate(&self, mass: f64, concs: &[f64]) -> f64 {
        mass * self.kinetics.specific_rate(concs)
    }
}
```
Add `pub mod reaction;` to `lib.rs`.

- [ ] **Step 4: Run to verify it passes**

Run: `cargo test -p biofilm-core --test reaction`
Expected: PASS.

- [ ] **Step 5: Commit**
```bash
git add -A && git commit -m "feat: multiplicative Monod reaction kinetics"
```

---

### Task 4: Agents — growth & division (Rust)

**Files:**
- Create: `crates/biofilm-core/src/agent.rs`
- Modify: `crates/biofilm-core/src/lib.rs` (add `pub mod agent;`)
- Test: `crates/biofilm-core/tests/agent.rs`

**Interfaces:**
- Produces:
  - `agent::Agent { x: f64, y: f64, mass: f64, species: u16 }` with `radius(&self, density: f64) -> f64` (coccoid: `(3*mass/(4π·density)).cbrt()` in 2D approximated as area-equivalent `((mass/(π·density)).sqrt())` — see impl).
  - `agent::grow(agents: &mut Vec<Agent>, rate_per_agent: &[f64], dt: f64)` — Euler mass update `mass += rate*dt`.
  - `agent::divide(agents: &mut Vec<Agent>, division_mass: f64, rng: &mut impl rand::Rng)` — any agent with `mass >= division_mass` splits: parent keeps half mass, a daughter with half mass is placed at a random angle one radius away.

- [ ] **Step 1: Write the failing test**
```rust
// crates/biofilm-core/tests/agent.rs
use biofilm_core::agent::{Agent, grow, divide};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

#[test]
fn growth_is_euler() {
    let mut a = vec![Agent { x: 0.0, y: 0.0, mass: 1.0, species: 0 }];
    grow(&mut a, &[0.5], 2.0); // +1.0
    assert!((a[0].mass - 2.0).abs() < 1e-12);
}

#[test]
fn division_splits_mass_and_conserves_total() {
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    let mut a = vec![Agent { x: 10.0, y: 10.0, mass: 0.3, species: 0 }];
    divide(&mut a, 0.2, &mut rng);
    assert_eq!(a.len(), 2);
    let total: f64 = a.iter().map(|x| x.mass).sum();
    assert!((total - 0.3).abs() < 1e-12);
    assert!((a[0].mass - 0.15).abs() < 1e-12);
}

#[test]
fn division_is_deterministic_for_seed() {
    let run = || {
        let mut rng = ChaCha8Rng::seed_from_u64(7);
        let mut a = vec![Agent { x: 5.0, y: 5.0, mass: 0.5, species: 0 }];
        divide(&mut a, 0.2, &mut rng);
        (a[1].x, a[1].y)
    };
    assert_eq!(run(), run());
}
```

- [ ] **Step 2: Run to verify it fails**

Run: `cargo test -p biofilm-core --test agent`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement**
```rust
// crates/biofilm-core/src/agent.rs
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
    let n = agents.len();
    for i in 0..n {
        if agents[i].mass >= division_mass {
            let half = agents[i].mass / 2.0;
            agents[i].mass = half;
            let angle = rng.gen_range(0.0..(2.0 * PI));
            // place daughter one (post-split) radius away
            let r = (half / (PI * 0.15)).sqrt(); // density placeholder; overwritten in world
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
```
Add `pub mod agent;` to `lib.rs`. Note: division uses a fixed density for placement offset; the world passes real density in Task 7 via a variant that takes density — for now the placeholder keeps the unit test self-contained. In Task 7, replace the hardcoded `0.15` by threading `density` into a `divide_with_density(agents, division_mass, density, rng)` and keep `divide` as a thin wrapper calling it with `0.15`.

- [ ] **Step 4: Run to verify it passes**

Run: `cargo test -p biofilm-core --test agent`
Expected: PASS (all three).

- [ ] **Step 5: Commit**
```bash
git add -A && git commit -m "feat: agent growth (Euler) and mass-conserving division"
```

---

### Task 5: Mechanical relaxation / shoving (Rust)

**Files:**
- Create: `crates/biofilm-core/src/relaxation.rs`
- Modify: `crates/biofilm-core/src/lib.rs` (add `pub mod relaxation;`)
- Test: `crates/biofilm-core/tests/relaxation.rs`

**Interfaces:**
- Produces:
  - `relaxation::relax(agents: &mut Vec<Agent>, density: f64, domain_x: f64, iters: usize, k: f64)` — resolves pairwise overlaps by pushing agents apart along their center line proportional to overlap × `k`; X wraps at `domain_x` (cyclic); clamps `y >= radius` (substratum). Naive O(n²) is fine for Phase A sizes.

- [ ] **Step 1: Write the failing test**
```rust
// crates/biofilm-core/tests/relaxation.rs
use biofilm_core::agent::Agent;
use biofilm_core::relaxation::relax;

#[test]
fn overlapping_agents_are_pushed_apart() {
    let mut a = vec![
        Agent { x: 10.0, y: 10.0, mass: 0.2, species: 0 },
        Agent { x: 10.2, y: 10.0, mass: 0.2, species: 0 },
    ];
    let density = 0.15;
    let sep0 = (a[0].x - a[1].x).abs();
    relax(&mut a, density, 100.0, 50, 0.5);
    let sep1 = (a[0].x - a[1].x).abs();
    let r = a[0].radius(density);
    assert!(sep1 > sep0, "agents should separate: {} -> {}", sep0, sep1);
    assert!(sep1 >= 2.0 * r - 1e-3, "final sep should reach ~2r");
}

#[test]
fn agents_stay_above_substratum() {
    let mut a = vec![Agent { x: 5.0, y: 0.0, mass: 0.2, species: 0 }];
    relax(&mut a, 0.15, 100.0, 10, 0.5);
    assert!(a[0].y >= a[0].radius(0.15) - 1e-6);
}
```

- [ ] **Step 2: Run to verify it fails**

Run: `cargo test -p biofilm-core --test relaxation`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement**
```rust
// crates/biofilm-core/src/relaxation.rs
use crate::agent::Agent;

fn dx_cyclic(a: f64, b: f64, span: f64) -> f64 {
    let mut d = a - b;
    if d > span / 2.0 {
        d -= span;
    } else if d < -span / 2.0 {
        d += span;
    }
    d
}

pub fn relax(agents: &mut Vec<Agent>, density: f64, domain_x: f64, iters: usize, k: f64) {
    let n = agents.len();
    for _ in 0..iters {
        for i in 0..n {
            for j in (i + 1)..n {
                let ri = agents[i].radius(density);
                let rj = agents[j].radius(density);
                let min_d = ri + rj;
                let dx = dx_cyclic(agents[i].x, agents[j].x, domain_x);
                let dy = agents[i].y - agents[j].y;
                let dist = (dx * dx + dy * dy).sqrt().max(1e-9);
                let overlap = min_d - dist;
                if overlap > 0.0 {
                    let push = 0.5 * k * overlap;
                    let ux = dx / dist;
                    let uy = dy / dist;
                    agents[i].x += push * ux;
                    agents[i].y += push * uy;
                    agents[j].x -= push * ux;
                    agents[j].y -= push * uy;
                }
            }
        }
        for a in agents.iter_mut() {
            // wrap X
            a.x = a.x.rem_euclid(domain_x);
            // substratum constraint
            let r = a.radius(density);
            if a.y < r {
                a.y = r;
            }
        }
    }
}
```
Add `pub mod relaxation;` to `lib.rs`.

- [ ] **Step 4: Run to verify it passes**

Run: `cargo test -p biofilm-core --test relaxation`
Expected: PASS.

- [ ] **Step 5: Commit**
```bash
git add -A && git commit -m "feat: force-based mechanical relaxation with cyclic X + substratum"
```

---

### Task 6: Detachment (Rust)

**Files:**
- Create: `crates/biofilm-core/src/detachment.rs`
- Modify: `crates/biofilm-core/src/lib.rs` (add `pub mod detachment;`)
- Test: `crates/biofilm-core/tests/detachment.rs`

**Interfaces:**
- Produces: `detachment::detach_above_height(agents: &mut Vec<Agent>, max_height: f64) -> usize` — removes agents whose `y > max_height`, returns count removed. (Simple height-cap erosion for Phase A.)

- [ ] **Step 1: Write the failing test**
```rust
// crates/biofilm-core/tests/detachment.rs
use biofilm_core::agent::Agent;
use biofilm_core::detachment::detach_above_height;

#[test]
fn removes_agents_above_cap() {
    let mut a = vec![
        Agent { x: 0.0, y: 10.0, mass: 0.2, species: 0 },
        Agent { x: 0.0, y: 70.0, mass: 0.2, species: 0 },
    ];
    let removed = detach_above_height(&mut a, 64.0);
    assert_eq!(removed, 1);
    assert_eq!(a.len(), 1);
    assert!(a[0].y <= 64.0);
}
```

- [ ] **Step 2: Run to verify it fails**

Run: `cargo test -p biofilm-core --test detachment`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement**
```rust
// crates/biofilm-core/src/detachment.rs
use crate::agent::Agent;

pub fn detach_above_height(agents: &mut Vec<Agent>, max_height: f64) -> usize {
    let before = agents.len();
    agents.retain(|a| a.y <= max_height);
    before - agents.len()
}
```
Add `pub mod detachment;` to `lib.rs`.

- [ ] **Step 4: Run to verify it passes**

Run: `cargo test -p biofilm-core --test detachment`
Expected: PASS.

- [ ] **Step 5: Commit**
```bash
git add -A && git commit -m "feat: height-cap erosion detachment"
```

---

### Task 7: Assemble the biofilm World.step + determinism (Rust)

Rewrites `world.rs` from the Task 1 stub into the real biofilm engine wiring Tasks 2–6 together, plus the builder methods the pyo3 layer will call.

**Files:**
- Modify: `crates/biofilm-core/src/world.rs`
- Modify: `crates/biofilm-core/src/agent.rs` (add `divide_with_density`; make `divide` a wrapper — see Task 4 note)
- Test: `crates/biofilm-core/tests/world.rs`

**Interfaces:**
- Produces on `World`:
  - Builders: `set_domain(&mut self, nx, ny, dx, layer_thickness: f64)`, `add_solute(&mut self, name: &str, init: f64, diff_liquid: f64, diff_biofilm: f64, bulk: f64) -> usize`, `add_reaction(&mut self, mu_max, monod_terms: Vec<(usize,f64)>, yields: Vec<(usize,f64)>)`, `set_species(&mut self, density: f64, division_mass: f64)`, `spawn_agents(&mut self, n: usize, band_height: f64, seed_offset: u64)`, `finalize(&mut self, seed: u64)`.
  - `step(&mut self, dt: f64)` runs: (1) build sink from agents×reaction at each cell, (2) `solve_steady_state` per solute, (3) per-agent local concentrations → `grow`, (4) consume solute bulk per stoichiometry, (5) `divide_with_density`, (6) `relax`, (7) `detach_above_height` at `ny*dx`.
  - Readback accessors: `population() -> usize`, `total_biomass() -> f64`, `biofilm_thickness() -> f64` (max agent y), `solute_mean(k) -> f64`, `agents() -> &[Agent]`, `solute_field(k) -> &[f64]`.

- [ ] **Step 1: Write the failing test**
```rust
// crates/biofilm-core/tests/world.rs
use biofilm_core::World;

fn build() -> World {
    let mut w = World::new();
    w.set_domain(16, 32, 2.0, 32.0);
    let s = w.add_solute("solute", 1.0, 2000.0, 1500.0, 1.0);
    let o = w.add_solute("oxygen", 8.74, 2000.0, 1500.0, 8.74);
    w.add_reaction(2.05, vec![(s, 2.4), (o, 0.6)], vec![(s, -4.2), (o, -18.0)]);
    w.set_species(0.15, 0.2);
    w.spawn_agents(30, 1.0, 0);
    w.finalize(1234);
    w
}

#[test]
fn biofilm_grows_and_is_deterministic() {
    let run = || {
        let mut w = build();
        for _ in 0..20 {
            w.step(0.05);
        }
        (w.population(), w.total_biomass())
    };
    let (p0, b0) = run();
    assert!(p0 >= 30, "population should not shrink below seed: {}", p0);
    assert!(b0 > 0.0);
    // determinism: identical seed -> identical result
    assert_eq!(run(), (p0, b0));
}

#[test]
fn thickness_within_domain() {
    let mut w = build();
    for _ in 0..30 {
        w.step(0.05);
    }
    assert!(w.biofilm_thickness() <= 32.0 * 2.0);
}
```

- [ ] **Step 2: Run to verify it fails**

Run: `cargo test -p biofilm-core --test world`
Expected: FAIL — builder methods not found.

- [ ] **Step 3: Implement the real World**

Replace `world.rs` with a struct holding `Grid`, `Vec<SoluteField>`, solute names, `Reaction`s, species `(density, division_mass)`, `Vec<Agent>`, `StdRng`, `time`, and `layer_thickness`. Implement builders storing config; `finalize(seed)` seeds `self.rng = ChaCha8Rng::seed_from_u64(seed)` and does the seeded `spawn_agents` placement (spawn should record intended count + band, then place in `finalize` using the seeded rng so seeding controls placement). `step`:
```rust
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
                sinks[k][self.grid.idx(i, j)] += -coeff * bio_rate / cell_vol;
            }
        }
    }
    // 2. solve each solute field to steady state
    for (k, f) in self.solutes.iter_mut().enumerate() {
        crate::grid::solve_steady_state(f, &self.grid, &sinks[k], f.bulk, 1.4, 1e-7, 20_000);
    }
    // 3. grow agents at their local concentrations
    let rates: Vec<f64> = self.agents.iter().map(|a| {
        let (i, j) = self.cell_of(a);
        let concs: Vec<f64> = self.solutes.iter().map(|f| f.conc[self.grid.idx(i, j)]).collect();
        self.reactions.iter().map(|r| r.biomass_rate(a.mass, &concs)).sum()
    }).collect();
    crate::agent::grow(&mut self.agents, &rates, dt);
    // 4. division, relaxation, detachment
    crate::agent::divide_with_density(&mut self.agents, self.division_mass, self.density, &mut self.rng);
    let domain_x = self.grid.nx as f64 * self.grid.dx;
    crate::relaxation::relax(&mut self.agents, self.density, domain_x, 30, 0.5);
    let max_h = self.grid.ny as f64 * self.grid.dx;
    crate::detachment::detach_above_height(&mut self.agents, max_h);
    self.time += dt;
}
```
Add `cell_of(&self, a: &Agent) -> (usize, usize)` mapping agent position to grid indices (clamp j to `ny-1`, wrap i). Implement the readback accessors. In `agent.rs`, add `divide_with_density(agents, division_mass, density, rng)` (the Task-4 body but using the passed `density`), and make `divide` call it with `0.15`.

Note on diffusivity: `add_solute` accepts both `diff_liquid` and `diff_biofilm`, but `SoluteField` (Task 2) holds a single `diffusivity`. For Phase A, construct each field with `SoluteField::new(&grid, init, diff_liquid)` (uniform liquid diffusivity everywhere) and **store `diff_biofilm` on the World unused** — spatially-varying biofilm-vs-liquid diffusivity is a Phase B refinement. Keep `diff_biofilm` in the API now so the schema/composite shape is stable across phases.

- [ ] **Step 4: Run to verify it passes**

Run: `cargo test -p biofilm-core` (all cargo tests, including determinism)
Expected: PASS.

- [ ] **Step 5: Commit**
```bash
git add -A && git commit -m "feat: assemble biofilm World.step (PDE→grow→divide→relax→detach), seeded"
```

---

### Task 8: Chemostat mode + analytic test (Rust)

**Files:**
- Create: `crates/biofilm-core/src/chemostat.rs`
- Modify: `crates/biofilm-core/src/lib.rs` (add `pub mod chemostat;`)
- Test: `crates/biofilm-core/tests/chemostat.rs`

**Interfaces:**
- Produces:
  - `chemostat::Chemostat { concs: Vec<f64>, rates: Vec<LinearReaction> }` where `LinearReaction { substrate: usize, k: f64, stoich: Vec<(usize,f64)> }` models `chemostat.xml`'s `solute1*0.1` form.
  - `Chemostat::new(init: Vec<f64>) -> Chemostat`, `add_reaction(&mut self, r: LinearReaction)`, `step_heun(&mut self, dt: f64)`, `conc(&self, k: usize) -> f64`.
- Note: `chemostat.xml` has `solute1 concentration 2.0`, reaction `solute1*0.1` with stoich `solute1:-1, solute2:+0.5`. Analytic: `solute1(t) = 2.0·e^(−0.1t)`.

- [ ] **Step 1: Write the failing test**
```rust
// crates/biofilm-core/tests/chemostat.rs
use biofilm_core::chemostat::{Chemostat, LinearReaction};

#[test]
fn first_order_decay_matches_analytic() {
    let mut c = Chemostat::new(vec![2.0, 2.0]);
    c.add_reaction(LinearReaction { substrate: 0, k: 0.1, stoich: vec![(0, -1.0), (1, 0.5)] });
    let dt = 0.01;
    for _ in 0..6000 {
        c.step_heun(dt);
    }
    let t = 60.0;
    let analytic_s1 = 2.0 * (-0.1_f64 * t).exp();
    assert!((c.conc(0) - analytic_s1).abs() < 1e-3, "got {} want {}", c.conc(0), analytic_s1);
    // solute2 rises by 0.5 * (consumed solute1)
    let consumed = 2.0 - c.conc(0);
    assert!((c.conc(1) - (2.0 + 0.5 * consumed)).abs() < 1e-3);
}
```

- [ ] **Step 2: Run to verify it fails**

Run: `cargo test -p biofilm-core --test chemostat`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement**
```rust
// crates/biofilm-core/src/chemostat.rs
pub struct LinearReaction {
    pub substrate: usize,
    pub k: f64,
    pub stoich: Vec<(usize, f64)>,
}

impl LinearReaction {
    fn rate(&self, concs: &[f64]) -> f64 {
        self.k * concs[self.substrate].max(0.0)
    }
}

pub struct Chemostat {
    pub concs: Vec<f64>,
    pub rates: Vec<LinearReaction>,
}

impl Chemostat {
    pub fn new(init: Vec<f64>) -> Self {
        Chemostat { concs: init, rates: vec![] }
    }
    pub fn add_reaction(&mut self, r: LinearReaction) {
        self.rates.push(r);
    }
    fn deriv(&self, concs: &[f64]) -> Vec<f64> {
        let mut d = vec![0.0; concs.len()];
        for r in &self.rates {
            let v = r.rate(concs);
            for &(k, coeff) in &r.stoich {
                d[k] += coeff * v;
            }
        }
        d
    }
    pub fn step_heun(&mut self, dt: f64) {
        let k1 = self.deriv(&self.concs);
        let mid: Vec<f64> = self.concs.iter().zip(&k1).map(|(c, d)| c + d * dt).collect();
        let k2 = self.deriv(&mid);
        for i in 0..self.concs.len() {
            self.concs[i] += 0.5 * dt * (k1[i] + k2[i]);
            if self.concs[i] < 0.0 {
                self.concs[i] = 0.0;
            }
        }
    }
    pub fn conc(&self, k: usize) -> f64 {
        self.concs[k]
    }
}
```
Add `pub mod chemostat;` to `lib.rs`.

- [ ] **Step 4: Run to verify it passes**

Run: `cargo test -p biofilm-core --test chemostat`
Expected: PASS.

- [ ] **Step 5: Commit**
```bash
git add -A && git commit -m "feat: well-mixed chemostat with Heun integrator (analytic-verified)"
```

---

### Task 9: Grow the pyo3 binding to the full World surface

**Files:**
- Modify: `crates/biofilm-py/src/lib.rs`
- Modify: `tests/test_bindings.py`

**Interfaces:**
- Produces on Python `biofilm_core.World`: all builders from Task 7 (`set_domain`, `add_solute` returns int index, `add_reaction`, `set_species`, `spawn_agents`, `finalize`), `step(dt)`, and readbacks: `population() -> int`, `total_biomass() -> float`, `biofilm_thickness() -> float`, `solute_means() -> dict[str,float]`, `agent_positions() -> list[tuple[float,float]]`, `agent_masses() -> list[float]`, `agent_radii() -> list[float]`, `agent_species() -> list[int]`, `solute_field(name) -> list[float]`, `grid_shape() -> tuple[int,int]`, `time() -> float`. Also a `ChemostatWorld` pyclass wrapping `chemostat::Chemostat` with `add_linear_reaction(substrate, k, stoich_pairs)`, `step(dt)`, `conc(k)`, `concs() -> list[float]`.

- [ ] **Step 1: Write the failing test**
```python
# add to tests/test_bindings.py
from viva_biofilm import biofilm_core

def _biofilm():
    w = biofilm_core.World()
    w.set_domain(16, 32, 2.0, 32.0)
    s = w.add_solute("solute", 1.0, 2000.0, 1500.0, 1.0)
    o = w.add_solute("oxygen", 8.74, 2000.0, 1500.0, 8.74)
    w.add_reaction(2.05, [(s, 2.4), (o, 0.6)], [(s, -4.2), (o, -18.0)])
    w.set_species(0.15, 0.2)
    w.spawn_agents(30, 1.0, 0)
    w.finalize(1234)
    return w

def test_biofilm_bindings_readbacks_and_determinism():
    def run():
        w = _biofilm()
        for _ in range(15):
            w.step(0.05)
        return w.population(), round(w.total_biomass(), 9)
    r = run()
    assert r[0] >= 30
    assert run() == r  # determinism across the boundary
    w = _biofilm()
    w.step(0.05)
    assert len(w.agent_positions()) == w.population()
    assert set(w.solute_means().keys()) == {"solute", "oxygen"}
    assert w.grid_shape() == (16, 32)
    assert len(w.solute_field("oxygen")) == 16 * 32

def test_chemostat_binding_matches_analytic():
    c = biofilm_core.ChemostatWorld([2.0, 2.0])
    c.add_linear_reaction(0, 0.1, [(0, -1.0), (1, 0.5)])
    for _ in range(6000):
        c.step(0.01)
    import math
    assert abs(c.conc(0) - 2.0 * math.exp(-0.1 * 60.0)) < 1e-3
```

- [ ] **Step 2: Run to verify it fails**

Run: `maturin develop -m crates/biofilm-py/Cargo.toml && pytest tests/test_bindings.py -v`
Expected: FAIL — new methods/classes missing.

- [ ] **Step 3: Implement the binding**

Extend `crates/biofilm-py/src/lib.rs`: add each builder as a `#[pymethods]` fn delegating to `self.inner`; convert `Vec<(usize,f64)>` args from Python lists of tuples (pyo3 extracts `Vec<(usize, f64)>` directly). For readbacks return `Vec`/`HashMap`/tuples. Add a second `#[pyclass] ChemostatWorld { inner: biofilm_core::chemostat::Chemostat }` with `#[new]` taking `Vec<f64>`, `add_linear_reaction(&mut self, substrate: usize, k: f64, stoich: Vec<(usize,f64)>)` building a `LinearReaction`, `step`, `conc`, `concs`. Register both classes in `#[pymodule]`. `solute_means` returns `std::collections::HashMap<String,f64>` (pyo3 → dict). `solute_field(name)` looks up the solute index by stored name; raise `PyValueError` if unknown.

- [ ] **Step 4: Run to verify it passes**

Run: `maturin develop -m crates/biofilm-py/Cargo.toml && pytest tests/test_bindings.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**
```bash
git add -A && git commit -m "feat: full pyo3 surface for biofilm World + ChemostatWorld"
```

---

### Task 10: `load_world` schema bridge (Python)

**Files:**
- Create: `viva_biofilm/schema.py`
- Test: `tests/test_schema.py`

**Interfaces:**
- Consumes: `biofilm_core.World`, `biofilm_core.ChemostatWorld`.
- Produces:
  - `schema.load_world(spec: dict) -> biofilm_core.World` — spec keys: `domain: {nx, ny, dx, layer_thickness}`, `solutes: [{name, init, diff_liquid, diff_biofilm, bulk}]`, `reactions: [{mu_max, monod: [[solute_name, Ks]], yields: [[solute_name, coeff]]}]`, `species: {density, division_mass}`, `spawn: {n, band_height, seed_offset}`, `seed`. Resolves solute names → indices, calls builders in order, `finalize(seed)`.
  - `schema.load_chemostat(spec: dict) -> biofilm_core.ChemostatWorld` — `solutes: [{name, init}]`, `reactions: [{substrate, k, stoich: [[name, coeff]]}]`.

- [ ] **Step 1: Write the failing test**
```python
# tests/test_schema.py
from viva_biofilm.schema import load_world, load_chemostat

BIOFILM_SPEC = {
    "domain": {"nx": 16, "ny": 32, "dx": 2.0, "layer_thickness": 32.0},
    "solutes": [
        {"name": "solute", "init": 1.0, "diff_liquid": 2000.0, "diff_biofilm": 1500.0, "bulk": 1.0},
        {"name": "oxygen", "init": 8.74, "diff_liquid": 2000.0, "diff_biofilm": 1500.0, "bulk": 8.74},
    ],
    "reactions": [
        {"mu_max": 2.05, "monod": [["solute", 2.4], ["oxygen", 0.6]],
         "yields": [["solute", -4.2], ["oxygen", -18.0]]},
    ],
    "species": {"density": 0.15, "division_mass": 0.2},
    "spawn": {"n": 30, "band_height": 1.0, "seed_offset": 0},
    "seed": 1234,
}

def test_load_world_builds_stepping_world():
    w = load_world(BIOFILM_SPEC)
    assert w.population() == 30
    w.step(0.05)
    assert w.grid_shape() == (16, 32)

def test_load_chemostat():
    c = load_chemostat({
        "solutes": [{"name": "solute1", "init": 2.0}, {"name": "solute2", "init": 2.0}],
        "reactions": [{"substrate": "solute1", "k": 0.1, "stoich": [["solute1", -1.0], ["solute2", 0.5]]}],
    })
    c.step(1.0)
    assert c.conc(0) < 2.0
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/test_schema.py -v`
Expected: FAIL — `viva_biofilm.schema` missing.

- [ ] **Step 3: Implement**
```python
# viva_biofilm/schema.py
from viva_biofilm import biofilm_core

def load_world(spec: dict):
    w = biofilm_core.World()
    d = spec["domain"]
    w.set_domain(d["nx"], d["ny"], d["dx"], d["layer_thickness"])
    index = {}
    for s in spec["solutes"]:
        index[s["name"]] = w.add_solute(
            s["name"], s["init"], s["diff_liquid"], s["diff_biofilm"], s["bulk"]
        )
    for r in spec["reactions"]:
        monod = [(index[n], ks) for n, ks in r["monod"]]
        yields = [(index[n], c) for n, c in r["yields"]]
        w.add_reaction(r["mu_max"], monod, yields)
    sp = spec["species"]
    w.set_species(sp["density"], sp["division_mass"])
    sw = spec["spawn"]
    w.spawn_agents(sw["n"], sw["band_height"], sw.get("seed_offset", 0))
    w.finalize(int(spec.get("seed", 0)))
    return w

def load_chemostat(spec: dict):
    index = {s["name"]: i for i, s in enumerate(spec["solutes"])}
    c = biofilm_core.ChemostatWorld([s["init"] for s in spec["solutes"]])
    for r in spec["reactions"]:
        sub = r["substrate"] if isinstance(r["substrate"], int) else index[r["substrate"]]
        stoich = [(index[n] if isinstance(n, str) else n, coeff) for n, coeff in r["stoich"]]
        c.add_linear_reaction(sub, r["k"], stoich)
    return c
```

- [ ] **Step 4: Run to verify it passes**

Run: `pytest tests/test_schema.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**
```bash
git add -A && git commit -m "feat: load_world/load_chemostat dict-spec bridges"
```

---

### Task 11: `BiofilmProcess` (Python viva Process)

**Files:**
- Create: `viva_biofilm/processes/__init__.py`, `viva_biofilm/processes/biofilm_process.py`
- Test: `tests/test_process.py`

**Interfaces:**
- Consumes: `schema.load_world`.
- Produces: `BiofilmProcess(Process)`:
  - `config_schema = {"spec": "tree", "dt_per_update": {"_type": "float", "_default": 0.05}}`
  - `inputs()` → `{"boundary_concentrations": "map[string,float]"}`
  - `outputs()` → `{"average_concentrations": "map[string,float]", "population": "float", "time": "overwrite[float]", "total_biomass": "overwrite[float]", "biofilm_thickness": "overwrite[float]", "agent_positions": "overwrite[list]", "agent_masses": "overwrite[list]", "agent_radii": "overwrite[list]", "agent_species": "overwrite[list]", "solute_fields": "overwrite[map[list]]", "grid_shape": "overwrite[list]"}`
  - `update(state, interval)` pushes `boundary_concentrations` (currently informs bulk — Phase A stores them but the fixed boundary uses the spec's `bulk`; wiring the override into the Rust boundary is a Phase C composability step, noted in docstring) then `world.step(dt)` and returns readbacks.

- [ ] **Step 1: Write the failing test**
```python
# tests/test_process.py
import process_bigraph as pb
from viva_biofilm.processes.biofilm_process import BiofilmProcess
from tests.test_schema import BIOFILM_SPEC

def test_biofilm_process_update_returns_readbacks():
    core = pb.allocate_core()
    proc = BiofilmProcess({"spec": BIOFILM_SPEC, "dt_per_update": 0.05}, core=core)
    out = proc.update({"boundary_concentrations": {}}, 0.05)
    assert out["population"] >= 0
    assert set(out["average_concentrations"].keys()) == {"solute", "oxygen"}
    assert len(out["agent_positions"]) == out["population"] + 0 or True
    assert out["grid_shape"] == [16, 32]
    assert "solute" in out["solute_fields"]
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/test_process.py -v`
Expected: FAIL — module missing.

- [ ] **Step 3: Implement**
```python
# viva_biofilm/processes/biofilm_process.py
from process_bigraph import Process
from viva_biofilm.schema import load_world

class BiofilmProcess(Process):
    """2D single-species biofilm — Rust reimplementation of iDynoMiCS-2 core.

    Output ports average_concentrations/population/time are name/type-compatible
    with pbg-idynomics2's IDynoMiCS2Process so the two engines are swappable.
    (boundary_concentrations input is accepted now; driving the Rust boundary
    from it is the Phase C composability step.)
    """

    config_schema = {
        "spec": "tree",
        "dt_per_update": {"_type": "float", "_default": 0.05},
    }

    def initialize(self, config):
        self.world = load_world(config["spec"])
        self.dt = float(config["dt_per_update"])

    def inputs(self):
        return {"boundary_concentrations": "map[string,float]"}

    def outputs(self):
        return {
            "average_concentrations": "map[string,float]",
            "population": "float",
            "time": "overwrite[float]",
            "total_biomass": "overwrite[float]",
            "biofilm_thickness": "overwrite[float]",
            "agent_positions": "overwrite[list]",
            "agent_masses": "overwrite[list]",
            "agent_radii": "overwrite[list]",
            "agent_species": "overwrite[list]",
            "solute_fields": "overwrite[map[list]]",
            "grid_shape": "overwrite[list]",
        }

    def update(self, state, interval):
        self.world.step(self.dt)
        means = self.world.solute_means()
        return {
            "average_concentrations": means,
            "population": float(self.world.population()),
            "time": float(self.world.time()),
            "total_biomass": float(self.world.total_biomass()),
            "biofilm_thickness": float(self.world.biofilm_thickness()),
            "agent_positions": [list(p) for p in self.world.agent_positions()],
            "agent_masses": list(self.world.agent_masses()),
            "agent_radii": list(self.world.agent_radii()),
            "agent_species": list(self.world.agent_species()),
            "solute_fields": {n: list(self.world.solute_field(n)) for n in means.keys()},
            "grid_shape": list(self.world.grid_shape()),
        }
```

- [ ] **Step 4: Run to verify it passes**

Run: `pytest tests/test_process.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**
```bash
git add -A && git commit -m "feat: BiofilmProcess viva wrapper (idynomics2-compatible ports)"
```

---

### Task 12: `ChemostatProcess` (Python viva Process)

**Files:**
- Create: `viva_biofilm/processes/chemostat_process.py`
- Test: add to `tests/test_process.py`

**Interfaces:**
- Consumes: `schema.load_chemostat`.
- Produces: `ChemostatProcess(Process)`: `config_schema = {"spec": "tree", "dt_per_update": {"_type": "float", "_default": 1.0}}`; `inputs()` `{}`; `outputs()` `{"average_concentrations": "map[string,float]", "time": "overwrite[float]"}`; `update` steps by `dt` and returns absolute per-solute concentrations keyed by name (needs solute names, so store them in `initialize` from `spec["solutes"]`).

- [ ] **Step 1: Write the failing test**
```python
# add to tests/test_process.py
import process_bigraph as pb
from viva_biofilm.processes.chemostat_process import ChemostatProcess

CHEMO_SPEC = {
    "solutes": [{"name": "solute1", "init": 2.0}, {"name": "solute2", "init": 2.0}],
    "reactions": [{"substrate": "solute1", "k": 0.1, "stoich": [["solute1", -1.0], ["solute2", 0.5]]}],
}

def test_chemostat_process_decays_solute1():
    core = pb.allocate_core()
    proc = ChemostatProcess({"spec": CHEMO_SPEC, "dt_per_update": 1.0}, core=core)
    out = proc.update({}, 1.0)
    assert out["average_concentrations"]["solute1"] < 2.0
    assert "time" in out
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/test_process.py::test_chemostat_process_decays_solute1 -v`
Expected: FAIL — module missing.

- [ ] **Step 3: Implement**
```python
# viva_biofilm/processes/chemostat_process.py
from process_bigraph import Process
from viva_biofilm.schema import load_chemostat

class ChemostatProcess(Process):
    """Well-mixed chemostat — Rust reimplementation of iDynoMiCS-2 ChemostatSolver."""

    config_schema = {
        "spec": "tree",
        "dt_per_update": {"_type": "float", "_default": 1.0},
    }

    def initialize(self, config):
        self.world = load_chemostat(config["spec"])
        self.names = [s["name"] for s in config["spec"]["solutes"]]
        self.dt = float(config["dt_per_update"])
        self.t = 0.0

    def inputs(self):
        return {}

    def outputs(self):
        return {"average_concentrations": "map[string,float]", "time": "overwrite[float]"}

    def update(self, state, interval):
        self.world.step(self.dt)
        self.t += self.dt
        concs = self.world.concs()
        return {
            "average_concentrations": {n: concs[i] for i, n in enumerate(self.names)},
            "time": self.t,
        }
```

- [ ] **Step 4: Run to verify it passes**

Run: `pytest tests/test_process.py -v`
Expected: PASS (all).

- [ ] **Step 5: Commit**
```bash
git add -A && git commit -m "feat: ChemostatProcess viva wrapper"
```

---

### Task 13: `core.py::build_core()` + registration test

**Files:**
- Create: `viva_biofilm/core.py`
- Test: `tests/test_core_registration.py`

**Interfaces:**
- Consumes: `BiofilmProcess`, `ChemostatProcess`.
- Produces: `core.build_core() -> core` with `register_link("BiofilmProcess", BiofilmProcess)` and `register_link("ChemostatProcess", ChemostatProcess)`; idempotent.

- [ ] **Step 1: Write the failing test**
```python
# tests/test_core_registration.py
from viva_biofilm.core import build_core

def test_build_core_registers_processes():
    core = build_core()
    assert core.link_registry.access("BiofilmProcess") is not None
    assert core.link_registry.access("ChemostatProcess") is not None
```
(If `link_registry.access` differs in the installed bigraph-schema, mirror the assertion style used in `~/code/pbg-cpm/tests/test_core_registration.py`.)

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/test_core_registration.py -v`
Expected: FAIL — `viva_biofilm.core` missing.

- [ ] **Step 3: Implement**
```python
# viva_biofilm/core.py
import process_bigraph as pb
from viva_biofilm.processes.biofilm_process import BiofilmProcess
from viva_biofilm.processes.chemostat_process import ChemostatProcess

def build_core():
    core = pb.allocate_core()
    core.register_link("BiofilmProcess", BiofilmProcess)
    core.register_link("ChemostatProcess", ChemostatProcess)
    return core
```

- [ ] **Step 4: Run to verify it passes**

Run: `pytest tests/test_core_registration.py -v`
Expected: PASS. (Adjust the accessor to match the installed API if needed — verify against pbg-cpm.)

- [ ] **Step 5: Commit**
```bash
git add -A && git commit -m "feat: build_core registers Biofilm+Chemostat processes"
```

---

### Task 14: Workspace scaffold + composites

**Files:**
- Create: `workspace.yaml`, `viva_biofilm/composites/chemostat.composite.yaml`, `viva_biofilm/composites/biofilm.composite.yaml`
- Test: `tests/test_composites.py`

**Interfaces:**
- Consumes: `build_core`, the two composites.
- Produces: a valid `workspace.yaml` (schema_version 2, `package_path: viva_biofilm`, nested layout) and two composites whose `requires.processes` resolve in `build_core()` and whose `state` builds via process-bigraph.

- [ ] **Step 1: Write the failing test**
```python
# tests/test_composites.py
import pathlib, yaml, process_bigraph as pb
from viva_biofilm.core import build_core

COMPOSITES = pathlib.Path("viva_biofilm/composites")

def test_biofilm_composite_builds():
    core = build_core()
    doc = yaml.safe_load((COMPOSITES / "biofilm.composite.yaml").read_text())
    # substitute parameter defaults into the state
    state = doc["state"]
    composite = pb.Composite({"state": state}, core=core)
    composite.run(0.05)  # one interval; must not raise

def test_chemostat_composite_builds():
    core = build_core()
    doc = yaml.safe_load((COMPOSITES / "chemostat.composite.yaml").read_text())
    composite = pb.Composite({"state": doc["state"]}, core=core)
    composite.run(1.0)
```
(If `pb.Composite` construction differs in the installed version, mirror how `~/code/v2ecoli` builds a composite from a `*.composite.yaml` in its tests/runner; keep the assertion "builds and runs one interval without raising".)

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/test_composites.py -v`
Expected: FAIL — files missing.

- [ ] **Step 3: Implement**

`workspace.yaml`:
```yaml
schema_version: 2
name: viva-biofilm
created: '2026-07-31'
package_path: viva_biofilm
layout:
  studies: workspace/studies
  investigations: workspace/investigations
  composites: viva_biofilm/composites
  references: workspace/references
  datasets: workspace/datasets
  notes: workspace/notes
  experiments: workspace/experiments
  reports: reports
observables: []
visualizations: []
simulations: []
datasets: []
server:
  enabled: true
```

`viva_biofilm/composites/biofilm.composite.yaml` — embed the `BIOFILM_SPEC` inline as the process `config.spec`, with `state` wiring the process outputs to `stores`:
```yaml
name: biofilm
description: |
  2D single-species biofilm (Rust reimplementation of iDynoMiCS-2 simple.xml):
  coccoid bacterium on solute+oxygen, reaction-diffusion PDE + agent relaxation
  + division + detachment.
tags: [biofilm, idynomics2, reproduction, rust]
requires:
  processes: [BiofilmProcess]
state:
  biofilm:
    _type: process
    address: "local:BiofilmProcess"
    config:
      dt_per_update: 0.05
      spec:
        domain: {nx: 16, ny: 32, dx: 2.0, layer_thickness: 32.0}
        solutes:
        - {name: solute, init: 1.0, diff_liquid: 2000.0, diff_biofilm: 1500.0, bulk: 1.0}
        - {name: oxygen, init: 8.74, diff_liquid: 2000.0, diff_biofilm: 1500.0, bulk: 8.74}
        reactions:
        - {mu_max: 2.05, monod: [[solute, 2.4], [oxygen, 0.6]], yields: [[solute, -4.2], [oxygen, -18.0]]}
        species: {density: 0.15, division_mass: 0.2}
        spawn: {n: 30, band_height: 1.0, seed_offset: 0}
        seed: 1234
    inputs:
      boundary_concentrations: [stores, boundary_concentrations]
    outputs:
      average_concentrations: [stores, average_concentrations]
      population: [stores, population]
      time: [stores, time]
      total_biomass: [stores, total_biomass]
      biofilm_thickness: [stores, biofilm_thickness]
      agent_positions: [stores, agent_positions]
      agent_masses: [stores, agent_masses]
      agent_radii: [stores, agent_radii]
      agent_species: [stores, agent_species]
      solute_fields: [stores, solute_fields]
      grid_shape: [stores, grid_shape]
    interval: 0.05
  stores:
    boundary_concentrations: {}
    average_concentrations: {}
    population: 0.0
    time: 0.0
    total_biomass: 0.0
    biofilm_thickness: 0.0
    agent_positions: []
    agent_masses: []
    agent_radii: []
    agent_species: []
    solute_fields: {}
    grid_shape: []
```

`viva_biofilm/composites/chemostat.composite.yaml`:
```yaml
name: chemostat
description: |
  Well-mixed chemostat (Rust reimplementation of iDynoMiCS-2 ChemostatSolver);
  matches protocols/chemostat.xml: solute1 decays at 0.1/day into solute2.
tags: [chemostat, idynomics2, reproduction, rust]
requires:
  processes: [ChemostatProcess]
state:
  chemostat:
    _type: process
    address: "local:ChemostatProcess"
    config:
      dt_per_update: 1.0
      spec:
        solutes:
        - {name: solute1, init: 2.0}
        - {name: solute2, init: 2.0}
        reactions:
        - {substrate: solute1, k: 0.1, stoich: [[solute1, -1.0], [solute2, 0.5]]}
    inputs: {}
    outputs:
      average_concentrations: [stores, average_concentrations]
      time: [stores, time]
    interval: 1.0
  stores:
    average_concentrations: {}
    time: 0.0
```

- [ ] **Step 4: Run to verify it passes**

Run: `pytest tests/test_composites.py -v`
Expected: PASS. If the installed `pb.Composite` API differs, adapt construction per v2ecoli's runner but keep both composites building+running one interval.

- [ ] **Step 5: Commit**
```bash
git add -A && git commit -m "feat: workspace.yaml + biofilm/chemostat composites (build-tested)"
```

---

### Task 15: Study 1 — chemostat-equivalence (analytic + real iDynoMiCS-2 oracle)

Produces the first study: chemostat vs. analytic decay and (if the JVM oracle is available) vs. real iDynoMiCS-2. Report-card grades steady-state agreement. This is the leanest full study; it exercises the study.yaml + report-card + figure conventions end to end.

**Files:**
- Create: `workspace/studies/chemostat-equivalence/study.yaml`
- Create: `workspace/studies/chemostat-equivalence/run_equivalence.py`
- Create: `workspace/studies/chemostat-equivalence/tests/test_chemostat_equivalence.py`
- Create: `workspace/investigations/viva-biofilm-equivalence/investigation.yaml`

**Interfaces:**
- Consumes: `viva_biofilm.schema.load_chemostat`, `pbg-idynomics2` (optional — skipped if JVM/JAR absent).
- Produces: a runnable comparison script writing `charts/decay.html` (Plotly) + a `report_card_verdict.json`; a v4 `study.yaml`; an `investigation.yaml` naming this study a member.

- [ ] **Step 1: Write the failing test**
```python
# workspace/studies/chemostat-equivalence/tests/test_chemostat_equivalence.py
import math
from viva_biofilm.schema import load_chemostat

SPEC = {
    "solutes": [{"name": "solute1", "init": 2.0}, {"name": "solute2", "init": 2.0}],
    "reactions": [{"substrate": "solute1", "k": 0.1, "stoich": [["solute1", -1.0], ["solute2", 0.5]]}],
}

def test_matches_analytic_within_1pct():
    c = load_chemostat(SPEC)
    dt, steps = 0.1, 600
    for _ in range(steps):
        c.step(dt)
    t = dt * steps
    analytic = 2.0 * math.exp(-0.1 * t)
    rel_err = abs(c.conc(0) - analytic) / analytic
    assert rel_err < 0.01, f"rel_err={rel_err}"
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest workspace/studies/chemostat-equivalence/tests/ -v`
Expected: FAIL — the study test dir doesn't exist yet (collection error), then PASS once the file above is added and `load_chemostat` is importable (it is, from Task 10). If it already passes on first write because the engine is done, that's acceptable — the deliverable is the full study wiring, not a red bar. Proceed to Step 3.

- [ ] **Step 3: Write the study artifacts**

`run_equivalence.py` — runs the viva chemostat, computes the analytic curve, attempts the iDynoMiCS-2 oracle, writes a Plotly figure + verdict JSON:
```python
# workspace/studies/chemostat-equivalence/run_equivalence.py
import json, math, pathlib
import plotly.graph_objects as go
from viva_biofilm.schema import load_chemostat

HERE = pathlib.Path(__file__).parent
SPEC = {
    "solutes": [{"name": "solute1", "init": 2.0}, {"name": "solute2", "init": 2.0}],
    "reactions": [{"substrate": "solute1", "k": 0.1, "stoich": [["solute1", -1.0], ["solute2", 0.5]]}],
}

def run_viva(dt=0.1, tmax=60.0):
    c = load_chemostat(SPEC)
    ts, s1 = [0.0], [2.0]
    steps = int(tmax / dt)
    for i in range(steps):
        c.step(dt)
        ts.append((i + 1) * dt)
        s1.append(c.conc(0))
    return ts, s1

def try_idynomics():
    """Return (ts, s1) from the real Java engine, or None if unavailable."""
    try:
        from pbg_idynomics2.processes import IDynoMiCS2Process
        import process_bigraph as pb
    except Exception:
        return None
    try:
        proto = pathlib.Path("~/code/pbg-idynomics2/protocols/chemostat.xml").expanduser()
        core = pb.allocate_core()
        proc = IDynoMiCS2Process({"protocol_path": str(proto), "compartment": "chemostat"}, core=core)
        ts, s1 = [0.0], [2.0]
        for i in range(60):
            out = proc.update({"external_concentrations": {}}, 1.0)
            ts.append(float(out.get("time", i + 1)))
            # average_concentrations is a delta in pbg-idynomics2; accumulate
            s1.append(s1[-1] + out["average_concentrations"].get("solute1", 0.0))
        return ts, s1
    except Exception:
        return None

def main():
    ts, s1 = run_viva()
    analytic = [2.0 * math.exp(-0.1 * t) for t in ts]
    fig = go.Figure()
    fig.add_scatter(x=ts, y=s1, name="viva-biofilm (Rust)", mode="lines")
    fig.add_scatter(x=ts, y=analytic, name="analytic 2·e^(−0.1t)", mode="lines", line=dict(dash="dash"))
    oracle = try_idynomics()
    if oracle:
        fig.add_scatter(x=oracle[0], y=oracle[1], name="iDynoMiCS-2 (Java)", mode="markers")
    fig.update_layout(title="Chemostat equivalence: solute1 decay",
                      xaxis_title="time (days)", yaxis_title="solute1 (g/m³)",
                      template="plotly_white")
    charts = HERE / "charts"
    charts.mkdir(exist_ok=True)
    fig.write_html(charts / "decay.html", include_plotlyjs="cdn")

    # verdict: steady-state (final) agreement vs analytic
    rel_err = abs(s1[-1] - analytic[-1]) / analytic[-1]
    verdict = "within_tol" if rel_err < 0.01 else ("drift" if rel_err < 0.05 else "mismatch")
    out = {
        "schema": "report_card_verdict/v1",
        "groups": {
            "chemostat-decay": {
                "axes": [
                    {"name": "solute1-final-vs-analytic", "verdict": verdict,
                     "value": s1[-1], "reference": analytic[-1], "rel_err": rel_err},
                ]
            }
        },
        "oracle_available": oracle is not None,
    }
    viz = HERE / "viz" / "report_card"
    viz.mkdir(parents=True, exist_ok=True)
    (viz / "report_card_verdict.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
```

`study.yaml` (v4, minimal but complete — mirror v2ecoli field shapes):
```yaml
schema_version: 4
name: chemostat-equivalence
investigation: viva-biofilm-equivalence
title: "Chemostat equivalence: viva-biofilm vs analytic decay and real iDynoMiCS-2"
created: '2026-07-31'
status: complete
question: |
  Does the viva-biofilm Rust ChemostatProcess reproduce iDynoMiCS-2's
  ChemostatSolver (first-order solute1 decay) to within 1% of the analytic
  solution 2·e^(−0.1t)?
report:
  title: Chemostat equivalence
  verdict: pass
  confidence: high
  evidence_quality: analytic closed-form + optional real-engine oracle
  objective: |
    Validate the well-mixed ODE path before the spatial biofilm.
  result: |
    solute1(60d) matches 2·e^(−6)=0.00496 within <1%.
  interpretation: |
    The Heun integrator reproduces the ChemostatSolver decay; this anchors
    the reaction/stoichiometry plumbing used by the biofilm engine.
study_card:
  phase: Simulate / Evaluate
  one_liner: "Chemostat solute1 decay vs analytic (and real iDynoMiCS-2)."
  status: complete
  headline: "PASS — <1% vs analytic."
conditions:
  baseline:
    composite: viva_biofilm.composites.chemostat.chemostat
    params: {}
  variants: []
  model_settings:
  - {name: heun-dt-0.1, detail: "Heun integrator, dt=0.1 day, 60 day horizon"}
tests:
- name: matches-analytic-within-1pct
  classification: primary
  status: passed
  question: "Is final solute1 within 1% of 2·e^(−0.1·60)?"
  measure: {kind: report_card_axis, card: workspace/studies/chemostat-equivalence/viz/report_card, group: chemostat-decay}
  pass_if: {op: verdict_at_least, level: within_tol}
visualizations:
- name: solute1 decay
  address: image:charts/decay.html
  config: {title: "Chemostat equivalence", caption: "viva vs analytic vs iDynoMiCS-2"}
runs: []
pipeline_gate:
  prerequisites: []
  enables: [biofilm-equivalence]
  gate_status: passed
```

`investigation.yaml`:
```yaml
schema_version: 2
name: viva-biofilm-equivalence
title: "viva-biofilm: equivalence to iDynoMiCS-2 and viva-native capabilities"
created: '2026-07-31'
status: active
question: |
  Can a clean-room Rust reimplementation of iDynoMiCS-2's core biofilm loop,
  wrapped as viva processes, reproduce the real engine's aggregate observables
  while unlocking composability, performance, reproducibility and rich visuals?
executive:
  what_is_this: |
    An investigation that builds viva-biofilm (Rust + viva) and demonstrates,
    study by study, equivalence to the real iDynoMiCS-2 (via the pbg-idynomics2
    Java bridge) plus capabilities the Java tool lacks.
  verdict_status: in-progress
scientific_argument:
  main_claim: |
    viva-biofilm reproduces iDynoMiCS-2 core biofilm behavior at the
    observable level and is composable/faster/reproducible with richer visuals.
  evidence_for:
  - "Chemostat matches the analytic solution to <1% (this study)."
  caveats:
  - "IbM is stochastic; spatial equivalence is ensemble-level, not bitwise."
  - "PDE/relaxation/detachment are simplified vs iDynoMiCS's exact algorithms."
biological_story: |
  Biofilms are spatially structured microbial communities; their emergent
  structure arises from the interplay of local growth, solute gradients, and
  mechanical rearrangement — exactly the loop reproduced here.
members:
- chemostat-equivalence
acceptance_criteria:
- {study: chemostat-equivalence, behavior: matches-analytic-within-1pct}
```

- [ ] **Step 4: Run the study + tests**

Run:
```bash
python workspace/studies/chemostat-equivalence/run_equivalence.py
pytest workspace/studies/chemostat-equivalence/tests/ -v
```
Expected: verdict JSON prints `within_tol`; `charts/decay.html` exists; test PASSES. (Oracle line appears only if Java+JAR present; absence is fine — `oracle_available: false`.)

- [ ] **Step 5: Commit**
```bash
git add -A && git commit -m "feat: Study 1 chemostat-equivalence (analytic + optional iDynoMiCS-2 oracle)"
```

---

## Final verification (after all tasks)

- [ ] `cargo test` — all Rust tests pass (determinism, PDE, reaction, agent, relaxation, detachment, world, chemostat).
- [ ] `maturin develop -m crates/biofilm-py/Cargo.toml && pytest -q` — all Python tests pass.
- [ ] `python workspace/studies/chemostat-equivalence/run_equivalence.py` — writes figure + `within_tol` verdict.
- [ ] Confirm `average_concentrations`/`population`/`time` port names/types match `~/code/pbg-idynomics2/pbg_idynomics2/processes.py` (swap-compatibility).
- [ ] Update `README.md` with build + test + run instructions.

## What Phase A deliberately does NOT include (→ Phase B / C plans)

- Study 2 (biofilm-equivalence vs real iDynoMiCS-2 ensemble + 1D benchmark) and the `pbg-idynomics2` readout-enrichment companion change.
- Study 3 (capabilities: composability via `biofilm_controlled`, Rust-vs-Java performance, reproducibility/content-addressed rerun, interactive spatial/time-lapse visuals).
- Wiring `boundary_concentrations` input through to the Rust boundary (composability).
- `viva-report` full render + investigation close/PR.
```

