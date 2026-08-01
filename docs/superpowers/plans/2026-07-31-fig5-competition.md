# viva-biofilm — Fig 5 reproduction (Rate- vs Yield-Strategist competition) — Plan 2

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Reproduce **Figure 5** of Cockx et al. 2024 (iDynoMiCS 2.0, PLOS Comp Biol) — the 2D competition between a **Rate Strategist (RS)** and a **Yield Strategist (YS)** — by extending the viva-biofilm engine to **multi-strategy competition** and building a study that shows the competition outcome depends on **seeding density**.

**Architecture:** Extend the existing 2D Rust engine so each agent grows by its own species/strategy's kinetics (per-species µmax, yields, density, division mass), add a distributed (alternating) spawner and per-strategy readbacks, expose them through pyo3, extend the Python schema, and build a `fig5-rs-ys-competition` study under a new `idynomics2-reproductions` investigation. Spherical (coccoid) agents, shoving mechanics, single shared substrate (oxygen) — all 2D, all within the current engine's paradigm.

**Tech Stack:** Rust (biofilm-core), pyo3/maturin, process-bigraph, Plotly, pytest.

## Global Constraints

- Branch: `fig5-competition` (in-place in `~/code/viva-biofilm` to reuse the maturin-built `.venv`). Rebuild after Rust changes: `source .venv/bin/activate && maturin develop -m crates/biofilm-py/Cargo.toml`.
- Rust core stays pure Rust; determinism preserved (seeded RNG; identical seed ⇒ identical run).
- **Backward compatibility is mandatory:** the existing single-species API (`set_species`, `add_reaction`, `spawn_agents`) must keep working and ALL existing tests (Rust + the 42 pytest) must stay green. New multi-species API is additive.
- Import shared specs in tests from `viva_biofilm.run`, NEVER `from tests.test_schema import ...` (unum ships a shadowing site-packages `tests/`). Registry access is a plain dict (`"Name" in core.link_registry`).
- Viz follows the existing `viva_biofilm/viz.py` conventions (Viridis/qualitative palettes, `plotly_white`, units, `constrain="domain"`, height set). Read `dataviz` guidance for any new chart.
- Study/investigation conventions mirror the merged `workspace/studies/spatial-biofilm-growth/` (v4 study.yaml, report_card_verdict/v1, embed_visualizations) and its investigation.
- **Fidelity honesty:** the paper's exact Monod parameters (RS/YS µmax, Ks, yields) live in S1 Text Table K, which is NOT available. Use physically reasonable values consistent with the paper's qualitative description (RS = higher µmax, same specific affinity/Ks, LOWER yield; YS = lower µmax, HIGHER yield) and the fixed oxygen bulk `S_ox,bulk = 1 mg/L`. The study MUST state this as a fidelity caveat and report the ACTUAL measured outcome — do NOT fabricate or force an exact match to the paper's density-dependent flip. Reproducing the *capability* (multi-strategy competition, outcome varies with seeding density) is the deliverable; an exact quantitative match is explicitly out of scope without Table K.

## Reference (from the paper, Fig 5, p.17)
- 2D, coccoid agents; single substrate oxygen; bulk 1 mg/L.
- Two strategies: **RS** (Rate) higher µmax, lower yield; **YS** (Yield) lower µmax, higher yield ("altruistic"/efficient).
- Distributed **alternating, equidistant** spawner at increasing density: **5, 10, 50 cells per strategy**.
- Domain ~100×100 grid elements; runs to 3 weeks (21 d).
- Finding: outcome (which strategy dominates) is sensitive to seeding density (and to shoving-vs-FbM mechanics — FbM is out of scope for this plan; shoving only).

---

### Task 1: Per-species (per-strategy) model in the Rust core

Give each agent its own strategy's kinetics + density + division mass. Backward-compatible.

**Files:**
- Modify: `crates/biofilm-core/src/world.rs`
- Test: `crates/biofilm-core/tests/species.rs`

**Interfaces (additive):**
- New `Species { density: f64, division_mass: f64, reactions: Vec<Reaction> }` and `species: Vec<Species>` on `World`.
- `add_species(&mut self, density: f64, division_mass: f64) -> usize` — pushes a `Species` (empty reactions), returns its index.
- `add_reaction_for(&mut self, species_idx: usize, mu_max: f64, monod_terms: Vec<(usize,f64)>, yields: Vec<(usize,f64)>)` — appends a reaction to that species.
- **Backward-compat shims:** `set_species(density, division_mass)` becomes: if `species` is empty, push one; else set `species[0]`'s density/division_mass (keep the world-level `density`/`division_mass` fields in sync with species[0] for any existing readback that uses them, OR migrate those readbacks). `add_reaction(...)` appends to species[0] (auto-create species[0] if empty). Existing single-species tests must pass unchanged.
- `step` changes: each agent grows by `self.species[a.species as usize].reactions` at its local concentrations; sink accumulates every agent's consumption (by its species' reactions); division uses `self.species[a.species as usize].division_mass`; agent radius uses `self.species[a.species as usize].density`. (Where the code currently uses the global `self.reactions`/`self.division_mass`/`self.density`, route through the agent's species.)

- [ ] **Step 1: Write the failing test**
```rust
// crates/biofilm-core/tests/species.rs
use biofilm_core::World;

#[test]
fn two_species_grow_by_their_own_mu_max() {
    let mut w = World::new();
    w.set_domain(8, 16, 2.0, 16.0);
    let o = w.add_solute("oxygen", 1.0, 2000.0, 1500.0, 1.0);
    let fast = w.add_species(0.15, 1e9); // huge division mass so they don't divide, just grow
    let slow = w.add_species(0.15, 1e9);
    w.add_reaction_for(fast, 4.0, vec![(o, 0.1)], vec![(o, -1.0)]);
    w.add_reaction_for(slow, 1.0, vec![(o, 0.1)], vec![(o, -1.0)]);
    // seed one agent of each species at the same spot-ish
    w.spawn_distributed(fast, 1, 1.0, 0);
    w.spawn_distributed(slow, 1, 1.0, 1);
    w.finalize(1);
    let m0: Vec<f64> = w.agents().iter().map(|a| a.mass).collect();
    for _ in 0..5 { w.step(0.02); }
    let m1: Vec<f64> = w.agents().iter().map(|a| a.mass).collect();
    // both grew; the fast-species agent gained more mass than the slow one
    let gained: Vec<f64> = m1.iter().zip(&m0).map(|(a,b)| a-b).collect();
    assert!(gained.iter().all(|&g| g > 0.0), "both should grow: {:?}", gained);
    // identify which agent is which species
    let sp: Vec<u16> = w.agents().iter().map(|a| a.species).collect();
    let fast_gain: f64 = gained.iter().zip(&sp).filter(|(_,s)| **s == fast as u16).map(|(g,_)| *g).sum();
    let slow_gain: f64 = gained.iter().zip(&sp).filter(|(_,s)| **s == slow as u16).map(|(g,_)| *g).sum();
    assert!(fast_gain > slow_gain, "fast species should gain more: {} vs {}", fast_gain, slow_gain);
}

#[test]
fn backward_compat_single_species_still_works() {
    // old API path: set_species + add_reaction + spawn_agents
    let mut w = World::new();
    w.set_domain(8, 16, 2.0, 16.0);
    let o = w.add_solute("oxygen", 1.0, 2000.0, 1500.0, 1.0);
    w.add_reaction(2.0, vec![(o, 0.1)], vec![(o, -1.0)]);
    w.set_species(0.15, 0.2);
    w.spawn_agents(10, 1.0, 0);
    w.finalize(1);
    let p0 = w.population();
    for _ in 0..10 { w.step(0.05); }
    assert!(w.population() >= p0);
}
```
(This test also references `spawn_distributed` from Task 2 — write it now; the test compiles once both tasks land. If you prefer, split: implement `spawn_distributed` as a thin stub in Task 1 sufficient for this test, then flesh it out in Task 2. Simplest: implement `spawn_distributed` fully here and let Task 2 add the readbacks + a dedicated placement test.)

- [ ] **Step 2: Run to verify it fails** — `cargo test -p biofilm-core --test species` → FAIL (missing methods).

- [ ] **Step 3: Implement** the `Species` struct + `species: Vec<Species>`, `add_species`, `add_reaction_for`, the `set_species`/`add_reaction` backward-compat shims, and route `step`'s growth/division/radius through the agent's species. Keep the sink accumulation iterating all agents (each using its own species' reactions). Also implement `spawn_distributed(species_idx, n, band_height, seed_offset)` (records a pending spawn tagged with the species; placement in `finalize` seeds agents of that species — alternating/eq-spaced along x within the band; deterministic from seed+offset).

- [ ] **Step 4: Run** — `cargo test -p biofilm-core` (species tests + ALL existing tests pass, incl. determinism/gradient/world). Rebuild `maturin develop -m crates/biofilm-py/Cargo.toml`, then `pytest -q` (all 42 still pass — backward compat).

- [ ] **Step 5: Commit** — `git commit -m "feat: per-species kinetics (RS/YS strategies) in Rust core, backward-compatible"`.

---

### Task 2: Distributed spawner + per-strategy readbacks (Rust)

**Files:**
- Modify: `crates/biofilm-core/src/world.rs`
- Test: `crates/biofilm-core/tests/competition.rs`

**Interfaces:**
- `spawn_distributed(species_idx, n, band_height, seed_offset)` (if not fully done in Task 1): places `n` agents of `species_idx` at alternating, roughly equidistant x-positions along the substratum band, deterministic.
- `population_of(&self, species_idx: usize) -> usize` and `biomass_of(&self, species_idx: usize) -> f64` — per-strategy readouts.

- [ ] **Step 1: Write the failing test**
```rust
// crates/biofilm-core/tests/competition.rs
use biofilm_core::World;
fn two_strategy_world(n_each: usize) -> World {
    let mut w = World::new();
    w.set_domain(50, 50, 2.0, 40.0);
    let o = w.add_solute("oxygen", 1.0, 2000.0, 1500.0, 1.0);
    let rs = w.add_species(0.15, 0.2);
    let ys = w.add_species(0.15, 0.2);
    // RS: higher mu_max, lower yield (more oxygen consumed per biomass);
    // YS: lower mu_max, higher yield (less oxygen per biomass).
    w.add_reaction_for(rs, 3.0, vec![(o, 0.3)], vec![(o, -2.0)]);
    w.add_reaction_for(ys, 1.5, vec![(o, 0.3)], vec![(o, -1.0)]);
    w.spawn_distributed(rs, n_each, 1.0, 0);
    w.spawn_distributed(ys, n_each, 1.0, 1);
    w.finalize(42);
    w
}
#[test]
fn spawns_equal_counts_and_tracks_per_strategy() {
    let w = two_strategy_world(10);
    assert_eq!(w.population_of(0), 10);
    assert_eq!(w.population_of(1), 10);
    assert_eq!(w.population(), 20);
    assert!(w.biomass_of(0) > 0.0 && w.biomass_of(1) > 0.0);
}
#[test]
fn competition_is_deterministic() {
    let run = || { let mut w = two_strategy_world(10); for _ in 0..30 { w.step(0.05); } (w.population_of(0), w.population_of(1)) };
    assert_eq!(run(), run());
}
```

- [ ] **Step 2: Run to verify it fails** — `cargo test -p biofilm-core --test competition`.
- [ ] **Step 3: Implement** `population_of`/`biomass_of` (count/sum agents whose `species` matches) and finalize `spawn_distributed` placement (alternating equidistant x across `[0, nx*dx]`, y within band, seeded).
- [ ] **Step 4: Run** — `cargo test -p biofilm-core` all pass (determinism included).
- [ ] **Step 5: Commit** — `git commit -m "feat: distributed spawner + per-strategy population/biomass readbacks"`.

---

### Task 3: pyo3 binding for the multi-strategy API

**Files:**
- Modify: `crates/biofilm-py/src/lib.rs`
- Modify: `tests/test_bindings.py`

**Interfaces (Python `biofilm_core.World`):** `add_species(density, division_mass) -> int`, `add_reaction_for(species_idx, mu_max, monod_terms, yields)`, `spawn_distributed(species_idx, n, band_height, seed_offset)`, `population_of(species_idx) -> int`, `biomass_of(species_idx) -> float`. (Existing methods unchanged.)

- [ ] **Step 1: Write the failing test** (append to `tests/test_bindings.py`)
```python
def test_multi_strategy_bindings():
    from viva_biofilm import biofilm_core
    w = biofilm_core.World()
    w.set_domain(50, 50, 2.0, 40.0)
    o = w.add_solute("oxygen", 1.0, 2000.0, 1500.0, 1.0)
    rs = w.add_species(0.15, 0.2); ys = w.add_species(0.15, 0.2)
    w.add_reaction_for(rs, 3.0, [(o, 0.3)], [(o, -2.0)])
    w.add_reaction_for(ys, 1.5, [(o, 0.3)], [(o, -1.0)])
    w.spawn_distributed(rs, 8, 1.0, 0); w.spawn_distributed(ys, 8, 1.0, 1)
    w.finalize(42)
    assert w.population_of(0) == 8 and w.population_of(1) == 8
    for _ in range(20): w.step(0.05)
    assert w.population_of(0) >= 8 and w.population_of(1) >= 8
```
- [ ] **Step 2: Fail** — `maturin develop -m crates/biofilm-py/Cargo.toml && pytest tests/test_bindings.py -v`.
- [ ] **Step 3: Implement** the five pyo3 methods (delegate to inner; extract `Vec<(usize,f64)>` for reactions). Track species count in the wrapper if needed for validation.
- [ ] **Step 4: Pass** — rebuild + `pytest -q` (all pass).
- [ ] **Step 5: Commit** — `git commit -m "feat: pyo3 bindings for multi-strategy competition API"`.

---

### Task 4: Python schema — competition spec

**Files:**
- Modify: `viva_biofilm/schema.py`, `viva_biofilm/run.py`
- Test: `tests/test_competition.py`

**Interfaces:**
- `schema.load_world` accepts a multi-species spec: a top-level `strategies: [{name, density, division_mass, monod:[[solute,Ks]], mu_max, yields:[[solute,coeff]], spawn_n}]` block (when present, uses `add_species`/`add_reaction_for`/`spawn_distributed` instead of the single-species path). Keep the existing single-species path working when `strategies` is absent.
- `run.competition_spec(n_each, rs=None, ys=None, nx=50, ny=50, seed=42) -> dict` — builds a two-strategy (RS/YS) spec over oxygen (bulk 1.0), with sensible default RS/YS params (RS higher µmax + lower yield; YS lower µmax + higher yield), seeding `n_each` of each via the distributed spawner.
- `run.run_competition(spec, n_steps, dt, snapshot_every) -> list[dict]` — like `run_biofilm` but snapshots also include `pop_by_strategy` and `biomass_by_strategy` (lists indexed by species) and per-agent `species`.

- [ ] **Step 1: Write the failing test**
```python
# tests/test_competition.py
from viva_biofilm.run import competition_spec, run_competition
def test_competition_spec_runs_two_strategies():
    spec = competition_spec(n_each=8, seed=7)
    snaps = run_competition(spec, n_steps=20, dt=0.05, snapshot_every=10)
    last = snaps[-1]
    assert len(last["pop_by_strategy"]) == 2
    assert sum(last["pop_by_strategy"]) == last["population"]
    assert last["pop_by_strategy"][0] >= 8 and last["pop_by_strategy"][1] >= 8
```
- [ ] **Step 2: Fail** — `pytest tests/test_competition.py -v`.
- [ ] **Step 3: Implement** the schema `strategies` path, `competition_spec`, `run_competition`.
- [ ] **Step 4: Pass** — `pytest -q` (all pass).
- [ ] **Step 5: Commit** — `git commit -m "feat: competition_spec/run_competition + multi-strategy schema path"`.

---

### Task 5: Study — fig5-rs-ys-competition (+ idynomics2-reproductions investigation)

**Files:**
- Create: `workspace/investigations/idynomics2-reproductions/investigation.yaml`
- Create: `workspace/studies/fig5-rs-ys-competition/{study.yaml, run_study.py, tests/test_fig5.py}`
- Add competition figures to `viva_biofilm/viz.py`: `strategy_colony_figure(snapshot)` (agents colored by strategy — RS one hue, YS another, from a qualitative palette) and `competition_outcome_figure(results)` (final RS vs YS fraction vs seeding density).

**Deliverable:** run RS-vs-YS at seeding densities **[5, 10, 50] per strategy** on a ~100×100 grid, 3-week (21 d) runs, measure per-strategy final population + biomass fraction, and render the outcome. Calibrate RS/YS params toward a density-dependent outcome; REPORT the actual measured outcome with fidelity caveats (Table K unavailable).

- [ ] **Step 1: Write the failing test**
```python
# workspace/studies/fig5-rs-ys-competition/tests/test_fig5.py
from viva_biofilm.run import competition_spec, run_competition
def test_competition_produces_an_outcome_at_two_densities():
    outcomes = {}
    for n in (5, 50):
        snaps = run_competition(competition_spec(n_each=n, seed=3), n_steps=60, dt=0.05, snapshot_every=60)
        last = snaps[-1]
        rs, ys = last["biomass_by_strategy"]
        outcomes[n] = rs / (rs + ys)  # RS biomass fraction
    # both strategies persist and a fraction is measurable at both densities
    assert all(0.0 <= f <= 1.0 for f in outcomes.values())
    # the outcome is density-dependent (fractions differ between the two densities)
    assert abs(outcomes[5] - outcomes[50]) > 1e-3
```
(If density-dependence is weak with the default params, tune RS/YS params in `competition_spec` until the fractions differ meaningfully between low and high density — that IS the Fig-5 reproduction. If you cannot get a clear density effect, report DONE_WITH_CONCERNS with the measured fractions rather than weakening the test.)

- [ ] **Step 2: Run** — `pytest workspace/studies/fig5-rs-ys-competition/tests/ -v`.
- [ ] **Step 3: Implement**
  - `viz.strategy_colony_figure` (RS vs YS distinct colors; domain box; constrain=domain; height) and `viz.competition_outcome_figure` (RS biomass-fraction vs seeding density, with the 0.5 line marked).
  - `run_study.py`: sweep densities [5,10,50], 3-week runs, collect per-strategy population/biomass over time; write `charts/`: `outcome_vs_density.html`, `colony_density5.html`/`colony_density50.html` (spatial, colored by strategy), `fraction_over_time.html`; write `viz/report_card/report_card_verdict.json` (group `competition`, axes: `both-strategies-simulated` within_tol; `outcome-is-density-dependent` within_tol if RS-fraction differs across densities else drift; `matches-paper-qualitatively` — a HONESTLY-graded axis: within_tol only if the density trend matches the paper's direction, else drift with a note). Mirror charts to `reports/figures/fig5-rs-ys-competition/`. Print a summary table (density → RS fraction).
  - `study.yaml` v4: baseline composite may be a new `competition` composite OR reference the run script (`canonical_runs`); question/hypothesis about density-dependent competition; report_card_axis measure at group `competition`; embed_visualizations; narrative with the **fidelity caveat** (Table K params unavailable; qualitative reproduction). 
  - `investigation.yaml` (schema_version 2, `idynomics2-reproductions`, three narrative sections framing the goal as reproducing iDynoMiCS-2 paper figures; `members: [fig5-rs-ys-competition]`; note Figs 3 & 6 as planned future members needing 3D + filaments).
- [ ] **Step 4: Run** — `python workspace/studies/fig5-rs-ys-competition/run_study.py` (charts + verdict written; density trend printed); `pytest -q` (all pass).
- [ ] **Step 5: Commit** — `git commit -m "feat: Fig 5 RS-vs-YS competition study + idynomics2-reproductions investigation"`.

---

## Final verification (after all tasks)
- [ ] `cargo test -p biofilm-core` all pass (incl. backward-compat single-species + determinism).
- [ ] `maturin develop -m crates/biofilm-py/Cargo.toml && pytest -q` all pass (existing 42 + new competition tests).
- [ ] `python workspace/studies/fig5-rs-ys-competition/run_study.py` writes the outcome-vs-density chart + strategy-colored colonies + verdict.
- [ ] The study renders in the workbench (`load_spec` accepts it; investigation discovered) and eyeball the strategy-colored colony + outcome chart.
- [ ] The Phase-A/capabilities swap-contract + single-species behavior are unchanged (backward compat).

## Out of scope (later phases)
- **Fig 3** (3D nitrifier biofilm) and **Fig 6** (3D filaments + force-based mechanics) — both require the **3D engine** (next plan) and, for Fig 6, non-spherical morphology + FbM.
- Fig 5's **shoving-vs-FbM comparison** and **10-week** extension, and exact Table K parameters — refinements once the core competition reproduction lands.
