# viva-biofilm — Design Spec

**Date:** 2026-07-31
**Status:** approved (design); pending implementation plan
**Author:** Eran Agmon + Claude

## 1. Purpose & one-line

Re-implement the core of the [iDynoMiCS-2](https://github.com/kreft/iDynoMiCS-2)
individual-based biofilm modeling framework (Kreft Lab) as a **viva-compatible
Rust library**, wrapped as process-bigraph (viva) `Process`es inside a
**vivarium-workspace**, and demonstrate — in a single investigation with three
studies — that it is **equivalent** to the real iDynoMiCS-2 on aggregate
observables while unlocking **viva-native capabilities** (composability,
performance, reproducibility, rich visuals) the Java tool does not offer.

This is a **clean-room reproduction in Rust** (viva-expert `--reproduce`
fidelity tier), *not* a wrap of the Java engine. The real Java engine is used
only as a **reference oracle** for equivalence, via the pre-existing
`pbg-idynomics2` bridge.

## 2. Context & prior art on this machine

- **`pbg-cpm`** (`~/code/pbg-cpm`) — the mechanical **template** to copy: a
  Cargo workspace with a pure-Rust core crate + a pyo3 binding crate, built as
  one wheel by maturin, plus a Python `Process` that drives a long-lived Rust
  `World` each `update()`. Copy its layout, `pyproject.toml`/`Cargo.toml`
  wiring, `schema.py::load_world` shape, and 3-layer test structure verbatim.
- **`pbg-idynomics2`** (`~/code/pbg-idynomics2`) — the **reference oracle**:
  binds the *real* iDynoMiCS-2 (July 2025 release) Java engine via JPype as
  `IDynoMiCS2Process`. Ships two protocol files that define our equivalence
  targets exactly:
  - `protocols/chemostat.xml` — well-mixed `ChemostatSolver` ODE, one linear
    reaction. Trivial deterministic warm-up with an analytic steady state.
  - `protocols/simple.xml` — 2D biofilm: coccoid `bacterium` growing by Monod
    kinetics on `solute` + `oxygen`, `AgentRelaxation` (shoving) + `PDEWrapper`
    (reaction-diffusion) on a 32×64 µm rectangle (X cyclic, Y fixed-boundary
    layer, thickness 32 µm), 30 randomly spawned agents, `divisionMass 0.2 pg`,
    density `0.15`.
  - Its process ports: `inputs = {external_concentrations}`,
    `outputs = {average_concentrations (delta), population (delta),
    time (overwrite[float])}`. **Coarse** — see §8 dependency.
- **`v2ecoli`** (`~/code/v2ecoli`) — the mature workspace whose conventions we
  copy for `workspace.yaml`, `core.py::build_core()`, `*.composite.yaml`,
  `study.yaml` (v4), investigation narrative spine, report-card evaluators, and
  Plotly figure embedding.
- **viva-superpowers skills** (`~/code/pbg-superpowers/skills/viva-*`) —
  `viva-workspace`, `viva-investigation`, `viva-study`, `viva-report` scaffold
  and lint the research state. Use the `/viva-*` skill forms to generate current
  pinned schemas, then fill by pattern-matching v2ecoli.

## 3. iDynoMiCS-2 model components being reproduced (increment 1)

Scope decision: **core biofilm** = chemostat warm-up + 2D single-species
biofilm. Out of scope for increment 1: multiple species, pH dynamics, 3D, a
general reaction-expression parser.

The reproduced `World.step(dt)` runs iDynoMiCS's ProcessManager order:

1. **Reaction-diffusion PDE** (`PDEWrapper` analog) — solutes on a 2D
   rectangular grid, solved to **quasi-steady-state** (diffusion fast vs.
   growth): `∇·(D∇C) − R(C) = 0`. Boundary conditions: X cyclic, substratum
   (Y=0) no-flux, top a fixed-concentration boundary layer. Separate biofilm
   vs. liquid diffusivity. Reaction sink `R` accumulated from agents mapped to
   grid cells. **Solver: red-black Gauss-Seidel / SOR relaxation to a residual
   tolerance.** *(Simplification vs. iDynoMiCS's MgFAS multigrid; observable-
   level equivalence is the target, not bit-equivalence.)*
2. **Agent growth** (`AgentGrowth` / reactions) — each agent integrates its
   Monod rate law over `dt` at its local solute concentrations, updating mass
   (and radius from `mass/density`). Solute fields consumed per stoichiometry.
3. **Division** — agent at/above `divisionMass` splits into two daughters with
   randomized placement (seeded RNG).
4. **Mechanical relaxation** (`AgentRelaxation` analog) — force-based overlap
   resolution (spring-like repulsion) iterated to a tolerance, with a substratum
   constraint. *(Simplification vs. iDynoMiCS's EULER/HEUN/SHOVE integrators.)*
5. **Detachment** — erosion/height-based biomass removal at the biofilm
   interface. *(Simplified detachment model.)*

**Reaction kinetics:** a parameterized multi-Monod rate law
`mass · μmax · (S/(S+Ks)) · (O/(O+Kox))` with configurable constants +
stoichiometry, matching `simple.xml` exactly. No general expression parser.

**Chemostat mode:** well-mixed, no grid — integrate solute ODEs with a Heun
integrator + agent growth, matching `ChemostatSolver`.

**Units:** fixed internal system (µm, pg, g/m³, days); convert iDynoMiCS's
unit-tagged params (`[g/m+3]`, `[pg]`, `[um+2/s]`, `[d-1]`, `[mg/l]`) at the
schema boundary. Must match iDynoMiCS units for equivalence.

**Determinism:** seeded RNG (`rand`); identical seed → bit-identical run,
asserted in Rust tests (mirrors `pbg-cpm`).

## 4. Architecture — four layers

### Layer 1 — Rust core (`crates/biofilm-core`, pure Rust, no pyo3)
Modules: `solute`/`grid`, `pde`, `reaction`, `agent`, `relaxation`,
`detachment`, `chemostat`, `world`. `World::step(dt)` runs the order in §3.
`World` owns grid + agents + params + seeded RNG.

### Layer 2 — pyo3 binding (`crates/biofilm-py`, `crate-type=["cdylib"]`, `[lib] name = "biofilm_core"`)
`#[pyclass] World` with a two-phase lifecycle (builder → `finalize(seed)` →
stepping), exposing:
- **Builders:** `add_solute`, `add_species`, `add_reaction`, `set_domain`,
  `set_boundary`, `spawn_agents`, `finalize`.
- **Advance:** `step(dt)`.
- **Readbacks (plain `Vec`/scalars over pyo3, no numpy/serialization):**
  `agent_positions() -> Vec<(f64,f64)>`, `agent_masses()`, `agent_radii()`,
  `agent_species() -> Vec<u16>`, `solute_field(name) -> Vec<f32>`,
  `grid_shape() -> (usize,usize)`, `solute_means() -> map`,
  `population() -> usize`, `biofilm_thickness() -> f64`,
  `total_biomass() -> f64`, `time() -> f64`, `snapshot()`.
- `#[pymodule] fn biofilm_core`.

### Layer 3 — Python package (`viva_biofilm/`, = workspace `package_path`)
- `schema.py::load_world(spec_dict)` — dict spec → ordered `World.*` builder
  calls → `finalize(seed)` (copy `pbg-cpm/cpm/schema.py` shape).
- `processes/biofilm_process.py::BiofilmProcess(Process)`:
  - `config_schema`: `{spec: "tree", dt_per_update: float, seed: integer}`.
  - `inputs`: `{boundary_concentrations: "map[string,float]"}` — lets an
    external process drive the environment (composability hook).
  - `outputs` (all `overwrite[...]`): `average_concentrations` (map),
    `population`, `time`, **plus superset** `agent_positions`, `agent_masses`,
    `agent_radii`, `agent_species`, `solute_fields` (map[name]→list),
    `biofilm_thickness`, `total_biomass`.
  - `initialize` builds the Rust `World` once; `update` pushes boundary inputs →
    `world.step(dt)` → returns readbacks.
- `processes/chemostat_process.py::ChemostatProcess(Process)` — well-mixed mode.
- `core.py::build_core()` — `register_link("BiofilmProcess", ...)`,
  `register_link("ChemostatProcess", ...)`; idempotent.

**Port-compatibility decision:** the shared output ports
(`average_concentrations`, `population`, `time`) are **deliberately identical to
`pbg-idynomics2`'s** so the Rust engine and the Java engine are drop-in
swappable into the same composite/store paths — this is what makes clean
side-by-side equivalence runs possible. viva-biofilm exposes a strict superset.

### Layer 4 — vivarium-workspace
- `workspace.yaml`: `schema_version: 2`, `package_path: viva_biofilm`, nested
  `layout:` (studies/investigations/composites/... under `workspace/`),
  `server.enabled: true`.
- Composites (`viva_biofilm/composites/*.composite.yaml`):
  - `chemostat.composite.yaml` — `ChemostatProcess`.
  - `biofilm.composite.yaml` — `BiofilmProcess` (the `simple.xml` equivalent).
  - `biofilm_controlled.composite.yaml` — `BiofilmProcess` + a controller
    process wired into `boundary_concentrations` (composability showcase).

## 5. Investigation & studies

One investigation: **`viva-biofilm-equivalence`** (git branch = worktree =
investigation slug). Required narrative spine: `executive`,
`scientific_argument`, `biological_story`.

### Study 1 — chemostat-equivalence (warm-up, near-exact)
`ChemostatProcess` vs **analytic steady state** and vs **real iDynoMiCS-2**
(`chemostat.xml`). Deterministic ODE → tolerance can be tight. Report card:
steady-state solute concentrations `within_tol`.

### Study 2 — biofilm-equivalence (headline)
`BiofilmProcess` vs **real iDynoMiCS-2** (`simple.xml`) and vs a **1D biofilm
benchmark** (Wanner–Gujer style). IbM is stochastic → **ensemble-level
equivalence**: multi-seed mean±CI envelopes of
- total biomass(t),
- max biofilm thickness(t),
- substrate penetration profile (C vs. depth),
- agent count(t).

Equivalence criterion: viva-biofilm's envelope overlaps iDynoMiCS-2's within a
tolerance band; analytic anchor = reaction-diffusion penetration depth. Rich
visuals: spatial colony view (agents colored by state), solute-field heatmap,
time-lapse. Report card: each observable graded `within_tol | drift | mismatch`.

### Study 3 — capabilities-showcase
Demonstrates the viva-native payoff (all four selected):
- **Composability** — `biofilm_controlled` composite: external controller
  perturbs boundary O2/substrate; biofilm responds. iDynoMiCS can't natively do
  this coupling.
- **Performance** — Rust vs Java wall-time and agents/step scaling on
  `simple.xml`; a concrete throughput number.
- **Reproducibility** — same seed → identical run hash; content-addressed
  rerun (viva reproducible-rerun spine).
- **Rich visuals** — interactive Plotly spatial/field/time-lapse views vs.
  iDynoMiCS's static SVG.

## 6. Equivalence methodology (because IbM is stochastic)

- **Within viva-biofilm:** seeded, bit-reproducible (Rust test asserts
  `run(seed) == run(seed)`).
- **Cross-engine:** different RNG/algorithms → compare **aggregate observables
  over multi-seed ensembles** (mean ± CI), not per-agent trajectories.
  Equivalence = envelope overlap within tolerance band.
- **Analytic anchors:** chemostat steady state (exact); biofilm substrate
  penetration depth (reaction-diffusion boundary-layer estimate).

## 7. Testing (three layers, per `pbg-cpm`)

- **Rust** (`cargo test`, `crates/biofilm-core/tests/*.rs`): per-module +
  **determinism** (`assert_eq!(run(seed), run(seed))`) + **chemostat-analytic**
  (steady state matches closed form) + **mass-conservation** (solute consumed =
  biomass produced per stoichiometry).
- **Bindings** (`tests/test_bindings.py`): construct `biofilm_core.World`, step,
  assert snapshot shape + determinism — no process-bigraph.
- **Process** (`tests/test_process.py`): `pb.allocate_core()` +
  `proc.update({}, dt)` returns expected readback keys/shapes.
- **Registration** (`tests/test_core_registration.py`):
  `viva_biofilm.core.build_core()` registers `BiofilmProcess`/`ChemostatProcess`
  so `local:BiofilmProcess` addresses resolve.
- **Study report-card** pytest under `<study>/tests/` using the workbench `run`
  fixture; verdicts → `report_card_verdict.json`.

## 8. Known dependency / risk

- **`pbg-idynomics2` readout enrichment (blocks Study 2).** Study 2 needs total
  biomass, max biofilm thickness, and substrate-vs-depth profiles *out of the
  Java engine*, but `IDynoMiCS2Process` currently emits only
  `average_concentrations` + `population` + `time`. Requires a **small companion
  change to `pbg-idynomics2`**: read the Java `AgentContainer` (total biomass,
  max agent Y) and the environment solute grid (profile), exposed as additional
  output ports port-compatible with `BiofilmProcess`. This is the one cross-repo
  dependency; done as a minor companion PR (own worktree).
- **Fidelity simplifications** (PDE relaxation solver vs. MgFAS; force-based vs.
  SHOVE relaxation; simplified detachment) — acceptable because the equivalence
  target is observable-level, but each is documented as a fidelity note in the
  study narrative so claims stay honest.
- **JVM/Java availability** for the oracle (Java 11 + cached JAR) on whatever
  host runs Study 2.

## 9. Build order / phasing

One investigation, delivered in three increments:

- **Phase A** — Rust core + pyo3 binding + `BiofilmProcess`/`ChemostatProcess` +
  workspace scaffold + composites + **Study 1 (chemostat-equivalence)**.
- **Phase B** — **Study 2 (biofilm-equivalence)** + the `pbg-idynomics2`
  readout-enrichment companion change.
- **Phase C** — **Study 3 (capabilities-showcase)** + `viva-report` render +
  investigation close (PR, never auto-merge).

## 10. Explicit non-goals (increment 1)

Multiple species; pH dynamics; 3D compartments; general reaction-expression
parser; exact numerical reproduction of iDynoMiCS's multigrid/SHOVE algorithms;
detachment beyond a simple erosion model. These are candidate follow-on
increments, not part of this spec.
