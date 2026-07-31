# viva-biofilm Capabilities Investigation — Plan 1 (viva-native)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Build the `viva-biofilm-capabilities` investigation: calibrate the biofilm engine so it grows a visually sensible structure with real solute gradients, add a reusable visualization module, and ship three studies — **spatial-biofilm-growth** (colony map + solute heatmaps + time-lapse), **runtime-and-scaling** (3 axes), and **composability** (biofilm + boundary controller) — with beautiful interactive Plotly figures, report cards, and narrative, viewable standalone and on the workbench.

**Architecture:** Builds on the merged Phase-A engine (Rust `biofilm-core` + pyo3 `biofilm_core` + `BiofilmProcess`/`ChemostatProcess` + workspace). Adds: a unit-reconciliation fix to the Rust PDE, a runtime boundary-concentration hook (Rust→binding→process), a Python `run_biofilm` snapshot driver, a `viz.py` Plotly module, a controller process, and three study directories under `workspace/studies/`.

**Tech Stack:** Rust (biofilm-core), pyo3/maturin, process-bigraph, Plotly, vivarium-workbench, pytest. Interactive HTML figures via Plotly (`include_plotlyjs="cdn"`).

## Global Constraints

- Branch: `capabilities` (in-place in `~/code/viva-biofilm` to reuse the maturin-built `.venv`; do NOT create a separate worktree — the extension `.so` lives in this checkout's `viva_biofilm/`). Rebuild after Rust changes: `source .venv/bin/activate && maturin develop -m crates/biofilm-py/Cargo.toml`.
- Rust core stays pure Rust (no pyo3 in `biofilm-core`); determinism preserved (seeded RNG, identical seed ⇒ identical run).
- Do NOT break the Phase-A swap-contract: `BiofilmProcess`/`ChemostatProcess` output ports `average_concentrations` (delta, map), `population` (delta, float), `time`, and the superset `overwrite[...]` ports keep their names/types/semantics. New ports may be ADDED.
- **Visualizations: read the `dataviz` skill BEFORE writing any chart code.** Theme-aware, accessible, one coherent visual system. Interactive Plotly, `template="plotly_white"` base but follow dataviz palette guidance. Every figure self-contained HTML (`include_plotlyjs="cdn"`).
- Study/investigation conventions mirror Phase A and `~/code/v2ecoli` (study.yaml v4 shapes, investigation.yaml with `executive`/`scientific_argument`/`biological_story`, report_card_verdict/v1 JSON, charts/ + viz/report_card/).
- Timing uses `time.perf_counter()` in normal run scripts (NOT in any workflow context).
- Commit after each task. Verify before claiming done.
- The single investigation this plan builds: **`viva-biofilm-capabilities`** (created in Task 4, referenced by all three studies).

---

### Task 1: Reconcile the unit system + calibrate the biofilm (Rust)

The Phase-A engine mixes diffusivity in µm²/**s** with growth/sink in per-**day**, and the reaction sink is in pg/µm³/day but used against g/m³ fields (the ~1e6 finding). Make the quasi-steady PDE dimensionally consistent (time in **days** throughout) so a *partial* solute gradient forms and the biofilm grows into a real structure. Exact iDynoMiCS match is NOT required — the target is a **visually sensible, growing biofilm with a partial (not flat, not fully-depleted) substrate gradient**.

**Files:**
- Modify: `crates/biofilm-core/src/world.rs` (sink units + diffusivity handling), possibly `crates/biofilm-core/src/grid.rs`
- Test: `crates/biofilm-core/tests/gradient.rs`

**Interfaces:**
- Produces (no signature changes to `new`/`step`/`time`/builders): internal unit convention documented at the top of `world.rs` — length µm, time **days**, concentration g/m³, mass pg. Diffusivity passed to the solver in **µm²/day**. Reaction sink in **g/m³/day** = `-coeff * bio_rate[pg/day] / cell_vol[µm³] * 1e6` (pg/µm³ → g/m³).

- [ ] **Step 1: Write the failing/validation test**
```rust
// crates/biofilm-core/tests/gradient.rs
use biofilm_core::World;

fn build_dev_biofilm() -> World {
    let mut w = World::new();
    w.set_domain(16, 32, 2.0, 32.0);
    // diffusivities are given in µm²/s at the schema layer; here pass the SAME
    // numbers the schema will pass. Task-1's job is to make the solver treat
    // time consistently in days (see world.rs unit note).
    let s = w.add_solute("solute", 1.0, 2000.0, 1500.0, 1.0);
    let o = w.add_solute("oxygen", 8.74, 2000.0, 1500.0, 8.74);
    w.add_reaction(2.05, vec![(s, 2.4), (o, 0.6)], vec![(s, -4.2), (o, -18.0)]);
    w.set_species(0.15, 0.2);
    w.spawn_agents(30, 1.0, 0);
    w.finalize(1234);
    w
}

#[test]
fn developed_biofilm_has_partial_substrate_gradient() {
    let mut w = build_dev_biofilm();
    for _ in 0..40 { w.step(0.05); }
    // A biofilm consuming substrate must draw the substratum concentration
    // DOWN relative to the bulk boundary, but NOT to zero everywhere
    // (a flat field or a fully-depleted field are both calibration failures).
    let sub_bottom = w.solute_row_mean("solute", 0);       // near substratum
    let sub_top = w.solute_row_mean("solute", 31);         // near boundary
    assert!(sub_top > sub_bottom, "expected gradient: top {} > bottom {}", sub_top, sub_bottom);
    assert!(sub_bottom < 0.9 * sub_top, "gradient too weak (nearly flat): {} vs {}", sub_bottom, sub_top);
    assert!(sub_bottom > 0.0, "substrate fully depleted — sink too strong");
}

#[test]
fn biofilm_grows_over_time() {
    let mut w = build_dev_biofilm();
    let p0 = w.population();
    for _ in 0..40 { w.step(0.05); }
    assert!(w.population() > p0, "biofilm should grow: {} -> {}", p0, w.population());
    assert!(w.total_biomass() > 0.0);
}
```

- [ ] **Step 2: Run to see the current (mis-calibrated) behavior**

Run: `cargo test -p biofilm-core --test gradient`
Expected: likely FAIL — either `solute_row_mean` doesn't exist yet (add it), or the gradient is flat/fully-depleted under the current units. Add `pub fn solute_row_mean(&self, name: &str, j: usize) -> f64` (mean of solute field row j; resolve name→index; average over i). Then observe whether the gradient assertions fail.

- [ ] **Step 3: Reconcile units + calibrate**

At the top of `world.rs`, add a unit-convention doc comment (µm, days, g/m³, pg). In `step`/`add_solute`, ensure the diffusivity handed to `solve_steady_state` is in **µm²/day** — convert the incoming µm²/s value once (`* 86400.0`) where the field is constructed (document it). In the sink accumulation, multiply by the `1e6` pg/µm³→g/m³ factor:
```rust
// sink [g/m³/day] = -coeff * bio_rate[pg/day] / cell_vol[µm³] * PG_PER_UM3_TO_G_PER_M3
const PG_PER_UM3_TO_G_PER_M3: f64 = 1.0e6;
sinks[k][self.grid.idx(i, j)] += -coeff * bio_rate / cell_vol * PG_PER_UM3_TO_G_PER_M3;
```
Then TUNE toward the validation test's "partial gradient + growth" target. Levers, in order of preference: (a) confirm the µm²/s→µm²/day diffusivity conversion; (b) if the gradient is still too weak/strong, the dominant balance is `D·∇²C ≈ sink`, so the penetration depth scales as `sqrt(D·C_bulk / sink)` — adjust nothing arbitrary, but you MAY expose/adjust the SOR `max_iter`/tolerance if convergence is the issue. Do NOT fudge the reaction constants away from the `simple.xml` values (mu_max 2.05, Ks 2.4/0.6, yields -4.2/-18) — those are the science. Document in the code comment what the final unit chain is and why the gradient lands where it does.

- [ ] **Step 4: Run to verify it passes**

Run: `cargo test -p biofilm-core --test gradient` then `cargo test -p biofilm-core` (whole crate — determinism/mass-conservation must still pass).
Expected: PASS. The biofilm grows and shows a partial substrate gradient.

- [ ] **Step 5: Rebuild the extension + commit**
```bash
source .venv/bin/activate && maturin develop -m crates/biofilm-py/Cargo.toml
git add -A && git commit -m "fix: reconcile PDE unit system (days) + sink g/m³ conversion; biofilm grows with partial gradient"
```

---

### Task 2: `run_biofilm` snapshot driver (Python)

A reusable helper that runs the biofilm and collects time-stamped snapshots for the viz + studies. Uses `load_world` directly (not the Process) for a tight loop.

**Files:**
- Create: `viva_biofilm/run.py`
- Test: `tests/test_run.py`

**Interfaces:**
- Consumes: `viva_biofilm.schema.load_world`.
- Produces: `run.run_biofilm(spec: dict, n_steps: int, snapshot_every: int = 1) -> list[dict]` — steps the world `n_steps` times (using `spec["dt"]` or a `dt` arg), capturing at t=0 and every `snapshot_every` steps a snapshot dict: `{"time": float, "population": int, "total_biomass": float, "biofilm_thickness": float, "agents": {"x": [...], "y": [...], "radius": [...], "mass": [...], "species": [...]}, "solutes": {name: {"field": [...], "nx": int, "ny": int}}}`. Radius computed from the binding's `agent_radii()`. Also `run.default_spec(**overrides) -> dict` returning the canonical biofilm spec (the `simple.xml` equivalent) with keyword overrides (nx, ny, dx, n_agents, seed, dt).

- [ ] **Step 1: Write the failing test**
```python
# tests/test_run.py
from viva_biofilm.run import run_biofilm, default_spec

def test_run_biofilm_collects_snapshots():
    spec = default_spec(nx=16, ny=32, n_agents=30, seed=7)
    snaps = run_biofilm(spec, n_steps=10, snapshot_every=5, dt=0.05)
    # t=0 plus steps 5 and 10 -> 3 snapshots
    assert len(snaps) == 3
    s = snaps[-1]
    assert set(s.keys()) >= {"time", "population", "total_biomass", "biofilm_thickness", "agents", "solutes"}
    assert len(s["agents"]["x"]) == s["population"]
    assert len(s["agents"]["radius"]) == s["population"]
    assert "solute" in s["solutes"] and s["solutes"]["solute"]["nx"] == 16
    assert len(s["solutes"]["solute"]["field"]) == 16 * 32

def test_run_biofilm_is_deterministic():
    spec = default_spec(seed=7)
    a = run_biofilm(spec, n_steps=8, snapshot_every=8, dt=0.05)[-1]
    b = run_biofilm(spec, n_steps=8, snapshot_every=8, dt=0.05)[-1]
    assert a["population"] == b["population"]
    assert a["agents"]["x"] == b["agents"]["x"]
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/test_run.py -v` → FAIL (module missing).

- [ ] **Step 3: Implement `viva_biofilm/run.py`**

`default_spec(**overrides)` returns the canonical biofilm spec dict (same fields as Phase-A `BIOFILM_SPEC`: domain, solutes solute+oxygen, growth reaction, species, spawn, seed) merged with overrides (`nx`,`ny`,`dx`,`n_agents`→spawn.n, `seed`, etc.). `run_biofilm(spec, n_steps, snapshot_every, dt=0.05)`: `w = load_world(spec)`; snapshot at step 0; loop `w.step(dt)`, and every `snapshot_every` steps append a snapshot built from `w.agent_positions()` (split into x/y), `w.agent_radii()`, `w.agent_masses()`, `w.agent_species()`, `w.population()`, `w.total_biomass()`, `w.biofilm_thickness()`, `w.time()`, and for each solute name `w.solute_field(name)` + `w.grid_shape()`.

- [ ] **Step 4: Run to verify it passes**

Run: `pytest tests/test_run.py -v` → PASS.

- [ ] **Step 5: Commit**
```bash
git add -A && git commit -m "feat: run_biofilm snapshot driver + default_spec"
```

---

### Task 3: `viz.py` visualization module (Plotly)

**READ THE `dataviz` SKILL FIRST.** Produce the reusable figure builders. Aesthetics matter — this is the "beautiful visuals" deliverable.

**Files:**
- Create: `viva_biofilm/viz.py`
- Test: `tests/test_viz.py`

**Interfaces:**
- Consumes: snapshot dicts from `run.run_biofilm`.
- Produces (each returns a `plotly.graph_objects.Figure`):
  - `colony_figure(snapshot, color_by="mass", title=None) -> Figure` — agents as filled circles at (x,y), sized by radius (use marker size scaled to the domain), colored by `mass` or `"local_substrate"` (sampled from the solute field at the agent's cell) with a colorbar; draw the domain box + substratum line; equal aspect; theme-aware.
  - `solute_field_figure(snapshot, name, title=None) -> Figure` — `go.Heatmap` of the solute field reshaped to (ny, nx), y-axis = height above substratum, with a colorbar labeled g/m³; sequential palette per dataviz.
  - `timelapse_figure(snapshots, color_by="mass", title=None) -> Figure` — animated colony scatter across snapshots (Plotly `frames` + a play/pause slider), consistent axis ranges across frames.
  - `growth_curves_figure(snapshots) -> Figure` — population, total_biomass, biofilm_thickness vs time (shared x, secondary axes or small-multiples).
  - `save_html(fig, path)` — writes self-contained interactive HTML (`include_plotlyjs="cdn"`).

- [ ] **Step 1: Read the dataviz skill, then write the failing test**

Invoke the `dataviz` skill and follow its palette/mark/theme guidance. Then:
```python
# tests/test_viz.py
from viva_biofilm.run import run_biofilm, default_spec
from viva_biofilm import viz
import plotly.graph_objects as go

def _snaps():
    return run_biofilm(default_spec(seed=3), n_steps=10, snapshot_every=5, dt=0.05)

def test_colony_figure_has_agent_trace():
    fig = viz.colony_figure(_snaps()[-1], color_by="mass")
    assert isinstance(fig, go.Figure)
    assert len(fig.data) >= 1  # at least the agent scatter

def test_solute_field_figure_is_heatmap():
    fig = viz.solute_field_figure(_snaps()[-1], "solute")
    assert any(t.type == "heatmap" for t in fig.data)

def test_timelapse_has_frames():
    fig = viz.timelapse_figure(_snaps())
    assert len(fig.frames) >= 2

def test_growth_curves_and_save(tmp_path):
    snaps = _snaps()
    fig = viz.growth_curves_figure(snaps)
    assert isinstance(fig, go.Figure)
    p = tmp_path / "g.html"
    viz.save_html(fig, str(p))
    assert p.exists() and p.stat().st_size > 0
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/test_viz.py -v` → FAIL (module missing).

- [ ] **Step 3: Implement `viva_biofilm/viz.py`** following the dataviz skill (palette, theme-aware light/dark where feasible in Plotly, clear axis titles/units, colorbars). Keep each function focused. `color_by="local_substrate"` samples `snapshot["solutes"]["solute"]["field"]` at each agent's grid cell (compute cell from x,y,dx,nx). Marker sizing: convert agent radius (µm) to pixels via the axis range so circles read true-to-scale.

- [ ] **Step 4: Run to verify it passes**

Run: `pytest tests/test_viz.py -v` → PASS.

- [ ] **Step 5: Commit**
```bash
git add -A && git commit -m "feat: viz.py Plotly figures (colony, solute heatmap, time-lapse, growth curves)"
```

---

### Task 4: Study — spatial-biofilm-growth (+ investigation)

The headline space-simulation study. Runs a developed biofilm and renders the beautiful figures; creates the investigation.

**Files:**
- Create: `workspace/investigations/viva-biofilm-capabilities/investigation.yaml`
- Create: `workspace/studies/spatial-biofilm-growth/study.yaml`
- Create: `workspace/studies/spatial-biofilm-growth/run_study.py`
- Create: `workspace/studies/spatial-biofilm-growth/tests/test_spatial.py`

**Interfaces:**
- Consumes: `run.run_biofilm`, `viz.*`.
- Produces: `run_study.py` writes `charts/colony_final.html`, `charts/solute_substrate.html`, `charts/solute_oxygen.html`, `charts/timelapse.html`, `charts/growth_curves.html`, and `viz/report_card/report_card_verdict.json` (axes: final population in a sensible band, biofilm_thickness > 0, substrate gradient present = within_tol). study.yaml v4 + investigation.yaml.

- [ ] **Step 1: Write the failing test**
```python
# workspace/studies/spatial-biofilm-growth/tests/test_spatial.py
from viva_biofilm.run import run_biofilm, default_spec

def test_developed_biofilm_grows_and_has_gradient():
    snaps = run_biofilm(default_spec(nx=32, ny=48, n_agents=40, seed=11), n_steps=60, snapshot_every=15, dt=0.05)
    first, last = snaps[0], snaps[-1]
    assert last["population"] > first["population"]          # grew
    assert last["biofilm_thickness"] > 0.0
    # substrate gradient: mean of bottom third < mean of top third of the field
    f = last["solutes"]["solute"]; nx, ny = f["nx"], f["ny"]
    field = f["field"]
    bottom = sum(field[0:nx*(ny//3)]) / (nx*(ny//3))
    top = sum(field[nx*2*(ny//3):nx*ny]) / (nx*(ny - 2*(ny//3)))
    assert top > bottom
```

- [ ] **Step 2: Run to verify it fails/passes**

Run: `pytest workspace/studies/spatial-biofilm-growth/tests/ -v`. It may PASS immediately (engine calibrated in Task 1) — acceptable; the deliverable is the full study. If it FAILS on gradient/growth, that indicates Task-1 calibration regressed — investigate before proceeding.

- [ ] **Step 3: Write the study artifacts**

`run_study.py`: `snaps = run_biofilm(default_spec(nx=32, ny=48, n_agents=40, seed=11), n_steps=120, snapshot_every=10, dt=0.05)`; build + `viz.save_html` the five figures into `charts/`; compute report-card verdict (population band e.g. `> 40` (grew) → within_tol; thickness > 0 → within_tol; gradient `top>bottom` → within_tol else mismatch); write `viz/report_card/report_card_verdict.json` (schema report_card_verdict/v1, group `spatial-structure`). Write `study.yaml` (v4: question, conditions.baseline.composite `viva_biofilm.composites.biofilm.biofilm`, tests with a report_card_axis measure at the verdict dir, visualizations listing the charts, embed_visualizations for the interactive HTML, pipeline_gate). Write `investigation.yaml` (schema_version 2, name `viva-biofilm-capabilities`, the three narrative sections, `members: [spatial-biofilm-growth]` — later tasks append the other studies).

- [ ] **Step 4: Run the study + tests**
```bash
python workspace/studies/spatial-biofilm-growth/run_study.py
pytest workspace/studies/spatial-biofilm-growth/tests/ -v
```
Expected: five `charts/*.html` exist; verdict JSON written; test passes. Open `charts/colony_final.html` and `charts/timelapse.html` to eyeball quality (note in report if a figure looks degenerate).

- [ ] **Step 5: Commit**
```bash
git add -A && git commit -m "feat: spatial-biofilm-growth study (colony/heatmap/time-lapse) + capabilities investigation"
```

---

### Task 5: Study — runtime-and-scaling (3 axes)

**Files:**
- Create: `workspace/studies/runtime-and-scaling/study.yaml`
- Create: `workspace/studies/runtime-and-scaling/run_study.py`
- Create: `workspace/studies/runtime-and-scaling/tests/test_scaling.py`

**Interfaces:**
- Consumes: `run.run_biofilm`/`load_world`, `viz`/Plotly, `time.perf_counter`.
- Produces: a benchmarking script that measures wall-time and throughput while sweeping three axes, writes `charts/scaling_grid.html`, `charts/scaling_population.html`, `charts/scaling_duration.html`, `charts/throughput.html`, and a report_card verdict recording measured throughput (agent-steps/sec) + that scaling curves were produced.

- [ ] **Step 1: Write the failing test**
```python
# workspace/studies/runtime-and-scaling/tests/test_scaling.py
import time
from viva_biofilm.schema import load_world
from viva_biofilm.run import default_spec

def test_step_timing_is_measurable_and_bounded():
    w = load_world(default_spec(nx=16, ny=32, n_agents=30, seed=1))
    t0 = time.perf_counter()
    for _ in range(5):
        w.step(0.05)
    dt = time.perf_counter() - t0
    assert dt > 0.0
    assert dt < 60.0, f"5 steps took {dt:.1f}s — unexpectedly slow at the small grid"

def test_larger_grid_costs_more_than_small():
    def timeit(nx, ny):
        w = load_world(default_spec(nx=nx, ny=ny, n_agents=30, seed=1))
        t0 = time.perf_counter()
        for _ in range(3):
            w.step(0.05)
        return time.perf_counter() - t0
    small = timeit(16, 32)
    big = timeit(64, 128)
    assert big > small  # more cells -> more PDE work
```

- [ ] **Step 2: Run to verify it fails/behaves**

Run: `pytest workspace/studies/runtime-and-scaling/tests/ -v`. These may pass immediately (engine exists). If `test_larger_grid_costs_more` is flaky due to timing noise, widen the ratio margin (assert `big > small * 1.2`). If a large grid is extremely slow (>60s for a few steps), that is a real finding — report DONE_WITH_CONCERNS and note the SOR cost; do NOT silently shrink the sweep.

- [ ] **Step 3: Write the benchmarking script + study**

`run_study.py`: three sweeps, each timing N steps with `time.perf_counter()`:
1. **grid**: nx×ny in `[(16,32),(32,64),(64,128),(96,192)]`, fixed 30 agents, plot wall-time/step vs cells (log-log).
2. **population**: fixed grid, initial `n_agents` in `[10,30,60,120,240]` (or grow to targets), plot wall-time/step vs agent count.
3. **duration**: fixed grid+agents, total steps in `[20,40,80,160]`, plot total wall-time vs steps (should be ~linear).
Also a throughput figure (agent-steps/sec). Write the four `charts/*.html` (Plotly, dataviz-styled, log axes where appropriate) and a report_card verdict recording the measured peak throughput and that all three curves were produced. Write `study.yaml` (v4; baseline composite `viva_biofilm.composites.biofilm.biofilm`; a test asserting throughput was recorded / curves exist via report_card_axis; visualizations + embed_visualizations; pipeline_gate prerequisites `[spatial-biofilm-growth]`). Keep each sweep's per-point step count modest (e.g. 10) so the whole script runs in a few minutes; `log()`/print what sizes ran and the timings.

- [ ] **Step 4: Run + tests**
```bash
python workspace/studies/runtime-and-scaling/run_study.py
pytest workspace/studies/runtime-and-scaling/tests/ -v
```
Expected: four charts + verdict written; tests pass. Add `runtime-and-scaling` to the investigation's `members`.

- [ ] **Step 5: Commit**
```bash
git add -A && git commit -m "feat: runtime-and-scaling study (grid/population/duration sweeps + throughput)"
```

---

### Task 6: Runtime boundary-concentration hook (Rust → binding → process)

Wire the Phase-A `boundary_concentrations` input through to the Rust boundary so an external process can perturb the environment. This is the composability plumbing.

**Files:**
- Modify: `crates/biofilm-core/src/world.rs` (add `set_bulk`), `crates/biofilm-py/src/lib.rs` (expose it), `viva_biofilm/processes/biofilm_process.py` (apply the input)
- Test: `crates/biofilm-core/tests/boundary.rs`, add to `tests/test_process.py`

**Interfaces:**
- Produces: Rust `World::set_bulk(&mut self, name_index: usize, value: f64)` sets a solute's `bulk` (the top Dirichlet value used each `step`). pyo3 `World.set_bulk_by_name(name: str, value: float)` (resolve name→index in the wrapper; raise `PyValueError` on unknown). `BiofilmProcess.update` reads `state["boundary_concentrations"]` (map name→value) and calls `set_bulk_by_name` for each before stepping.

- [ ] **Step 1: Write the failing tests**
```rust
// crates/biofilm-core/tests/boundary.rs
use biofilm_core::World;
#[test]
fn set_bulk_changes_steady_state_toward_new_boundary() {
    let mut w = World::new();
    w.set_domain(8, 16, 2.0, 16.0);
    let s = w.add_solute("solute", 1.0, 2000.0, 1500.0, 1.0);
    w.set_species(0.15, 0.2);
    w.finalize(1);
    w.step(0.05);
    let before = w.solute_row_mean("solute", 15);
    w.set_bulk(s, 5.0);
    w.step(0.05);
    let after = w.solute_row_mean("solute", 15);
    assert!(after > before, "raising bulk should raise the boundary row: {} -> {}", before, after);
}
```
```python
# add to tests/test_process.py
def test_biofilm_process_honors_boundary_concentrations():
    import process_bigraph as pb
    from viva_biofilm.processes.biofilm_process import BiofilmProcess
    from tests.test_schema import BIOFILM_SPEC
    core = pb.allocate_core()
    proc = BiofilmProcess({"spec": BIOFILM_SPEC, "dt_per_update": 0.05}, core=core)
    out0 = proc.update({"boundary_concentrations": {}}, 0.05)
    # push oxygen boundary up; the average oxygen delta should reflect the change over subsequent steps
    out1 = proc.update({"boundary_concentrations": {"oxygen": 20.0}}, 0.05)
    assert "average_concentrations" in out1  # does not raise; input accepted and applied
```

- [ ] **Step 2: Run to verify they fail**

Run: `cargo test -p biofilm-core --test boundary` (FAIL: `set_bulk` missing) and `pytest tests/test_process.py::test_biofilm_process_honors_boundary_concentrations` (FAIL until process reads the input + binding exposes `set_bulk_by_name`).

- [ ] **Step 3: Implement**

Rust `world.rs`: `pub fn set_bulk(&mut self, k: usize, value: f64) { self.solutes[k].bulk = value; }`. pyo3 `lib.rs`: `fn set_bulk_by_name(&mut self, name: &str, value: f64) -> PyResult<()>` resolving via the wrapper's name→index map (raise `PyValueError` if unknown). `biofilm_process.py` `update`: `for name, val in (state or {}).get("boundary_concentrations", {}).items(): self.world.set_bulk_by_name(name, float(val))` BEFORE `self.world.step(self.dt)`. Update the class docstring (boundary hook now live, not deferred). Rebuild: `maturin develop -m crates/biofilm-py/Cargo.toml`.

- [ ] **Step 4: Run to verify they pass**

Run: `cargo test -p biofilm-core` (all), rebuild, `pytest -q` (all). Expected PASS.

- [ ] **Step 5: Commit**
```bash
git add -A && git commit -m "feat: runtime boundary-concentration hook (set_bulk) wired through to BiofilmProcess"
```

---

### Task 7: Controller process + composite + composability study

**Files:**
- Create: `viva_biofilm/processes/controller_process.py`
- Create: `viva_biofilm/composites/biofilm_controlled.composite.yaml`
- Modify: `viva_biofilm/core.py` (register `BoundaryControllerProcess`)
- Create: `workspace/studies/composability/{study.yaml, run_study.py, tests/test_composability.py}`
- Test: add controller registration to `tests/test_core_registration.py`

**Interfaces:**
- Produces: `BoundaryControllerProcess(Process)` — config `{schedule: "tree", solute: string}`; no inputs; output `{"boundary_concentrations": "map[string,float]"}`; `update` emits the scheduled boundary value for the current time (piecewise-constant schedule of `[time, value]` points). `core.build_core()` also registers it. `biofilm_controlled.composite.yaml` wires controller.boundary_concentrations → biofilm.boundary_concentrations via a shared store.

- [ ] **Step 1: Write the failing tests**
```python
# tests/test_core_registration.py — add
def test_controller_registered():
    from viva_biofilm.core import build_core
    core = build_core()
    assert "BoundaryControllerProcess" in core.link_registry
```
```python
# workspace/studies/composability/tests/test_composability.py
import process_bigraph as pb
from viva_biofilm.processes.controller_process import BoundaryControllerProcess

def test_controller_emits_scheduled_value():
    core = pb.allocate_core()
    proc = BoundaryControllerProcess({"schedule": [[0.0, 8.74], [1.0, 2.0]], "solute": "oxygen"}, core=core)
    out_early = proc.update({}, 0.0)
    assert out_early["boundary_concentrations"]["oxygen"] == 8.74
```

- [ ] **Step 2: Run to verify they fail**

Run the two tests → FAIL (module/registration missing).

- [ ] **Step 3: Implement**

`controller_process.py`: `BoundaryControllerProcess` tracks time `t`; `update(state, interval)` advances `t += interval`, finds the latest schedule point with `time <= t`, returns `{"boundary_concentrations": {self.solute: value}}`. Register in `core.py`. `biofilm_controlled.composite.yaml`: two process nodes (controller + biofilm) sharing a `boundary_concentrations` store (controller outputs → store; biofilm inputs ← store). `run_study.py`: run the controlled composite over time (via `run_biofilm`-style loop that also steps the controller, or drive the composite), stepping oxygen boundary down at t=1d and back up at t=3d; collect population/biomass/oxygen-mean vs time; render `charts/response.html` (growth + the perturbation schedule overlaid) and a before/after colony snapshot pair; write a report_card verdict asserting the biofilm RESPONDS (e.g. growth-rate slows during the low-oxygen window). study.yaml v4 (baseline composite `viva_biofilm.composites.biofilm_controlled.biofilm_controlled`), narrative. Add `composability` to the investigation `members`.

- [ ] **Step 4: Run + tests**
```bash
maturin develop -m crates/biofilm-py/Cargo.toml   # only if Rust changed (it didn't in this task)
python workspace/studies/composability/run_study.py
pytest -q
```
Expected: charts + verdict written; the response figure shows a visible dip during the perturbation; all tests pass.

- [ ] **Step 5: Commit**
```bash
git add -A && git commit -m "feat: boundary controller process + biofilm_controlled composite + composability study"
```

---

### Task 8: Workbench hosting + embed + render

Make the investigation viewable on the vivarium-workbench dashboard and embed the interactive figures.

**Files:**
- Modify: `pyproject.toml` (add `vivarium-workbench` to dev deps), `workspace.yaml` (ensure layout/observables consistent), study `study.yaml` files (`embed_visualizations` pointing at `reports/figures/<study>/...` or the `charts/` HTML), `README.md` (how to view)
- Create: `reports/figures/<study>/*.html` (copies/links of the interactive figures, per the v2ecoli embed convention) if the workbench expects them there
- Test: `tests/test_workspace_serves.py` (smoke)

**Interfaces:**
- Produces: `vivarium-workbench` installed in `.venv`; the workspace lints/serves; each study embeds its interactive HTML; a rendered report.

- [ ] **Step 1: Install the workbench + write a smoke test**
```bash
source .venv/bin/activate
uv pip install -e ~/code/vivarium-workbench   # editable from local checkout
python -c "import vivarium_workbench; print('workbench ok')"
```
If the install hits numpy/numba conflicts (known risk), try `uv pip install -e ~/code/vivarium-workbench --no-deps` then add only the missing runtime deps it names; document what was needed. If it cannot be made to work cleanly, STOP and report DONE_WITH_CONCERNS — do NOT downgrade numpy in a way that breaks the maturin-built extension (verify `python -c "from viva_biofilm import biofilm_core"` still imports after any dep change).
```python
# tests/test_workspace_serves.py
import subprocess, sys
def test_workspace_lints_or_report_renders():
    # Prefer the workspace report renderer; assert it produces index.html without raising.
    from pathlib import Path
    from vivarium_workbench.lib.report import render_workspace_report
    render_workspace_report(Path("."))
    assert Path("reports/index.html").exists()
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/test_workspace_serves.py -v` → FAIL until workbench installed + embeds wired. (If the renderer import path differs, mirror the exact call used in `~/code/v2ecoli` — read how it renders — and keep the assertion "index.html is produced".)

- [ ] **Step 3: Wire embeds + add workbench dep**

Add `vivarium-workbench` to `pyproject.toml` `[project.optional-dependencies].dev`. In each study.yaml add `embed_visualizations` entries pointing at the study's interactive charts (follow the v2ecoli convention — `reports/figures/<study>/<name>.html` or a `charts/` URL). Copy or symlink the study `charts/*.html` into `reports/figures/<study>/` if the renderer serves from there. Update `README.md` with: build (`maturin develop`), test (`pytest`), run studies (`python workspace/studies/<s>/run_study.py`), and serve the workbench (`vivarium-workbench serve --workspace .` — confirm the exact command from the workbench CLI).

- [ ] **Step 4: Render + verify**

Run: `python -c "from pathlib import Path; from vivarium_workbench.lib.report import render_workspace_report; render_workspace_report(Path('.'))"` then `pytest tests/test_workspace_serves.py -v`. Confirm `reports/index.html` exists and references the three studies. Optionally launch `vivarium-workbench serve --workspace .` and confirm it boots (note the URL in the report).

- [ ] **Step 5: Commit**
```bash
git add -A && git commit -m "feat: workbench hosting — install dep, embed interactive figures, render report"
```

---

## Final verification (after all tasks)

- [ ] `cargo test -p biofilm-core` — all Rust tests pass (incl. gradient, boundary, determinism).
- [ ] `maturin develop -m crates/biofilm-py/Cargo.toml && pytest -q` — all Python tests pass.
- [ ] Each study's `run_study.py` runs and writes its charts + verdict.
- [ ] `reports/index.html` renders and shows the `viva-biofilm-capabilities` investigation with all three studies.
- [ ] Eyeball `charts/colony_final.html`, `charts/timelapse.html`, `charts/solute_substrate.html`, the scaling curves, and the composability response — confirm they are visually sensible, not degenerate.
- [ ] Swap-contract intact: `BiofilmProcess`/`ChemostatProcess` Phase-A ports unchanged in name/type/semantics.

## Out of scope (→ Plan 2)

- Perf/equivalence vs the REAL Java iDynoMiCS-2 (needs a JVM + Java 11 + `pbg-idynomics2` working here). That is the separate Plan 2, gated on getting the Java engine running on this machine.
