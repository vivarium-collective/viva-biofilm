import copy

from viva_biofilm.schema import load_world

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
    # Wanner-Gujer surface erosion rate (Task 2), units 1/(um*day). Without
    # this, Task 1's fix (substrate limitation genuinely biting -> linear,
    # not exponential, growth) still leaves thickness growing unboundedly
    # (a surface active layer marching upward forever). Calibrated by sweep
    # at this spec's kinetics/dt=0.05: k_det=0.01 plateaus biofilm_thickness
    # in the ~30-35um range by ~t=15-20 days (see
    # .superpowers/sdd/substrate-limitation/task2-erosion-report.md).
    "detachment_rate": 0.01,
}


def default_spec(**overrides) -> dict:
    """Return the canonical biofilm spec, deep-copied, with keyword overrides applied.

    Supported override keys: nx, ny, dx, layer_thickness (-> domain),
    n_agents, band_height, seed_offset (-> spawn), seed, dt (passthrough).
    """
    spec = copy.deepcopy(BIOFILM_SPEC)

    domain_keys = {"nx", "ny", "dx", "layer_thickness"}
    spawn_map = {"n_agents": "n", "band_height": "band_height", "seed_offset": "seed_offset"}

    for key, value in overrides.items():
        if key in domain_keys:
            spec["domain"][key] = value
        elif key in spawn_map:
            spec["spawn"][spawn_map[key]] = value
        elif key == "seed":
            spec["seed"] = value
        elif key == "dt":
            spec["dt"] = value
        else:
            spec[key] = value

    return spec


def _snapshot(w) -> dict:
    positions = w.agent_positions()
    xs = [p[0] for p in positions]
    ys = [p[1] for p in positions]
    nx, ny = w.grid_shape()

    solutes = {}
    for name in w.solute_means().keys():
        solutes[name] = {
            "field": w.solute_field(name),
            "nx": nx,
            "ny": ny,
        }

    return {
        "time": w.time(),
        "population": w.population(),
        "total_biomass": w.total_biomass(),
        "biofilm_thickness": w.biofilm_thickness(),
        "agents": {
            "x": xs,
            "y": ys,
            "radius": w.agent_radii(),
            "mass": w.agent_masses(),
            "species": w.agent_species(),
        },
        "solutes": solutes,
    }


# RS-vs-YS competition (Fig. 5 reproduction) — S1 Text Table K / Kreft 2004,
# converted to engine units (length um, time DAYS, conc g/m^3 == mg/L, mass
# pg). Rate Strategist: higher mu_max, lower yield. Yield Strategist: lower
# mu_max, higher (more efficient) yield. Consistency check baked into the
# numbers: RS mu_max is exactly 2x YS mu_max; RS yield coeff is exactly half
# YS's; RS and YS share the same specific affinity by construction (each
# strategy's mu_max/Kox ratio is equal — 6.566 in these day-unit numbers, the
# engine-unit counterpart of the paper's 0.274 hourly-unit figure); RS
# consumes 2x the oxygen per unit biomass that YS does.
RATE_STRATEGIST = {
    "name": "RS",
    "density": 0.1363,
    "division_mass": 0.08,
    "mu_max": 3.9398,
    "monod": [["oxygen", 0.6]],
    "yields": [["oxygen", -43.478]],
}
YIELD_STRATEGIST = {
    "name": "YS",
    "density": 0.1363,
    "division_mass": 0.08,
    "mu_max": 1.9699,
    "monod": [["oxygen", 0.3]],
    "yields": [["oxygen", -21.739]],
}


def competition_spec(
    n_each: int,
    rs: dict | None = None,
    ys: dict | None = None,
    seed: int = 42,
    nx: int = 128,
    ny: int = 64,
    dx: float = 1.5625,
) -> dict:
    """Build a two-strategy (RS vs YS) competition spec over oxygen.

    Domain defaults to a ~200um-wide 2D slab (nx=128, ny=64, dx=1.5625um,
    layer_thickness=40um). Oxygen is the sole solute (init/bulk 1.0,
    diffusivity 2000 um^2/s in both liquid and biofilm phases — the paper
    gives a single value). `n_each` agents of each strategy are seeded via
    the distributed spawner; RS is strategy/species index 0, YS is index 1.

    `rs`/`ys` optionally override the default Table-K RS/YS parameter
    dicts (each shaped like `RATE_STRATEGIST`/`YIELD_STRATEGIST`).
    """
    rs_params = {**RATE_STRATEGIST, **(rs or {})}
    ys_params = {**YIELD_STRATEGIST, **(ys or {})}

    return {
        "domain": {"nx": nx, "ny": ny, "dx": dx, "layer_thickness": 40.0},
        "solutes": [
            {"name": "oxygen", "init": 1.0, "diff_liquid": 2000.0, "diff_biofilm": 2000.0, "bulk": 1.0},
        ],
        "strategies": [
            {**rs_params, "spawn_n": n_each, "seed_offset": 0},
            {**ys_params, "spawn_n": n_each, "seed_offset": 1},
        ],
        "seed": seed,
        # Wanner-Gujer surface erosion rate (Task 2), units 1/(um*day).
        # Same balance used for BIOFILM_SPEC (erosion engages once
        # Delta = k_det*h^2*dt approaches an agent's own radius r; RS/YS
        # r ~ 0.35-0.43um at this dt=1/24) predicts a threshold around
        # k_det ~ 0.013-0.02. Confirmed empirically on a reduced-domain
        # proxy (same species/dt, smaller nx/ny for tractable runtime):
        # k_det=0.013 showed no measurable effect through day 37.5
        # (right at threshold), k_det=0.02 measurably suppressed thickness
        # by day 37.5 (21.0um vs 23.0um no-erosion control) and was still
        # trending flatter — see task2-erosion-report.md for the full
        # numbers and the caveat that a full multi-week plateau (as opposed
        # to this trend) was not confirmed at the PRODUCTION nx=128,ny=64
        # domain due to runtime cost.
        "detachment_rate": 0.02,
    }


def run_competition(
    spec: dict, n_steps: int, dt: float = 1 / 24, snapshot_every: int = 1
) -> list[dict]:
    """Run a multi-strategy competition world, collecting snapshots.

    Like `run_biofilm`, but each snapshot also carries `pop_by_strategy`
    and `biomass_by_strategy` (lists indexed by species/strategy index, RS
    first at 0, YS at 1) alongside the usual per-agent `species` array.
    Default dt = 1/24 day (paper's Delta-t = 1 hour).
    """
    step_dt = spec.get("dt", dt)
    w = load_world(spec)
    n_strategies = len(spec.get("strategies", []))

    def snap() -> dict:
        s = _snapshot(w)
        s["pop_by_strategy"] = [w.population_of(i) for i in range(n_strategies)]
        s["biomass_by_strategy"] = [w.biomass_of(i) for i in range(n_strategies)]
        return s

    snapshots = [snap()]
    for i in range(1, n_steps + 1):
        w.step(step_dt)
        if i % snapshot_every == 0:
            snapshots.append(snap())

    if n_steps % snapshot_every != 0:
        snapshots.append(snap())

    return snapshots


def run_biofilm(spec: dict, n_steps: int, snapshot_every: int = 1, dt: float = 0.05) -> list[dict]:
    """Run the biofilm world for n_steps, collecting snapshots.

    Captures a snapshot at t=0 and every `snapshot_every` steps thereafter.
    """
    step_dt = spec.get("dt", dt)
    w = load_world(spec)

    snapshots = [_snapshot(w)]
    for i in range(1, n_steps + 1):
        w.step(step_dt)
        if i % snapshot_every == 0:
            snapshots.append(_snapshot(w))

    if n_steps % snapshot_every != 0:
        snapshots.append(_snapshot(w))

    return snapshots
