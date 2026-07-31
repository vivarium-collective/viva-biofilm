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
