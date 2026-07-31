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
