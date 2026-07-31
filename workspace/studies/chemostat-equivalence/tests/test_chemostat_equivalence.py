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
