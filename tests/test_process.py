import pytest

import process_bigraph as pb
from viva_biofilm.processes.biofilm_process import BiofilmProcess
from viva_biofilm.processes.chemostat_process import ChemostatProcess
from viva_biofilm.run import BIOFILM_SPEC

CHEMO_SPEC = {
    "solutes": [{"name": "solute1", "init": 2.0}, {"name": "solute2", "init": 2.0}],
    "reactions": [{"substrate": "solute1", "k": 0.1, "stoich": [["solute1", -1.0], ["solute2", 0.5]]}],
}

def test_biofilm_process_update_returns_readbacks():
    core = pb.allocate_core()
    proc = BiofilmProcess({"spec": BIOFILM_SPEC, "dt_per_update": 0.05}, core=core)
    out = proc.update({"boundary_concentrations": {}}, 0.05)
    # population is now a per-step DELTA (accumulate port); a single early
    # step shouldn't lose agents, but it won't have grown to >=30 either.
    assert out["population"] >= 0.0
    assert set(out["average_concentrations"].keys()) == {"solute", "oxygen"}
    # agent_positions/masses are ABSOLUTE (overwrite) superset ports — the
    # true current agent count lives there, not in the population delta.
    assert len(out["agent_positions"]) == len(out["agent_masses"])
    assert len(out["agent_positions"]) >= 30
    assert len(out["agent_masses"]) >= 30
    assert out["grid_shape"] == [16, 32]
    assert "solute" in out["solute_fields"]
    assert len(out["solute_fields"]["solute"]) == 16 * 32

def test_biofilm_process_honors_boundary_concentrations():
    # average_concentrations is a per-step DELTA (accumulate) port. BIOFILM_SPEC's
    # oxygen bulk starts at 8.74, and grid.rs force-sets the Dirichlet top row to
    # `bulk` on every solve — so once the field is near steady state (delta ~0),
    # raising the bulk mid-run must pull the field up, producing a clearly
    # POSITIVE oxygen delta on the very next update. Two identically-seeded
    # processes (same spec -> same deterministic RNG/agent placement) are warmed
    # up in lockstep with empty boundary_concentrations; only one then gets the
    # push. If the wiring were broken (loop never runs, wrong arg order, the
    # pyo3 call silently swallowed), the pushed run would be indistinguishable
    # from the control — this comparison catches that.
    def make_proc():
        return BiofilmProcess({"spec": BIOFILM_SPEC, "dt_per_update": 0.05}, core=pb.allocate_core())

    proc_pushed = make_proc()
    proc_control = make_proc()

    for _ in range(3):
        proc_pushed.update({"boundary_concentrations": {}}, 0.05)
        proc_control.update({"boundary_concentrations": {}}, 0.05)

    out_pushed = proc_pushed.update({"boundary_concentrations": {"oxygen": 20.0}}, 0.05)
    out_control = proc_control.update({"boundary_concentrations": {}}, 0.05)

    assert out_pushed["average_concentrations"]["oxygen"] > 0.0
    assert out_pushed["average_concentrations"]["oxygen"] > out_control["average_concentrations"]["oxygen"]


def test_biofilm_process_boundary_concentrations_unknown_solute_raises():
    core = pb.allocate_core()
    proc = BiofilmProcess({"spec": BIOFILM_SPEC, "dt_per_update": 0.05}, core=core)
    with pytest.raises(ValueError):
        proc.update({"boundary_concentrations": {"nonesuch": 1.0}}, 0.05)


def test_chemostat_process_decays_solute1():
    core = pb.allocate_core()
    proc = ChemostatProcess({"spec": CHEMO_SPEC, "dt_per_update": 1.0}, core=core)
    out = proc.update({}, 1.0)
    # average_concentrations is now a per-step DELTA (accumulate port);
    # solute1 is consumed, so the delta must be negative.
    assert out["average_concentrations"]["solute1"] < 0.0
    assert "time" in out
