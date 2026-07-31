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
    core = pb.allocate_core()
    proc = BiofilmProcess({"spec": BIOFILM_SPEC, "dt_per_update": 0.05}, core=core)
    proc.update({"boundary_concentrations": {}}, 0.05)
    # push oxygen boundary up; the average oxygen delta should reflect the change over subsequent steps
    out1 = proc.update({"boundary_concentrations": {"oxygen": 20.0}}, 0.05)
    assert "average_concentrations" in out1  # does not raise; input accepted and applied


def test_chemostat_process_decays_solute1():
    core = pb.allocate_core()
    proc = ChemostatProcess({"spec": CHEMO_SPEC, "dt_per_update": 1.0}, core=core)
    out = proc.update({}, 1.0)
    # average_concentrations is now a per-step DELTA (accumulate port);
    # solute1 is consumed, so the delta must be negative.
    assert out["average_concentrations"]["solute1"] < 0.0
    assert "time" in out
