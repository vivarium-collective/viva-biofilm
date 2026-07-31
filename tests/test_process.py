import process_bigraph as pb
from viva_biofilm.processes.biofilm_process import BiofilmProcess
from viva_biofilm.processes.chemostat_process import ChemostatProcess
from tests.test_schema import BIOFILM_SPEC

CHEMO_SPEC = {
    "solutes": [{"name": "solute1", "init": 2.0}, {"name": "solute2", "init": 2.0}],
    "reactions": [{"substrate": "solute1", "k": 0.1, "stoich": [["solute1", -1.0], ["solute2", 0.5]]}],
}

def test_biofilm_process_update_returns_readbacks():
    core = pb.allocate_core()
    proc = BiofilmProcess({"spec": BIOFILM_SPEC, "dt_per_update": 0.05}, core=core)
    out = proc.update({"boundary_concentrations": {}}, 0.05)
    assert out["population"] >= 30
    assert set(out["average_concentrations"].keys()) == {"solute", "oxygen"}
    assert len(out["agent_positions"]) == int(out["population"])
    assert len(out["agent_masses"]) == int(out["population"])
    assert out["grid_shape"] == [16, 32]
    assert "solute" in out["solute_fields"]
    assert len(out["solute_fields"]["solute"]) == 16 * 32

def test_chemostat_process_decays_solute1():
    core = pb.allocate_core()
    proc = ChemostatProcess({"spec": CHEMO_SPEC, "dt_per_update": 1.0}, core=core)
    out = proc.update({}, 1.0)
    assert out["average_concentrations"]["solute1"] < 2.0
    assert "time" in out
