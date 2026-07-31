import process_bigraph as pb
from viva_biofilm.processes.biofilm_process import BiofilmProcess
from tests.test_schema import BIOFILM_SPEC

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
