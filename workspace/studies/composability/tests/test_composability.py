import process_bigraph as pb

from viva_biofilm.processes.controller_process import BoundaryControllerProcess
from viva_biofilm.processes.biofilm_process import BiofilmProcess
from viva_biofilm.run import default_spec


def test_controller_emits_scheduled_value():
    core = pb.allocate_core()
    proc = BoundaryControllerProcess({"schedule": [[0.0, 8.74], [1.0, 2.0]], "solute": "oxygen"}, core=core)
    out_early = proc.update({}, 0.0)
    assert out_early["boundary_concentrations"]["oxygen"] == 8.74
    out_late = proc.update({}, 1.5)
    assert out_late["boundary_concentrations"]["oxygen"] == 2.0


def _run_coupled(schedule: list, n_steps: int, dt: float = 0.05, seed: int = 1234):
    """Drive controller -> biofilm in lockstep: controller.update() first each
    step, its boundary_concentrations output fed straight into
    biofilm.update()'s boundary_concentrations input (mirrors the
    biofilm_controlled composite's shared store, driven manually)."""
    core = pb.allocate_core()
    controller = BoundaryControllerProcess({"schedule": schedule, "solute": "oxygen"}, core=core)
    biofilm = BiofilmProcess({"spec": default_spec(seed=seed), "dt_per_update": dt}, core=core)
    for _ in range(n_steps):
        boundary = controller.update({}, dt)["boundary_concentrations"]
        biofilm.update({"boundary_concentrations": boundary}, dt)
    return biofilm.world.population(), biofilm.world.total_biomass()


def test_low_oxygen_perturbation_reduces_growth_vs_control():
    # Same seed both runs -- only the oxygen boundary schedule differs. Driving
    # oxygen far below its Monod half-saturation (ks=0.6) from t=0 should
    # visibly slow growth relative to an un-perturbed control held at the
    # spec's normal bulk oxygen (8.74).
    perturbed_pop, perturbed_biomass = _run_coupled([[0.0, 0.05]], n_steps=40)
    control_pop, control_biomass = _run_coupled([[0.0, 8.74]], n_steps=40)

    assert perturbed_pop <= control_pop
    assert perturbed_biomass < control_biomass
