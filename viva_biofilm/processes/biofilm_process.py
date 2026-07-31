from process_bigraph import Process
from viva_biofilm.schema import load_world


class BiofilmProcess(Process):
    """2D single-species biofilm — Rust reimplementation of iDynoMiCS-2 core.

    Output ports average_concentrations/population/time are name/type-compatible
    with pbg-idynomics2's IDynoMiCS2Process so the two engines are swappable.
    (boundary_concentrations input is accepted now; driving the Rust boundary
    from it is the Phase C composability step.)
    """

    config_schema = {
        "spec": "tree",
        "dt_per_update": {"_type": "float", "_default": 0.05},
    }

    def initialize(self, config):
        self.world = load_world(config["spec"])
        self.dt = float(config["dt_per_update"])

    def inputs(self):
        return {"boundary_concentrations": "map[string,float]"}

    def outputs(self):
        return {
            "average_concentrations": "map[string,float]",
            "population": "float",
            "time": "overwrite[float]",
            "total_biomass": "overwrite[float]",
            "biofilm_thickness": "overwrite[float]",
            "agent_positions": "overwrite[list]",
            "agent_masses": "overwrite[list]",
            "agent_radii": "overwrite[list]",
            "agent_species": "overwrite[list]",
            "solute_fields": "overwrite[map[list]]",
            "grid_shape": "overwrite[list]",
        }

    def update(self, state, interval):
        self.world.step(self.dt)
        means = self.world.solute_means()
        return {
            "average_concentrations": means,
            "population": float(self.world.population()),
            "time": float(self.world.time()),
            "total_biomass": float(self.world.total_biomass()),
            "biofilm_thickness": float(self.world.biofilm_thickness()),
            "agent_positions": [list(p) for p in self.world.agent_positions()],
            "agent_masses": list(self.world.agent_masses()),
            "agent_radii": list(self.world.agent_radii()),
            "agent_species": list(self.world.agent_species()),
            "solute_fields": {n: list(self.world.solute_field(n)) for n in means.keys()},
            "grid_shape": list(self.world.grid_shape()),
        }
