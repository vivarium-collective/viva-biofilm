from process_bigraph import Process
from viva_biofilm.schema import load_chemostat


class ChemostatProcess(Process):
    """Well-mixed chemostat — Rust reimplementation of iDynoMiCS-2 ChemostatSolver."""

    config_schema = {
        "spec": "tree",
        "dt_per_update": {"_type": "float", "_default": 1.0},
    }

    def initialize(self, config):
        self.world = load_chemostat(config["spec"])
        self.names = [s["name"] for s in config["spec"]["solutes"]]
        self.dt = float(config["dt_per_update"])
        self.t = 0.0

    def inputs(self):
        return {}

    def outputs(self):
        return {"average_concentrations": "map[string,float]", "time": "overwrite[float]"}

    def update(self, state, interval):
        self.world.step(self.dt)
        self.t += self.dt
        concs = self.world.concs()
        return {
            "average_concentrations": {n: concs[i] for i, n in enumerate(self.names)},
            "time": self.t,
        }
