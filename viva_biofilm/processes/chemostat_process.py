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
        # Baseline for the accumulate-typed average_concentrations port so
        # update() can emit per-step DELTAS — mirrors pbg-idynomics2's
        # IDynoMiCS2Process._prev_concns.
        concs = self.world.concs()
        self._prev = {name: concs[i] for i, name in enumerate(self.names)}

    def inputs(self):
        return {}

    def outputs(self):
        return {"average_concentrations": "map[string,float]", "time": "overwrite[float]"}

    def initial_state(self):
        # Seed the accumulate-typed store key before the first delta lands.
        return {
            "average_concentrations": dict(self._prev),
            "time": 0.0,
        }

    def update(self, state, interval):
        self.world.step(self.dt)
        self.t += self.dt
        concs = self.world.concs()
        cur = {name: concs[i] for i, name in enumerate(self.names)}

        d = {n: cur[n] - self._prev.get(n, 0.0) for n in cur}
        self._prev = cur

        return {
            "average_concentrations": d,
            "time": self.t,
        }
