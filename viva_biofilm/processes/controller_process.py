from process_bigraph import Process


class BoundaryControllerProcess(Process):
    """Perturbs a biofilm's boundary solute concentration on a schedule.

    No inputs; emits `boundary_concentrations` (map name->value) holding the
    current scheduled value for `solute`. The schedule is a list of
    `[time, value]` points, applied piecewise-constant: the emitted value is
    that of the latest point with `time <= t` (the first point's value before
    the first point is reached). Couples with BiofilmProcess's
    `boundary_concentrations` input port (biofilm_process.py) — wired through
    a shared store in biofilm_controlled.composite.yaml — to perturb the
    biofilm's environment at runtime via the Rust `set_bulk_by_name` hook,
    without touching biofilm internals.
    """

    config_schema = {
        "schedule": "tree",
        "solute": {"_type": "string", "_default": "oxygen"},
    }

    def initialize(self, config):
        schedule = [(float(t), float(v)) for t, v in config["schedule"]]
        if not schedule:
            raise ValueError("BoundaryControllerProcess requires a non-empty schedule")
        self.schedule = sorted(schedule, key=lambda point: point[0])
        self.solute = config["solute"]
        self.t = 0.0

    def inputs(self):
        return {}

    def outputs(self):
        # overwrite[...]: this port must SET the boundary each update, not
        # accumulate -- plain "map[string,float]" defaults to additive apply
        # in this schema system (mirrors biofilm_process.py's absolute ports
        # like `time`/`total_biomass`, which use the same overwrite[...] form
        # to distinguish themselves from the accumulate-typed delta ports).
        return {"boundary_concentrations": "overwrite[map[string,float]]"}

    def _value_at(self, t: float) -> float:
        # Latest point with time <= t; before the first point, its value.
        value = self.schedule[0][1]
        for time_point, point_value in self.schedule:
            if time_point <= t:
                value = point_value
            else:
                break
        return value

    def initial_state(self):
        # Seed the store with the t=0 boundary value before the first update.
        return {"boundary_concentrations": {self.solute: self._value_at(self.t)}}

    def update(self, state, interval):
        self.t += interval
        return {"boundary_concentrations": {self.solute: self._value_at(self.t)}}
