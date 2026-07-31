from viva_biofilm import biofilm_core

def test_world_steps_advance_time():
    w = biofilm_core.World()
    assert w.time() == 0.0
    w.step(0.5)
    w.step(0.5)
    assert abs(w.time() - 1.0) < 1e-12

def _biofilm():
    w = biofilm_core.World()
    w.set_domain(16, 32, 2.0, 32.0)
    s = w.add_solute("solute", 1.0, 2000.0, 1500.0, 1.0)
    o = w.add_solute("oxygen", 8.74, 2000.0, 1500.0, 8.74)
    w.add_reaction(2.05, [(s, 2.4), (o, 0.6)], [(s, -4.2), (o, -18.0)])
    w.set_species(0.15, 0.2)
    w.spawn_agents(30, 1.0, 0)
    w.finalize(1234)
    return w

def test_biofilm_bindings_readbacks_and_determinism():
    def run():
        w = _biofilm()
        for _ in range(15):
            w.step(0.05)
        return w.population(), round(w.total_biomass(), 9)
    r = run()
    assert r[0] >= 30
    assert run() == r  # determinism across the boundary
    w = _biofilm()
    w.step(0.05)
    assert len(w.agent_positions()) == w.population()
    assert set(w.solute_means().keys()) == {"solute", "oxygen"}
    assert w.grid_shape() == (16, 32)
    assert len(w.solute_field("oxygen")) == 16 * 32

def test_chemostat_binding_matches_analytic():
    c = biofilm_core.ChemostatWorld([2.0, 2.0])
    c.add_linear_reaction(0, 0.1, [(0, -1.0), (1, 0.5)])
    for _ in range(6000):
        c.step(0.01)
    import math
    assert abs(c.conc(0) - 2.0 * math.exp(-0.1 * 60.0)) < 1e-3
