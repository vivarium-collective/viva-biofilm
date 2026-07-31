from viva_biofilm.run import run_biofilm, default_spec

def test_run_biofilm_collects_snapshots():
    spec = default_spec(nx=16, ny=32, n_agents=30, seed=7)
    snaps = run_biofilm(spec, n_steps=10, snapshot_every=5, dt=0.05)
    # t=0 plus steps 5 and 10 -> 3 snapshots
    assert len(snaps) == 3
    s = snaps[-1]
    assert set(s.keys()) >= {"time", "population", "total_biomass", "biofilm_thickness", "agents", "solutes"}
    assert len(s["agents"]["x"]) == s["population"]
    assert len(s["agents"]["radius"]) == s["population"]
    assert "solute" in s["solutes"] and s["solutes"]["solute"]["nx"] == 16
    assert len(s["solutes"]["solute"]["field"]) == 16 * 32

def test_run_biofilm_is_deterministic():
    spec = default_spec(seed=7)
    a = run_biofilm(spec, n_steps=8, snapshot_every=8, dt=0.05)[-1]
    b = run_biofilm(spec, n_steps=8, snapshot_every=8, dt=0.05)[-1]
    assert a["population"] == b["population"]
    assert a["agents"]["x"] == b["agents"]["x"]

def test_run_biofilm_always_snapshots_the_final_step():
    # n_steps=100 is not a multiple of snapshot_every=15 (100 % 15 == 10), so
    # the last periodic snapshot would land at step 90 unless run_biofilm
    # appends a final snapshot for step 100.
    spec = default_spec(nx=16, ny=32, n_agents=30, seed=7)
    dt = 0.05
    snaps = run_biofilm(spec, n_steps=100, snapshot_every=15, dt=dt)
    assert abs(snaps[-1]["time"] - 100 * dt) < 1e-9
