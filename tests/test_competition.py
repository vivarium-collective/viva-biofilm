from viva_biofilm.run import competition_spec, run_competition


def test_competition_spec_runs_two_strategies():
    spec = competition_spec(n_each=8, seed=7)
    snaps = run_competition(spec, n_steps=20, dt=0.05, snapshot_every=10)
    last = snaps[-1]
    assert len(last["pop_by_strategy"]) == 2
    assert sum(last["pop_by_strategy"]) == last["population"]
    assert last["pop_by_strategy"][0] >= 8 and last["pop_by_strategy"][1] >= 8
