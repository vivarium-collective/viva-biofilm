import time

from viva_biofilm.run import default_spec, run_biofilm


def test_custom_pde_block_runs():
    spec = default_spec(nx=16, ny=32, n_agents=20, seed=3)
    spec["pde"] = {"tol": 1e-5, "max_iter": 5000, "omega": 1.7}
    snaps = run_biofilm(spec, n_steps=5, dt=0.05)
    assert len(snaps) == 6
    assert snaps[-1]["population"] >= 20


def test_pde_block_defaults_when_unset_match_fast_defaults():
    # No "pde" key at all -> World falls back to the fast defaults
    # (tol=1e-4, max_iter=2000, omega=1.8) baked into World::new().
    spec = default_spec(nx=16, ny=32, n_agents=20, seed=3)
    assert "pde" not in spec
    snaps = run_biofilm(spec, n_steps=5, dt=0.05)
    assert snaps[-1]["population"] >= 20


def test_looser_tolerance_is_not_slower_and_yields_valid_gradient():
    tight = default_spec(nx=32, ny=48, n_agents=40, seed=11)
    tight["pde"] = {"tol": 1e-7, "max_iter": 20_000, "omega": 1.4}
    t0 = time.perf_counter()
    run_biofilm(tight, n_steps=20, dt=0.05)
    tight_time = time.perf_counter() - t0

    loose = default_spec(nx=32, ny=48, n_agents=40, seed=11)
    loose["pde"] = {"tol": 1e-4, "max_iter": 2_000, "omega": 1.8}
    t0 = time.perf_counter()
    snaps = run_biofilm(loose, n_steps=20, dt=0.05)
    loose_time = time.perf_counter() - t0

    assert loose_time <= tight_time * 1.1, (
        f"looser tolerance should not be slower: loose={loose_time:.3f}s tight={tight_time:.3f}s"
    )

    field = snaps[-1]["solutes"]["solute"]["field"]
    assert max(field) > 0, "solute field should not be fully depleted"
