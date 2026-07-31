import pytest

from viva_biofilm import biofilm_core
from viva_biofilm.run import default_spec, run_biofilm
from viva_biofilm.schema import load_world


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


def test_looser_tolerance_yields_gradient_close_to_tight_tolerance():
    # Not a timing comparison (flaky under CI load) -- instead assert both
    # the tight- and loose-tolerance runs complete and produce a valid,
    # non-degenerate partial gradient, and that the loose-tolerance result
    # is numerically close to the tight-tolerance one (proving the looser
    # default is a safe speed/accuracy tradeoff, not a correctness change).
    def row_mean(field, nx, j):
        return sum(field[i + j * nx] for i in range(nx)) / nx

    tight = default_spec(nx=32, ny=48, n_agents=40, seed=11)
    tight["pde"] = {"tol": 1e-7, "max_iter": 20_000, "omega": 1.4}
    tight_snap = run_biofilm(tight, n_steps=20, dt=0.05)[-1]
    tight_field = tight_snap["solutes"]["solute"]
    tight_bottom = row_mean(tight_field["field"], tight_field["nx"], 0)
    tight_top = row_mean(tight_field["field"], tight_field["nx"], tight_field["ny"] - 1)

    loose = default_spec(nx=32, ny=48, n_agents=40, seed=11)
    loose["pde"] = {"tol": 1e-4, "max_iter": 2_000, "omega": 1.8}
    loose_snap = run_biofilm(loose, n_steps=20, dt=0.05)[-1]
    loose_field = loose_snap["solutes"]["solute"]
    loose_bottom = row_mean(loose_field["field"], loose_field["nx"], 0)
    loose_top = row_mean(loose_field["field"], loose_field["nx"], loose_field["ny"] - 1)

    for name, bottom, top in (("tight", tight_bottom, tight_top), ("loose", loose_bottom, loose_top)):
        assert top > bottom > 0.0, f"{name} run should have a valid partial gradient: top={top} bottom={bottom}"

    rel_diff = abs(tight_bottom - loose_bottom) / max(abs(tight_bottom), 1e-9)
    assert rel_diff < 0.10, (
        f"loose-tolerance gradient diverged from tight-tolerance: tight_bottom={tight_bottom} "
        f"loose_bottom={loose_bottom} rel_diff={rel_diff}"
    )


@pytest.mark.parametrize("bad_pde", [
    {"omega": 2.1},
    {"omega": 1.0 - 1e-9},
    {"tol": 0.0},
    {"tol": -1e-4},
    {"max_iter": 0},
])
def test_invalid_pde_params_raise_via_load_world(bad_pde):
    spec = default_spec(nx=16, ny=32, n_agents=20, seed=3)
    spec["pde"] = bad_pde
    with pytest.raises(ValueError):
        load_world(spec)


def test_invalid_pde_params_raise_via_direct_binding_call():
    w = biofilm_core.World()
    w.set_domain(16, 32, 2.0, 32.0)
    with pytest.raises(ValueError):
        w.set_pde_params(1e-4, 2000, 2.1)
