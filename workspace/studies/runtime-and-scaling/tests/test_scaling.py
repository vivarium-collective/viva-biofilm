import time

from viva_biofilm.schema import load_world
from viva_biofilm.run import default_spec


def test_step_timing_is_measurable_and_bounded():
    w = load_world(default_spec(nx=16, ny=32, n_agents=40, seed=1))
    t0 = time.perf_counter()
    for _ in range(5):
        w.step(0.05)
    dt = time.perf_counter() - t0
    assert dt > 0.0
    assert dt < 30.0, f"5 steps took {dt:.1f}s -- unexpectedly slow at the small grid"


def test_larger_grid_costs_more_than_small():
    def timeit(nx, ny):
        w = load_world(default_spec(nx=nx, ny=ny, n_agents=40, seed=1))
        t0 = time.perf_counter()
        for _ in range(4):
            w.step(0.05)
        return time.perf_counter() - t0

    small = timeit(16, 32)
    big = timeit(64, 96)
    # more cells -> more PDE work; margin guards against timing noise on tiny grids
    assert big > small * 1.2, f"expected big grid ({big:.4f}s) > 1.2x small grid ({small:.4f}s)"
