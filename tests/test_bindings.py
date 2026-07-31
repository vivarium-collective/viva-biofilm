from viva_biofilm import biofilm_core

def test_world_steps_advance_time():
    w = biofilm_core.World()
    assert w.time() == 0.0
    w.step(0.5)
    w.step(0.5)
    assert abs(w.time() - 1.0) < 1e-12
