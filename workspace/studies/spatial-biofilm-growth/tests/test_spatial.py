from viva_biofilm.run import run_biofilm, default_spec

def test_developed_biofilm_grows_and_has_gradient():
    snaps = run_biofilm(default_spec(nx=32, ny=48, n_agents=40, seed=11), n_steps=60, snapshot_every=15, dt=0.05)
    first, last = snaps[0], snaps[-1]
    assert last["population"] > first["population"]          # grew
    assert last["biofilm_thickness"] > 0.0
    # substrate gradient: mean of bottom third < mean of top third of the field
    f = last["solutes"]["solute"]; nx, ny = f["nx"], f["ny"]
    field = f["field"]
    bottom = sum(field[0:nx*(ny//3)]) / (nx*(ny//3))
    top = sum(field[nx*2*(ny//3):nx*ny]) / (nx*(ny - 2*(ny//3)))
    assert top > bottom
