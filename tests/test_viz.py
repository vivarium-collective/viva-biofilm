from viva_biofilm.run import run_biofilm, default_spec
from viva_biofilm import viz
import plotly.graph_objects as go


def _snaps():
    return run_biofilm(default_spec(seed=3), n_steps=10, snapshot_every=5, dt=0.05)


def test_colony_figure_has_agent_trace():
    fig = viz.colony_figure(_snaps()[-1], color_by="mass")
    assert isinstance(fig, go.Figure)
    assert len(fig.data) >= 1  # at least the agent scatter


def test_solute_field_figure_is_heatmap():
    fig = viz.solute_field_figure(_snaps()[-1], "solute")
    assert any(t.type == "heatmap" for t in fig.data)


def test_timelapse_has_frames():
    fig = viz.timelapse_figure(_snaps())
    assert len(fig.frames) >= 2


def test_growth_curves_and_save(tmp_path):
    snaps = _snaps()
    fig = viz.growth_curves_figure(snaps)
    assert isinstance(fig, go.Figure)
    p = tmp_path / "g.html"
    viz.save_html(fig, str(p))
    assert p.exists() and p.stat().st_size > 0
