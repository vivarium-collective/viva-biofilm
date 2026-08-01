from viva_biofilm.run import run_biofilm, default_spec, competition_spec, run_competition
from viva_biofilm import viz
import plotly.graph_objects as go


def _snaps():
    return run_biofilm(default_spec(seed=3), n_steps=10, snapshot_every=5, dt=0.05)


def _competition_snaps():
    return run_competition(competition_spec(n_each=5, seed=3), n_steps=5, dt=1 / 24, snapshot_every=5)


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


def test_strategy_colony_figure_has_two_strategy_traces():
    fig = viz.strategy_colony_figure(_competition_snaps()[-1])
    assert isinstance(fig, go.Figure)
    marker_traces = [t for t in fig.data if t.mode and "markers" in t.mode]
    assert len(marker_traces) == 2  # RS + YS scatter traces
    names = {t.name.split(" ")[0] for t in marker_traces}
    assert names == {"RS", "YS"}


def test_competition_outcome_figure_has_tie_line_and_range():
    fig = viz.competition_outcome_figure([
        {"n_each": 5, "rs_fraction": 0.3},
        {"n_each": 10, "rs_fraction": 0.55},
        {"n_each": 50, "rs_fraction": 0.7},
    ])
    assert isinstance(fig, go.Figure)
    assert any(t.type == "scatter" for t in fig.data)
    assert tuple(fig.layout.yaxis.range) == (0, 1)
    tie_lines = [t for t in fig.data if t.name == "tie (0.5)"]
    assert len(tie_lines) == 1
    assert list(tie_lines[0].y) == [0.5, 0.5]


def test_save_png(tmp_path):
    fig = viz.growth_curves_figure(_snaps())
    p = tmp_path / "g.png"
    viz.save_png(fig, str(p))
    assert p.exists() and p.stat().st_size > 0
