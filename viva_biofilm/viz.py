"""Plotly visualization builders for viva_biofilm snapshots.

Each function consumes snapshot dict(s) produced by ``viva_biofilm.run.run_biofilm``
and returns a ``plotly.graph_objects.Figure``. Design choices follow the project's
dataviz guidance: perceptually-uniform sequential palettes (Viridis/Cividis) for
continuous encodings, a small fixed-order qualitative palette for species identity,
a clean ``plotly_white`` theme, explicit units on every axis/colorbar, and
true-to-scale marker sizing for the colony scatter.
"""

from __future__ import annotations

import plotly.graph_objects as go

# Sequential palette for continuous encodings (mass, local substrate, solute field).
SEQUENTIAL_SCALE = "Viridis"

# Fixed-order qualitative palette for categorical (species) encoding. Chosen from
# an accessible, colorblind-safe set — never cycled, assigned in this fixed order.
SPECIES_PALETTE = ["#1B9E77", "#D95F02", "#7570B3", "#E7298A", "#66A61E", "#E6AB02"]

PAPER_BGCOLOR = "white"
PLOT_BGCOLOR = "#F7F7F7"
FONT = dict(size=13, color="#222222")
TITLE_FONT = dict(size=17, color="#111111")


def _base_layout(title: str | None) -> dict:
    layout = dict(
        template="plotly_white",
        paper_bgcolor=PAPER_BGCOLOR,
        plot_bgcolor=PLOT_BGCOLOR,
        font=FONT,
        margin=dict(l=70, r=40, t=60, b=60),
    )
    if title:
        layout["title"] = dict(text=title, font=TITLE_FONT)
    return layout


def _sample_field(field: list[float], nx: int, ny: int, x: float, y: float, dx: float) -> float:
    i = min(max(int(x / dx), 0), nx - 1)
    j = min(max(int(y / dx), 0), ny - 1)
    idx = i + j * nx
    return field[idx]


def _agent_color_values(snapshot: dict, color_by: str, dx: float) -> tuple[list[float], str, str]:
    """Return (values, colorbar_title, colorscale) for the given color_by mode."""
    agents = snapshot["agents"]
    if color_by == "mass":
        return list(agents["mass"]), "mass (pg)", SEQUENTIAL_SCALE
    if color_by == "local_substrate":
        solute = snapshot["solutes"]["solute"]
        field, nx, ny = solute["field"], solute["nx"], solute["ny"]
        values = [
            _sample_field(field, nx, ny, x, y, dx)
            for x, y in zip(agents["x"], agents["y"])
        ]
        return values, "local substrate (g/m³)", SEQUENTIAL_SCALE
    raise ValueError(f"unsupported color_by: {color_by!r}")


def _domain_extent(snapshot: dict, dx: float) -> tuple[float, float]:
    solute = next(iter(snapshot["solutes"].values()))
    x_max = solute["nx"] * dx
    y_max = solute["ny"] * dx
    return x_max, y_max


def _radius_to_pixel_size(radii: list[float], x_max: float, plot_px: float = 520.0) -> list[float]:
    """Convert agent radii (µm, data units) to marker `size` (diameter, px).

    Plotly scatter marker `size` is a fixed pixel diameter independent of axis
    zoom, so we approximate "true to scale" by mapping the data-space diameter
    through the *initial* px-per-data-unit ratio implied by the axis range and
    the figure's plot width. This keeps circles readable and proportionate to
    each other without requiring custom shapes.
    """
    if x_max <= 0:
        return [8.0 for _ in radii]
    px_per_unit = plot_px / x_max
    return [max(2.0 * r * px_per_unit, 3.0) for r in radii]


def _colony_traces(snapshot: dict, color_by: str, dx: float, x_max: float, y_max: float,
                    showscale: bool = True) -> list[go.Scatter]:
    agents = snapshot["agents"]
    values, colorbar_title, colorscale = _agent_color_values(snapshot, color_by, dx)
    sizes = _radius_to_pixel_size(agents["radius"], x_max)

    agent_trace = go.Scatter(
        x=agents["x"],
        y=agents["y"],
        mode="markers",
        marker=dict(
            size=sizes,
            color=values,
            colorscale=colorscale,
            showscale=showscale,
            colorbar=dict(title=colorbar_title) if showscale else None,
            line=dict(width=0.6, color="rgba(30,30,30,0.35)"),
            opacity=0.9,
        ),
        name="agents",
        text=[f"r={r:.2f} µm" for r in agents["radius"]],
        hovertemplate="x=%{x:.1f} µm<br>y=%{y:.1f} µm<br>%{text}<extra></extra>",
    )

    substratum_trace = go.Scatter(
        x=[0, x_max],
        y=[0, 0],
        mode="lines",
        line=dict(color="#8B5A2B", width=3),
        name="substratum",
        hoverinfo="skip",
        showlegend=False,
    )

    domain_box_trace = go.Scatter(
        x=[0, x_max, x_max, 0, 0],
        y=[0, 0, y_max, y_max, 0],
        mode="lines",
        line=dict(color="rgba(80,80,80,0.4)", width=1, dash="dot"),
        name="domain",
        hoverinfo="skip",
        showlegend=False,
    )

    return [domain_box_trace, substratum_trace, agent_trace]


def colony_figure(snapshot: dict, color_by: str = "mass", title: str | None = None,
                   dx: float = 2.0) -> go.Figure:
    """Colony scatter: agents as filled circles at (x, y), sized true-to-scale by
    radius, colored by `mass` or `local_substrate`, with domain box + substratum line.
    """
    x_max, y_max = _domain_extent(snapshot, dx)
    traces = _colony_traces(snapshot, color_by, dx, x_max, y_max)

    fig_title = title or f"Colony — t={snapshot['time']:.2f} d, n={snapshot['population']}"
    layout = _base_layout(fig_title)
    layout.update(
        xaxis=dict(title="x (µm)", range=[0, x_max], showgrid=False, zeroline=False),
        yaxis=dict(
            title="height above substratum (µm)",
            range=[0, y_max],
            scaleanchor="x",
            scaleratio=1,
            showgrid=False,
            zeroline=False,
        ),
        showlegend=False,
    )
    return go.Figure(data=traces, layout=layout)


def solute_field_figure(snapshot: dict, name: str, title: str | None = None,
                         dx: float = 2.0) -> go.Figure:
    """Heatmap of a solute field, y-axis = height above substratum (bottom row = 0)."""
    solute = snapshot["solutes"][name]
    field, nx, ny = solute["field"], solute["nx"], solute["ny"]

    z = [[field[i + j * nx] for i in range(nx)] for j in range(ny)]
    xs = [i * dx for i in range(nx)]
    ys = [j * dx for j in range(ny)]

    heatmap = go.Heatmap(
        x=xs,
        y=ys,
        z=z,
        colorscale=SEQUENTIAL_SCALE,
        colorbar=dict(title="g/m³"),
        hovertemplate="x=%{x:.1f} µm<br>y=%{y:.1f} µm<br>%{z:.3f} g/m³<extra></extra>",
    )

    fig_title = title or f"{name} field — t={snapshot['time']:.2f} d"
    layout = _base_layout(fig_title)
    layout.update(
        xaxis=dict(title="x (µm)", showgrid=False, zeroline=False),
        yaxis=dict(
            title="height above substratum (µm)",
            showgrid=False,
            zeroline=False,
            scaleanchor="x",
            scaleratio=1,
        ),
    )
    return go.Figure(data=[heatmap], layout=layout)


def timelapse_figure(snapshots: list[dict], color_by: str = "mass", title: str | None = None,
                      dx: float = 2.0) -> go.Figure:
    """Animated colony scatter across snapshots with play/pause + slider, fixed axes."""
    if not snapshots:
        raise ValueError("timelapse_figure requires at least one snapshot")

    # Global domain extent (same domain across the run, but computed once to be safe).
    x_max = max(_domain_extent(s, dx)[0] for s in snapshots)
    y_max = max(_domain_extent(s, dx)[1] for s in snapshots)

    frames = []
    for i, snap in enumerate(snapshots):
        traces = _colony_traces(snap, color_by, dx, x_max, y_max, showscale=(i == 0))
        frames.append(go.Frame(data=traces, name=str(i),
                                layout=dict(title=dict(text=f"t={snap['time']:.2f} d, n={snap['population']}"))))

    initial_traces = _colony_traces(snapshots[0], color_by, dx, x_max, y_max, showscale=True)

    fig_title = title or "Colony time-lapse"
    layout = _base_layout(fig_title)
    layout.update(
        xaxis=dict(title="x (µm)", range=[0, x_max], showgrid=False, zeroline=False),
        yaxis=dict(
            title="height above substratum (µm)",
            range=[0, y_max],
            scaleanchor="x",
            scaleratio=1,
            showgrid=False,
            zeroline=False,
        ),
        showlegend=False,
        updatemenus=[
            dict(
                type="buttons",
                showactive=False,
                x=0.05,
                y=-0.12,
                xanchor="left",
                yanchor="top",
                direction="left",
                buttons=[
                    dict(
                        label="Play",
                        method="animate",
                        args=[None, dict(frame=dict(duration=400, redraw=True), fromcurrent=True,
                                          transition=dict(duration=0))],
                    ),
                    dict(
                        label="Pause",
                        method="animate",
                        args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate")],
                    ),
                ],
            )
        ],
        sliders=[
            dict(
                active=0,
                x=0.15,
                y=-0.12,
                len=0.8,
                xanchor="left",
                yanchor="top",
                currentvalue=dict(prefix="t = ", suffix=" d", visible=True),
                steps=[
                    dict(
                        label=f"{snap['time']:.2f}",
                        method="animate",
                        args=[[str(i)], dict(mode="immediate",
                                              frame=dict(duration=300, redraw=True),
                                              transition=dict(duration=0))],
                    )
                    for i, snap in enumerate(snapshots)
                ],
            )
        ],
    )

    return go.Figure(data=initial_traces, layout=layout, frames=frames)


def growth_curves_figure(snapshots: list[dict]) -> go.Figure:
    """Population, total biomass, and biofilm thickness vs. time as stacked
    small-multiples sharing an x-axis (avoids dual/triple-axis scale distortion).
    """
    from plotly.subplots import make_subplots

    times = [s["time"] for s in snapshots]
    population = [s["population"] for s in snapshots]
    biomass = [s["total_biomass"] for s in snapshots]
    thickness = [s["biofilm_thickness"] for s in snapshots]

    colors = SPECIES_PALETTE[:3]

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        subplot_titles=("Population", "Total biomass (pg)", "Biofilm thickness (µm)"),
    )

    fig.add_trace(
        go.Scatter(x=times, y=population, mode="lines+markers", name="population",
                    line=dict(color=colors[0], width=2), marker=dict(size=6)),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(x=times, y=biomass, mode="lines+markers", name="total biomass",
                    line=dict(color=colors[1], width=2), marker=dict(size=6)),
        row=2, col=1,
    )
    fig.add_trace(
        go.Scatter(x=times, y=thickness, mode="lines+markers", name="biofilm thickness",
                    line=dict(color=colors[2], width=2), marker=dict(size=6)),
        row=3, col=1,
    )

    layout = _base_layout("Growth curves")
    fig.update_layout(**layout, showlegend=False, height=700)
    fig.update_xaxes(title_text="time (days)", row=3, col=1, showgrid=False, zeroline=False)
    fig.update_xaxes(showgrid=False, zeroline=False, row=1, col=1)
    fig.update_xaxes(showgrid=False, zeroline=False, row=2, col=1)
    fig.update_yaxes(title_text="agents", row=1, col=1, showgrid=True, gridcolor="#E5E5E5")
    fig.update_yaxes(title_text="pg", row=2, col=1, showgrid=True, gridcolor="#E5E5E5")
    fig.update_yaxes(title_text="µm", row=3, col=1, showgrid=True, gridcolor="#E5E5E5")

    return fig


def save_html(fig: go.Figure, path: str) -> None:
    """Write a self-contained interactive HTML file (Plotly.js loaded from CDN)."""
    fig.write_html(path, include_plotlyjs="cdn")
