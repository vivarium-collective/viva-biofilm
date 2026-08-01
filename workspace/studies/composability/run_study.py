"""Composability study driver.

Demonstrates that the Task 7 BoundaryControllerProcess composes with the
Task 6 boundary_concentrations hook on BiofilmProcess: the controller drives
the oxygen boundary down at t=1d and restores it at t=3d, and the biofilm
visibly slows its growth in response, relative to an un-perturbed control run
from the same seed.

Driving strategy: MANUAL LOOP, not the pb.Composite runtime. The
biofilm_controlled.composite.yaml wiring (controller.outputs.boundary_concentrations
-> [stores, boundary_concentrations] -> biofilm.inputs.boundary_concentrations)
is exercised directly by tests/test_composites.py (build + propagation smoke
test). Here the two Process instances are stepped by hand -- controller
first each iteration, its output fed straight into the biofilm's
boundary_concentrations input -- because it makes the per-step handoff and
the absolute-value snapshotting (reading proc.world directly, mirroring
viva_biofilm.run's _snapshot convention) explicit and easy to chart, without
needing to reach into pb.Composite's internal store representation for every
sample point.
"""
import json
import pathlib
import shutil

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import process_bigraph as pb

from viva_biofilm.processes.controller_process import BoundaryControllerProcess
from viva_biofilm.processes.biofilm_process import BiofilmProcess
from viva_biofilm.run import default_spec
from viva_biofilm import viz
from viva_biofilm.emit import emit_run

HERE = pathlib.Path(__file__).parent
CHARTS = HERE / "charts"
REPORT_CARD = HERE / "viz" / "report_card"
EMBED_DIR = HERE.parents[2] / "reports" / "figures" / "composability"

DT = 0.05
N_STEPS = 80  # 4 days
SEED = 1234

# Schedule: normal oxygen (8.74, the spec's default bulk) until t=1d, dropped
# to 0.5 -- well below the Monod ks_oxygen=0.6 half-saturation, so growth is
# strongly oxygen-limited -- until t=3d, then restored.
SCHEDULE = [[0.0, 8.74], [1.0, 0.5], [3.0, 8.74]]
SOLUTE = "oxygen"
LOW_WINDOW = (1.0, 3.0)

CHART_FILES = ["response.html", "colony_before.html", "colony_after.html"]


def _save(fig, stem: str) -> None:
    """Write both the interactive HTML and a static PNG sibling into CHARTS.

    The static PNG is what the dashboard's Charts panel actually renders --
    discover_static_study_charts only picks up *.svg/*.png/*.gif under
    charts/, not embed_visualizations (interactive HTML).
    """
    viz.save_html(fig, str(CHARTS / f"{stem}.html"))
    viz.save_png(fig, str(CHARTS / f"{stem}.png"))


def _trace_point(world, boundary_value: float) -> dict:
    return {
        "time": world.time(),
        "population": world.population(),
        "total_biomass": world.total_biomass(),
        "oxygen_mean": world.solute_means()["oxygen"],
        "boundary_oxygen": boundary_value,
    }


def _colony_snapshot(world) -> dict:
    """Snapshot shape expected by viz.colony_figure (mirrors viva_biofilm.run._snapshot)."""
    positions = world.agent_positions()
    xs = [p[0] for p in positions]
    ys = [p[1] for p in positions]
    nx, ny = world.grid_shape()
    solutes = {
        name: {"field": world.solute_field(name), "nx": nx, "ny": ny}
        for name in world.solute_means().keys()
    }
    return {
        "time": world.time(),
        "population": world.population(),
        "total_biomass": world.total_biomass(),
        "biofilm_thickness": world.biofilm_thickness(),
        "agents": {
            "x": xs, "y": ys,
            "radius": world.agent_radii(), "mass": world.agent_masses(),
            "species": world.agent_species(),
        },
        "solutes": solutes,
    }


def run_perturbed(n_steps: int = N_STEPS, dt: float = DT, seed: int = SEED):
    core = pb.allocate_core()
    controller = BoundaryControllerProcess({"schedule": SCHEDULE, "solute": SOLUTE}, core=core)
    biofilm = BiofilmProcess({"spec": default_spec(seed=seed), "dt_per_update": dt}, core=core)

    trace = [_trace_point(biofilm.world, SCHEDULE[0][1])]
    colony_before = None
    for _ in range(n_steps):
        boundary = controller.update({}, dt)["boundary_concentrations"]
        biofilm.update({"boundary_concentrations": boundary}, dt)
        trace.append(_trace_point(biofilm.world, boundary[SOLUTE]))
        if colony_before is None and biofilm.world.time() >= LOW_WINDOW[0]:
            # First snapshot at/after the oxygen drop -- the "before" state,
            # i.e. the colony as grown under normal oxygen, right as the
            # perturbation begins.
            colony_before = _colony_snapshot(biofilm.world)

    colony_after = _colony_snapshot(biofilm.world)  # final state, after restore + recovery
    return trace, colony_before, colony_after


def run_control(n_steps: int = N_STEPS, dt: float = DT, seed: int = SEED):
    core = pb.allocate_core()
    biofilm = BiofilmProcess({"spec": default_spec(seed=seed), "dt_per_update": dt}, core=core)

    trace = [_trace_point(biofilm.world, SCHEDULE[0][1])]
    for _ in range(n_steps):
        biofilm.update({"boundary_concentrations": {}}, dt)  # no perturbation
        trace.append(_trace_point(biofilm.world, SCHEDULE[0][1]))
    return trace


def response_figure(perturbed: list[dict], control: list[dict]) -> go.Figure:
    p_color, c_color = viz.SPECIES_PALETTE[0], viz.SPECIES_PALETTE[1]

    times_p = [s["time"] for s in perturbed]
    times_c = [s["time"] for s in control]

    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.06,
        subplot_titles=("Population", "Total biomass (pg)", "Oxygen boundary concentration (g/m³)"),
    )

    for row, key in ((1, "population"), (2, "total_biomass")):
        fig.add_trace(
            go.Scatter(x=times_p, y=[s[key] for s in perturbed], mode="lines", name="perturbed",
                       line=dict(color=p_color, width=2), legendgroup="perturbed",
                       showlegend=(row == 1)),
            row=row, col=1,
        )
        fig.add_trace(
            go.Scatter(x=times_c, y=[s[key] for s in control], mode="lines", name="control",
                       line=dict(color=c_color, width=2, dash="dash"), legendgroup="control",
                       showlegend=(row == 1)),
            row=row, col=1,
        )

    fig.add_trace(
        go.Scatter(x=times_p, y=[s["boundary_oxygen"] for s in perturbed], mode="lines",
                   name="oxygen boundary (perturbed)", line=dict(color=p_color, width=2, shape="hv"),
                   legendgroup="perturbed", showlegend=False),
        row=3, col=1,
    )
    fig.add_trace(
        go.Scatter(x=times_c, y=[s["boundary_oxygen"] for s in control], mode="lines",
                   name="oxygen boundary (control, constant)", line=dict(color=c_color, width=2, dash="dash"),
                   legendgroup="control", showlegend=False),
        row=3, col=1,
    )

    layout = dict(
        template="plotly_white", paper_bgcolor="white", plot_bgcolor="#F7F7F7",
        font=dict(size=13, color="#222222"),
        title=dict(text="Biofilm response to a scheduled oxygen boundary perturbation",
                   font=dict(size=17, color="#111111")),
        margin=dict(l=70, r=40, t=60, b=60), height=760,
        legend=dict(orientation="h", y=1.06),
    )
    fig.update_layout(**layout)
    for row in (1, 2, 3):
        fig.add_vrect(x0=LOW_WINDOW[0], x1=LOW_WINDOW[1], fillcolor="rgba(217,95,2,0.08)",
                      line_width=0, row=row, col=1)
        fig.update_xaxes(showgrid=False, zeroline=False, row=row, col=1)
    fig.update_xaxes(title_text="time (days)", row=3, col=1)
    fig.update_yaxes(title_text="agents", row=1, col=1, showgrid=True, gridcolor="#E5E5E5")
    fig.update_yaxes(title_text="pg", row=2, col=1, showgrid=True, gridcolor="#E5E5E5")
    fig.update_yaxes(title_text="g/m³", row=3, col=1, showgrid=True, gridcolor="#E5E5E5")

    return fig


def build_verdict(perturbed: list[dict], control: list[dict]) -> dict:
    # Axis 1: the controller actually drove the boundary -- during the low
    # window it must read the scheduled low value, and after restore it must
    # be back at the normal bulk.
    during_low = [s["boundary_oxygen"] for s in perturbed if LOW_WINDOW[0] < s["time"] <= LOW_WINDOW[1]]
    after_restore = [s["boundary_oxygen"] for s in perturbed if s["time"] > LOW_WINDOW[1]]
    low_value = min(during_low) if during_low else float("nan")
    restored_value = after_restore[-1] if after_restore else float("nan")
    low_err = abs(low_value - SCHEDULE[1][1])
    restore_err = abs(restored_value - SCHEDULE[2][1])
    if low_err < 0.01 and restore_err < 0.01:
        controller_verdict = "within_tol"
    elif low_err < 0.5 and restore_err < 0.5:
        controller_verdict = "drift"
    else:
        controller_verdict = "mismatch"

    # Axis 2: the biofilm responds -- perturbed final biomass/population must
    # be below the un-perturbed control's, same seed, same window.
    p_final, c_final = perturbed[-1], control[-1]
    biomass_ratio = p_final["total_biomass"] / c_final["total_biomass"] if c_final["total_biomass"] else float("inf")
    if p_final["population"] < c_final["population"] and biomass_ratio < 0.95:
        response_verdict = "within_tol"
    elif p_final["population"] <= c_final["population"] and biomass_ratio < 1.0:
        response_verdict = "drift"
    else:
        response_verdict = "mismatch"

    return {
        "schema": "report_card_verdict/v1",
        "groups": {
            "composability": {
                "axes": [
                    {
                        "name": "controller-drives-boundary",
                        "verdict": controller_verdict,
                        "value": low_value,
                        "reference": SCHEDULE[1][1],
                        "restored_value": restored_value,
                        "restored_reference": SCHEDULE[2][1],
                    },
                    {
                        "name": "biofilm-responds",
                        "verdict": response_verdict,
                        "value": p_final["total_biomass"],
                        "reference": c_final["total_biomass"],
                        "perturbed_population": p_final["population"],
                        "control_population": c_final["population"],
                        "biomass_ratio": biomass_ratio,
                    },
                ]
            }
        },
    }


def main() -> None:
    CHARTS.mkdir(parents=True, exist_ok=True)
    REPORT_CARD.mkdir(parents=True, exist_ok=True)

    perturbed, colony_before, colony_after = run_perturbed()
    control = run_control()

    _save(response_figure(perturbed, control), "response")
    _save(viz.colony_figure(colony_before, color_by="mass",
                             title=f"Colony before perturbation (t={colony_before['time']:.2f} d)"),
          "colony_before")
    _save(viz.colony_figure(colony_after, color_by="mass",
                             title=f"Colony after perturbation (t={colony_after['time']:.2f} d)"),
          "colony_after")

    EMBED_DIR.mkdir(parents=True, exist_ok=True)
    for name in CHART_FILES:
        shutil.copy2(CHARTS / name, EMBED_DIR / name)

    verdict = build_verdict(perturbed, control)
    (REPORT_CARD / "report_card_verdict.json").write_text(json.dumps(verdict, indent=2))

    # Register both arms as runs for the dashboard's Runs tab.
    emit_run(HERE, spec_id="composability", snaps=perturbed,
             run_id="perturbed", label="perturbed (O2 dip)", reset=True)
    emit_run(HERE, spec_id="composability", snaps=control,
             run_id="control", label="control (constant O2)", reset=False)

    p_final, c_final = perturbed[-1], control[-1]
    axes = verdict["groups"]["composability"]["axes"]
    print(
        f"controller: low-window boundary={axes[0]['value']:.3f} (ref {axes[0]['reference']}), "
        f"restored={axes[0]['restored_value']:.3f} (ref {axes[0]['restored_reference']}) "
        f"-> {axes[0]['verdict']}"
    )
    print(
        f"response: perturbed final pop={p_final['population']} biomass={p_final['total_biomass']:.2f} pg  |  "
        f"control final pop={c_final['population']} biomass={c_final['total_biomass']:.2f} pg  |  "
        f"biomass ratio={axes[1]['biomass_ratio']:.3f} -> {axes[1]['verdict']}"
    )


if __name__ == "__main__":
    main()
