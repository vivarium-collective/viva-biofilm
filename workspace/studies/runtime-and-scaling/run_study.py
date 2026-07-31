"""Runtime-and-scaling study driver.

Measures wall-time and throughput of the viva_biofilm Rust engine across
three FIXED sweep axes -- grid size, initial population, and run duration --
and writes scaling curves + a throughput figure, plus a report-card verdict
for the study's ``performance`` test group.

Sweep sizes are fixed by the task brief; do NOT widen/narrow them:
  grid:       nx,ny in [(16,32),(32,48),(48,64),(64,96),(96,128)], 40 agents, 40 steps
  population: grid 48x64, n_agents in [10,20,40,80,160], 40 steps
  duration:   grid 48x64, 40 agents, steps in [20,40,80,160]
"""
import json
import pathlib
import shutil
import time

import numpy as np

from viva_biofilm.schema import load_world
from viva_biofilm.run import default_spec
from viva_biofilm import viz
import plotly.graph_objects as go

HERE = pathlib.Path(__file__).parent
CHARTS = HERE / "charts"
REPORT_CARD = HERE / "viz" / "report_card"
EMBED_DIR = HERE.parents[2] / "reports" / "figures" / "runtime-and-scaling"

DT = 0.05

GRID_SIZES = [(16, 32), (32, 48), (48, 64), (64, 96), (96, 128)]
GRID_N_AGENTS = 40
GRID_N_STEPS = 40

POP_GRID = (48, 64)
POP_N_AGENTS = [10, 20, 40, 80, 160]
POP_N_STEPS = 40

DUR_GRID = (48, 64)
DUR_N_AGENTS = 40
DUR_STEPS = [20, 40, 80, 160]

CHART_FILES = [
    "scaling_grid.html",
    "scaling_population.html",
    "scaling_duration.html",
    "throughput.html",
]

SLOW_THRESHOLD_S = 30.0


def _save(fig, stem: str) -> None:
    """Write both the interactive HTML and a static PNG sibling into CHARTS.

    The static PNG is what the dashboard's Charts panel actually renders --
    discover_static_study_charts only picks up *.svg/*.png/*.gif under
    charts/, not embed_visualizations (interactive HTML).
    """
    viz.save_html(fig, str(CHARTS / f"{stem}.html"))
    viz.save_png(fig, str(CHARTS / f"{stem}.png"))


def time_run(nx: int, ny: int, n_agents: int, n_steps: int, seed: int = 1) -> dict:
    """Build a world and time n_steps of w.step(DT). Returns measurement dict.

    If a single configuration exceeds SLOW_THRESHOLD_S it is still recorded
    (not silently skipped) -- see brief: report, don't shrink the sweep.
    """
    w = load_world(default_spec(nx=nx, ny=ny, n_agents=n_agents, seed=seed))
    initial_pop = w.population()
    t0 = time.perf_counter()
    for _ in range(n_steps):
        w.step(DT)
    dt = time.perf_counter() - t0
    final_pop = w.population()
    # agent-steps/sec: use the mean of initial/final population as the
    # per-step agent count estimate (population grows over the run).
    mean_pop = (initial_pop + final_pop) / 2.0
    agent_steps = mean_pop * n_steps
    return {
        "nx": nx,
        "ny": ny,
        "cells": nx * ny,
        "n_agents_initial": n_agents,
        "n_agents_final": final_pop,
        "n_steps": n_steps,
        "wall_time_s": dt,
        "wall_time_per_step_s": dt / n_steps,
        "throughput_agent_steps_per_s": agent_steps / dt if dt > 0 else float("inf"),
        "slow": dt > SLOW_THRESHOLD_S,
    }


def run_grid_sweep() -> list[dict]:
    results = []
    for nx, ny in GRID_SIZES:
        r = time_run(nx, ny, GRID_N_AGENTS, GRID_N_STEPS)
        results.append(r)
        flag = "  ** SLOW **" if r["slow"] else ""
        print(
            f"  grid {nx}x{ny} (cells={r['cells']:6d}): "
            f"total={r['wall_time_s']:.4f}s  per-step={r['wall_time_per_step_s']*1000:.3f}ms{flag}"
        )
    return results


def run_population_sweep() -> list[dict]:
    nx, ny = POP_GRID
    results = []
    for n_agents in POP_N_AGENTS:
        r = time_run(nx, ny, n_agents, POP_N_STEPS)
        results.append(r)
        flag = "  ** SLOW **" if r["slow"] else ""
        print(
            f"  population n0={n_agents:4d} (final={r['n_agents_final']:4d}): "
            f"total={r['wall_time_s']:.4f}s  per-step={r['wall_time_per_step_s']*1000:.3f}ms{flag}"
        )
    return results


def run_duration_sweep() -> list[dict]:
    nx, ny = DUR_GRID
    results = []
    for n_steps in DUR_STEPS:
        r = time_run(nx, ny, DUR_N_AGENTS, n_steps)
        results.append(r)
        flag = "  ** SLOW **" if r["slow"] else ""
        print(
            f"  duration steps={n_steps:4d} (final pop={r['n_agents_final']:4d}): "
            f"total={r['wall_time_s']:.4f}s  per-step={r['wall_time_per_step_s']*1000:.3f}ms{flag}"
        )
    return results


def _base_layout(title: str) -> dict:
    return dict(
        template="plotly_white",
        paper_bgcolor="white",
        plot_bgcolor="#F7F7F7",
        font=dict(size=13, color="#222222"),
        title=dict(text=title, font=dict(size=17, color="#111111")),
        margin=dict(l=70, r=40, t=60, b=60),
        height=460,
    )


def grid_scaling_figure(results: list[dict]) -> go.Figure:
    cells = [r["cells"] for r in results]
    per_step_ms = [r["wall_time_per_step_s"] * 1000 for r in results]
    fig = go.Figure(
        data=[
            go.Scatter(
                x=cells,
                y=per_step_ms,
                mode="lines+markers",
                line=dict(color=viz.SPECIES_PALETTE[0], width=2),
                marker=dict(size=9),
                text=[f"{r['nx']}x{r['ny']}" for r in results],
                hovertemplate="grid=%{text}<br>cells=%{x}<br>%{y:.4f} ms/step<extra></extra>",
            )
        ]
    )
    layout = _base_layout(f"Grid-size scaling ({GRID_N_AGENTS} agents, {GRID_N_STEPS} steps)")
    layout.update(
        xaxis=dict(title="grid cells (nx * ny)", type="log", showgrid=True, gridcolor="#E5E5E5"),
        yaxis=dict(title="wall time / step (ms)", type="log", showgrid=True, gridcolor="#E5E5E5"),
    )
    return go.Figure(data=fig.data, layout=layout)


def population_scaling_figure(results: list[dict]) -> go.Figure:
    final_pop = [r["n_agents_final"] for r in results]
    per_step_ms = [r["wall_time_per_step_s"] * 1000 for r in results]
    fig = go.Figure(
        data=[
            go.Scatter(
                x=final_pop,
                y=per_step_ms,
                mode="lines+markers",
                line=dict(color=viz.SPECIES_PALETTE[1], width=2),
                marker=dict(size=9),
                text=[f"n0={r['n_agents_initial']}" for r in results],
                hovertemplate="%{text}<br>final population=%{x}<br>%{y:.4f} ms/step<extra></extra>",
            )
        ]
    )
    layout = _base_layout(f"Population scaling (grid {POP_GRID[0]}x{POP_GRID[1]}, {POP_N_STEPS} steps)")
    layout.update(
        xaxis=dict(title="final agent count", type="log", showgrid=True, gridcolor="#E5E5E5"),
        yaxis=dict(title="wall time / step (ms)", type="log", showgrid=True, gridcolor="#E5E5E5"),
    )
    return go.Figure(data=fig.data, layout=layout)


def duration_scaling_figure(results: list[dict]) -> go.Figure:
    steps = [r["n_steps"] for r in results]
    total_s = [r["wall_time_s"] for r in results]
    fig = go.Figure(
        data=[
            go.Scatter(
                x=steps,
                y=total_s,
                mode="lines+markers",
                line=dict(color=viz.SPECIES_PALETTE[2], width=2),
                marker=dict(size=9),
                text=[f"final pop={r['n_agents_final']}" for r in results],
                hovertemplate="steps=%{x}<br>total=%{y:.4f} s<br>%{text}<extra></extra>",
            )
        ]
    )
    layout = _base_layout(f"Duration scaling (grid {DUR_GRID[0]}x{DUR_GRID[1]}, {DUR_N_AGENTS} agents)")
    layout.update(
        xaxis=dict(title="steps", showgrid=True, gridcolor="#E5E5E5"),
        yaxis=dict(title="total wall time (s)", showgrid=True, gridcolor="#E5E5E5"),
    )
    return go.Figure(data=fig.data, layout=layout)


def throughput_figure(grid_r: list[dict], pop_r: list[dict], dur_r: list[dict]) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=[r["cells"] for r in grid_r],
            y=[r["throughput_agent_steps_per_s"] for r in grid_r],
            mode="lines+markers",
            name="grid sweep (vs cells)",
            line=dict(color=viz.SPECIES_PALETTE[0], width=2),
            marker=dict(size=9),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[r["n_agents_final"] for r in pop_r],
            y=[r["throughput_agent_steps_per_s"] for r in pop_r],
            mode="lines+markers",
            name="population sweep (vs final agents)",
            line=dict(color=viz.SPECIES_PALETTE[1], width=2),
            marker=dict(size=9),
        )
    )
    layout = _base_layout("Throughput (agent-steps / sec)")
    layout.update(
        # Both traces share this axis as a generic "sweep size" scale (grid
        # cells for the grid sweep, final agent count for the population
        # sweep) -- see per-trace names/hover for which metric applies.
        xaxis=dict(title="sweep size (grid cells or final agent count, log)", type="log",
                    showgrid=True, gridcolor="#E5E5E5"),
        yaxis=dict(title="agent-steps / sec", type="log", showgrid=True, gridcolor="#E5E5E5"),
        showlegend=True,
        legend=dict(orientation="h", y=-0.18),
    )
    return go.Figure(data=fig.data, layout=layout)


def _linearity_r2(xs: list[float], ys: list[float]) -> float:
    """R^2 of an ordinary-least-squares linear fit y = a*x + b."""
    x = np.array(xs, dtype=float)
    y = np.array(ys, dtype=float)
    A = np.vstack([x, np.ones_like(x)]).T
    slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]
    pred = slope * x + intercept
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0


def build_verdict(grid_r: list[dict], pop_r: list[dict], dur_r: list[dict], peak_throughput: float) -> dict:
    throughput_verdict = "within_tol" if peak_throughput > 0 else "mismatch"

    grid_verdict = "within_tol" if len(grid_r) == len(GRID_SIZES) and all(r["wall_time_s"] > 0 for r in grid_r) else "mismatch"
    pop_verdict = "within_tol" if len(pop_r) == len(POP_N_AGENTS) and all(r["wall_time_s"] > 0 for r in pop_r) else "mismatch"

    r2 = _linearity_r2([r["n_steps"] for r in dur_r], [r["wall_time_s"] for r in dur_r])
    if r2 >= 0.9:
        duration_verdict = "within_tol"
    elif r2 >= 0.6:
        duration_verdict = "drift"
    else:
        duration_verdict = "mismatch"

    return {
        "schema": "report_card_verdict/v1",
        "groups": {
            "performance": {
                "axes": [
                    {
                        "name": "throughput-measured",
                        "verdict": throughput_verdict,
                        "value": peak_throughput,
                        "reference": 0.0,
                        "units": "agent-steps/sec",
                    },
                    {
                        "name": "grid-scaling-produced",
                        "verdict": grid_verdict,
                        "value": len(grid_r),
                        "reference": len(GRID_SIZES),
                    },
                    {
                        "name": "population-scaling-produced",
                        "verdict": pop_verdict,
                        "value": len(pop_r),
                        "reference": len(POP_N_AGENTS),
                    },
                    {
                        "name": "duration-near-linear",
                        "verdict": duration_verdict,
                        "value": r2,
                        "reference": 0.9,
                        "note": (
                            "total wall-time vs steps R^2 of a linear fit; population "
                            "grows during the run (division), which compounds per-step "
                            "cost and can pull this below strict linearity"
                        ),
                    },
                ]
            }
        },
    }


def main() -> None:
    CHARTS.mkdir(parents=True, exist_ok=True)
    REPORT_CARD.mkdir(parents=True, exist_ok=True)

    print("Grid-size sweep (fixed 40 agents, 40 steps):")
    grid_r = run_grid_sweep()
    print("Population sweep (fixed grid 48x64, 40 steps):")
    pop_r = run_population_sweep()
    print("Duration sweep (fixed grid 48x64, 40 agents):")
    dur_r = run_duration_sweep()

    _save(grid_scaling_figure(grid_r), "scaling_grid")
    _save(population_scaling_figure(pop_r), "scaling_population")
    _save(duration_scaling_figure(dur_r), "scaling_duration")
    _save(throughput_figure(grid_r, pop_r, dur_r), "throughput")

    EMBED_DIR.mkdir(parents=True, exist_ok=True)
    for name in CHART_FILES:
        shutil.copy2(CHARTS / name, EMBED_DIR / name)

    all_results = grid_r + pop_r + dur_r
    peak_throughput = max(r["throughput_agent_steps_per_s"] for r in all_results)

    verdict = build_verdict(grid_r, pop_r, dur_r, peak_throughput)
    (REPORT_CARD / "report_card_verdict.json").write_text(json.dumps(verdict, indent=2))

    slow = [r for r in all_results if r["slow"]]
    print()
    print("=" * 72)
    print(f"peak throughput: {peak_throughput:,.0f} agent-steps/sec")
    print(f"grid sweep:       {grid_r[0]['wall_time_per_step_s']*1000:.4f} ms/step @ {grid_r[0]['cells']} cells  ->  "
          f"{grid_r[-1]['wall_time_per_step_s']*1000:.4f} ms/step @ {grid_r[-1]['cells']} cells")
    print(f"population sweep: {pop_r[0]['wall_time_per_step_s']*1000:.4f} ms/step @ {pop_r[0]['n_agents_final']} agents  ->  "
          f"{pop_r[-1]['wall_time_per_step_s']*1000:.4f} ms/step @ {pop_r[-1]['n_agents_final']} agents")
    print(f"duration sweep:   {dur_r[0]['wall_time_s']:.4f}s @ {dur_r[0]['n_steps']} steps  ->  "
          f"{dur_r[-1]['wall_time_s']:.4f}s @ {dur_r[-1]['n_steps']} steps  "
          f"(linearity R^2={verdict['groups']['performance']['axes'][3]['value']:.3f})")
    if slow:
        print(f"** {len(slow)} configuration(s) exceeded the {SLOW_THRESHOLD_S}s slow threshold -- see results above **")
    print("=" * 72)


if __name__ == "__main__":
    main()
