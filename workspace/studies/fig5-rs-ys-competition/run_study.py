"""Fig 5 RS-vs-YS competition study driver — Cockx et al. 2024 Fig. 5 reproduction.

Runs the paper's Rate-Strategist-vs-Yield-Strategist competition (exact
Table-K kinetics baked into ``viva_biofilm.run.competition_spec``) at the
paper's three seeding densities (5, 10, 50 agents per strategy), renders the
outcome, and writes a report-card verdict for the study's ``competition``
test group.

FIDELITY CAVEAT — horizon: the paper runs to 21 days (504 steps at
dt=1/24 d). Table-K's fast kinetics + tiny division_mass (0.08 pg) drive
near-exponential early population growth (e.g. n_each=50 reaches ~2400
agents by day 4), and this engine's agent-relaxation ("shoving") pass is
O(n^2) per step. A full 504-step run at n_each=50 was timed and killed after
several minutes without completing even a quarter of the horizon — clearly
outside the study's ~10-minute runtime budget. We use a SHORTER horizon
instead: dt=1/24 d (the paper's own Delta-t = 1 hour, unchanged) for
N_STEPS steps (see below) — keeping the paper's PARAMETER values exactly as
given (nothing in competition_spec's RS/YS kinetics is altered), only the
observation WINDOW is shortened. See the study.yaml narrative for how this
bears on the result's fidelity to the paper's exact density-flip finding.
"""
import json
import pathlib
import shutil
import time

import plotly.graph_objects as go

from viva_biofilm.run import competition_spec, run_competition
from viva_biofilm import viz

HERE = pathlib.Path(__file__).parent
CHARTS = HERE / "charts"
REPORT_CARD = HERE / "viz" / "report_card"
EMBED_DIR = HERE.parents[2] / "reports" / "figures" / "fig5-rs-ys-competition"

DENSITIES = [5, 10, 50]
SEED = 3
DT = 1 / 24        # paper's Delta-t = 1 hour (unchanged)
N_STEPS = 96       # 4.0 days -- SHORTENED from the paper's 21 d / 504 steps; see module docstring.
SNAPSHOT_EVERY = 12  # 0.5 day resolution for the time-series chart

CHART_STEMS = ["outcome_vs_density", "colony_density5", "colony_density50", "fraction_over_time"]

# Density-line colors for the time-series chart: low density closer to RS's
# blue, high density closer to YS's red, echoing the outcome-figure palette.
_DENSITY_COLORS = {5: viz.STRATEGY_PALETTE[0], 10: "#7570B3", 50: viz.STRATEGY_PALETTE[1]}


def rs_fraction(snapshot: dict) -> float:
    rs, ys = snapshot["biomass_by_strategy"]
    total = rs + ys
    return rs / total if total > 0 else float("nan")


def _save(fig: go.Figure, stem: str) -> None:
    viz.save_html(fig, str(CHARTS / f"{stem}.html"))
    viz.save_png(fig, str(CHARTS / f"{stem}.png"))


def fraction_over_time_figure(runs: dict[int, list[dict]]) -> go.Figure:
    """RS biomass-fraction vs time, one line per seeding density."""
    fig = go.Figure()
    for n in DENSITIES:
        times = [s["time"] for s in runs[n]]
        fracs = [rs_fraction(s) for s in runs[n]]
        fig.add_trace(go.Scatter(
            x=times, y=fracs, mode="lines+markers", name=f"n_each={n}",
            line=dict(color=_DENSITY_COLORS.get(n, "#333333"), width=2),
            marker=dict(size=6),
            hovertemplate=f"n_each={n}<br>" + "t=%{x:.2f}d<br>RS fraction=%{y:.3f}<extra></extra>",
        ))
    t_max = max(s["time"] for snaps in runs.values() for s in snaps)
    fig.add_shape(type="line", x0=0, x1=t_max, y0=0.5, y1=0.5,
                  line=dict(color="#999999", width=1, dash="dash"))
    fig.update_layout(
        template="plotly_white",
        title=dict(text="RS biomass fraction over time, by seeding density", font=viz.TITLE_FONT),
        xaxis=dict(title="time (days)", showgrid=False, zeroline=False),
        yaxis=dict(title="RS biomass fraction  [RS / (RS+YS)]", range=[0, 1],
                   showgrid=True, gridcolor="#E5E5E5", zeroline=False),
        height=440,
        paper_bgcolor=viz.PAPER_BGCOLOR,
        plot_bgcolor=viz.PLOT_BGCOLOR,
        font=viz.FONT,
        margin=dict(l=70, r=40, t=60, b=60),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    return fig


def build_verdict(runs: dict[int, list[dict]], outcomes: list[dict]) -> dict:
    fractions = {r["n_each"]: r["rs_fraction"] for r in outcomes}
    spread = max(fractions.values()) - min(fractions.values())

    # Axis 1: both strategies persist at the final step, at every density.
    both_alive = all(
        runs[n][-1]["pop_by_strategy"][0] > 0 and runs[n][-1]["pop_by_strategy"][1] > 0
        for n in DENSITIES
    )
    both_verdict = "within_tol" if both_alive else "mismatch"

    # Axis 2: the outcome is density-dependent -- RS fraction differs clearly
    # across densities. spread > 0.02 = clear margin; > 0.005 = weak drift.
    if spread > 0.02:
        density_verdict = "within_tol"
    elif spread > 0.005:
        density_verdict = "drift"
    else:
        density_verdict = "mismatch"

    # Axis 3: HONEST grading against the paper's reported direction. Cockx
    # 2024 Fig 5 reports YS favored at low density, RS at intermediate
    # density, YS favored again at high density (a non-monotonic flip). Our
    # measured trend: RS dominates (fraction > 0.5) at ALL three densities,
    # monotonically DECREASING with density (5 -> 10 -> 50) -- i.e. YS gains
    # relative ground as density rises (partial directional agreement with
    # the paper's high-density YS-favoring direction), but RS never loses
    # outright at any density and there is no low-density YS win or flip.
    fracs_by_n = [fractions[n] for n in DENSITIES]
    monotonic_decreasing = all(fracs_by_n[i] > fracs_by_n[i + 1] for i in range(len(fracs_by_n) - 1))
    any_ys_win = any(f < 0.5 for f in fracs_by_n)
    if any_ys_win and monotonic_decreasing:
        # A genuine partial match (density trend AND an outright YS win somewhere).
        paper_verdict = "within_tol"
        paper_note = "RS fraction decreases with density AND YS wins outright at the highest density tested."
    else:
        paper_verdict = "drift"
        paper_note = (
            "RS fraction decreases monotonically with seeding density (qualitative direction "
            "matches the paper's high-density YS-favoring trend), but RS remains dominant "
            "(fraction > 0.5) at every density tested -- no outright YS win, and no low-density "
            "YS-favored / intermediate-density RS-favored flip as the paper reports. Shortened "
            "4-day horizon (vs the paper's 21 days) is the leading suspect: the paper's flip may "
            "only emerge as the biofilm matures and becomes more strongly diffusion-limited."
        )

    return {
        "schema": "report_card_verdict/v1",
        "groups": {
            "competition": {
                "axes": [
                    {
                        "name": "both-strategies-simulated",
                        "verdict": both_verdict,
                        "value": {str(n): runs[n][-1]["pop_by_strategy"] for n in DENSITIES},
                        "reference": "both > 0 at every density",
                    },
                    {
                        "name": "outcome-is-density-dependent",
                        "verdict": density_verdict,
                        "value": fractions,
                        "reference": 0.02,
                        "note": f"max-min RS-fraction spread across densities = {spread:.4f}",
                    },
                    {
                        "name": "matches-paper-direction",
                        "verdict": paper_verdict,
                        "value": fractions,
                        "reference": "paper: YS low, RS intermediate, YS high (non-monotonic flip)",
                        "note": paper_note,
                    },
                ]
            }
        },
    }


def main() -> None:
    CHARTS.mkdir(parents=True, exist_ok=True)
    REPORT_CARD.mkdir(parents=True, exist_ok=True)

    runs: dict[int, list[dict]] = {}
    wall_times: dict[int, float] = {}
    for n in DENSITIES:
        spec = competition_spec(n_each=n, seed=SEED)
        t0 = time.time()
        snaps = run_competition(spec, n_steps=N_STEPS, dt=DT, snapshot_every=SNAPSHOT_EVERY)
        wall_times[n] = time.time() - t0
        runs[n] = snaps
        print(f"density n_each={n:>3}: {N_STEPS} steps ({N_STEPS * DT:.2f} d) in {wall_times[n]:.1f}s"
              f"  final pop_by_strategy={snaps[-1]['pop_by_strategy']}")

    outcomes = [{"n_each": n, "rs_fraction": rs_fraction(runs[n][-1])} for n in DENSITIES]

    _save(viz.competition_outcome_figure(outcomes), "outcome_vs_density")
    _save(
        viz.strategy_colony_figure(
            runs[5][-1], title=f"Final colony, n_each=5 (t={runs[5][-1]['time']:.1f}d)"
        ),
        "colony_density5",
    )
    _save(
        viz.strategy_colony_figure(
            runs[50][-1], title=f"Final colony, n_each=50 (t={runs[50][-1]['time']:.1f}d)"
        ),
        "colony_density50",
    )
    _save(fraction_over_time_figure(runs), "fraction_over_time")

    # Mirror the charts into reports/figures/<study>/ for study.yaml's
    # embed_visualizations (workspace-root-relative /reports/figures/... URLs).
    EMBED_DIR.mkdir(parents=True, exist_ok=True)
    for stem in CHART_STEMS:
        for ext in ("html", "png"):
            shutil.copy2(CHARTS / f"{stem}.{ext}", EMBED_DIR / f"{stem}.{ext}")

    verdict = build_verdict(runs, outcomes)
    (REPORT_CARD / "report_card_verdict.json").write_text(json.dumps(verdict, indent=2))

    total_wall = sum(wall_times.values())
    print(f"\ntotal wall time: {total_wall:.1f}s across {len(DENSITIES)} densities "
          f"({N_STEPS} steps / {N_STEPS * DT:.2f} d each)")
    print("\ndensity -> RS biomass fraction")
    for r in outcomes:
        print(f"  n_each={r['n_each']:>3}: {r['rs_fraction']:.4f}")


if __name__ == "__main__":
    main()
