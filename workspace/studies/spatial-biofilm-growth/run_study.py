"""Spatial biofilm growth study driver.

Runs a developed 2D biofilm (viva_biofilm Rust engine) with FIXED parameters
long enough to reach a visible thickness and a clear substrate-limitation
gradient, renders the colony / solute-field / time-lapse / growth-curve
figures via ``viz.py``, and writes a report-card verdict for the study's
``spatial-structure`` test group.

Parameters are fixed (do NOT sweep): nx=32, ny=48, n_agents=40, seed=11,
n_steps=150, snapshot_every=15, dt=0.05.
"""
import json
import pathlib
import shutil

from viva_biofilm.run import run_biofilm, default_spec
from viva_biofilm import viz

HERE = pathlib.Path(__file__).parent
CHARTS = HERE / "charts"
REPORT_CARD = HERE / "viz" / "report_card"
EMBED_DIR = HERE.parents[2] / "reports" / "figures" / "spatial-biofilm-growth"

DX = 2.0

# Fixed run parameters -- see task brief: n_steps=150, snapshot_every=15,
# dt=0.05 yields ~640 agents, ~11.4 um thickness, substratum ~ 0.39x boundary.
N_STEPS = 150
SNAPSHOT_EVERY = 15
DT = 0.05

CHART_FILES = [
    "colony_final.html",
    "colony_substrate.html",
    "solute_substrate.html",
    "solute_oxygen.html",
    "timelapse.html",
    "growth_curves.html",
]


def gradient_stats(snapshot: dict) -> tuple[float, float, float]:
    """Return (substratum_third_mean, boundary_third_mean, ratio) for 'solute'."""
    f = snapshot["solutes"]["solute"]
    nx, ny = f["nx"], f["ny"]
    field = f["field"]
    substratum = sum(field[0:nx * (ny // 3)]) / (nx * (ny // 3))
    boundary = sum(field[nx * 2 * (ny // 3):nx * ny]) / (nx * (ny - 2 * (ny // 3)))
    ratio = substratum / boundary if boundary else float("inf")
    return substratum, boundary, ratio


def build_verdict(snaps: list[dict]) -> dict:
    first, last = snaps[0], snaps[-1]
    substratum, boundary, ratio = gradient_stats(last)

    # Axis 1: population grew (final > initial).
    pop_verdict = "within_tol" if last["population"] > first["population"] else "mismatch"

    # Axis 2: biofilm thickness is positive.
    thickness_verdict = "within_tol" if last["biofilm_thickness"] > 0.0 else "mismatch"

    # Axis 3: a clear substrate gradient (boundary noticeably richer than
    # substratum). Thresholds: ratio < 0.7 -> clear margin (within_tol);
    # ratio < 1.0 -> some gradient but weak (drift); else no gradient (mismatch).
    if ratio < 0.7:
        gradient_verdict = "within_tol"
    elif ratio < 1.0:
        gradient_verdict = "drift"
    else:
        gradient_verdict = "mismatch"

    return {
        "schema": "report_card_verdict/v1",
        "groups": {
            "spatial-structure": {
                "axes": [
                    {
                        "name": "population-grew",
                        "verdict": pop_verdict,
                        "value": last["population"],
                        "reference": first["population"],
                    },
                    {
                        "name": "thickness-positive",
                        "verdict": thickness_verdict,
                        "value": last["biofilm_thickness"],
                        "reference": 0.0,
                    },
                    {
                        "name": "substrate-gradient-present",
                        "verdict": gradient_verdict,
                        "value": ratio,
                        "reference": 0.7,
                        "substratum_mean": substratum,
                        "boundary_mean": boundary,
                    },
                ]
            }
        },
    }


def main() -> None:
    spec = default_spec(nx=32, ny=48, n_agents=40, seed=11)
    snaps = run_biofilm(spec, n_steps=N_STEPS, snapshot_every=SNAPSHOT_EVERY, dt=DT)
    last = snaps[-1]

    CHARTS.mkdir(parents=True, exist_ok=True)
    REPORT_CARD.mkdir(parents=True, exist_ok=True)

    viz.save_html(viz.colony_figure(last, color_by="mass", dx=DX), str(CHARTS / "colony_final.html"))
    viz.save_html(viz.colony_figure(last, color_by="local_substrate", dx=DX), str(CHARTS / "colony_substrate.html"))
    viz.save_html(viz.solute_field_figure(last, "solute", dx=DX), str(CHARTS / "solute_substrate.html"))
    viz.save_html(viz.solute_field_figure(last, "oxygen", dx=DX), str(CHARTS / "solute_oxygen.html"))
    viz.save_html(viz.timelapse_figure(snaps, color_by="mass", dx=DX), str(CHARTS / "timelapse.html"))
    viz.save_html(viz.growth_curves_figure(snaps), str(CHARTS / "growth_curves.html"))

    # Mirror the charts into reports/figures/<study>/ for study.yaml's
    # embed_visualizations (workspace-root-relative /reports/figures/... URLs).
    EMBED_DIR.mkdir(parents=True, exist_ok=True)
    for name in CHART_FILES:
        shutil.copy2(CHARTS / name, EMBED_DIR / name)

    verdict = build_verdict(snaps)
    (REPORT_CARD / "report_card_verdict.json").write_text(json.dumps(verdict, indent=2))

    substratum, boundary, ratio = gradient_stats(last)
    print(
        f"final population: {last['population']} (from {snaps[0]['population']})  "
        f"thickness: {last['biofilm_thickness']:.2f} um  "
        f"substratum/boundary ratio: {ratio:.3f} "
        f"(substratum={substratum:.4f}, boundary={boundary:.4f})"
    )


if __name__ == "__main__":
    main()
