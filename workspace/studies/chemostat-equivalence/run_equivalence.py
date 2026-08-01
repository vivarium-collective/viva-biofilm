import json, math, pathlib
import plotly.graph_objects as go
from viva_biofilm.schema import load_chemostat
from viva_biofilm import viz
from viva_biofilm.emit import emit_run

HERE = pathlib.Path(__file__).parent
SPEC = {
    "solutes": [{"name": "solute1", "init": 2.0}, {"name": "solute2", "init": 2.0}],
    "reactions": [{"substrate": "solute1", "k": 0.1, "stoich": [["solute1", -1.0], ["solute2", 0.5]]}],
}

def run_viva(dt=0.1, tmax=60.0):
    c = load_chemostat(SPEC)
    ts, s1 = [0.0], [2.0]
    steps = int(tmax / dt)
    for i in range(steps):
        c.step(dt)
        ts.append((i + 1) * dt)
        s1.append(c.conc(0))
    return ts, s1

def try_idynomics():
    """Return (ts, s1) from the real Java engine, or None if unavailable."""
    try:
        from pbg_idynomics2.processes import IDynoMiCS2Process
        import process_bigraph as pb
    except Exception:
        return None
    try:
        proto = pathlib.Path("~/code/pbg-idynomics2/protocols/chemostat.xml").expanduser()
        core = pb.allocate_core()
        proc = IDynoMiCS2Process({"protocol_path": str(proto), "compartment": "chemostat"}, core=core)
        ts, s1 = [0.0], [2.0]
        for i in range(60):
            out = proc.update({"external_concentrations": {}}, 1.0)
            ts.append(float(out.get("time", i + 1)))
            # average_concentrations is a delta in pbg-idynomics2; accumulate
            s1.append(s1[-1] + out["average_concentrations"].get("solute1", 0.0))
        return ts, s1
    except Exception:
        return None

def main():
    ts, s1 = run_viva()
    analytic = [2.0 * math.exp(-0.1 * t) for t in ts]
    fig = go.Figure()
    fig.add_scatter(x=ts, y=s1, name="viva-biofilm (Rust)", mode="lines")
    fig.add_scatter(x=ts, y=analytic, name="analytic 2·e^(−0.1t)", mode="lines", line=dict(dash="dash"))
    oracle = try_idynomics()
    if oracle:
        fig.add_scatter(x=oracle[0], y=oracle[1], name="iDynoMiCS-2 (Java)", mode="markers")
    fig.update_layout(title="Chemostat equivalence: solute1 decay",
                      xaxis_title="time (days)", yaxis_title="solute1 (g/m³)",
                      template="plotly_white")
    charts = HERE / "charts"
    charts.mkdir(exist_ok=True)
    fig.write_html(charts / "decay.html", include_plotlyjs="cdn")
    # Static PNG sibling -- the dashboard's Charts panel (discover_static_study_charts)
    # only picks up *.svg/*.png/*.gif under charts/, not embed_visualizations.
    viz.save_png(fig, str(charts / "decay.png"))

    # verdict: an explicit equivalence check -- simulated final solute1 vs. the
    # analytic closed-form solution 2*e^(-0.1*t). value = simulated, reference
    # = analytic, delta/rel_err = the equivalence metrics themselves.
    delta = s1[-1] - analytic[-1]
    rel_err = abs(delta) / analytic[-1]
    verdict = "within_tol" if rel_err < 0.01 else ("drift" if rel_err < 0.05 else "mismatch")
    out = {
        "schema": "report_card_verdict/v1",
        "groups": {
            "chemostat-decay": {
                "axes": [
                    {"name": "solute1-final-vs-analytic", "verdict": verdict,
                     "value": s1[-1], "reference": analytic[-1],
                     "delta": delta, "rel_err": rel_err, "units": "g/m^3"},
                ]
            }
        },
        "oracle_available": oracle is not None,
    }
    report_card_dir = HERE / "viz" / "report_card"
    report_card_dir.mkdir(parents=True, exist_ok=True)
    (report_card_dir / "report_card_verdict.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))

    # Register the decay simulation as a run for the dashboard's Runs tab.
    snaps = [{"time": t, "solute1": v, "solute1_analytic": a}
             for t, v, a in zip(ts, s1, analytic)]
    emit_run(HERE, spec_id="chemostat-equivalence", snaps=snaps)

if __name__ == "__main__":
    main()
