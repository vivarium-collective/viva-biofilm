# viva-biofilm

Rust reimplementation of the iDynoMiCS-2 individual-based biofilm framework,
wrapped as viva (process-bigraph) processes in a vivarium-workspace.

See `docs/superpowers/specs/2026-07-31-viva-biofilm-design.md` for the design.

## Workbench

Build, test, run, and view this workspace via
[vivarium-workbench](https://github.com/vivarium-collective/vivarium-workbench):

```bash
source .venv/bin/activate

# build the Rust extension
maturin develop -m crates/biofilm-py/Cargo.toml

# run the test suite
pytest -q

# run a study (writes charts + verdict under workspace/studies/<study>/charts/)
python workspace/studies/spatial-biofilm-growth/run_study.py

# serve the dashboard
vivarium-workbench serve --workspace . --port 8099
```

The `viva-biofilm-capabilities` investigation
(`workspace/investigations/viva-biofilm-capabilities/`) collects three
studies: **spatial-biofilm-growth** (a developed 2D biofilm with a real
substrate gradient), **runtime-and-scaling** (throughput and scaling curves
for the Rust engine), and **composability** (a boundary-controller process
perturbing the biofilm's oxygen boundary at runtime and driving a measurable
growth response). Each study embeds its interactive Plotly figures — see
`reports/figures/<study>/*.html`.
