"""Emit a study's per-step snapshots as a first-class *simulation run*.

A viva-biofilm study is executed by a ``run_study.py`` script that produces
figures directly. To also make it show up as a real run in the
vivarium-workbench dashboard (the "Runs" tab, and its per-study run listing),
each run must leave two artifacts on disk that the dashboard reads:

  1. an emitter directory ``<study_dir>/out/<run_id>/`` holding the run's
     time-series data as parquet (this is the "emit"), and
  2. a ``runs_meta`` row in ``<study_dir>/runs.db`` describing the run.

At publish time the dashboard regenerates its ``.pbg/runs.jsonl`` index from
every study's ``runs.db`` (``build_simulations_data`` →
``backfill_index_into_jsonl``) and serves it as ``api/simulations.json`` — so
committing ``runs.db`` (+ the parquet) is all that's needed for the read-only
dashboard to list the run. Clicking a run navigates to the study's own page
(and its Visualizations tab), so no particular parquet column schema is
required; we emit a compact scalar-observables table.

This helper is intentionally self-contained (only ``pandas`` + ``pyarrow``) so a
study script never needs the workbench package installed to emit. The
``runs_meta`` DDL is vendored verbatim from
``vivarium_workbench/lib/run_registry.py`` — keep it in sync if that schema
changes (the dashboard only reads these columns).
"""
from __future__ import annotations

import shutil
import sqlite3
import time
from pathlib import Path

import pandas as pd

# Vendored verbatim from vivarium_workbench/lib/run_registry.py::RUNS_META_DDL.
# The dashboard ALTERs in extra nullable columns later; a superset row is fine.
_RUNS_META_DDL = """
CREATE TABLE IF NOT EXISTS runs_meta (
    run_id        TEXT PRIMARY KEY,
    spec_id       TEXT NOT NULL,
    label         TEXT,
    params_json   TEXT,
    started_at    REAL NOT NULL,
    completed_at  REAL,
    n_steps       INTEGER,
    status        TEXT NOT NULL,
    sim_name      TEXT,
    generation_id TEXT,
    emitter_path  TEXT
);
"""


# Snapshot keys that need structured flattening rather than a scalar copy.
_STRUCTURED_KEYS = {"pop_by_strategy", "biomass_by_strategy", "solutes", "agents"}


def _observables_frame(snaps: list[dict]) -> pd.DataFrame:
    """Flatten a study's per-step snapshot dicts into a scalar time-series table.

    Generic: every top-level scalar key is copied as its own column (so any
    study's observables — biofilm population/biomass/thickness, chemostat solute
    concentrations, etc. — carry through), plus special handling for
    ``pop_by_strategy`` / ``biomass_by_strategy`` (per-strategy columns) and
    ``solutes`` (per-solute field mean/min/max). Spatial detail (agent
    positions, full field grids) is intentionally dropped — it's already in the
    committed charts — so the parquet stays tiny.
    """
    n_strat = max((len(s.get("pop_by_strategy") or []) for s in snaps), default=0)
    rows = []
    for step, s in enumerate(snaps):
        row = {"step": step}
        for k, v in s.items():
            if k in _STRUCTURED_KEYS:
                continue
            if v is None or isinstance(v, (int, float, bool)):
                row[k] = v
        pbs = s.get("pop_by_strategy") or []
        bbs = s.get("biomass_by_strategy") or []
        for i in range(n_strat):
            row[f"pop_strategy_{i}"] = pbs[i] if i < len(pbs) else None
            row[f"biomass_strategy_{i}"] = bbs[i] if i < len(bbs) else None
        for name, fld in (s.get("solutes") or {}).items():
            field = (fld or {}).get("field") or []
            if field:
                row[f"solute_{name}_mean"] = sum(field) / len(field)
                row[f"solute_{name}_min"] = min(field)
                row[f"solute_{name}_max"] = max(field)
        rows.append(row)
    return pd.DataFrame(rows)


def emit_run(study_dir, spec_id: str, snaps: list[dict], *,
             run_id: str = "baseline", label: str | None = None,
             reset: bool = True) -> str:
    """Emit one run for a study: parquet under ``out/<run_id>/`` + a runs_meta row.

    Parameters
    ----------
    study_dir : path to the study directory (the one holding ``study.yaml``).
    spec_id   : display/spec identifier for the run (use the study slug).
    snaps     : the study's list of per-step snapshot dicts.
    run_id    : run directory / id (default ``"baseline"``). Use distinct ids
                for a multi-run study (e.g. one per swept condition).
    label     : human label shown in the dashboard (defaults to ``run_id``).
    reset     : when True (default) the call first wipes ``out/`` and
                ``runs.db`` so re-running a study leaves no stale rows. Pass
                ``reset=False`` for the 2nd..Nth run of a multi-run study.

    Returns the ``run_id``.
    """
    study_dir = Path(study_dir)
    runs_db = study_dir / "runs.db"
    out_root = study_dir / "out"
    if reset:
        if out_root.exists():
            shutil.rmtree(out_root)
        if runs_db.exists():
            runs_db.unlink()

    # The dashboard folds runs by run_id GLOBALLY across the workspace (run_log
    # fold + simulations_index dedup key on run_id alone), so two studies using
    # the same short id (e.g. both "baseline") would collide and one would drop.
    # Namespace the id with the study slug (spec_id) to keep it workspace-unique;
    # keep the caller's short id as the display label.
    full_id = run_id if run_id.startswith(f"{spec_id}-") else f"{spec_id}-{run_id}"
    run_dir = out_root / full_id
    run_dir.mkdir(parents=True, exist_ok=True)
    _observables_frame(snaps).to_parquet(run_dir / "observables.parquet", index=False)

    now = time.time()
    conn = sqlite3.connect(runs_db)
    try:
        conn.executescript(_RUNS_META_DDL)
        conn.execute(
            "INSERT OR REPLACE INTO runs_meta"
            "(run_id, spec_id, label, started_at, completed_at, n_steps, status, emitter_path)"
            " VALUES(?,?,?,?,?,?,?,?)",
            (full_id, spec_id, label or run_id, now, now, len(snaps),
             "completed", f"out/{full_id}"),
        )
        conn.commit()
    finally:
        conn.close()
    return full_id
