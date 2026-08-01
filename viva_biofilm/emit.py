"""Emit a study's per-step snapshots as a first-class *simulation run*.

A viva-biofilm study is executed by a ``run_study.py`` script that drives the
Rust world directly and produces figures. To make it a real run in the
vivarium-workbench dashboard — listed in the "Runs" tab AND explorable in the
Data Explorer / native time-series charts — each run leaves a single SQLite
store ``<study_dir>/runs.db`` holding two tables:

  1. ``runs_meta`` — one row describing the run (drives the Runs tab; the
     dashboard regenerates ``.pbg/runs.jsonl`` from these rows at publish time
     via ``build_simulations_data`` and serves ``api/simulations.json``).
  2. ``history`` — the run's time-series in the exact schema
     ``process_bigraph.emitter.SQLiteEmitter`` writes
     (``simulation_id, step, global_time, state``-JSON), so the store resolves
     as ``kind="sqlite"`` with real data (``explorer_data._resolve_run_source``
     + ``_run_has_data``) and the explorer/charts can read it.

**Why not the real SQLiteEmitter?** The in-composite RAM/Parquet emitters
``history.append(tree_copy(full_state))`` every tick (tens of GB in v2ecoli).
We instead write the SQLiteEmitter's *output format* directly from the study's
already-bounded, subsampled scalar snapshots (``snapshot_every``; agent arrays
dropped, solute fields reduced to mean/min/max) — a single ``executemany``
bulk insert, so there is no per-tick accumulation. Self-contained (stdlib
``sqlite3`` + ``json``; no workbench import). The ``runs_meta`` DDL is vendored
verbatim from ``vivarium_workbench/lib/run_registry.py``.
"""
from __future__ import annotations

import json
import shutil
import sqlite3
import time
from pathlib import Path

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

# The history schema process_bigraph.emitter.SQLiteEmitter writes; the explorer
# reads global_time + json_extract(state, '$.<observable>') from it.
_HISTORY_DDL = """
CREATE TABLE IF NOT EXISTS history (
    simulation_id TEXT,
    step          INTEGER,
    global_time   REAL,
    state         TEXT
);
"""


# Snapshot keys that need structured flattening rather than a scalar copy.
_STRUCTURED_KEYS = {"pop_by_strategy", "biomass_by_strategy", "solutes", "agents"}


def _observable_state(snap: dict) -> dict:
    """Reduce one snapshot to a flat dict of scalar observables for history.state.

    Generic: every top-level scalar carries through (biofilm
    population/biomass/thickness, chemostat solute concentrations, ...), plus
    ``pop_by_strategy`` / ``biomass_by_strategy`` as per-strategy scalars and
    each solute field reduced to mean/min/max. Spatial detail (agent positions,
    full field grids) is dropped — it's already in the committed charts — so
    each state JSON stays tiny.
    """
    state = {}
    for k, v in snap.items():
        if k in _STRUCTURED_KEYS:
            continue
        if isinstance(v, bool) or isinstance(v, (int, float)):
            state[k] = v
    pbs = snap.get("pop_by_strategy") or []
    bbs = snap.get("biomass_by_strategy") or []
    for i in range(len(pbs)):
        state[f"pop_strategy_{i}"] = pbs[i]
    for i in range(len(bbs)):
        state[f"biomass_strategy_{i}"] = bbs[i]
    for name, fld in (snap.get("solutes") or {}).items():
        field = (fld or {}).get("field") or []
        if field:
            state[f"solute_{name}_mean"] = sum(field) / len(field)
            state[f"solute_{name}_min"] = min(field)
            state[f"solute_{name}_max"] = max(field)
    return state


def emit_run(study_dir, spec_id: str, snaps: list[dict], *,
             run_id: str = "baseline", label: str | None = None,
             reset: bool = True) -> str:
    """Emit one run for a study into ``<study_dir>/runs.db`` (a SQLite store).

    Writes a ``runs_meta`` row (Runs tab) and a ``history`` table (Data
    Explorer / native time-series charts) in the ``process_bigraph`` SQLite
    emitter schema. The whole time-series is bulk-inserted from the already-
    subsampled scalar snapshots, so there is no per-tick accumulation.

    Parameters
    ----------
    study_dir : path to the study directory (the one holding ``study.yaml``).
    spec_id   : spec/study identifier for the run (use the study slug).
    snaps     : the study's list of per-step snapshot dicts.
    run_id    : short run id; the stored id is namespaced ``<spec_id>-<run_id>``
                so it is unique across the workspace (the dashboard folds runs by
                run_id globally). Use distinct short ids for a multi-run study.
    label     : human label shown in the dashboard (defaults to the short id).
    reset     : when True (default) the call first deletes ``runs.db`` so a
                re-run leaves no stale rows. Pass ``reset=False`` for the
                2nd..Nth run of a multi-run study.

    Returns the stored (namespaced) run id.
    """
    study_dir = Path(study_dir)
    study_dir.mkdir(parents=True, exist_ok=True)
    runs_db = study_dir / "runs.db"
    # Legacy out/ parquet dir from an earlier emit format — remove if present.
    out_root = study_dir / "out"
    if reset:
        if out_root.exists():
            shutil.rmtree(out_root)
        if runs_db.exists():
            runs_db.unlink()

    # Namespace the id with the study slug so it's workspace-unique (the run_log
    # fold + simulations_index dedup key on run_id alone across ALL studies).
    full_id = run_id if run_id.startswith(f"{spec_id}-") else f"{spec_id}-{run_id}"
    now = time.time()

    conn = sqlite3.connect(runs_db)
    try:
        conn.executescript(_RUNS_META_DDL + _HISTORY_DDL)
        conn.execute(
            "INSERT OR REPLACE INTO runs_meta"
            "(run_id, spec_id, label, started_at, completed_at, n_steps, status)"
            " VALUES(?,?,?,?,?,?,?)",
            (full_id, spec_id, label or run_id, now, now, len(snaps), "completed"),
        )
        conn.executemany(
            "INSERT INTO history(simulation_id, step, global_time, state) VALUES(?,?,?,?)",
            [(full_id, step, float(s.get("time", step)),
              json.dumps(_observable_state(s)))
             for step, s in enumerate(snaps)],
        )
        conn.commit()
    finally:
        conn.close()
    return full_id
