"""Tests for viva_biofilm.emit — study runs registered for the dashboard."""
import json
import sqlite3

from viva_biofilm.emit import emit_run, _observable_state


def _biofilm_snaps():
    return [
        {"time": 0.0, "population": 40, "total_biomass": 4.0, "biofilm_thickness": 1.0,
         "solutes": {"oxygen": {"field": [1.0, 0.5, 0.0], "nx": 3, "ny": 1}}},
        {"time": 1.0, "population": 80, "total_biomass": 9.0, "biofilm_thickness": 2.0,
         "solutes": {"oxygen": {"field": [0.8, 0.4, 0.1], "nx": 3, "ny": 1}}},
    ]


def test_emit_run_sqlite_writes_runs_meta_and_history(tmp_path):
    # run_id is namespaced with the study slug so it's workspace-unique.
    run_id = emit_run(tmp_path, spec_id="my-study", snaps=_biofilm_snaps(),
                      emitter="sqlite")
    assert run_id == "my-study-baseline"

    runs_db = tmp_path / "runs.db"
    assert runs_db.is_file()
    conn = sqlite3.connect(runs_db)
    rows = conn.execute(
        "SELECT run_id, spec_id, label, status, n_steps FROM runs_meta").fetchall()
    assert rows == [("my-study-baseline", "my-study", "baseline", "completed", 2)]

    # history in the SQLiteEmitter schema: one row per step, state a JSON dict.
    hist = conn.execute(
        "SELECT simulation_id, step, global_time, state FROM history ORDER BY step").fetchall()
    assert len(hist) == 2
    assert hist[0][:3] == ("my-study-baseline", 0, 0.0)
    st = json.loads(hist[1][3])
    assert st["population"] == 80 and st["biofilm_thickness"] == 2.0
    for k in ("solute_oxygen_mean", "solute_oxygen_min", "solute_oxygen_max"):
        assert k in st


def _biofilm_snaps_n(n):
    return [{"time": float(i), "population": 40 * (i + 1),
             "total_biomass": 4.0 * (i + 1), "biofilm_thickness": 1.0 + i}
            for i in range(n)]


def test_emit_run_xarray_writes_zarr_store(tmp_path):
    # Default emitter is xarray → a runs.<full_id>.zarr store + runs_meta row,
    # and NO history table (data lives in the zarr). Needs >= the emitter buffer
    # size (3) of snapshots; the studies always have many more.
    run_id = emit_run(tmp_path, spec_id="my-study", snaps=_biofilm_snaps_n(6))
    assert run_id == "my-study-baseline"
    assert (tmp_path / "runs.my-study-baseline.zarr").is_dir()

    conn = sqlite3.connect(tmp_path / "runs.db")
    rows = conn.execute(
        "SELECT run_id, status, n_steps, sim_name FROM runs_meta").fetchall()
    assert rows == [("my-study-baseline", "completed", 6, "my-study-baseline")]
    tables = {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'")}
    assert "history" not in tables


def test_emit_run_xarray_handles_short_runs(tmp_path):
    # The XArray store flushes on close even for very short runs, so no fallback
    # is triggered (a 2-snapshot run still produces a valid zarr).
    emit_run(tmp_path, spec_id="s", snaps=_biofilm_snaps_n(2))  # default xarray
    assert (tmp_path / "runs.s-baseline.zarr").is_dir()


def test_observable_state_carries_arbitrary_scalars():
    # A chemostat-style study emits plain scalar observables (no biofilm fields).
    st = _observable_state({"time": 1.0, "solute1": 1.81, "solute1_analytic": 1.81})
    assert st == {"time": 1.0, "solute1": 1.81, "solute1_analytic": 1.81}


def test_emit_run_multi_run_accumulates(tmp_path):
    emit_run(tmp_path, spec_id="s", snaps=_biofilm_snaps(), run_id="a",
             reset=True, emitter="sqlite")
    emit_run(tmp_path, spec_id="s", snaps=_biofilm_snaps(), run_id="b",
             reset=False, emitter="sqlite")
    ids = {r[0] for r in sqlite3.connect(tmp_path / "runs.db").execute(
        "SELECT run_id FROM runs_meta")}
    assert ids == {"s-a", "s-b"}
    # both runs' history present
    counts = dict(sqlite3.connect(tmp_path / "runs.db").execute(
        "SELECT simulation_id, count(*) FROM history GROUP BY simulation_id"))
    assert counts == {"s-a": 2, "s-b": 2}


def test_emit_run_ids_are_study_namespaced_for_global_uniqueness(tmp_path):
    # Two studies both using the default "baseline" must NOT collide on run_id
    # (the dashboard folds runs by run_id globally across the workspace).
    a = emit_run(tmp_path / "study_a", spec_id="study-a", snaps=_biofilm_snaps())
    b = emit_run(tmp_path / "study_b", spec_id="study-b", snaps=_biofilm_snaps())
    assert a != b
    assert a == "study-a-baseline" and b == "study-b-baseline"


def test_emit_run_reset_clears_stale(tmp_path):
    emit_run(tmp_path, spec_id="s", snaps=_biofilm_snaps(), run_id="old",
             reset=True, emitter="sqlite")
    # A fresh reset run must drop the stale "old" row + its history.
    emit_run(tmp_path, spec_id="s", snaps=_biofilm_snaps(), run_id="new",
             reset=True, emitter="sqlite")
    conn = sqlite3.connect(tmp_path / "runs.db")
    ids = {r[0] for r in conn.execute("SELECT run_id FROM runs_meta")}
    assert ids == {"s-new"}
    hist_ids = {r[0] for r in conn.execute("SELECT DISTINCT simulation_id FROM history")}
    assert hist_ids == {"s-new"}


def test_emit_run_reset_clears_stale_zarr(tmp_path):
    # Default (xarray) reset must also drop a stale runs.*.zarr store.
    emit_run(tmp_path, spec_id="s", snaps=_biofilm_snaps(), run_id="old", reset=True)
    emit_run(tmp_path, spec_id="s", snaps=_biofilm_snaps(), run_id="new", reset=True)
    assert (tmp_path / "runs.s-new.zarr").is_dir()
    assert not (tmp_path / "runs.s-old.zarr").exists()
