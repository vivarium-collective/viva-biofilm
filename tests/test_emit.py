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


def test_emit_run_writes_runs_meta_and_history(tmp_path):
    # run_id is namespaced with the study slug so it's workspace-unique.
    run_id = emit_run(tmp_path, spec_id="my-study", snaps=_biofilm_snaps())
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
    # No legacy parquet dir is written.
    assert not (tmp_path / "out").exists()


def test_observable_state_carries_arbitrary_scalars():
    # A chemostat-style study emits plain scalar observables (no biofilm fields).
    st = _observable_state({"time": 1.0, "solute1": 1.81, "solute1_analytic": 1.81})
    assert st == {"time": 1.0, "solute1": 1.81, "solute1_analytic": 1.81}


def test_emit_run_multi_run_accumulates(tmp_path):
    emit_run(tmp_path, spec_id="s", snaps=_biofilm_snaps(), run_id="a", reset=True)
    emit_run(tmp_path, spec_id="s", snaps=_biofilm_snaps(), run_id="b", reset=False)
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
    emit_run(tmp_path, spec_id="s", snaps=_biofilm_snaps(), run_id="old", reset=True)
    # A fresh reset run must drop the stale "old" row + its history.
    emit_run(tmp_path, spec_id="s", snaps=_biofilm_snaps(), run_id="new", reset=True)
    conn = sqlite3.connect(tmp_path / "runs.db")
    ids = {r[0] for r in conn.execute("SELECT run_id FROM runs_meta")}
    assert ids == {"s-new"}
    hist_ids = {r[0] for r in conn.execute("SELECT DISTINCT simulation_id FROM history")}
    assert hist_ids == {"s-new"}
