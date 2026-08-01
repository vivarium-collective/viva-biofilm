"""Tests for viva_biofilm.emit — study runs registered for the dashboard."""
import sqlite3

import pandas as pd

from viva_biofilm.emit import emit_run, _observables_frame


def _biofilm_snaps():
    return [
        {"time": 0.0, "population": 40, "total_biomass": 4.0, "biofilm_thickness": 1.0,
         "solutes": {"oxygen": {"field": [1.0, 0.5, 0.0], "nx": 3, "ny": 1}}},
        {"time": 1.0, "population": 80, "total_biomass": 9.0, "biofilm_thickness": 2.0,
         "solutes": {"oxygen": {"field": [0.8, 0.4, 0.1], "nx": 3, "ny": 1}}},
    ]


def test_emit_run_writes_parquet_and_runs_meta(tmp_path):
    # run_id is namespaced with the study slug so it's workspace-unique.
    run_id = emit_run(tmp_path, spec_id="my-study", snaps=_biofilm_snaps())
    assert run_id == "my-study-baseline"

    parquet = tmp_path / "out" / "my-study-baseline" / "observables.parquet"
    assert parquet.is_file()
    df = pd.read_parquet(parquet)
    assert len(df) == 2
    # scalars carried through + solute field summarized
    for col in ("time", "population", "total_biomass", "biofilm_thickness",
                "solute_oxygen_mean", "solute_oxygen_min", "solute_oxygen_max"):
        assert col in df.columns

    runs_db = tmp_path / "runs.db"
    assert runs_db.is_file()
    rows = sqlite3.connect(runs_db).execute(
        "SELECT run_id, spec_id, label, status, n_steps, emitter_path FROM runs_meta").fetchall()
    assert rows == [("my-study-baseline", "my-study", "baseline", "completed", 2,
                     "out/my-study-baseline")]


def test_observables_frame_carries_arbitrary_scalars():
    # A chemostat-style study emits plain scalar observables (no biofilm fields).
    snaps = [{"time": 0.0, "solute1": 2.0, "solute1_analytic": 2.0},
             {"time": 1.0, "solute1": 1.81, "solute1_analytic": 1.81}]
    df = _observables_frame(snaps)
    assert list(df.columns) == ["step", "time", "solute1", "solute1_analytic"]
    assert df["solute1"].tolist() == [2.0, 1.81]


def test_emit_run_multi_run_accumulates(tmp_path):
    emit_run(tmp_path, spec_id="s", snaps=_biofilm_snaps(), run_id="a", reset=True)
    emit_run(tmp_path, spec_id="s", snaps=_biofilm_snaps(), run_id="b", reset=False)
    ids = {r[0] for r in sqlite3.connect(tmp_path / "runs.db").execute(
        "SELECT run_id FROM runs_meta")}
    assert ids == {"s-a", "s-b"}
    assert (tmp_path / "out" / "s-a" / "observables.parquet").is_file()
    assert (tmp_path / "out" / "s-b" / "observables.parquet").is_file()


def test_emit_run_ids_are_study_namespaced_for_global_uniqueness(tmp_path):
    # Two studies both using the default "baseline" must NOT collide on run_id
    # (the dashboard folds runs by run_id globally across the workspace).
    a = emit_run(tmp_path / "study_a", spec_id="study-a", snaps=_biofilm_snaps())
    b = emit_run(tmp_path / "study_b", spec_id="study-b", snaps=_biofilm_snaps())
    assert a != b
    assert a == "study-a-baseline" and b == "study-b-baseline"


def test_emit_run_reset_clears_stale(tmp_path):
    emit_run(tmp_path, spec_id="s", snaps=_biofilm_snaps(), run_id="old", reset=True)
    # A fresh reset run must drop the stale "old" row + its out dir.
    emit_run(tmp_path, spec_id="s", snaps=_biofilm_snaps(), run_id="new", reset=True)
    ids = {r[0] for r in sqlite3.connect(tmp_path / "runs.db").execute(
        "SELECT run_id FROM runs_meta")}
    assert ids == {"s-new"}
    assert not (tmp_path / "out" / "s-old").exists()
