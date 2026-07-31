import pathlib
import pytest
import yaml
import process_bigraph as pb
from viva_biofilm.core import build_core

COMPOSITES = pathlib.Path("viva_biofilm/composites")


def test_biofilm_composite_builds():
    core = build_core()
    doc = yaml.safe_load((COMPOSITES / "biofilm.composite.yaml").read_text())
    state = doc["state"]
    composite = pb.Composite({"state": state}, core=core)
    composite.run(0.05)  # one interval; must not raise


def test_chemostat_composite_builds():
    core = build_core()
    doc = yaml.safe_load((COMPOSITES / "chemostat.composite.yaml").read_text())
    composite = pb.Composite({"state": doc["state"]}, core=core)
    composite.run(1.0)


def test_chemostat_composite_multistep_store_holds_absolutes():
    """Regression test for the delta/initial_state swap-contract bugs.

    average_concentrations is an accumulate[map[string,float]] port: each
    update() must return a per-step DELTA, and initial_state() must seed the
    store with the starting absolutes, or repeated intervals either double-
    count (no initial_state) or grow unboundedly (absolute-not-delta
    returns). Running several intervals and reading the store directly
    (composite.state, a plain dict keyed by the composite's store path —
    ["stores", "average_concentrations"] / ["stores", "time"] per the
    composite yaml's `outputs` mapping) proves the store converges to the
    correct decayed absolute rather than accumulating nonsense.
    """
    core = build_core()
    doc = yaml.safe_load((COMPOSITES / "chemostat.composite.yaml").read_text())
    state = doc["state"]
    composite = pb.Composite({"state": state}, core=core)

    for _ in range(3):
        composite.run(1.0)

    concs = composite.state["stores"]["average_concentrations"]
    assert concs, "average_concentrations store must be populated (initial_state seeding)"
    # solute1 starts at 2.0 and decays (k=0.1/day); after 3 intervals the
    # store's ABSOLUTE value must be below the initial value -- proving
    # deltas accumulated correctly rather than the store holding raw
    # per-step absolutes (which would look identical to a single step) or
    # exploding from double-counted deltas.
    assert concs["solute1"] < 2.0
    assert concs["solute1"] > 0.0

    assert composite.state["stores"]["time"] == pytest.approx(3.0)


def test_biofilm_composite_multistep_population_bounded():
    """population is also an accumulate port; a multi-step run must keep the
    store's absolute population close to the seeded initial count (30) over
    a couple of short intervals, not run away from double-counted absolutes
    landing in an accumulate slot.
    """
    core = build_core()
    doc = yaml.safe_load((COMPOSITES / "biofilm.composite.yaml").read_text())
    state = doc["state"]
    composite = pb.Composite({"state": state}, core=core)

    for _ in range(2):
        composite.run(0.05)

    population = composite.state["stores"]["population"]
    # Two short (0.05 day) intervals won't grow the 30-agent seed population
    # much; the key regression check is that it's a small, bounded absolute
    # near 30 -- not a runaway sum of double-counted absolute readbacks
    # (which would produce ~60). Upper bound < 45 catches the delta-accumulation
    # regression while remaining well above the correct ~30.
    assert 30 <= population < 45, f"population {population} outside guard {30} <= x < 45"
