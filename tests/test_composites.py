import pathlib
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
