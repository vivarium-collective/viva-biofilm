"""build_core() — bigraph-schema core with viva-biofilm's own processes registered.

Editable-installed workspace processes are not auto-discovered by
``process_bigraph.allocate_core()``, so composites addressing them as
``local:<ProcessName>`` need them registered explicitly here.
"""
import process_bigraph as pb

from viva_biofilm.processes.biofilm_process import BiofilmProcess
from viva_biofilm.processes.chemostat_process import ChemostatProcess


def build_core():
    core = pb.allocate_core()
    core.register_link("BiofilmProcess", BiofilmProcess)
    core.register_link("ChemostatProcess", ChemostatProcess)
    return core
