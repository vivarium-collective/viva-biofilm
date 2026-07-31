"""build_core() must register BiofilmProcess and ChemostatProcess.

Composites address processes as `local:<ProcessName>`, which resolves via
`core.link_registry`. Empirically (this repo's installed bigraph-schema,
verified via a python REPL against `process_bigraph.allocate_core()`),
`link_registry` is a plain `dict` — subscript access (`core.link_registry[name]`)
and `.get(name)` both work, but `.access(...)` does NOT exist on it
(`AttributeError: 'dict' object has no attribute 'access'`). So this test
asserts via subscript/`in`/`.get`, not `.access(...)`.
"""

from viva_biofilm.core import build_core
from viva_biofilm.processes.biofilm_process import BiofilmProcess
from viva_biofilm.processes.chemostat_process import ChemostatProcess


def test_build_core_registers_biofilm_and_chemostat_processes():
    core = build_core()

    for name, cls in (
        ("BiofilmProcess", BiofilmProcess),
        ("ChemostatProcess", ChemostatProcess),
    ):
        assert name in core.link_registry, (
            f"build_core() did not register {name!r}; composites addressing "
            f"`local:{name}` will fail to resolve with 'no link found at address'."
        )
        assert core.link_registry[name] is cls, (
            f"core.link_registry[{name!r}] does not resolve to {cls!r}"
        )
        assert core.link_registry.get(name) is cls


def test_build_core_is_idempotent():
    # Calling build_core() twice (e.g. once per test module) must not raise.
    core1 = build_core()
    core2 = build_core()
    assert "BiofilmProcess" in core1.link_registry
    assert "BiofilmProcess" in core2.link_registry
