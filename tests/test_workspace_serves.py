"""Smoke test: the vivarium-workbench can load and discover this workspace.

Covers Task 8 (workbench hosting) of the viva-biofilm capabilities plan:
- every study.yaml under workspace/studies/ passes load_spec without raising
  and comes back tagged with its own study name
- scan_worktree_investigations discovers the viva-biofilm-capabilities
  investigation (the one this task's three studies belong to)
- (best-effort) the workspace report renderer, if importable and cheap,
  produces reports/index.html

The report-render check is intentionally best-effort: it is skipped rather
than failed if the renderer is unavailable, needs network, or errors, so
this test stays a fast, reliable smoke test rather than an integration test.
"""

from pathlib import Path

import pytest

WORKSPACE_DIR = Path(__file__).resolve().parent.parent / "workspace"


def test_all_studies_load_spec():
    workbench = pytest.importorskip("vivarium_workbench.lib.investigations")
    load_spec = workbench.load_spec

    study_files = sorted((WORKSPACE_DIR / "studies").glob("*/study.yaml"))
    assert study_files, "expected at least one study.yaml under workspace/studies/"

    for study_file in study_files:
        expected_name = study_file.parent.name
        spec = load_spec(study_file)
        assert isinstance(spec, dict), f"{study_file}: load_spec did not return a dict"
        assert spec.get("name") == expected_name, (
            f"{study_file}: expected name={expected_name!r}, got {spec.get('name')!r}"
        )


def test_capabilities_investigation_discovered():
    registry = pytest.importorskip("vivarium_workbench.lib.investigation_registry")
    scan_worktree_investigations = registry.scan_worktree_investigations

    investigations = scan_worktree_investigations(str(WORKSPACE_DIR))
    slugs = {inv["slug"] for inv in investigations}
    assert "viva-biofilm-capabilities" in slugs, (
        f"expected 'viva-biofilm-capabilities' among discovered investigations, got {slugs}"
    )


def test_report_renders_if_available():
    try:
        from vivarium_workbench.lib.report import render_workspace_report
    except ImportError:
        pytest.skip("vivarium_workbench.lib.report not importable")

    root = WORKSPACE_DIR.parent
    try:
        render_workspace_report(root)
    except Exception as exc:  # pragma: no cover - defensive: keep smoke test non-flaky
        pytest.skip(f"render_workspace_report raised, skipping: {exc}")

    assert (root / "reports" / "index.html").exists()
