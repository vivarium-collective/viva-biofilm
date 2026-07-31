#!/usr/bin/env bash
# Build the read-only viva-biofilm dashboard snapshot — a self-contained static
# SPA bundle (all investigations + studies + assets) that anyone can browse
# with no server. Built by vivarium-workbench-publish; the same build is used
# locally (to practice/preview) and by
# .github/workflows/publish-dashboard.yml (to publish to gh-pages:dashboard/).
#
# Usage:
#   .venv/bin/python -m vivarium_workbench ... (or, after `source .venv/bin/activate`)
#   scripts/publish_dashboard.sh [OUT_DIR]
#     OUT_DIR defaults to reports/published/dashboard
#
# Preview locally:
#   scripts/publish_dashboard.sh /tmp/dash
#   python -m http.server -d /tmp/dash 8080   # -> http://localhost:8080/
#
# Notes:
#   * --base-path /viva-biofilm/dashboard rewrites root-absolute URLs for
#     GitHub Pages' project subpath (served at
#     <user>.github.io/viva-biofilm/dashboard/).
#   * bigraph-loom source maps (~8MB, half the bundle) are stripped — a
#     read-only viewer never needs them.
#   * Needs `vivarium-workbench-publish` on PATH (via the activated .venv:
#     .venv/bin/vivarium-workbench-publish).
#   * The Rust extension (viva_biofilm.biofilm_core) MUST already be built
#     (`maturin develop -m crates/biofilm-py/Cargo.toml`) before this runs —
#     composite discovery imports it during build_core() registration.
set -euo pipefail

WS_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="${1:-$WS_ROOT/reports/published/dashboard}"
BASE_PATH="/viva-biofilm/dashboard"
INTERACTIVE_URL="https://github.com/vivarium-collective/viva-biofilm"

rm -rf "$OUT"
# The workspace's own package must be importable for build_core() registration.
PYTHONPATH="$WS_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
  vivarium-workbench-publish \
    --workspace "$WS_ROOT" \
    --out "$OUT" \
    --base-path "$BASE_PATH" \
    --interactive-url "$INTERACTIVE_URL"

# Strip bigraph-loom source maps — not needed for the read-only viewer.
find "$OUT" -name '*.map' -delete
touch "$OUT/.nojekyll"

echo "built read-only dashboard bundle at $OUT ($(du -sh "$OUT" | cut -f1))"
