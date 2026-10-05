#!/bin/zsh
# Run one NRT cycle, then publish the new snapshot (triggers the Pages deploy).
cd "$(dirname "$0")/.." || exit 1
.venv/bin/python src/update_2026_nrt.py || exit 1
git add -f data/processed/gujarat_2026_inference.csv data/processed/gujarat_2026_inference_freshness.json
git diff --cached --quiet && exit 0
git commit -qm "Auto-refresh 2026 inference snapshot" && git push -q
