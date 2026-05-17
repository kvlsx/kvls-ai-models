#!/usr/bin/env bash
# Fetches 100 deterministic 448×448 CC0 photos from Picsum into
# `calibration/` for use by scripts/quantize_static.py. Re-running is
# idempotent: existing files are kept.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p calibration
for i in $(seq 1 100); do
    name="calibration/img_$(printf '%03d' $i).jpg"
    [[ -f "$name" ]] && continue
    curl -sL -o "$name" "https://picsum.photos/seed/kvls$i/448/448" &
    (( i % 10 == 0 )) && wait
done
wait
echo "calibration set: $(ls calibration/*.jpg | wc -l) images"
