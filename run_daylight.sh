#!/usr/bin/env bash
set -euo pipefail

# Run daylight branch only
export ENABLE_DAYLIGHT=1
export ENABLE_THERMAL=0

exec python3 dual_cam_pipeline.py
