#!/usr/bin/env bash
set -euo pipefail

# Run thermal branch only
export ENABLE_DAYLIGHT=0
export ENABLE_THERMAL=1

exec python3 dual_cam_pipeline.py
