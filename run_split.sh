#!/usr/bin/env bash
set -euo pipefail

# Launch daylight and thermal in separate processes
export ENABLE_DAYLIGHT=1
export ENABLE_THERMAL=0
python3 dual_cam_pipeline.py &
DAY_PID=$!

export ENABLE_DAYLIGHT=0
export ENABLE_THERMAL=1
python3 dual_cam_pipeline.py &
THERM_PID=$!

cleanup() {
	kill $DAY_PID $THERM_PID 2>/dev/null || true
	wait $DAY_PID $THERM_PID 2>/dev/null || true
}

trap cleanup INT TERM EXIT

wait $DAY_PID $THERM_PID
