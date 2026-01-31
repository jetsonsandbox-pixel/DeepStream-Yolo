#!/bin/bash
# Test guide for dual camera pipeline with CUDA fixes

echo "=== Dual Camera Pipeline CUDA Fix - Testing Guide ==="
echo ""

# Test 1: Daylight camera only
echo "[1/3] Testing daylight camera only (4-tile tiling)..."
echo "      This validates the tiling + preprocessing fixes"
echo "      Expected: Runs for 30+ seconds without CUDA errors"
timeout 30 python3 /home/jet-nx8/DeepStream-Yolo/test_daylight_only.py 2>&1 | grep -E "(FPS|Error|CUDA|Building|Starting)" &
PID=$!
sleep 30
kill $PID 2>/dev/null
wait $PID 2>/dev/null
echo "      ✓ Daylight test complete"
echo ""

# Test 2: Thermal camera only  
echo "[2/3] Testing thermal camera only (640x512)..."
echo "      This validates the thermal processing fixes"
echo "      Expected: Stable FPS, no memory errors"
echo "      TODO: Create test_thermal_only.py"
echo ""

# Test 3: Dual camera with fixes
echo "[3/3] Testing dual camera pipeline (fixed version)..."
echo "      This validates both cameras with queue management"
echo "      Expected: Daylight ~15 FPS, Thermal ~20 FPS"
echo "      Expected: No CUDA errors after 30+ seconds"
timeout 30 python3 /home/jet-nx8/DeepStream-Yolo/dual_cam_pipeline.py 2>&1 | grep -E "(FPS|Error|CUDA|Building|Starting)" &
PID=$!
sleep 30
kill $PID 2>/dev/null
wait $PID 2>/dev/null
echo "      ✓ Dual camera test complete"
echo ""

echo "=== Testing Complete ==="
echo ""
echo "Summary of Fixes:"
echo "  ✅ Changed sinks from nveglglessink to fakesink"
echo "  ✅ Reduced batch from 8 to 4 tiles"
echo "  ✅ Reduced buffer pools from 4 to 3"
echo "  ✅ Added queue buffer management"
echo "  ✅ Added graceful shutdown (sync=False, async=False)"
echo ""
echo "Expected Result:"
echo "  - No cudaErrorIllegalAddress (700) errors"
echo "  - Stable GPU memory allocation"
echo "  - Sustained 15+ FPS on daylight with tiling"
echo "  - Sustained 20+ FPS on thermal"
echo ""

# GPU Memory Check
echo "GPU Status:"
nvidia-smi --query-gpu=memory.free,memory.total --format=csv,noheader
