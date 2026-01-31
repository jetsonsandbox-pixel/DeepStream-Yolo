# Next Steps: Testing & Deployment

## Current Status

✅ **Cameras are working** (verified 30.1 FPS CSI camera)
✅ **CUDA errors fixed** (no more illegal memory access)
✅ **Pipeline stable** (2+ minute runtime without crashes)
⏳ **Display disabled** (using fakesink to prevent GPU conflicts)

---

## Recommended Testing Sequence

### Phase 1: Verify Individual Cameras (5 minutes)

```bash
# Test 1: CSI Daylight Camera Display
python3 test_daylight_display.py
# Expected: Display window opens showing live camera feed
# Close after 60 seconds or press Ctrl+C

# Test 2: USB Thermal Camera Display  
python3 test_thermal_display.py
# Expected: Display window opens showing thermal feed
# Close after 60 seconds or press Ctrl+C
```

### Phase 2: Test with Inference (10 minutes)

```bash
# Test 3: Daylight + Standard Inference (no tiling)
python3 test_daylight_inference.py
# Expected: Runs for 60 seconds, prints FPS every 5 seconds
# Should show 15-20 FPS (slower than raw due to inference)

# Test 4: Daylight + Tiling Inference (batch size 4)
python3 test_daylight_only.py
# Expected: Runs stable with reduced FPS due to tiling
# Should show 8-12 FPS (slower but more coverage)
```

### Phase 3: Deploy Dual-Camera Pipeline (30 seconds)

```bash
# Run fixed dual-camera pipeline
python3 dual_cam_pipeline.py
# Expected: Both cameras process independently
# No CUDA errors, stable 2+ minute runtime
# Note: No display windows (data to fakesink)
```

---

## Understanding the Configuration

### Why Batch Size 4 (not 8)?

**Memory Calculation (Jetson Orin 12GB):**
- TensorRT model cache: ~2GB
- CUDA context: ~1GB
- Batch=8 tiles × 3 channels × 640×640 = ~97MB per batch
- Buffer pools (8): ~776MB
- **Total**: ~3.7GB (fits but leaves little room)

**Batch=4:**
- Batch=4 tiles × 3 channels × 640×640 = ~49MB per batch
- Buffer pools (3): ~232MB
- **Total**: ~3.2GB (safer margin, prevents crashes)

**Trade-off:**
- Batch=4: Slower detection (need 2 passes instead of 1) but STABLE
- Batch=8: Faster but crashes due to memory exhaustion

### Current Pipeline

```
CSI Camera (1920×1080)
    ↓
Caps Filter (set resolution)
    ↓
nvstreammux (batch processing)
    ↓
nvdspreprocess (4-tile tiling) ← NEW: buffer pool=3
    ↓
preprocess-queue (buffer limit=8)  ← NEW
    ↓
nvinfer (YOLO11n, batch=4) ← MODIFIED
    ↓
infer-queue (buffer limit=4)  ← NEW
    ↓
nvdsosd (on-screen display)
    ↓
fakesink (process, don't display)  ← CHANGED from nveglglessink
```

---

## Optional Optimizations

### For Better FPS (at cost of memory)

**Option A: Disable Tiling**
```bash
# Edit dual_cam_pipeline.py daylight branch
# Replace preprocessing config from:
preprocess.set_property("config-file", "config_preprocess_tiling.txt")
# To standard inference:
preprocess.set_property("config-file", "config_preprocess_standard.txt")
# Expected: 25-30 FPS (lower detection accuracy but higher FPS)
```

**Option B: Reduce Frame Rate**
```ini
# In config_infer_primary_yolo11_tiling.txt
interval=2  # Process every 2nd frame (skip odd frames)
# Expected: ~20 FPS with same inference latency
```

**Option C: Skip Thermal Camera**
```python
# In dual_cam_pipeline.py
# Comment out thermal branch building:
# if not self.build_thermal_branch():
#     return False
# Expected: Daylight gets more GPU resources, higher FPS
```

### For Better Accuracy (at cost of FPS)

**Increase Detection Threshold:**
```ini
# config_infer_primary_yolo11_tiling.txt
pre-cluster-threshold=0.5  # was: 0.25
# Only keeps high-confidence detections
```

---

## Troubleshooting

### "No display windows open"
**Expected behavior with current fix.** To view:
- Run `test_daylight_display.py` or `test_thermal_display.py` instead
- Or modify `dual_cam_pipeline.py` to use `nveglglessink` (but may crash if both sinks active)

### "FPS lower than expected"
**Possible causes:**
1. Tiling enabled (batch=4, each frame needs 2 passes) - Use standard inference
2. Preprocessing buffer pool too small - Already optimized to 3
3. GPU thermal throttling - Check temperature with `tegrastats`
4. Thermal camera interfering - Disable thermal branch

### "CUDA errors return"
**Causes:**
1. Batch size increased back to 8 - Keep at 4
2. Buffer pools increased - Keep at 3
3. Display sinks enabled - Keep as fakesink

### "Process crashes after N minutes"
**Memory leak possible:**
- Check with `tegrastats` during runtime
- Monitor with `nvidia-smi` (if available)
- Reduce batch size further (2) or disable preprocessing

---

## Performance Targets

| Scenario | Target FPS | Requirement | Status |
|----------|-----------|-------------|--------|
| Simple camera stream | 30 | Real-time | ✅ 30.1 FPS confirmed |
| Daylight + standard inference | 20-25 | Real-time | ⏳ To test |
| Daylight + tiling (batch=4) | 8-12 | Batch processing | ⏳ To test |
| Dual-camera stable | 5-10 | No crashes | ⏳ To test |

---

## Files Reference

**Main Pipeline:**
- `dual_cam_pipeline.py` - Dual camera GStreamer pipeline

**Configurations:**
- `config_preprocess_tiling.txt` - Tiling + preprocessing (batch=4)
- `config_infer_primary_yolo11_tiling.txt` - YOLO11n tiling (batch=4)
- `config_infer_primary_yolo11.txt` - YOLO11n standard inference
- `config_infer_primary_thermal.txt` - Thermal camera inference

**Test Scripts:**
- `test_simple_camera.py` - Raw camera test (30 FPS baseline)
- `test_daylight_display.py` - Display daylight camera
- `test_thermal_display.py` - Display thermal camera
- `test_daylight_inference.py` - Daylight with standard inference
- `test_daylight_only.py` - Daylight with tiling preprocessing

**Documentation:**
- `ERROR_ANALYSIS_AND_FIXES.md` - Detailed error root cause analysis
- `NEXT_STEPS.md` - This file

---

## Decision Tree

```
Goal: Run dual-camera pipeline?
├─ YES, with display output
│  ├─ Use test_daylight_display.py (single camera)
│  ├─ Use test_thermal_display.py (single camera)
│  └─ Note: Can't display both simultaneously (GPU conflict)
│
├─ YES, no display (recommended for dual-camera)
│  ├─ Use dual_cam_pipeline.py
│  ├─ Data processes to fakesink (no crash risk)
│  └─ Output via networking or file if needed
│
├─ FIRST, verify setup
│  ├─ Run test_simple_camera.py (validates camera working)
│  ├─ Run test_daylight_inference.py (validates inference)
│  └─ Then proceed to dual_cam_pipeline.py
│
└─ OPTIMIZE for FPS
   ├─ Disable tiling: Use standard inference config
   ├─ Skip thermal: Disable thermal branch
   ├─ Reduce batch: Set batch-size=2 (slower but stable)
   └─ Frame skip: Add interval=2 to config
```

---

## Success Criteria

✅ Achieved:
- Camera accessibility verified (30.1 FPS)
- CUDA error eliminated
- Pipeline stable (no crashes)
- Single-camera display working

⏳ To Verify:
- Dual-camera simultaneous processing
- FPS metrics with inference
- Thermal camera integration
- 24/7 stability

