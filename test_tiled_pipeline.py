#!/usr/bin/env python3
"""
Quick test of the tiled inference pipeline
Tests each component and then runs end-to-end
"""

import sys
import cv2
import numpy as np
from tiled_yolo_inference import (
    TileConfig, 
    TileExtractor, 
    TensorRTInference,
    DetectionMerger,
    TiledYOLOInference
)

def test_components():
    """Test individual components"""
    print("=" * 60)
    print("COMPONENT TESTS")
    print("=" * 60)
    
    # Test 1: Tile Configuration
    print("\n1. Testing Tile Configuration...")
    config = TileConfig()
    assert config.total_tiles == 8, f"Expected 8 tiles, got {config.total_tiles}"
    print(f"   ✓ Grid: {config.tiles_x}x{config.tiles_y} = {config.total_tiles} tiles")
    
    # Test 2: Tile Extraction
    print("\n2. Testing Tile Extraction...")
    extractor = TileExtractor(config)
    test_frame = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
    tiles = extractor.extract_tiles(test_frame)
    assert len(tiles) == 8, f"Expected 8 tiles, got {len(tiles)}"
    assert tiles[0][0].shape == (640, 640, 3), f"Wrong tile shape: {tiles[0][0].shape}"
    print(f"   ✓ Extracted {len(tiles)} tiles, shape: {tiles[0][0].shape}")
    
    # Test 3: TensorRT Engine
    print("\n3. Testing TensorRT Engine...")
    engine = TensorRTInference("model_b8_gpu0_fp32.engine")
    dummy_tile = np.zeros((640, 640, 3), dtype=np.uint8)
    output = engine.infer(dummy_tile)
    assert output.shape == (8400, 6), f"Wrong output shape: {output.shape}"
    print(f"   ✓ Inference successful, output shape: {output.shape}")
    
    # Test 4: YOLO Output Parsing
    print("\n4. Testing YOLO Output Parsing...")
    # Create fake detections
    fake_output = np.zeros((8400, 6), dtype=np.float32)
    fake_output[0] = [100, 100, 200, 200, 0.9, 0]  # High confidence detection
    fake_output[1] = [300, 300, 400, 400, 0.1, 1]  # Low confidence (should be filtered)
    
    from tiled_yolo_inference import TiledYOLOInference
    pipeline = TiledYOLOInference("model_b8_gpu0_fp32.engine")
    parsed = pipeline.parse_yolo_output(fake_output, conf_threshold=0.25)
    assert len(parsed) == 1, f"Expected 1 detection, got {len(parsed)}"
    assert abs(parsed[0][4] - 0.9) < 0.01, f"Wrong confidence: {parsed[0][4]}"  # Float comparison
    print(f"   ✓ Parsed {len(parsed)} detection from fake output")
    
    # Test 5: Detection Merging
    print("\n5. Testing Detection Merger...")
    merger = DetectionMerger()
    tile_dets = [
        (np.array([[100, 100, 200, 200, 0.9, 0]]), 0, 0),
        (np.array([[150, 150, 250, 250, 0.8, 0]]), 100, 100),  # Overlapping
    ]
    merged = merger.merge_tiles(tile_dets, config)
    print(f"   ✓ Merged {len(merged)} detections from 2 tiles")
    
    print("\n" + "=" * 60)
    print("✅ ALL COMPONENT TESTS PASSED!")
    print("=" * 60)
    return True

def test_video_inference(video_path: str, max_frames: int = 30):
    """Test on actual video"""
    print("\n" + "=" * 60)
    print("VIDEO INFERENCE TEST")
    print("=" * 60)
    
    # Check if video exists
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"   ✗ Cannot open video: {video_path}")
        return False
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"\nVideo Info:")
    print(f"   Resolution: {width}x{height}")
    print(f"   FPS: {fps}")
    print(f"   Total frames: {total_frames}")
    print(f"   Testing on: {max_frames} frames")
    
    # Initialize pipeline
    print(f"\nInitializing tiled inference pipeline...")
    pipeline = TiledYOLOInference("model_b8_gpu0_fp32.engine")
    
    # Process frames
    print(f"\nProcessing frames...")
    frame_count = 0
    total_detections = 0
    inference_times = []
    
    import time
    
    while frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Resize if needed
        if frame.shape[0] != 1080 or frame.shape[1] != 1920:
            frame = cv2.resize(frame, (1920, 1080))
        
        # Run tiled inference
        start_time = time.time()
        detections = pipeline.process_frame(frame)
        elapsed = time.time() - start_time
        
        inference_times.append(elapsed)
        total_detections += len(detections)
        frame_count += 1
        
        # Print progress every 10 frames
        if frame_count % 10 == 0:
            avg_time = np.mean(inference_times[-10:])
            avg_fps = 1.0 / avg_time if avg_time > 0 else 0
            print(f"   Frame {frame_count}/{max_frames}: {len(detections)} detections, "
                  f"{elapsed*1000:.1f}ms ({avg_fps:.1f} FPS)")
    
    cap.release()
    
    # Summary
    print(f"\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Frames processed: {frame_count}")
    print(f"Total detections: {total_detections}")
    print(f"Avg detections/frame: {total_detections/frame_count:.1f}")
    print(f"Avg inference time: {np.mean(inference_times)*1000:.1f}ms")
    print(f"Avg FPS: {1.0/np.mean(inference_times):.1f}")
    print(f"Min/Max FPS: {1.0/np.max(inference_times):.1f} / {1.0/np.min(inference_times):.1f}")
    print("=" * 60)
    
    return True

if __name__ == "__main__":
    try:
        # Run component tests
        if not test_components():
            print("\n❌ Component tests failed!")
            sys.exit(1)
        
        # Run video inference test
        video_path = "/home/jet-nx8/Sandbox/test-data/iphone_day_fpv_kushi_shogla_people_08_11_2025.MOV"
        if not test_video_inference(video_path, max_frames=30):
            print("\n❌ Video inference test failed!")
            sys.exit(1)
        
        print("\n✅ ALL TESTS PASSED!")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
