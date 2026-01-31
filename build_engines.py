#!/usr/bin/env python3
"""
Build TensorRT Engine from ONNX Model
Converts YOLO11n ONNX → TensorRT engine for inference
"""

import os
import tensorrt as trt
import numpy as np

def build_engine(onnx_file, engine_file, fp16=False, batch_size=1):
    """Build TensorRT engine from ONNX file"""
    
    print(f"Building TensorRT engine...")
    print(f"  Input ONNX: {onnx_file}")
    print(f"  Output Engine: {engine_file}")
    print(f"  Batch Size: {batch_size}")
    print(f"  Precision: {'FP16' if fp16 else 'FP32'}")
    
    # Create logger
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    
    # Create builder
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)
    
    # Parse ONNX
    print("\nParsing ONNX file...")
    with open(onnx_file, 'rb') as f:
        if not parser.parse(f.read()):
            print("ERROR: Failed to parse ONNX file")
            for error in range(parser.num_errors):
                print(f"  {parser.get_error(error)}")
            return False
    
    print("✓ ONNX parsed successfully")
    
    # Create config
    print("\nConfiguring TensorRT...")
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1GB workspace
    
    # Enable FP16
    if fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        print("✓ FP16 precision enabled")
    
    # Set batch size
    profile = builder.create_optimization_profile()
    profile.set_shape('images', (1, 3, 640, 640), (batch_size, 3, 640, 640), (batch_size, 3, 640, 640))
    config.add_optimization_profile(profile)
    
    # Build engine
    print(f"\nBuilding engine (this may take 1-2 minutes)...")
    engine = builder.build_serialized_network(network, config)
    
    if engine is None:
        print("ERROR: Failed to build engine")
        return False
    
    # Save engine
    print(f"\nSaving engine to {engine_file}...")
    with open(engine_file, 'wb') as f:
        f.write(engine)
    
    # Verify file exists
    if os.path.exists(engine_file):
        size_mb = os.path.getsize(engine_file) / (1024 * 1024)
        print(f"✓ Engine saved successfully ({size_mb:.1f}MB)")
        return True
    else:
        print("ERROR: Failed to save engine")
        return False

if __name__ == "__main__":
    import sys
    
    print("=== TensorRT Engine Builder ===\n")
    
    # Check if TensorRT is available
    try:
        import tensorrt as trt
        print(f"TensorRT version: {trt.__version__}\n")
    except ImportError:
        print("ERROR: TensorRT not installed")
        sys.exit(1)
    
    # Build engines
    onnx_file = "yolo11n-2025-11-07_v4-0a.onnx"
    
    if not os.path.exists(onnx_file):
        print(f"ERROR: {onnx_file} not found")
        sys.exit(1)
    
    # Build batch=1 FP32 engine for standard inference
    print("Building model_b1_gpu0_fp32.engine (batch=1, FP32)...")
    success1 = build_engine(onnx_file, "model_b1_gpu0_fp32.engine", fp16=False, batch_size=1)
    
    print("\n" + "="*50 + "\n")
    
    # Build batch=4 FP16 engine for tiling
    print("Building model_b4_gpu0_fp16.engine (batch=4, FP16)...")
    success2 = build_engine(onnx_file, "model_b4_gpu0_fp16.engine", fp16=True, batch_size=4)
    
    print("\n" + "="*50)
    if success1 and success2:
        print("\n✓ All engines built successfully!")
        sys.exit(0)
    else:
        print("\n✗ Engine build failed")
        sys.exit(1)
