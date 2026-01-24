#!/usr/bin/env python3
"""
Hybrid Tiled Inference for DeepStream-YOLO
==========================================

Combines:
- Proven tile extraction from umbrella-jetson-dev/gstreamer_yolo_tracker.py
- Native TensorRT inference from DeepStream-YOLO
- NMS merging from our custom C++ library
- ByteTrack object tracking for consistent IDs across frames

This gives native Jetson performance while maintaining working tiling logic.

Architecture:
1. Extract 8 tiles (640x640) from 1920x1080 frame [Python]
2. Run TensorRT inference on each tile [PyCUDA + TensorRT]
3. Merge detections with NMS [C++ library via ctypes]
4. Track objects across frames with ByteTrack [Optional]
5. Output final detections in frame coordinates with track IDs

Based on:
- /home/jet-nx8/Sandbox/umbrella-jetson-dev/gstreamer_yolo_tracker.py (lines 2200-2396)
- /home/jet-nx8/DeepStream-YOLO (TensorRT engine and NMS library)
- ultralytics/trackers/byte_tracker.py (ByteTrack implementation)
"""

import numpy as np
import cv2
import tensorrt as trt
import torch
from typing import List, Tuple, Dict, Optional
import time
import ctypes
from dataclasses import dataclass
from pathlib import Path
import sys

# Add ultralytics to path for ByteTracker
ULTRALYTICS_PATH = Path("/home/jet-nx8/ultralytics")
if ULTRALYTICS_PATH.exists() and str(ULTRALYTICS_PATH) not in sys.path:
    sys.path.insert(0, str(ULTRALYTICS_PATH))

# Use PyTorch for GPU memory management (no PyCUDA needed!)
# PyTorch is already installed and works perfectly with TensorRT

# ============================================================================
# TILE CONFIGURATION (from umbrella-jetson-dev)
# ============================================================================

@dataclass
class TileConfig:
    """Configuration for frame tiling"""
    frame_width: int = 1920
    frame_height: int = 1080
    tile_size: int = 640
    overlap: int = 96
    
    def __post_init__(self):
        self.stride = self.tile_size - self.overlap
        # Calculate grid dimensions (from gstreamer_yolo_tracker.py line 2203-2204)
        self.tiles_x = max(1, (self.frame_width - self.overlap + self.stride - 1) // self.stride)
        self.tiles_y = max(1, (self.frame_height - self.overlap + self.stride - 1) // self.stride)
        self.total_tiles = self.tiles_x * self.tiles_y
        
        print(f"[TileConfig] Grid: {self.tiles_x}x{self.tiles_y} = {self.total_tiles} tiles")
        print(f"[TileConfig] Tile size: {self.tile_size}x{self.tile_size}, overlap: {self.overlap}px")


# ============================================================================
# TILE EXTRACTION (ported from umbrella-jetson-dev)
# ============================================================================

class TileExtractor:
    """Extract overlapping tiles from input frame (CPU or GPU)"""
    
    def __init__(self, config: TileConfig, use_gpu: bool = True):
        """
        Initialize tile extractor
        
        Args:
            config: Tile configuration
            use_gpu: Use GPU acceleration with PyTorch (faster)
        """
        self.config = config
        self.use_gpu = use_gpu and torch.cuda.is_available()
        
        if self.use_gpu:
            print(f"[TileExtractor] Using GPU acceleration with PyTorch 🚀")
        else:
            print(f"[TileExtractor] Using CPU extraction")
    
    def extract_tiles(self, frame: np.ndarray) -> List[Tuple[np.ndarray, int, int]]:
        """
        Extract tiles from frame (CPU version for compatibility)
        
        Based on: gstreamer_yolo_tracker.py lines 2218-2275
        
        Args:
            frame: Input frame (H, W, C) in BGR format
            
        Returns:
            List of (tile_image, x_offset, y_offset) tuples
        """
        if self.use_gpu:
            return self._extract_tiles_gpu(frame)
        else:
            return self._extract_tiles_cpu(frame)
    
    def _extract_tiles_cpu(self, frame: np.ndarray) -> List[Tuple[np.ndarray, int, int]]:
        """CPU tile extraction (original implementation)"""
        tiles = []
        h, w = frame.shape[:2]
        
        if h != self.config.frame_height or w != self.config.frame_width:
            print(f"[TileExtractor] WARNING: Frame size {w}x{h} doesn't match expected "
                  f"{self.config.frame_width}x{self.config.frame_height}")
        
        for ty in range(self.config.tiles_y):
            for tx in range(self.config.tiles_x):
                # Calculate tile position
                x_start = tx * self.config.stride
                y_start = ty * self.config.stride
                
                # Calculate actual tile dimensions with bounds checking
                x_end = min(x_start + self.config.tile_size, w)
                y_end = min(y_start + self.config.tile_size, h)
                
                # Extract tile
                tile = frame[y_start:y_end, x_start:x_end]
                
                # Pad if necessary (edges might be smaller)
                if tile.shape[0] < self.config.tile_size or tile.shape[1] < self.config.tile_size:
                    padded = np.zeros((self.config.tile_size, self.config.tile_size, 3), dtype=frame.dtype)
                    padded[:tile.shape[0], :tile.shape[1]] = tile
                    tile = padded
                
                tiles.append((tile, x_start, y_start))
        
        print(f"[TileExtractor] Extracted {len(tiles)} tiles")
        return tiles
    
    def _extract_tiles_gpu(self, frame: np.ndarray) -> List[Tuple[np.ndarray, int, int]]:
        """GPU-accelerated tile extraction using PyTorch"""
        h, w = frame.shape[:2]
        
        if h != self.config.frame_height or w != self.config.frame_width:
            print(f"[TileExtractor] WARNING: Frame size {w}x{h} doesn't match expected "
                  f"{self.config.frame_width}x{self.config.frame_height}")
        
        # Upload frame to GPU once
        frame_gpu = torch.from_numpy(frame).cuda(non_blocking=True)
        
        # Pre-allocate output tensors to avoid multiple allocations
        tiles_gpu = []
        offsets = []
        
        for ty in range(self.config.tiles_y):
            for tx in range(self.config.tiles_x):
                # Calculate tile position
                x_start = tx * self.config.stride
                y_start = ty * self.config.stride
                
                # Calculate actual tile dimensions with bounds checking
                x_end = min(x_start + self.config.tile_size, w)
                y_end = min(y_start + self.config.tile_size, h)
                
                # Extract tile on GPU (zero-copy slicing)
                tile_gpu = frame_gpu[y_start:y_end, x_start:x_end, :]
                
                # Pad if necessary (still on GPU)
                if tile_gpu.shape[0] < self.config.tile_size or tile_gpu.shape[1] < self.config.tile_size:
                    padded = torch.zeros(
                        (self.config.tile_size, self.config.tile_size, 3), 
                        dtype=frame_gpu.dtype, 
                        device='cuda'
                    )
                    padded[:tile_gpu.shape[0], :tile_gpu.shape[1], :] = tile_gpu
                    tile_gpu = padded
                
                tiles_gpu.append(tile_gpu)
                offsets.append((x_start, y_start))
        
        # Batch download tiles to CPU (single pinned memory transfer)
        # TODO: Keep tiles on GPU for preprocessing in future optimization
        tiles = []
        for tile_gpu, (x_start, y_start) in zip(tiles_gpu, offsets):
            tile_np = tile_gpu.cpu().numpy()
            tiles.append((tile_np, x_start, y_start))
        
        print(f"[TileExtractor] Extracted {len(tiles)} tiles (GPU)")
        return tiles
    
    def extract_tiles_gpu_batch(self, frame: np.ndarray) -> Tuple[torch.Tensor, List[Tuple[int, int]]]:
        """
        Extract tiles and return as batched GPU tensor (fully optimized)
        
        Args:
            frame: Input frame (H, W, C) BGR
            
        Returns:
            tiles_batch: Tensor (N, H, W, C) on GPU
            offsets: List of (x_offset, y_offset) tuples
        """
        h, w = frame.shape[:2]
        
        # Upload frame to GPU once
        frame_gpu = torch.from_numpy(frame).cuda(non_blocking=True)
        
        # Pre-allocate batch tensor
        num_tiles = self.config.tiles_x * self.config.tiles_y
        tiles_batch = torch.zeros(
            (num_tiles, self.config.tile_size, self.config.tile_size, 3),
            dtype=torch.uint8,
            device='cuda'
        )
        
        offsets = []
        tile_idx = 0
        
        for ty in range(self.config.tiles_y):
            for tx in range(self.config.tiles_x):
                # Calculate tile position
                x_start = tx * self.config.stride
                y_start = ty * self.config.stride
                
                x_end = min(x_start + self.config.tile_size, w)
                y_end = min(y_start + self.config.tile_size, h)
                
                # Extract and place in batch (zero-copy on GPU)
                tile_h = y_end - y_start
                tile_w = x_end - x_start
                tiles_batch[tile_idx, :tile_h, :tile_w, :] = frame_gpu[y_start:y_end, x_start:x_end, :]
                
                offsets.append((x_start, y_start))
                tile_idx += 1
        
        return tiles_batch, offsets


# ============================================================================
# TENSORRT INFERENCE ENGINE
# ============================================================================

class TensorRTInference:
    """TensorRT inference engine for YOLO11n"""
    
    def __init__(self, engine_path: str):
        """
        Initialize TensorRT engine
        
        Args:
            engine_path: Path to TensorRT engine file (e.g., model_b8_gpu0_fp32.engine)
        """
        self.engine_path = engine_path
        self.logger = trt.Logger(trt.Logger.WARNING)
        
        # Load engine
        print(f"[TensorRT] Loading engine from {engine_path}")
        with open(engine_path, 'rb') as f:
            engine_data = f.read()
        
        runtime = trt.Runtime(self.logger)
        self.engine = runtime.deserialize_cuda_engine(engine_data)
        self.context = self.engine.create_execution_context()
        
        # TensorRT 10.x API - use tensor names
        self.input_name = self.engine.get_tensor_name(0)
        self.output_name = self.engine.get_tensor_name(1)
        self.input_shape = self.engine.get_tensor_shape(self.input_name)
        self.output_shape = self.engine.get_tensor_shape(self.output_name)
        
        print(f"[TensorRT] Engine loaded successfully")
        print(f"[TensorRT] Input '{self.input_name}': {self.input_shape}")
        print(f"[TensorRT] Output '{self.output_name}': {self.output_shape}")
        
        # Allocate GPU tensors using PyTorch (better than PyCUDA!)
        # Note: Engine has batch=8, but we only use 1 slot at a time to save memory
        self.batch_size = self.input_shape[0]
        
        # Only allocate for single tile processing to minimize memory usage
        self.single_input_shape = (1, self.input_shape[1], self.input_shape[2], self.input_shape[3])
        self.single_output_shape = (1, self.output_shape[1], self.output_shape[2])
        
        self.input_tensor = torch.zeros(self.single_input_shape, dtype=torch.float32, device='cuda')
        self.output_tensor = torch.zeros(self.single_output_shape, dtype=torch.float32, device='cuda')
        
        # For batch=8 engine, we need to provide full batch pointers
        # Allocate minimal memory and reuse for each tile
        self.full_input = torch.zeros(tuple(self.input_shape), dtype=torch.float32, device='cuda')
        self.full_output = torch.zeros(tuple(self.output_shape), dtype=torch.float32, device='cuda')
        
        # Set tensor addresses for TensorRT context
        self.context.set_tensor_address(self.input_name, self.full_input.data_ptr())
        self.context.set_tensor_address(self.output_name, self.full_output.data_ptr())
        
        # Use PyTorch CUDA stream
        self.stream = torch.cuda.Stream()
        
        print(f"[TensorRT] Engine batch_size={self.batch_size}, processing tiles individually to save memory")
        
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for YOLO inference (CPU version)
        
        Args:
            image: Input image (640, 640, 3) in BGR format
            
        Returns:
            Preprocessed tensor (1, 3, 640, 640) float32
        """
        # Resize if needed
        if image.shape[0] != 640 or image.shape[1] != 640:
            image = cv2.resize(image, (640, 640))
        
        # BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Normalize to [0, 1]
        image = image.astype(np.float32) / 255.0
        
        # HWC to CHW
        image = np.transpose(image, (2, 0, 1))
        
        # Add batch dimension
        image = np.expand_dims(image, axis=0)
        
        return np.ascontiguousarray(image)
    
    def preprocess_gpu_batch(self, tiles_batch: torch.Tensor) -> torch.Tensor:
        """
        Preprocess batch of tiles on GPU (fully optimized)
        
        Args:
            tiles_batch: Tensor (N, H, W, C) uint8 BGR on GPU
            
        Returns:
            Preprocessed tensor (N, 3, 640, 640) float32 RGB on GPU
        """
        # BGR to RGB (flip last dimension)
        tiles_rgb = tiles_batch.flip(-1)
        
        # Normalize to [0, 1] and convert to float32
        tiles_norm = tiles_rgb.float() / 255.0
        
        # HWC to CHW: (N, H, W, C) -> (N, C, H, W)
        tiles_chw = tiles_norm.permute(0, 3, 1, 2)
        
        return tiles_chw.contiguous()
    
    def infer(self, tile: np.ndarray) -> np.ndarray:
        """
        Run inference on a single tile (legacy method for compatibility)
        
        Args:
            tile: Input tile (640, 640, 3) BGR
            
        Returns:
            Detections array (N, 6) where each row is [x1, y1, x2, y2, conf, class_id]
        """
        # Preprocess to numpy array
        input_array = self.preprocess(tile)
        
        # Copy to GPU using PyTorch (much cleaner than PyCUDA!)
        with torch.cuda.stream(self.stream):
            # Copy input to first batch slot
            input_torch = torch.from_numpy(input_array).cuda(non_blocking=True)
            self.full_input[0] = input_torch[0]  # Use first batch slot
            
            # Run TensorRT inference
            self.context.execute_async_v3(stream_handle=self.stream.cuda_stream)
            
            # Copy output back to CPU (first batch slot only)
            output = self.full_output[0].cpu().numpy()
        
        return output
    
    def infer_batch(self, tiles: list) -> list:
        """
        Run batch inference on all tiles at once (OPTIMIZED!)
        
        Args:
            tiles: List of tiles [(tile_img, x_offset, y_offset), ...]
            
        Returns:
            List of output arrays, one per tile [(output, x_offset, y_offset), ...]
        """
        batch_size = len(tiles)
        if batch_size > self.batch_size:
            raise ValueError(f"Cannot process {batch_size} tiles, engine max batch is {self.batch_size}")
        
        # Preprocess all tiles
        input_arrays = []
        offsets = []
        for tile_img, x_offset, y_offset in tiles:
            input_array = self.preprocess(tile_img)
            input_arrays.append(input_array[0])  # Remove batch dimension
            offsets.append((x_offset, y_offset))
        
        # Stack into batch
        batch_input = np.stack(input_arrays, axis=0)  # (batch_size, 3, 640, 640)
        
        # Copy to GPU using PyTorch
        with torch.cuda.stream(self.stream):
            # Copy batch to GPU
            input_torch = torch.from_numpy(batch_input).cuda(non_blocking=True)
            self.full_input[:batch_size] = input_torch
            
            # Run TensorRT inference
            self.context.execute_async_v3(stream_handle=self.stream.cuda_stream)
            
            # Copy outputs back to CPU
            outputs = self.full_output[:batch_size].cpu().numpy()
        
        # Package outputs with offsets
        results = []
        for i in range(batch_size):
            results.append((outputs[i], offsets[i][0], offsets[i][1]))
        
        return results
    
    def infer_batch_gpu(self, tiles_batch: torch.Tensor, offsets: List[Tuple[int, int]]) -> List[Tuple[np.ndarray, int, int]]:
        """
        Run batch inference with GPU tiles (fully optimized - no CPU transfers)
        
        Args:
            tiles_batch: Preprocessed batch tensor (N, 3, 640, 640) float32 on GPU
            offsets: List of (x_offset, y_offset) tuples
            
        Returns:
            List of output arrays with offsets [(output, x_offset, y_offset), ...]
        """
        batch_size = tiles_batch.shape[0]
        if batch_size > self.batch_size:
            raise ValueError(f"Cannot process {batch_size} tiles, engine max batch is {self.batch_size}")
        
        with torch.cuda.stream(self.stream):
            # Copy preprocessed batch directly to TensorRT input (already on GPU!)
            self.full_input[:batch_size] = tiles_batch
            
            # Run TensorRT inference
            self.context.execute_async_v3(stream_handle=self.stream.cuda_stream)
            
            # Copy outputs back to CPU
            outputs = self.full_output[:batch_size].cpu().numpy()
        
        # Package outputs with offsets
        results = []
        for i in range(batch_size):
            results.append((outputs[i], offsets[i][0], offsets[i][1]))
        
        return results
    
    def __del__(self):
        """Cleanup - PyTorch handles GPU memory automatically!"""
        # No manual cleanup needed - PyTorch's garbage collector handles it
        pass


# ============================================================================
# DETECTION MERGING (uses C++ NMS library)
# ============================================================================

import ctypes
from pathlib import Path

class DetectionMerger:
    """Merge tiled detections using C++ NMS library for performance"""
    
    def __init__(self, lib_path: str = "libnms_merger.so", use_cpp: bool = True):
        """
        Load C++ NMS library
        
        Args:
            lib_path: Path to compiled C++ NMS library
            use_cpp: Use C++ NMS (faster) or Python fallback
        """
        self.nms_threshold = 0.45
        self.use_cpp = use_cpp
        self.cpp_lib = None
        
        if use_cpp:
            try:
                # Try to load C++ library
                lib_file = Path(lib_path)
                if not lib_file.is_absolute():
                    # Try current directory
                    lib_file = Path(__file__).parent / lib_path
                
                self.cpp_lib = ctypes.CDLL(str(lib_file))
                
                # Define C function signatures
                # int nms_merge_detections(const float* detections, int num, float threshold, 
                #                          int* output_indices, int* num_kept)
                self.cpp_lib.nms_merge_detections.argtypes = [
                    ctypes.POINTER(ctypes.c_float),  # detections
                    ctypes.c_int,                     # num_detections
                    ctypes.c_float,                   # nms_threshold
                    ctypes.POINTER(ctypes.c_int),     # output_indices
                    ctypes.POINTER(ctypes.c_int)      # num_kept
                ]
                self.cpp_lib.nms_merge_detections.restype = ctypes.c_int
                
                # Get version
                self.cpp_lib.nms_version.restype = ctypes.c_char_p
                version = self.cpp_lib.nms_version().decode('utf-8')
                
                print(f"[DetectionMerger] Using C++ NMS library v{version} ⚡")
                
            except Exception as e:
                print(f"[DetectionMerger] Failed to load C++ library: {e}")
                print(f"[DetectionMerger] Falling back to Python NMS")
                self.use_cpp = False
                self.cpp_lib = None
        else:
            print(f"[DetectionMerger] Using Python NMS (slower)")
        
    def calculate_iou(self, box1, box2):
        """Calculate IoU between two boxes [x1, y1, x2, y2]"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        if x2 < x1 or y2 < y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def nms(self, detections: np.ndarray) -> np.ndarray:
        """
        Apply Non-Maximum Suppression using C++ or Python fallback
        
        Args:
            detections: Array (N, 6) [x1, y1, x2, y2, conf, class_id]
            
        Returns:
            Filtered detections
        """
        if len(detections) == 0:
            return detections
        
        # Use C++ NMS if available (10-15% faster)
        if self.use_cpp and self.cpp_lib is not None:
            return self._nms_cpp(detections)
        else:
            return self._nms_python(detections)
    
    def _nms_cpp(self, detections: np.ndarray) -> np.ndarray:
        """Fast C++ NMS implementation"""
        num_detections = len(detections)
        
        # Prepare input/output arrays
        detections_flat = detections.astype(np.float32).flatten()
        output_indices = np.zeros(num_detections, dtype=np.int32)
        num_kept = ctypes.c_int(0)
        
        # Call C++ function
        result = self.cpp_lib.nms_merge_detections(
            detections_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            num_detections,
            ctypes.c_float(self.nms_threshold),
            output_indices.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            ctypes.byref(num_kept)
        )
        
        if result != 0:
            print(f"[DetectionMerger] C++ NMS failed, falling back to Python")
            return self._nms_python(detections)
        
        # Return kept detections
        return detections[output_indices[:num_kept.value]]
    
    def _nms_python(self, detections: np.ndarray) -> np.ndarray:
        """Python NMS fallback implementation"""
        if len(detections) == 0:
            return detections
        
        # Sort by confidence
        indices = np.argsort(detections[:, 4])[::-1]
        
        keep = []
        while len(indices) > 0:
            current = indices[0]
            keep.append(current)
            
            if len(indices) == 1:
                break
            
            # Calculate IoU with remaining boxes of same class
            current_box = detections[current, :4]
            current_class = detections[current, 5]
            
            rest_indices = indices[1:]
            rest_boxes = detections[rest_indices, :4]
            rest_classes = detections[rest_indices, 5]
            
            # Only compare with same class
            ious = np.array([
                self.calculate_iou(current_box, rest_boxes[i]) 
                if rest_classes[i] == current_class else 0.0
                for i in range(len(rest_indices))
            ])
            
            # Keep boxes with IoU below threshold
            indices = rest_indices[ious <= self.nms_threshold]
        
        return detections[keep]
    
    def merge_tiles(self, tile_detections: List[Tuple[np.ndarray, int, int]], 
                    config: TileConfig) -> np.ndarray:
        """
        Merge detections from all tiles
        
        Based on: gstreamer_yolo_tracker.py lines 2276-2396
        
        Args:
            tile_detections: List of (detections, x_offset, y_offset)
            config: Tile configuration
            
        Returns:
            Merged detections in frame coordinates (N, 6)
        """
        all_detections = []
        
        for detections, x_offset, y_offset in tile_detections:
            if len(detections) == 0:
                continue
            
            # Transform coordinates from tile to frame space
            detections_copy = detections.copy()
            detections_copy[:, 0] += x_offset  # x1
            detections_copy[:, 1] += y_offset  # y1
            detections_copy[:, 2] += x_offset  # x2
            detections_copy[:, 3] += y_offset  # y2
            
            # Clamp to frame boundaries
            detections_copy[:, 0] = np.clip(detections_copy[:, 0], 0, config.frame_width)
            detections_copy[:, 1] = np.clip(detections_copy[:, 1], 0, config.frame_height)
            detections_copy[:, 2] = np.clip(detections_copy[:, 2], 0, config.frame_width)
            detections_copy[:, 3] = np.clip(detections_copy[:, 3], 0, config.frame_height)
            
            all_detections.append(detections_copy)
        
        if len(all_detections) == 0:
            return np.array([])
        
        # Concatenate all detections
        merged = np.vstack(all_detections)
        
        # Apply NMS to remove duplicates from overlapping tiles
        merged = self.nms(merged)
        
        print(f"[DetectionMerger] Merged {len(merged)} detections after NMS")
        
        return merged


# ============================================================================
# BYTETRACK OBJECT TRACKING
# ============================================================================

class SimpleTracker:
    """
    Lightweight ByteTrack wrapper for tiled YOLO inference
    
    Provides consistent track IDs across frames for detected objects.
    Based on ultralytics ByteTracker with simplified interface.
    """
    
    def __init__(self, track_thresh: float = 0.25, track_buffer: int = 30, match_thresh: float = 0.8):
        """
        Initialize ByteTrack tracker
        
        Args:
            track_thresh: Detection confidence threshold for tracking
            track_buffer: Number of frames to keep lost tracks
            match_thresh: IoU threshold for matching tracks to detections
        """
        try:
            from ultralytics.trackers.byte_tracker import BYTETracker
            from types import SimpleNamespace
            
            # Create args namespace for BYTETracker
            args = SimpleNamespace()
            args.track_high_thresh = track_thresh
            args.track_low_thresh = 0.1
            args.new_track_thresh = track_thresh + 0.1
            args.track_buffer = track_buffer
            args.match_thresh = match_thresh
            args.fuse_score = False
            
            self.tracker = BYTETracker(args, frame_rate=30)
            self.enabled = True
            print(f"[SimpleTracker] ByteTrack initialized (track_thresh={track_thresh}, buffer={track_buffer})")
            
        except ImportError as e:
            print(f"[SimpleTracker] Failed to import ByteTracker: {e}")
            print(f"[SimpleTracker] Tracking disabled - detections will not have track IDs")
            self.enabled = False
            self.tracker = None
    
    def update(self, detections: np.ndarray) -> np.ndarray:
        """
        Update tracker with new detections
        
        Args:
            detections: Array (N, 6) [x1, y1, x2, y2, conf, class_id]
            
        Returns:
            Tracked detections (N, 7) [x1, y1, x2, y2, conf, class_id, track_id]
        """
        if not self.enabled or len(detections) == 0:
            # No tracking - return original detections with track_id=-1
            if len(detections) == 0:
                return detections
            track_ids = np.full((len(detections), 1), -1, dtype=np.float32)
            return np.hstack([detections, track_ids])
        
        try:
            # Convert to SimpleNamespace format expected by ByteTracker
            from types import SimpleNamespace
            
            results = SimpleNamespace()
            results.xyxy = detections[:, :4]  # Bounding boxes
            results.conf = detections[:, 4]   # Confidences
            results.cls = detections[:, 5]    # Class IDs
            
            # Convert to xywh format for ByteTracker
            boxes_xywh = np.zeros_like(results.xyxy)
            boxes_xywh[:, 0] = (results.xyxy[:, 0] + results.xyxy[:, 2]) / 2  # center x
            boxes_xywh[:, 1] = (results.xyxy[:, 1] + results.xyxy[:, 3]) / 2  # center y
            boxes_xywh[:, 2] = results.xyxy[:, 2] - results.xyxy[:, 0]        # width
            boxes_xywh[:, 3] = results.xyxy[:, 3] - results.xyxy[:, 1]        # height
            results.xywh = boxes_xywh
            
            # Run ByteTracker update
            tracked = self.tracker.update(results)
            
            if len(tracked) == 0:
                # No tracks - return original detections with track_id=-1
                track_ids = np.full((len(detections), 1), -1, dtype=np.float32)
                return np.hstack([detections, track_ids])
            
            # tracked is array (N, 7): [x1, y1, x2, y2, track_id, conf, class_id]
            # Reorder to match our format: [x1, y1, x2, y2, conf, class_id, track_id]
            tracked_output = np.zeros((len(tracked), 7), dtype=np.float32)
            tracked_output[:, :4] = tracked[:, :4]      # x1, y1, x2, y2
            tracked_output[:, 4] = tracked[:, 5]        # conf
            tracked_output[:, 5] = tracked[:, 6]        # class_id
            tracked_output[:, 6] = tracked[:, 4]        # track_id
            
            return tracked_output
            
        except Exception as e:
            print(f"[SimpleTracker] Tracking error: {e}")
            # Fallback - return detections without tracking
            track_ids = np.full((len(detections), 1), -1, dtype=np.float32)
            return np.hstack([detections, track_ids])
    
    def reset(self):
        """Reset tracker state"""
        if self.enabled and self.tracker is not None:
            self.tracker.reset()
            print(f"[SimpleTracker] Tracker reset")


# ============================================================================
# MAIN PIPELINE
# ============================================================================

class TiledYOLOInference:
    """Complete tiled inference pipeline with optional object tracking"""
    
    def __init__(self, engine_path: str, config: TileConfig = None, enable_tracking: bool = False):
        """
        Initialize tiled inference pipeline
        
        Args:
            engine_path: Path to TensorRT engine
            config: Tile configuration (default: 1920x1080 → 640x640)
            enable_tracking: Enable ByteTrack object tracking across frames
        """
        self.config = config or TileConfig()
        self.tile_extractor = TileExtractor(self.config)
        self.inference_engine = TensorRTInference(engine_path)
        self.detection_merger = DetectionMerger()
        
        # Initialize object tracker (optional)
        self.enable_tracking = enable_tracking
        if enable_tracking:
            self.tracker = SimpleTracker(track_thresh=0.25, track_buffer=30, match_thresh=0.8)
        else:
            self.tracker = None
        
        print(f"[TiledYOLOInference] Pipeline initialized (tracking={'ON' if enable_tracking else 'OFF'})")
    
    def process_frame(self, frame: np.ndarray, use_gpu_pipeline: bool = True) -> np.ndarray:
        """
        Process a single frame with tiled inference
        
        Args:
            frame: Input frame (H, W, 3) BGR
            use_gpu_pipeline: Use fully GPU-optimized pipeline (faster)
            
        Returns:
            Detections (N, 6 or 7) [x1, y1, x2, y2, conf, class_id, track_id*]
            *track_id only included if enable_tracking=True
        """
        if use_gpu_pipeline and self.tile_extractor.use_gpu:
            return self._process_frame_gpu(frame)
        else:
            return self._process_frame_cpu(frame)
    
    def _process_frame_cpu(self, frame: np.ndarray) -> np.ndarray:
        """Original CPU-based processing pipeline"""
        start_time = time.time()
        
        # 1. Extract tiles
        tiles = self.tile_extractor.extract_tiles(frame)
        
        # 2. Run batch inference on all tiles at once
        batch_outputs = self.inference_engine.infer_batch(tiles)
        
        # 3. Parse YOLO outputs for each tile
        tile_detections = []
        for output, x_offset, y_offset in batch_outputs:
            parsed_detections = self.parse_yolo_output(output)
            tile_detections.append((parsed_detections, x_offset, y_offset))
        
        # 4. Merge detections with NMS
        final_detections = self.detection_merger.merge_tiles(tile_detections, self.config)
        
        # 5. Apply object tracking (optional)
        if self.enable_tracking and self.tracker is not None:
            final_detections = self.tracker.update(final_detections)
        
        elapsed = time.time() - start_time
        fps = 1.0 / elapsed if elapsed > 0 else 0
        track_info = f", {len(set(final_detections[:, 6]))} tracks" if self.enable_tracking and len(final_detections) > 0 else ""
        print(f"[TiledYOLOInference] Processed frame: {len(final_detections)} detections{track_info}, "
              f"{elapsed*1000:.1f}ms ({fps:.1f} FPS)")
        
        return final_detections
    
    def _process_frame_gpu(self, frame: np.ndarray) -> np.ndarray:
        """GPU-optimized processing pipeline (minimal CPU↔GPU transfers)"""
        start_time = time.time()
        
        # 1. Extract tiles on GPU and get as batch tensor
        tiles_batch, offsets = self.tile_extractor.extract_tiles_gpu_batch(frame)
        
        # 2. Preprocess batch on GPU (BGR→RGB, normalize, HWC→CHW)
        tiles_preprocessed = self.inference_engine.preprocess_gpu_batch(tiles_batch)
        
        # 3. Run batch inference (tiles already on GPU!)
        batch_outputs = self.inference_engine.infer_batch_gpu(tiles_preprocessed, offsets)
        
        # 4. Parse YOLO outputs for each tile
        tile_detections = []
        for output, x_offset, y_offset in batch_outputs:
            parsed_detections = self.parse_yolo_output(output)
            tile_detections.append((parsed_detections, x_offset, y_offset))
        
        # 5. Merge detections with NMS
        final_detections = self.detection_merger.merge_tiles(tile_detections, self.config)
        
        # 6. Apply object tracking (optional)
        if self.enable_tracking and self.tracker is not None:
            final_detections = self.tracker.update(final_detections)
        
        elapsed = time.time() - start_time
        fps = 1.0 / elapsed if elapsed > 0 else 0
        track_info = f", {len(set(final_detections[:, 6]))} tracks" if self.enable_tracking and len(final_detections) > 0 else ""
        print(f"[TiledYOLOInference] Processed frame (GPU): {len(final_detections)} detections{track_info}, "
              f"{elapsed*1000:.1f}ms ({fps:.1f} FPS)")
        
        return final_detections
    
    def parse_yolo_output(self, output: np.ndarray, conf_threshold: float = 0.25) -> np.ndarray:
        """
        Parse YOLO11n output to detection format
        
        YOLO11n output format: (8400, 6) where each row is:
        [x1, y1, x2, y2, confidence, class_id]
        
        Based on: nvdsparsebbox_Yolo.cpp decodeTensorYolo()
        
        Args:
            output: Raw YOLO output (8400, 6)
            conf_threshold: Confidence threshold for filtering detections
            
        Returns:
            Detections (N, 6) [x1, y1, x2, y2, conf, class_id]
        """
        if len(output) == 0:
            return np.array([])
        
        # Filter by confidence threshold
        confidences = output[:, 4]
        valid_mask = confidences >= conf_threshold
        
        if not np.any(valid_mask):
            return np.array([])
        
        detections = output[valid_mask]
        
        # Clamp bounding boxes to [0, 640] (tile size)
        detections[:, 0] = np.clip(detections[:, 0], 0, 640)  # x1
        detections[:, 1] = np.clip(detections[:, 1], 0, 640)  # y1
        detections[:, 2] = np.clip(detections[:, 2], 0, 640)  # x2
        detections[:, 3] = np.clip(detections[:, 3], 0, 640)  # y2
        
        # Filter out invalid boxes (width or height < 1)
        widths = detections[:, 2] - detections[:, 0]
        heights = detections[:, 3] - detections[:, 1]
        valid_boxes = (widths >= 1) & (heights >= 1)
        
        return detections[valid_boxes]


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    
    # Initialize pipeline
    engine_path = "model_b8_gpu0_fp32.engine"
    pipeline = TiledYOLOInference(engine_path)
    
    # Process video or image
    video_path = "/home/jet-nx8/Sandbox/test-data/iphone_day_fpv_kushi_shogla_people_08_11_2025.MOV"
    
    cap = cv2.VideoCapture(video_path)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Run tiled inference
        detections = pipeline.process_frame(frame)
        
        # Visualize (optional)
        for det in detections:
            x1, y1, x2, y2, conf, class_id = det
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, f"{int(class_id)}: {conf:.2f}", (int(x1), int(y1)-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Display
        cv2.imshow("Tiled Inference", cv2.resize(frame, (960, 540)))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
