#!/usr/bin/env python3
"""
Real-time Tiled Object Detection
Processes video streams with tiled YOLO inference on Jetson
"""

import cv2
import numpy as np
import argparse
import time
from pathlib import Path
from tiled_yolo_inference import TiledYOLOInference, TileConfig


class VideoProcessor:
    """Real-time video processor with tiled YOLO detection"""
    
    def __init__(self, 
                 engine_path: str,
                 input_source: str,
                 output_path: str = None,
                 display: bool = True,
                 conf_threshold: float = 0.25,
                 labels_path: str = "labels.txt"):
        """
        Initialize video processor
        
        Args:
            engine_path: Path to TensorRT engine file
            input_source: Video file path or camera index (0, 1, etc.)
            output_path: Optional path to save output video
            display: Whether to display results in window
            conf_threshold: Detection confidence threshold
            labels_path: Path to class labels file
        """
        self.input_source = input_source
        self.output_path = output_path
        self.display = display
        self.conf_threshold = conf_threshold
        
        # Load class labels
        self.labels = self._load_labels(labels_path)
        
        # Initialize tiled inference pipeline
        print(f"[VideoProcessor] Initializing tiled inference with engine: {engine_path}")
        config = TileConfig()  # 1920x1080 -> 640x640 tiles
        self.pipeline = TiledYOLOInference(engine_path, config)
        
        # Open video source
        self.cap = self._open_video_source()
        
        # Setup output video writer if requested
        self.writer = None
        if output_path:
            self.writer = self._setup_output_video()
        
        # Performance tracking
        self.frame_count = 0
        self.total_time = 0
        self.fps_history = []
        
        print(f"[VideoProcessor] Ready to process video")
    
    def _load_labels(self, labels_path: str) -> list:
        """Load class labels from file"""
        if Path(labels_path).exists():
            with open(labels_path, 'r') as f:
                labels = [line.strip() for line in f.readlines()]
            print(f"[VideoProcessor] Loaded {len(labels)} class labels")
            return labels
        else:
            print(f"[VideoProcessor] Warning: Labels file not found, using class indices")
            return []
    
    def _open_video_source(self):
        """Open video file or camera"""
        # Try to convert to int for camera index
        try:
            source = int(self.input_source)
            print(f"[VideoProcessor] Opening camera {source}")
        except ValueError:
            source = self.input_source
            print(f"[VideoProcessor] Opening video file: {source}")
        
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video source: {source}")
        
        # Get video properties
        self.frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.video_fps = cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"[VideoProcessor] Video properties:")
        print(f"   Resolution: {self.frame_width}x{self.frame_height}")
        print(f"   FPS: {self.video_fps:.2f}")
        if self.total_frames > 0:
            print(f"   Total frames: {self.total_frames}")
        
        return cap
    
    def _setup_output_video(self):
        """Setup video writer for output"""
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(
            self.output_path,
            fourcc,
            self.video_fps,
            (self.frame_width, self.frame_height)
        )
        
        if not writer.isOpened():
            print(f"[VideoProcessor] Warning: Could not open output video writer")
            return None
        
        print(f"[VideoProcessor] Saving output to: {self.output_path}")
        return writer
    
    def draw_detections(self, frame: np.ndarray, detections: np.ndarray) -> np.ndarray:
        """
        Draw bounding boxes and labels on frame
        
        Args:
            frame: Input frame (H, W, 3) BGR
            detections: Detections (N, 6) [x1, y1, x2, y2, conf, class_id]
            
        Returns:
            Frame with drawn detections
        """
        if len(detections) == 0:
            return frame
        
        # Color palette for different classes
        np.random.seed(42)
        colors = np.random.randint(0, 255, size=(len(self.labels) if self.labels else 80, 3), dtype=np.uint8)
        
        for det in detections:
            x1, y1, x2, y2, conf, class_id = det
            class_id = int(class_id)
            
            # Filter by confidence threshold
            if conf < self.conf_threshold:
                continue
            
            # Get color for this class
            if class_id < len(colors):
                color = tuple(map(int, colors[class_id]))
            else:
                color = (0, 255, 0)
            
            # Draw bounding box
            cv2.rectangle(frame, 
                         (int(x1), int(y1)), 
                         (int(x2), int(y2)), 
                         color, 2)
            
            # Prepare label text
            if self.labels and class_id < len(self.labels):
                label = f"{self.labels[class_id]}: {conf:.2f}"
            else:
                label = f"Class {class_id}: {conf:.2f}"
            
            # Draw label background
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame,
                         (int(x1), int(y1) - text_height - baseline - 5),
                         (int(x1) + text_width, int(y1)),
                         color, -1)
            
            # Draw label text
            cv2.putText(frame, label,
                       (int(x1), int(y1) - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                       (255, 255, 255), 1)
        
        return frame
    
    def draw_performance_stats(self, frame: np.ndarray, fps: float, det_count: int) -> np.ndarray:
        """Draw performance statistics overlay"""
        # Performance text
        stats = [
            f"FPS: {fps:.1f}",
            f"Detections: {det_count}",
            f"Frame: {self.frame_count}/{self.total_frames if self.total_frames > 0 else '?'}"
        ]
        
        # Draw semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (250, 90), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        # Draw text
        y_offset = 35
        for stat in stats:
            cv2.putText(frame, stat, (20, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                       (0, 255, 0), 2)
            y_offset += 25
        
        return frame
    
    def process_stream(self, max_frames: int = None):
        """
        Process video stream with tiled detection
        
        Args:
            max_frames: Optional limit on frames to process (None = all frames)
        """
        print(f"[VideoProcessor] Starting video processing...")
        print(f"   Press 'q' to quit, 's' to save screenshot")
        
        try:
            while True:
                # Read frame
                ret, frame = self.cap.read()
                if not ret:
                    print("[VideoProcessor] End of video or failed to read frame")
                    break
                
                self.frame_count += 1
                
                # Check frame limit
                if max_frames and self.frame_count > max_frames:
                    print(f"[VideoProcessor] Reached frame limit: {max_frames}")
                    break
                
                # Run tiled inference
                start_time = time.time()
                detections = self.pipeline.process_frame(frame)
                inference_time = time.time() - start_time
                
                # Calculate FPS
                fps = 1.0 / inference_time if inference_time > 0 else 0
                self.fps_history.append(fps)
                self.total_time += inference_time
                
                # Draw detections on frame
                output_frame = self.draw_detections(frame.copy(), detections)
                output_frame = self.draw_performance_stats(output_frame, fps, len(detections))
                
                # Write to output video
                if self.writer:
                    self.writer.write(output_frame)
                
                # Display frame
                if self.display:
                    cv2.imshow('Tiled YOLO Detection', output_frame)
                    
                    # Handle keyboard input
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        print("[VideoProcessor] User requested quit")
                        break
                    elif key == ord('s'):
                        screenshot_path = f"screenshot_{self.frame_count:06d}.jpg"
                        cv2.imwrite(screenshot_path, output_frame)
                        print(f"[VideoProcessor] Saved screenshot: {screenshot_path}")
                
                # Print progress every 30 frames
                if self.frame_count % 30 == 0:
                    avg_fps = np.mean(self.fps_history[-30:])
                    print(f"[VideoProcessor] Frame {self.frame_count}: "
                          f"{len(detections)} detections, "
                          f"{fps:.1f} FPS (avg: {avg_fps:.1f})")
        
        except KeyboardInterrupt:
            print("\n[VideoProcessor] Interrupted by user")
        
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Release resources and print summary"""
        print("\n[VideoProcessor] Processing complete!")
        print(f"   Total frames: {self.frame_count}")
        print(f"   Total time: {self.total_time:.1f}s")
        if self.frame_count > 0:
            avg_fps = self.frame_count / self.total_time if self.total_time > 0 else 0
            print(f"   Average FPS: {avg_fps:.1f}")
            if self.fps_history:
                print(f"   FPS range: {min(self.fps_history):.1f} - {max(self.fps_history):.1f}")
        
        # Release resources
        self.cap.release()
        if self.writer:
            self.writer.release()
            print(f"   Output saved to: {self.output_path}")
        
        if self.display:
            cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(
        description='Real-time tiled object detection with YOLO on Jetson',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--engine', type=str, 
                       default='model_b8_gpu0_fp32.engine',
                       help='Path to TensorRT engine file')
    
    parser.add_argument('--input', type=str,
                       required=True,
                       help='Input video file or camera index (0, 1, etc.)')
    
    parser.add_argument('--output', type=str,
                       default=None,
                       help='Output video file path (optional)')
    
    parser.add_argument('--no-display', action='store_true',
                       help='Disable display window (headless mode)')
    
    parser.add_argument('--conf', type=float,
                       default=0.25,
                       help='Detection confidence threshold')
    
    parser.add_argument('--labels', type=str,
                       default='labels.txt',
                       help='Path to class labels file')
    
    parser.add_argument('--max-frames', type=int,
                       default=None,
                       help='Maximum frames to process (None = all)')
    
    args = parser.parse_args()
    
    # Create processor
    processor = VideoProcessor(
        engine_path=args.engine,
        input_source=args.input,
        output_path=args.output,
        display=not args.no_display,
        conf_threshold=args.conf,
        labels_path=args.labels
    )
    
    # Process video stream
    processor.process_stream(max_frames=args.max_frames)


if __name__ == '__main__':
    main()
