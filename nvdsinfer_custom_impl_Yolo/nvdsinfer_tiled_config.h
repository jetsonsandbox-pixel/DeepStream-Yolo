/*
 * Copyright (c) 2025, Custom Implementation for Tiled Inference
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 */

#ifndef __NVDSINFER_TILED_CONFIG_H__
#define __NVDSINFER_TILED_CONFIG_H__

#include <algorithm>
#include <vector>

/**
 * Configuration structure for tiled inference
 * Based on proven implementation from umbrella-jetson-dev/gstreamer_yolo_tracker.py
 */
struct TileConfig {
    int frame_width;      // Original frame width (e.g., 1920)
    int frame_height;     // Original frame height (e.g., 1080)
    int tile_width;       // Tile width (e.g., 640)
    int tile_height;      // Tile height (e.g., 640)
    int overlap;          // Overlap in pixels (e.g., 96)
    int stride;           // Stride = tile_size - overlap
    int tiles_x;          // Number of tiles in X direction
    int tiles_y;          // Number of tiles in Y direction
    int total_tiles;      // Total number of tiles
    
    /**
     * Initialize tile configuration
     * 
     * @param frame_w Original frame width
     * @param frame_h Original frame height
     * @param tile_sz Tile size (default: 640 for YOLO11n)
     * @param ovlp Overlap in pixels (default: 96)
     */
    TileConfig(int frame_w = 1920, int frame_h = 1080, int tile_sz = 640, int ovlp = 96) {
        frame_width = frame_w;
        frame_height = frame_h;
        tile_width = tile_sz;
        tile_height = tile_sz;
        overlap = ovlp;
        stride = tile_sz - ovlp;
        
        // Calculate grid dimensions
        // Algorithm from gstreamer_yolo_tracker.py lines 2203-2204
        tiles_x = std::max(1, (frame_w - overlap + stride - 1) / stride);
        tiles_y = std::max(1, (frame_h - overlap + stride - 1) / stride);
        total_tiles = tiles_x * tiles_y;
    }
    
    /**
     * Get tile position and dimensions for a given tile index
     * 
     * @param tile_idx Tile index (0 to total_tiles-1)
     * @param x_start Output: X start position
     * @param y_start Output: Y start position
     * @param width Output: Actual tile width
     * @param height Output: Actual tile height
     */
    void getTileInfo(int tile_idx, int& x_start, int& y_start, int& width, int& height) const {
        if (tile_idx < 0 || tile_idx >= total_tiles) {
            x_start = y_start = width = height = 0;
            return;
        }
        
        int tile_row = tile_idx / tiles_x;
        int tile_col = tile_idx % tiles_x;
        
        x_start = tile_col * stride;
        y_start = tile_row * stride;
        
        int x_end = std::min(x_start + tile_width, frame_width);
        int y_end = std::min(y_start + tile_height, frame_height);
        
        width = x_end - x_start;
        height = y_end - y_start;
    }
    
    /**
     * Get scale factors for coordinate transformation
     * 
     * @param tile_idx Tile index
     * @param scale_x Output: X scale factor
     * @param scale_y Output: Y scale factor
     */
    void getScaleFactors(int tile_idx, float& scale_x, float& scale_y) const {
        int x_start, y_start, actual_width, actual_height;
        getTileInfo(tile_idx, x_start, y_start, actual_width, actual_height);
        
        // Scale factors for edge tiles that may be resized
        scale_x = static_cast<float>(actual_width) / tile_width;
        scale_y = static_cast<float>(actual_height) / tile_height;
    }
};

/**
 * Structure to hold tile information for processing
 */
struct TileInfo {
    int tile_id;
    int x_start;
    int y_start;
    int x_end;
    int y_end;
    int actual_width;
    int actual_height;
    float scale_x;
    float scale_y;
};

#endif  // __NVDSINFER_TILED_CONFIG_H__
