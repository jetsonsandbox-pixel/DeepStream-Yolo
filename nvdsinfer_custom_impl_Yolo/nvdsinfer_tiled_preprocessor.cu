/*
 * Copyright (c) 2025, Custom Implementation for Tiled Inference
 * CUDA kernel for GPU-accelerated tile extraction
 * 
 * Based on proven implementation from:
 * /home/jet-nx8/Sandbox/umbrella-jetson-dev/gstreamer_yolo_tracker.py
 * 
 * Implements tile extraction from 1920x1080 → 8×640×640 tiles with overlap
 */

#include "nvdsinfer_tiled_config.h"
#include <cuda_runtime.h>
#include <stdio.h>

/**
 * CUDA kernel to extract overlapping tiles from input frame
 * Runs on GPU for maximum performance
 * 
 * @param input Input frame buffer (RGB, HWC format)
 * @param output Output tiles buffer (8 tiles concatenated)
 * @param config Tile configuration
 */
__global__ void extractTilesKernel(
    const unsigned char* input,
    unsigned char* output,
    int input_width,
    int input_height,
    int tile_width,
    int tile_height,
    int stride,
    int tiles_x,
    int tiles_y)
{
    // Each block processes one tile
    int tile_idx = blockIdx.x;
    if (tile_idx >= tiles_x * tiles_y) return;
    
    // Calculate tile position in grid
    int tile_col = tile_idx % tiles_x;
    int tile_row = tile_idx / tiles_x;
    
    // Calculate tile boundaries in original frame
    int x_start = tile_col * stride;
    int y_start = tile_row * stride;
    int x_end = min(x_start + tile_width, input_width);
    int y_end = min(y_start + tile_height, input_height);
    
    int actual_width = x_end - x_start;
    int actual_height = y_end - y_start;
    
    // Each thread processes one pixel (RGB channels)
    int pixel_idx = threadIdx.x + blockIdx.y * blockDim.x;
    int pixels_per_tile = tile_width * tile_height;
    
    if (pixel_idx < pixels_per_tile) {
        int tile_y = pixel_idx / tile_width;
        int tile_x = pixel_idx % tile_width;
        
        // Calculate source coordinates in input frame
        int src_x = x_start + tile_x;
        int src_y = y_start + tile_y;
        
        // Handle edge tiles with zero padding if needed
        // Bounds check on BOTH tile boundaries AND input frame boundaries
        if (tile_x < actual_width && tile_y < actual_height &&
            src_x < input_width && src_y < input_height) {
            // Copy RGB channels from input to output
            // Input is HWC (RGB interleaved), output is also HWC per tile
            int src_idx = (src_y * input_width + src_x) * 3;  // RGB without pitch
            int dst_idx = (tile_idx * pixels_per_tile + pixel_idx) * 3;
            
            output[dst_idx + 0] = input[src_idx + 0];  // R
            output[dst_idx + 1] = input[src_idx + 1];  // G
            output[dst_idx + 2] = input[src_idx + 2];  // B
        } else {
            // Pad with zeros for edge tiles or out-of-bounds pixels
            int dst_idx = (tile_idx * pixels_per_tile + pixel_idx) * 3;
            output[dst_idx + 0] = 0;
            output[dst_idx + 1] = 0;
            output[dst_idx + 2] = 0;
        }
    }
    
    // Memory fence to ensure all writes complete before kernel exits
    __threadfence();
}

/**
 * Host function to launch tile extraction kernel
 * 
 * @param d_input Device pointer to input frame
 * @param d_output Device pointer to output tiles buffer
 * @param config Tile configuration
 * @param stream CUDA stream for async execution
 * @return true on success
 */
extern "C"
bool launchTileExtractionKernel(
    const unsigned char* d_input,
    unsigned char* d_output,
    const TileConfig& config,
    cudaStream_t stream = 0)
{
    if (!d_input || !d_output) {
        fprintf(stderr, "ERROR: Invalid input/output pointers for tile extraction\n");
        return false;
    }
    
    // Validate input buffer size expectations
    size_t expected_input_size = config.frame_width * config.frame_height * 3;
    size_t expected_output_size = config.total_tiles * config.tile_width * config.tile_height * 3;
    
    if (expected_input_size == 0 || expected_output_size == 0) {
        fprintf(stderr, "ERROR: Invalid buffer size calculation (input=%zu, output=%zu)\n",
                expected_input_size, expected_output_size);
        return false;
    }
    
    // Configure kernel launch parameters
    int pixels_per_tile = config.tile_width * config.tile_height;
    int threads_per_block = 256;
    
    // Block dimensions: one block per tile for X, Y blocks for pixels
    dim3 grid(config.total_tiles, (pixels_per_tile + threads_per_block - 1) / threads_per_block);
    dim3 block(threads_per_block);
    
    // Launch kernel
    extractTilesKernel<<<grid, block, 0, stream>>>(
        d_input,
        d_output,
        config.frame_width,
        config.frame_height,
        config.tile_width,
        config.tile_height,
        config.stride,
        config.tiles_x,
        config.tiles_y
    );
    
    // Check for kernel launch errors immediately
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "ERROR: Tile extraction kernel launch failed: %s\n", 
                cudaGetErrorString(err));
        return false;
    }
    
    // Don't synchronize here - let host handle it to avoid double sync
    return true;
}

/**
 * Alternative simpler implementation for CPU-based tile extraction
 * Fallback when CUDA acceleration is not available
 * 
 * @param input Input frame buffer
 * @param output Output tiles buffer
 * @param config Tile configuration
 * @return true on success
 */
extern "C"
bool extractTilesCPU(
    const unsigned char* input,
    unsigned char* output,
    const TileConfig& config)
{
    if (!input || !output) {
        return false;
    }
    
    int pixels_per_tile = config.tile_width * config.tile_height;
    
    for (int tile_idx = 0; tile_idx < config.total_tiles; ++tile_idx) {
        int x_start, y_start, actual_width, actual_height;
        config.getTileInfo(tile_idx, x_start, y_start, actual_width, actual_height);
        
        // Extract tile
        for (int y = 0; y < config.tile_height; ++y) {
            for (int x = 0; x < config.tile_width; ++x) {
                int dst_idx = (tile_idx * pixels_per_tile + y * config.tile_width + x) * 3;
                
                if (x < actual_width && y < actual_height) {
                    int src_x = x_start + x;
                    int src_y = y_start + y;
                    int src_idx = (src_y * config.frame_width + src_x) * 3;
                    
                    output[dst_idx + 0] = input[src_idx + 0];
                    output[dst_idx + 1] = input[src_idx + 1];
                    output[dst_idx + 2] = input[src_idx + 2];
                } else {
                    // Pad with zeros
                    output[dst_idx + 0] = 0;
                    output[dst_idx + 1] = 0;
                    output[dst_idx + 2] = 0;
                }
            }
        }
    }
    
    return true;
}
