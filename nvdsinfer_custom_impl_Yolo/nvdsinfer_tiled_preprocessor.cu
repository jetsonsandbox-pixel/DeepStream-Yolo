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
#include <cuda_fp16.h>
#include <stdio.h>

/**
 * CUDA kernel to extract overlapping tiles from input frame
 * Runs on GPU for maximum performance
 * 
 * @param input Input frame buffer (RGB, HWC format)
 * @param output Output tiles buffer (8 tiles concatenated)
 * @param config Tile configuration
 */
__global__ void extractTilesKernelFp16(
    const unsigned char* input,
    __half* output,
    int input_width,
    int input_height,
    int input_pitch,
    int tile_width,
    int tile_height,
    int stride,
    int tiles_x,
    int tiles_y,
    float scale_factor)
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
            src_x < input_width && src_y < input_height &&
            src_x >= 0 && src_y >= 0) {
            // Copy RGB channels from input to output (normalize to FP16)
            // Input is HWC (RGB interleaved)
            // Output is NCHW for TensorRT: [batch, channels, height, width]
            int src_idx = src_y * input_pitch + src_x * 3;  // RGB interleaved, pitch in bytes
            
            // NCHW layout: [batch, channels, height, width] = [8, 3, 640, 640]
            // For each channel c: dst_idx = tile_idx * (C * H * W) + c * (H * W) + pixel_idx
            int chw_offset = tile_idx * 3 * pixels_per_tile;  // offset to this tile
            output[chw_offset + 0 * pixels_per_tile + pixel_idx] = __float2half(static_cast<float>(input[src_idx + 0]) * scale_factor);  // R
            output[chw_offset + 1 * pixels_per_tile + pixel_idx] = __float2half(static_cast<float>(input[src_idx + 1]) * scale_factor);  // G
            output[chw_offset + 2 * pixels_per_tile + pixel_idx] = __float2half(static_cast<float>(input[src_idx + 2]) * scale_factor);  // B
        } else {
            // Pad with zeros for edge tiles or out-of-bounds pixels
            // NCHW layout padding
            int chw_offset = tile_idx * 3 * pixels_per_tile;
            output[chw_offset + 0 * pixels_per_tile + pixel_idx] = __float2half(0.0f);  // R
            output[chw_offset + 1 * pixels_per_tile + pixel_idx] = __float2half(0.0f);  // G
            output[chw_offset + 2 * pixels_per_tile + pixel_idx] = __float2half(0.0f);  // B
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
    __half* d_output,
    const TileConfig& config,
    int input_pitch,
    float scale_factor,
    cudaStream_t stream = 0)
{
    if (!d_input || !d_output) {
        fprintf(stderr, "ERROR: Invalid input/output pointers for tile extraction\n");
        return false;
    }
    
    // Validate input buffer size expectations
    size_t expected_input_size = config.frame_width * config.frame_height * 3;
    size_t expected_output_size = config.total_tiles * config.tile_width * config.tile_height * 3 * sizeof(__half);
    
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
    extractTilesKernelFp16<<<grid, block, 0, stream>>>(
        d_input,
        d_output,
        config.frame_width,
        config.frame_height,
        input_pitch,
        config.tile_width,
        config.tile_height,
        config.stride,
        config.tiles_x,
        config.tiles_y,
        scale_factor
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
 * CUDA kernel to convert UINT8 HWC input to FP16 NCHW output with normalization
 * Duplicates the single input frame to fill batch=8
 * 
 * @param input Input frame (640x640 RGB HWC UINT8)
 * @param output Output tensor (8x3x640x640 FP16 NCHW)
 * @param width Frame width (640)
 * @param height Frame height (640)
 * @param num_tiles Number of output tiles (8)
 * @param scale_factor Normalization factor (1/255)
 */
__global__ void uint8ToFp16Kernel(
    const unsigned char* __restrict__ input,
    __half* __restrict__ output,
    int width,
    int height,
    int num_tiles,
    float scale_factor)
{
    // Each thread handles one pixel across all tiles
    int pixel_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_pixels = width * height;
    
    if (pixel_idx >= total_pixels) return;
    
    // Calculate input pixel position (HWC format)
    int y = pixel_idx / width;
    int x = pixel_idx % width;
    int hwc_idx = (y * width + x) * 3;
    
    // Read RGB values once and convert to FP16 with normalization
    __half r = __float2half(static_cast<float>(input[hwc_idx + 0]) * scale_factor);
    __half g = __float2half(static_cast<float>(input[hwc_idx + 1]) * scale_factor);
    __half b = __float2half(static_cast<float>(input[hwc_idx + 2]) * scale_factor);
    
    // Write to all 8 tiles in NCHW format
    // Layout: [batch, channels, height, width] = [8, 3, 640, 640]
    int pixels_per_channel = total_pixels;  // 640*640
    int pixels_per_tile = 3 * pixels_per_channel;  // 3*640*640
    
    // Unroll loop for fixed 8 tiles
    int tile_offset;
    
    tile_offset = 0 * pixels_per_tile;
    output[tile_offset + 0 * pixels_per_channel + pixel_idx] = r;
    output[tile_offset + 1 * pixels_per_channel + pixel_idx] = g;
    output[tile_offset + 2 * pixels_per_channel + pixel_idx] = b;
    
    tile_offset = 1 * pixels_per_tile;
    output[tile_offset + 0 * pixels_per_channel + pixel_idx] = r;
    output[tile_offset + 1 * pixels_per_channel + pixel_idx] = g;
    output[tile_offset + 2 * pixels_per_channel + pixel_idx] = b;
    
    tile_offset = 2 * pixels_per_tile;
    output[tile_offset + 0 * pixels_per_channel + pixel_idx] = r;
    output[tile_offset + 1 * pixels_per_channel + pixel_idx] = g;
    output[tile_offset + 2 * pixels_per_channel + pixel_idx] = b;
    
    tile_offset = 3 * pixels_per_tile;
    output[tile_offset + 0 * pixels_per_channel + pixel_idx] = r;
    output[tile_offset + 1 * pixels_per_channel + pixel_idx] = g;
    output[tile_offset + 2 * pixels_per_channel + pixel_idx] = b;
    
    tile_offset = 4 * pixels_per_tile;
    output[tile_offset + 0 * pixels_per_channel + pixel_idx] = r;
    output[tile_offset + 1 * pixels_per_channel + pixel_idx] = g;
    output[tile_offset + 2 * pixels_per_channel + pixel_idx] = b;
    
    tile_offset = 5 * pixels_per_tile;
    output[tile_offset + 0 * pixels_per_channel + pixel_idx] = r;
    output[tile_offset + 1 * pixels_per_channel + pixel_idx] = g;
    output[tile_offset + 2 * pixels_per_channel + pixel_idx] = b;
    
    tile_offset = 6 * pixels_per_tile;
    output[tile_offset + 0 * pixels_per_channel + pixel_idx] = r;
    output[tile_offset + 1 * pixels_per_channel + pixel_idx] = g;
    output[tile_offset + 2 * pixels_per_channel + pixel_idx] = b;
    
    tile_offset = 7 * pixels_per_tile;
    output[tile_offset + 0 * pixels_per_channel + pixel_idx] = r;
    output[tile_offset + 1 * pixels_per_channel + pixel_idx] = g;
    output[tile_offset + 2 * pixels_per_channel + pixel_idx] = b;
}

/**
 * Host function to launch UINT8->FP16 conversion kernel
 */
extern "C"
void launchUint8ToFp16Kernel(
    const unsigned char* d_input,
    __half* d_output,
    int width,
    int height,
    int num_tiles,
    float scale_factor,
    cudaStream_t stream)
{
    int total_pixels = width * height;
    int threads_per_block = 256;
    int num_blocks = (total_pixels + threads_per_block - 1) / threads_per_block;
    
    uint8ToFp16Kernel<<<num_blocks, threads_per_block, 0, stream>>>(
        d_input, d_output, width, height, num_tiles, scale_factor);
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
        
        // Extract tile in NCHW format
        for (int y = 0; y < config.tile_height; ++y) {
            for (int x = 0; x < config.tile_width; ++x) {
                int pixel_idx = y * config.tile_width + x;
                int chw_offset = tile_idx * 3 * pixels_per_tile;
                
                if (x < actual_width && y < actual_height) {
                    int src_x = x_start + x;
                    int src_y = y_start + y;
                    int src_idx = (src_y * config.frame_width + src_x) * 3;
                    
                    // NCHW output
                    output[chw_offset + 0 * pixels_per_tile + pixel_idx] = input[src_idx + 0];
                    output[chw_offset + 1 * pixels_per_tile + pixel_idx] = input[src_idx + 1];
                    output[chw_offset + 2 * pixels_per_tile + pixel_idx] = input[src_idx + 2];
                } else {
                    // Pad with zeros (NCHW)
                    output[chw_offset + 0 * pixels_per_tile + pixel_idx] = 0;
                    output[chw_offset + 1 * pixels_per_tile + pixel_idx] = 0;
                    output[chw_offset + 2 * pixels_per_tile + pixel_idx] = 0;
                }
            }
        }
    }
    
    return true;
}
