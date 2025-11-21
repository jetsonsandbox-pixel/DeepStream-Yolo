/*
 * CUDA Tile Extraction Kernel for Python/PyTorch Integration
 * Simple interface for extracting overlapping tiles from frames
 */

#include <cuda_runtime.h>
#include <stdio.h>

/**
 * CUDA kernel to extract one tile from input frame
 * Input: BGR frame (H, W, 3) uint8
 * Output: Preprocessed tile (3, 640, 640) float32 normalized
 * 
 * Performs:
 * - Tile extraction with zero-padding if needed
 * - BGR -> RGB conversion
 * - Normalize to [0, 1] (divide by 255)
 * - HWC -> CHW reordering
 */
__global__ void extractAndPreprocessTile(
    const unsigned char* input,    // Input frame (H, W, 3) BGR uint8
    float* output,                  // Output tile (3, 640, 640) RGB float32
    int input_width,                // Frame width (e.g., 1920)
    int input_height,               // Frame height (e.g., 1080)
    int tile_x_offset,              // Tile X start position
    int tile_y_offset,              // Tile Y start position
    int tile_width,                 // Tile width (640)
    int tile_height)                // Tile height (640)
{
    // Each thread processes one pixel in the output tile
    int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    int out_y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (out_x >= tile_width || out_y >= tile_height) {
        return;
    }
    
    // Calculate source position in input frame
    int src_x = tile_x_offset + out_x;
    int src_y = tile_y_offset + out_y;
    
    // Check if source is within frame boundaries
    if (src_x >= 0 && src_x < input_width && src_y >= 0 && src_y < input_height) {
        // Valid pixel - read from input (BGR format)
        int input_idx = (src_y * input_width + src_x) * 3;
        unsigned char b = input[input_idx + 0];
        unsigned char g = input[input_idx + 1];
        unsigned char r = input[input_idx + 2];
        
        // Write to output in CHW format, RGB order, normalized
        int pixels = tile_width * tile_height;
        int output_idx = out_y * tile_width + out_x;
        
        output[0 * pixels + output_idx] = r / 255.0f;  // R channel
        output[1 * pixels + output_idx] = g / 255.0f;  // G channel
        output[2 * pixels + output_idx] = b / 255.0f;  // B channel
    } else {
        // Padding - write zeros
        int pixels = tile_width * tile_height;
        int output_idx = out_y * tile_width + out_x;
        
        output[0 * pixels + output_idx] = 0.0f;  // R
        output[1 * pixels + output_idx] = 0.0f;  // G
        output[2 * pixels + output_idx] = 0.0f;  // B
    }
}

/**
 * CUDA kernel to extract batch of tiles from input frame
 * Processes all 8 tiles in parallel
 * 
 * Input: BGR frame (H, W, 3) uint8
 * Output: Batch of preprocessed tiles (8, 3, 640, 640) float32
 */
__global__ void extractAndPreprocessTilesBatch(
    const unsigned char* input,    // Input frame (H, W, 3) BGR uint8
    float* output,                  // Output tiles (batch, 3, 640, 640) RGB float32
    int input_width,
    int input_height,
    const int* tile_offsets,        // Array of [x, y] offsets for each tile (num_tiles * 2)
    int num_tiles,
    int tile_width,
    int tile_height)
{
    // Tile index from Z dimension
    int tile_idx = blockIdx.z;
    if (tile_idx >= num_tiles) return;
    
    // Pixel position in output tile
    int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    int out_y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (out_x >= tile_width || out_y >= tile_height) {
        return;
    }
    
    // Get tile offsets
    int tile_x_offset = tile_offsets[tile_idx * 2 + 0];
    int tile_y_offset = tile_offsets[tile_idx * 2 + 1];
    
    // Calculate source position
    int src_x = tile_x_offset + out_x;
    int src_y = tile_y_offset + out_y;
    
    // Calculate output index
    int pixels_per_tile = tile_width * tile_height;
    int pixels_per_channel = pixels_per_tile;
    int output_base = tile_idx * 3 * pixels_per_tile;
    int output_idx = out_y * tile_width + out_x;
    
    if (src_x >= 0 && src_x < input_width && src_y >= 0 && src_y < input_height) {
        // Valid pixel - read BGR from input
        int input_idx = (src_y * input_width + src_x) * 3;
        unsigned char b = input[input_idx + 0];
        unsigned char g = input[input_idx + 1];
        unsigned char r = input[input_idx + 2];
        
        // Write RGB to output (CHW format, normalized)
        output[output_base + 0 * pixels_per_channel + output_idx] = r / 255.0f;
        output[output_base + 1 * pixels_per_channel + output_idx] = g / 255.0f;
        output[output_base + 2 * pixels_per_channel + output_idx] = b / 255.0f;
    } else {
        // Padding
        output[output_base + 0 * pixels_per_channel + output_idx] = 0.0f;
        output[output_base + 1 * pixels_per_channel + output_idx] = 0.0f;
        output[output_base + 2 * pixels_per_channel + output_idx] = 0.0f;
    }
}

/**
 * Host function to launch tile extraction kernel
 * 
 * @param d_input Device pointer to input frame (H, W, 3) BGR uint8
 * @param d_output Device pointer to output tile (3, 640, 640) float32
 * @param input_width Frame width
 * @param input_height Frame height
 * @param tile_x_offset Tile X offset in frame
 * @param tile_y_offset Tile Y offset in frame
 * @param tile_width Tile width (640)
 * @param tile_height Tile height (640)
 * @param stream CUDA stream
 * @return CUDA error code
 */
extern "C"
cudaError_t launchTileExtraction(
    const unsigned char* d_input,
    float* d_output,
    int input_width,
    int input_height,
    int tile_x_offset,
    int tile_y_offset,
    int tile_width,
    int tile_height,
    cudaStream_t stream)
{
    // Configure kernel: 16x16 threads per block
    dim3 block(16, 16);
    dim3 grid(
        (tile_width + block.x - 1) / block.x,
        (tile_height + block.y - 1) / block.y
    );
    
    extractAndPreprocessTile<<<grid, block, 0, stream>>>(
        d_input,
        d_output,
        input_width,
        input_height,
        tile_x_offset,
        tile_y_offset,
        tile_width,
        tile_height
    );
    
    return cudaGetLastError();
}

/**
 * Host function to launch batch tile extraction kernel
 * 
 * @param d_input Device pointer to input frame (H, W, 3) BGR uint8
 * @param d_output Device pointer to output tiles (batch, 3, 640, 640) float32
 * @param input_width Frame width
 * @param input_height Frame height
 * @param d_tile_offsets Device pointer to tile offsets array (num_tiles * 2)
 * @param num_tiles Number of tiles
 * @param tile_width Tile width
 * @param tile_height Tile height
 * @param stream CUDA stream
 * @return CUDA error code
 */
extern "C"
cudaError_t launchTileExtractionBatch(
    const unsigned char* d_input,
    float* d_output,
    int input_width,
    int input_height,
    const int* d_tile_offsets,
    int num_tiles,
    int tile_width,
    int tile_height,
    cudaStream_t stream)
{
    // Configure kernel: 16x16 threads per block, Z for tiles
    dim3 block(16, 16);
    dim3 grid(
        (tile_width + block.x - 1) / block.x,
        (tile_height + block.y - 1) / block.y,
        num_tiles
    );
    
    extractAndPreprocessTilesBatch<<<grid, block, 0, stream>>>(
        d_input,
        d_output,
        input_width,
        input_height,
        d_tile_offsets,
        num_tiles,
        tile_width,
        tile_height
    );
    
    return cudaGetLastError();
}

/**
 * Get version information
 */
extern "C"
const char* cuda_tile_version() {
    return "1.0.0";
}
