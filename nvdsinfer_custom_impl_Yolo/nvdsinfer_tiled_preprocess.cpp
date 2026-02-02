#include <algorithm>
#include <cuda_runtime_api.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <new>
#include <string>
#include <unordered_map>

#include "nvbufsurface.h"
#include "nvdsinfer_tiled_config.h"
#include "nvdspreprocess_interface.h"
#include "nvdspreprocess_lib.h"

// Global mutex to serialize CUDA operations across multiple pipeline instances
// This prevents race conditions when dual cameras run simultaneously
static std::mutex g_cuda_mutex;

// Forward declaration of CUDA kernel for UINT8->FP16 conversion
extern "C" void launchUint8ToFp16Kernel(
    const unsigned char* d_input,
    __half* d_output,
    int width,
    int height,
    int num_tiles,
    float scale_factor,
    cudaStream_t stream);

struct CustomCtx {
  TileConfig tile_config;
  cudaStream_t stream = nullptr;
};

namespace {

int parseInt(const std::unordered_map<std::string, std::string>& map,
             const char* key,
             int defaultValue) {
  auto it = map.find(key);
  if (it == map.end()) {
    return defaultValue;
  }
  try {
    return std::stoi(it->second);
  } catch (...) {
    return defaultValue;
  }
}

}  // namespace

bool configureTileConfig(CustomCtx* ctx,
                         const CustomInitParams& initparams) {
  int frameWidth = parseInt(initparams.user_configs, "frame-width", 1920);
  int frameHeight = parseInt(initparams.user_configs, "frame-height", 1080);
  int tileWidth = parseInt(initparams.user_configs, "tile-width", 640);
  int tileHeight = parseInt(initparams.user_configs, "tile-height", 640);
  int overlap = parseInt(initparams.user_configs, "tile-overlap", 96);

  ctx->tile_config = TileConfig(frameWidth, frameHeight, tileWidth, overlap);
  ctx->tile_config.tile_height = tileHeight;
  ctx->tile_config.stride = tileWidth - overlap;
  if (ctx->tile_config.stride <= 0) {
    ctx->tile_config.stride = tileWidth;
  }

  ctx->tile_config.tiles_x =
      std::max(1, (frameWidth - overlap + ctx->tile_config.stride - 1) /
                       ctx->tile_config.stride);
  ctx->tile_config.tiles_y = std::max(
      1, (frameHeight - overlap + ctx->tile_config.stride - 1) /
             ctx->tile_config.stride);
  ctx->tile_config.total_tiles = ctx->tile_config.tiles_x * ctx->tile_config.tiles_y;
  return true;
}

extern "C" CustomCtx* initLib(CustomInitParams initparams) {
  auto* ctx = new (std::nothrow) CustomCtx();
  if (!ctx) {
    return nullptr;
  }

  if (!configureTileConfig(ctx, initparams)) {
    delete ctx;
    return nullptr;
  }

  cudaError_t cudaErr = cudaStreamCreateWithFlags(&ctx->stream, cudaStreamDefault);
  if (cudaErr != cudaSuccess) {
    fprintf(stderr, "ERROR: Failed to create CUDA stream: %s\n", cudaGetErrorString(cudaErr));
    delete ctx;
    return nullptr;
  }

  fprintf(stderr, "INFO: Custom tiling library initialized - %d tiles (%dx%d)\n",
          ctx->tile_config.total_tiles, ctx->tile_config.tiles_x, ctx->tile_config.tiles_y);
  return ctx;
}

extern "C" void deInitLib(CustomCtx* ctx) {
  if (!ctx) {
    return;
  }
  if (ctx->stream) {
    cudaStreamDestroy(ctx->stream);
  }
  delete ctx;
}

extern "C" NvDsPreProcessStatus CustomTensorPreparation(
    CustomCtx* ctx,
    NvDsPreProcessBatch* batch,
    NvDsPreProcessCustomBuf*& buf,
    CustomTensorParams& tensorParam,
    NvDsPreProcessAcquirer* acquirer) {
  if (!ctx || !batch || batch->units.empty() || !acquirer) {
    fprintf(stderr, "ERROR: Invalid parameters in CustomTensorPreparation\n");
    return NVDSPREPROCESS_INVALID_PARAMS;
  }

  // Acquire mutex to serialize CUDA operations across multiple pipelines
  std::lock_guard<std::mutex> lock(g_cuda_mutex);

  // Clear any stale CUDA errors from previous operations
  cudaError_t prevErr = cudaGetLastError();
  if (prevErr != cudaSuccess) {
    fprintf(stderr, "WARN: Cleared previous CUDA error: %s\n", cudaGetErrorString(prevErr));
  }

  buf = acquirer->acquire();
  if (!buf || !buf->memory_ptr) {
    fprintf(stderr, "ERROR: Failed to acquire tensor buffer\n");
    return NVDSPREPROCESS_RESOURCE_ERROR;
  }

  const NvDsPreProcessUnit& unit = batch->units[0];
  
  // converted_frame_ptr is 640x640 RGB HWC UINT8 (scaled by nvdspreprocess)
  // We need to convert to FP16 NCHW and duplicate 8 times for batch=8 engine
  if (!unit.converted_frame_ptr) {
    fprintf(stderr, "ERROR: converted_frame_ptr is NULL\n");
    acquirer->release(buf);
    return NVDSPREPROCESS_INVALID_PARAMS;
  }

  const unsigned char* d_input =
      reinterpret_cast<const unsigned char*>(unit.converted_frame_ptr);
  __half* d_output = reinterpret_cast<__half*>(buf->memory_ptr);
  
  // Debug: Print pointer info (first frame only)
  static int frame_count = 0;
  frame_count++;
  
  if (frame_count <= 3 || frame_count % 100 == 0) {
    fprintf(stderr, "DEBUG[%d]: Input=%p, Output=%p\n", frame_count, (void*)d_input, (void*)d_output);
  }

  // Launch CUDA kernel to convert UINT8 HWC -> FP16 NCHW with normalization
  // Scale factor 1/255 = 0.00392156... to normalize 0-255 to 0.0-1.0
  const float scale_factor = 0.0039215697906911373f;
  const int width = 640;
  const int height = 640;
  const int num_tiles = 8;
  
  launchUint8ToFp16Kernel(d_input, d_output, width, height, num_tiles, scale_factor, ctx->stream);
  
  // Check for kernel errors
  cudaError_t kernelErr = cudaGetLastError();
  if (kernelErr != cudaSuccess) {
    fprintf(stderr, "ERROR: Conversion kernel failed: %s\n", cudaGetErrorString(kernelErr));
    acquirer->release(buf);
    return NVDSPREPROCESS_CUDA_ERROR;
  }

  // Synchronize stream
  cudaError_t syncErr = cudaStreamSynchronize(ctx->stream);
  if (syncErr != cudaSuccess) {
    fprintf(stderr, "ERROR: Stream synchronization failed: %s\n", cudaGetErrorString(syncErr));
    cudaGetLastError();
    acquirer->release(buf);
    return NVDSPREPROCESS_CUDA_ERROR;
  }

  // Report batch=8 to satisfy the engine
  tensorParam.params.network_input_shape[0] = 8;  // batch = 8
  tensorParam.params.network_input_shape[1] = 3;  // channels
  tensorParam.params.network_input_shape[2] = 640;  // height
  tensorParam.params.network_input_shape[3] = 640;  // width
  return NVDSPREPROCESS_SUCCESS;
}
