#include <algorithm>
#include <cuda_runtime_api.h>
#include <cstdio>
#include <cstdlib>
#include <new>
#include <string>
#include <unordered_map>

#include "nvbufsurface.h"
#include "nvdsinfer_tiled_config.h"
#include "nvdspreprocess_interface.h"
#include "nvdspreprocess_lib.h"

extern "C" bool launchTileExtractionKernel(
    const unsigned char* d_input,
    unsigned char* d_output,
    const TileConfig& config,
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
    delete ctx;
    return nullptr;
  }

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
    return NVDSPREPROCESS_INVALID_PARAMS;
  }

  buf = acquirer->acquire();
  if (!buf || !buf->memory_ptr) {
    return NVDSPREPROCESS_RESOURCE_ERROR;
  }

  const NvDsPreProcessUnit& unit = batch->units[0];
  if (!unit.converted_frame_ptr) {
    fprintf(stderr, "ERROR: converted_frame_ptr is NULL\n");
    acquirer->release(buf);
    return NVDSPREPROCESS_INVALID_PARAMS;
  }

  // Defensive: clear any pending CUDA errors before our operation
  cudaGetLastError();

  const unsigned char* d_input =
      reinterpret_cast<const unsigned char*>(unit.converted_frame_ptr);
  unsigned char* d_output = reinterpret_cast<unsigned char*>(buf->memory_ptr);
  
  // Validate pointers are accessible (will catch some memory issues early)
  cudaPointerAttributes inputAttrs, outputAttrs;
  cudaError_t checkErr = cudaPointerGetAttributes(&inputAttrs, d_input);
  if (checkErr != cudaSuccess) {
    fprintf(stderr, "ERROR: Input pointer validation failed: %s\n", cudaGetErrorString(checkErr));
    cudaGetLastError(); // Clear error
    acquirer->release(buf);
    return NVDSPREPROCESS_INVALID_PARAMS;
  }
  
  checkErr = cudaPointerGetAttributes(&outputAttrs, d_output);
  if (checkErr != cudaSuccess) {
    fprintf(stderr, "ERROR: Output pointer validation failed: %s\n", cudaGetErrorString(checkErr));
    cudaGetLastError(); // Clear error
    acquirer->release(buf);
    return NVDSPREPROCESS_RESOURCE_ERROR;
  }

  bool success = launchTileExtractionKernel(
      d_input, d_output, ctx->tile_config, ctx->stream);
  if (!success) {
    fprintf(stderr, "ERROR: Tile extraction kernel launch returned false\n");
    acquirer->release(buf);
    return NVDSPREPROCESS_CUSTOM_TENSOR_FAILED;
  }

  // Synchronize stream to ensure kernel completes before buffer is used
  cudaError_t syncErr = cudaStreamSynchronize(ctx->stream);
  if (syncErr != cudaSuccess) {
    fprintf(stderr, "ERROR: Stream synchronization failed: %s\n", cudaGetErrorString(syncErr));
    cudaGetLastError(); // Clear error
    acquirer->release(buf);
    return NVDSPREPROCESS_CUDA_ERROR;
  }
  
  // Also synchronize device to catch context-wide issues early
  cudaError_t deviceErr = cudaDeviceSynchronize();
  if (deviceErr != cudaSuccess) {
    fprintf(stderr, "ERROR: Device synchronization failed: %s\n", cudaGetErrorString(deviceErr));
    cudaGetLastError(); // Clear error
    acquirer->release(buf);
    return NVDSPREPROCESS_CUDA_ERROR;
  }

  tensorParam.params.network_input_shape[0] = ctx->tile_config.total_tiles;  // batch = 8
  tensorParam.params.network_input_shape[1] = 3;  // channels = 3 (RGB)
  tensorParam.params.network_input_shape[2] = ctx->tile_config.tile_height;  // height = 640
  tensorParam.params.network_input_shape[3] = ctx->tile_config.tile_width;   // width = 640
  return NVDSPREPROCESS_SUCCESS;
}
