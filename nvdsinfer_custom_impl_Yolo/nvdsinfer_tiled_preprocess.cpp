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

  cudaError_t cudaErr = cudaStreamCreateWithFlags(&ctx->stream, cudaStreamNonBlocking);
  if (cudaErr != cudaSuccess) {
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
    acquirer->release(buf);
    return NVDSPREPROCESS_INVALID_PARAMS;
  }

  const unsigned char* d_input =
      reinterpret_cast<const unsigned char*>(unit.converted_frame_ptr);
  unsigned char* d_output = reinterpret_cast<unsigned char*>(buf->memory_ptr);
  bool success = launchTileExtractionKernel(
      d_input, d_output, ctx->tile_config, ctx->stream);
  if (!success) {
    acquirer->release(buf);
    return NVDSPREPROCESS_CUSTOM_TENSOR_FAILED;
  }

  cudaError_t syncErr = cudaStreamSynchronize(ctx->stream);
  if (syncErr != cudaSuccess) {
    acquirer->release(buf);
    return NVDSPREPROCESS_CUDA_ERROR;
  }

  tensorParam.params.network_input_shape[0] = ctx->tile_config.total_tiles;
  return NVDSPREPROCESS_SUCCESS;
}
