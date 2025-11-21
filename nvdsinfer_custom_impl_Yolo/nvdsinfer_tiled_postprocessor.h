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

#ifndef __NVDSINFER_TILED_POSTPROCESSOR_H__
#define __NVDSINFER_TILED_POSTPROCESSOR_H__

#include <vector>
#include "nvdsinfer_custom_impl.h"
#include "nvdsinfer_tiled_config.h"

/**
 * Merge detections from multiple tiles into frame coordinates
 * 
 * @param tileDetections Vector of detection lists, one per tile
 * @param config Tile configuration with grid layout and overlap info
 * @param nmsThreshold IoU threshold for Non-Maximum Suppression
 * @return Merged and NMS-filtered detections in frame coordinates
 */
std::vector<NvDsInferParseObjectInfo> mergeTiledDetections(
    const std::vector<std::vector<NvDsInferParseObjectInfo>>& tileDetections,
    const TileConfig& config,
    float nmsThreshold = 0.45f
);

#endif // __NVDSINFER_TILED_POSTPROCESSOR_H__
