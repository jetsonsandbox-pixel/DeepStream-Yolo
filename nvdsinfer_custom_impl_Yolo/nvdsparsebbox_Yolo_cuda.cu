/*
 * Copyright (c) 2018-2024, NVIDIA CORPORATION. All rights reserved.
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
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 *
 * Edited by Marcos Luciano
 * https://www.github.com/marcoslucianops
 */

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include "nvdsinfer_custom_impl.h"

extern "C" bool
NvDsInferParseYoloCuda(std::vector<NvDsInferLayerInfo> const& outputLayersInfo, NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams, std::vector<NvDsInferParseObjectInfo>& objectList);

__global__ void decodeTensorYoloCuda(NvDsInferParseObjectInfo *binfo, const float* output, const uint outputSize,
    const uint netW, const uint netH, const float* preclusterThreshold, const uint numClasses)
{
  int x_id = blockIdx.x * blockDim.x + threadIdx.x;

  if (x_id >= outputSize) {
    return;
  }

  float maxProb = output[x_id * 6 + 4];
  int maxIndex = (int) output[x_id * 6 + 5];

  if (maxIndex < 0 || maxIndex >= (int) numClasses) {
    binfo[x_id].detectionConfidence = 0.0;
    return;
  }

  if (maxProb < preclusterThreshold[maxIndex]) {
    binfo[x_id].detectionConfidence = 0.0;
    return;
  }

  float bx1 = output[x_id * 6 + 0];
  float by1 = output[x_id * 6 + 1];
  float bx2 = output[x_id * 6 + 2];
  float by2 = output[x_id * 6 + 3];

  if (bx2 <= 1.5f && by2 <= 1.5f) {
    bx1 *= netW;
    bx2 *= netW;
    by1 *= netH;
    by2 *= netH;
  }

  bx1 = fminf(float(netW), fmaxf(float(0.0), bx1));
  by1 = fminf(float(netH), fmaxf(float(0.0), by1));
  bx2 = fminf(float(netW), fmaxf(float(0.0), bx2));
  by2 = fminf(float(netH), fmaxf(float(0.0), by2));

  binfo[x_id].left = bx1;
  binfo[x_id].top = by1;
  binfo[x_id].width = fminf(float(netW), fmaxf(float(0.0), bx2 - bx1));
  binfo[x_id].height = fminf(float(netH), fmaxf(float(0.0), by2 - by1));
  binfo[x_id].detectionConfidence = maxProb;
  binfo[x_id].classId = maxIndex;
}

static bool NvDsInferParseCustomYoloCuda(std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo, NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferParseObjectInfo>& objectList)
{
  if (outputLayersInfo.empty()) {
    std::cerr << "ERROR: Could not find output layer in bbox parsing" << std::endl;
    return false;
  }

  const NvDsInferLayerInfo& output = outputLayersInfo[0];
  uint batch = 1;
  uint outputSize = 0;

  if (output.inferDims.numDims == 3 && output.inferDims.d[2] == 6) {
    batch = output.inferDims.d[0];
    outputSize = output.inferDims.d[1];
  } else if (output.inferDims.numDims == 2 && output.inferDims.d[1] == 6) {
    outputSize = output.inferDims.d[0];
  } else if (output.inferDims.numDims == 1 && output.inferDims.d[0] % 6 == 0) {
    outputSize = output.inferDims.d[0] / 6;
  } else {
    std::cerr << "ERROR: Unexpected output dims for YOLO parser" << std::endl;
    return false;
  }

  thrust::device_vector<float> perClassPreclusterThreshold = detectionParams.perClassPreclusterThreshold;

  thrust::device_vector<NvDsInferParseObjectInfo> objects(outputSize);

  int threads_per_block = 1024;
  int number_of_blocks = ((outputSize) / threads_per_block) + 1;

  const float* base = (const float*) (output.buffer);
  const uint numClasses = static_cast<uint>(detectionParams.perClassPreclusterThreshold.size());
  for (uint b = 0; b < batch; ++b) {
    const float* batchPtr = base + (b * outputSize * 6);
    decodeTensorYoloCuda<<<number_of_blocks, threads_per_block>>>(
        thrust::raw_pointer_cast(objects.data()), batchPtr, outputSize, networkInfo.width,
            networkInfo.height, thrust::raw_pointer_cast(perClassPreclusterThreshold.data()),
            numClasses);
  }

  objectList.resize(outputSize);
  thrust::copy(objects.begin(), objects.end(), objectList.begin());

  return true;
}

extern "C" bool
NvDsInferParseYoloCuda(std::vector<NvDsInferLayerInfo> const& outputLayersInfo, NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams, std::vector<NvDsInferParseObjectInfo>& objectList)
{
  return NvDsInferParseCustomYoloCuda(outputLayersInfo, networkInfo, detectionParams, objectList);
}

CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseYoloCuda);
