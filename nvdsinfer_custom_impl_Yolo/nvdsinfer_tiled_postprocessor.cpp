/*
 * Copyright (c) 2025, Custom Implementation for Tiled Inference
 * Post-processing for merging tiled detections with NMS
 * 
 * Based on proven implementation from:
 * /home/jet-nx8/Sandbox/umbrella-jetson-dev/gstreamer_yolo_tracker.py
 * Lines 2276-2396: _merge_tile_detections()
 * 
 * Implements:
 * - Coordinate transformation from tile space to frame space
 * - Non-Maximum Suppression for duplicate removal
 * - Boundary clamping
 */

#include "nvdsinfer_tiled_config.h"
#include "nvdsinfer_custom_impl.h"
#include <algorithm>
#include <cmath>
#include <vector>

/**
 * Structure to hold a single detection
 */
struct Detection {
    float x1, y1, x2, y2;  // Bounding box coordinates
    float confidence;       // Detection confidence
    int class_id;          // Class label
    int tile_id;           // Source tile index
    
    Detection() : x1(0), y1(0), x2(0), y2(0), confidence(0), class_id(0), tile_id(0) {}
    
    Detection(float _x1, float _y1, float _x2, float _y2, float _conf, int _cls, int _tile)
        : x1(_x1), y1(_y1), x2(_x2), y2(_y2), confidence(_conf), class_id(_cls), tile_id(_tile) {}
};

/**
 * Calculate Intersection over Union (IoU) between two detections
 * Used for Non-Maximum Suppression
 * 
 * @param a First detection
 * @param b Second detection
 * @return IoU value [0.0, 1.0]
 */
static float calculateIoU(const Detection& a, const Detection& b) {
    float x1 = std::max(a.x1, b.x1);
    float y1 = std::max(a.y1, b.y1);
    float x2 = std::min(a.x2, b.x2);
    float y2 = std::min(a.y2, b.y2);
    
    // No overlap
    if (x2 < x1 || y2 < y1) return 0.0f;
    
    float intersection = (x2 - x1) * (y2 - y1);
    float area_a = (a.x2 - a.x1) * (a.y2 - a.y1);
    float area_b = (b.x2 - b.x1) * (b.y2 - b.y1);
    float union_area = area_a + area_b - intersection;
    
    return union_area > 0 ? intersection / union_area : 0.0f;
}

/**
 * Apply Non-Maximum Suppression to remove duplicate detections
 * Algorithm from gstreamer_yolo_tracker.py lines 2361-2374
 * 
 * @param detections Input detections (will be sorted by confidence)
 * @param nms_threshold IoU threshold for suppression (default: 0.45)
 * @return Filtered detections after NMS
 */
static std::vector<Detection> applyNMS(
    std::vector<Detection>& detections,
    float nms_threshold = 0.45f)
{
    if (detections.empty()) {
        return std::vector<Detection>();
    }
    
    // Sort by confidence descending
    std::sort(detections.begin(), detections.end(),
        [](const Detection& a, const Detection& b) {
            return a.confidence > b.confidence;
        });
    
    std::vector<Detection> result;
    std::vector<bool> suppressed(detections.size(), false);
    
    for (size_t i = 0; i < detections.size(); ++i) {
        if (suppressed[i]) continue;
        
        result.push_back(detections[i]);
        
        // Suppress overlapping boxes of the same class
        for (size_t j = i + 1; j < detections.size(); ++j) {
            if (suppressed[j]) continue;
            if (detections[i].class_id != detections[j].class_id) continue;
            
            float iou = calculateIoU(detections[i], detections[j]);
            if (iou > nms_threshold) {
                suppressed[j] = true;
            }
        }
    }
    
    return result;
}

/**
 * Transform detection coordinates from tile space to original frame space
 * Algorithm from gstreamer_yolo_tracker.py lines 2322-2357
 * 
 * @param tile_detections Detections in tile coordinates
 * @param config Tile configuration
 * @return Detections in frame coordinates
 */
static std::vector<Detection> transformTileDetections(
    const std::vector<Detection>& tile_detections,
    const TileConfig& config)
{
    std::vector<Detection> transformed;
    transformed.reserve(tile_detections.size());
    
    for (const auto& det : tile_detections) {
        Detection transformed_det = det;
        
        // Get tile position and scale factors
        int x_start, y_start, actual_width, actual_height;
        config.getTileInfo(det.tile_id, x_start, y_start, actual_width, actual_height);
        
        float scale_x = static_cast<float>(actual_width) / config.tile_width;
        float scale_y = static_cast<float>(actual_height) / config.tile_height;
        
        // Transform coordinates from tile space to frame space
        // Formula: orig_coord = (tile_coord * scale) + tile_offset
        transformed_det.x1 = (det.x1 * scale_x) + x_start;
        transformed_det.y1 = (det.y1 * scale_y) + y_start;
        transformed_det.x2 = (det.x2 * scale_x) + x_start;
        transformed_det.y2 = (det.y2 * scale_y) + y_start;
        
        // Clamp to frame boundaries (lines 2348-2351 in gstreamer_yolo_tracker.py)
        transformed_det.x1 = std::max(0.0f, std::min(transformed_det.x1, static_cast<float>(config.frame_width)));
        transformed_det.y1 = std::max(0.0f, std::min(transformed_det.y1, static_cast<float>(config.frame_height)));
        transformed_det.x2 = std::max(0.0f, std::min(transformed_det.x2, static_cast<float>(config.frame_width)));
        transformed_det.y2 = std::max(0.0f, std::min(transformed_det.y2, static_cast<float>(config.frame_height)));
        
        // Only keep valid boxes (lines 2354-2357)
        if (transformed_det.x2 > transformed_det.x1 && 
            transformed_det.y2 > transformed_det.y1) {
            transformed.push_back(transformed_det);
        }
    }
    
    return transformed;
}

/**
 * Merge detections from all tiles with NMS
 * Main function called by DeepStream inference engine
 * 
 * This function:
 * 1. Collects detections from all tiles
 * 2. Transforms coordinates to frame space
 * 3. Applies NMS to remove duplicates
 * 4. Converts to DeepStream format
 * 
 * @param tile_detections Vector of detections from each tile
 * @param config Tile configuration
 * @param nms_threshold IoU threshold for NMS (default: 0.45)
 * @param objectList Output: merged detections in DeepStream format
 * @return true on success
 */
extern "C"
bool mergeTiledDetections(
    const std::vector<std::vector<Detection>>& tile_detections,
    const TileConfig& config,
    float nms_threshold,
    std::vector<NvDsInferParseObjectInfo>& objectList)
{
    // Collect all detections from all tiles
    std::vector<Detection> all_detections;
    
    for (size_t tile_idx = 0; tile_idx < tile_detections.size(); ++tile_idx) {
        for (const auto& det : tile_detections[tile_idx]) {
            all_detections.push_back(det);
        }
    }
    
    if (all_detections.empty()) {
        return true;  // No detections is not an error
    }
    
    // Transform tile coordinates to frame coordinates
    auto transformed = transformTileDetections(all_detections, config);
    
    // Apply NMS to remove duplicates in overlap regions
    auto final_detections = applyNMS(transformed, nms_threshold);
    
    // Convert to DeepStream format
    for (const auto& det : final_detections) {
        NvDsInferParseObjectInfo obj;
        obj.left = det.x1;
        obj.top = det.y1;
        obj.width = det.x2 - det.x1;
        obj.height = det.y2 - det.y1;
        obj.detectionConfidence = det.confidence;
        obj.classId = det.class_id;
        objectList.push_back(obj);
    }
    
    return true;
}

/**
 * Helper function to convert NvDsInferParseObjectInfo to Detection
 * Used when integrating with existing YOLO parser
 * 
 * @param obj DeepStream object info
 * @param tile_id Source tile index
 * @return Detection structure
 */
static Detection convertToDetection(const NvDsInferParseObjectInfo& obj, int tile_id) {
    return Detection(
        obj.left,
        obj.top,
        obj.left + obj.width,
        obj.top + obj.height,
        obj.detectionConfidence,
        obj.classId,
        tile_id
    );
}

/**
 * Simplified wrapper that returns merged detections directly
 * 
 * @param tileDetections Vector of detection lists, one per tile  
 * @param config Tile configuration
 * @param nmsThreshold NMS IoU threshold
 * @return Merged and NMS-filtered detections
 */
std::vector<NvDsInferParseObjectInfo> mergeTiledDetections(
    const std::vector<std::vector<NvDsInferParseObjectInfo>>& tileDetections,
    const TileConfig& config,
    float nmsThreshold)
{
    // Convert to internal Detection format
    std::vector<std::vector<Detection>> tile_dets;
    tile_dets.resize(tileDetections.size());
    
    for (size_t tile_idx = 0; tile_idx < tileDetections.size(); ++tile_idx) {
        for (const auto& obj : tileDetections[tile_idx]) {
            tile_dets[tile_idx].push_back(convertToDetection(obj, tile_idx));
        }
    }
    
    // Merge and return
    std::vector<NvDsInferParseObjectInfo> result;
    mergeTiledDetections(tile_dets, config, nmsThreshold, result);
    return result;
}

/**
 * Wrapper function for integrating with existing NvDsInferParseYolo
 * Merges detections from batched inference (batch-size=8 tiles)
 * 
 * @param per_tile_objectLists Vector of object lists (one per tile)
 * @param config Tile configuration
 * @param nms_threshold NMS IoU threshold
 * @param merged_objectList Output: merged and filtered object list
 * @return true on success
 */
extern "C"
bool NvDsInferMergeTiledYoloDetections(
    const std::vector<std::vector<NvDsInferParseObjectInfo>>& per_tile_objectLists,
    const TileConfig& config,
    float nms_threshold,
    std::vector<NvDsInferParseObjectInfo>& merged_objectList)
{
    // Convert DeepStream format to internal Detection format
    std::vector<std::vector<Detection>> tile_detections;
    tile_detections.resize(per_tile_objectLists.size());
    
    for (size_t tile_idx = 0; tile_idx < per_tile_objectLists.size(); ++tile_idx) {
        for (const auto& obj : per_tile_objectLists[tile_idx]) {
            tile_detections[tile_idx].push_back(convertToDetection(obj, tile_idx));
        }
    }
    
    // Use main merging function
    return mergeTiledDetections(tile_detections, config, nms_threshold, merged_objectList);
}
