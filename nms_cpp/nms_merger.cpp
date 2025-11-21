/*
 * Fast C++ NMS Implementation for Tiled Detection Merging
 * Python-friendly interface using simple arrays
 */

#include <algorithm>
#include <vector>
#include <cmath>
#include <cstdint>

/**
 * Calculate Intersection over Union (IoU) between two bounding boxes
 * 
 * @param box1 First box [x1, y1, x2, y2, conf, class_id]
 * @param box2 Second box [x1, y1, x2, y2, conf, class_id]
 * @return IoU value [0.0, 1.0]
 */
static float calculateIoU(const float* box1, const float* box2) {
    float x1 = std::max(box1[0], box2[0]);
    float y1 = std::max(box1[1], box2[1]);
    float x2 = std::min(box1[2], box2[2]);
    float y2 = std::min(box1[3], box2[3]);
    
    if (x2 < x1 || y2 < y1) return 0.0f;
    
    float intersection = (x2 - x1) * (y2 - y1);
    float area1 = (box1[2] - box1[0]) * (box1[3] - box1[1]);
    float area2 = (box2[2] - box2[0]) * (box2[3] - box2[1]);
    float union_area = area1 + area2 - intersection;
    
    return union_area > 0 ? intersection / union_area : 0.0f;
}

/**
 * Apply Non-Maximum Suppression to detections
 * 
 * Input format: detections is array of shape (N, 6) where each row is:
 * [x1, y1, x2, y2, confidence, class_id]
 * 
 * @param detections Input detections (N x 6 array, row-major)
 * @param num_detections Number of input detections
 * @param nms_threshold IoU threshold for suppression
 * @param output_indices Output array for kept indices (must be pre-allocated, size N)
 * @param num_kept Output: number of detections kept after NMS
 * @return 0 on success, -1 on error
 */
extern "C"
int nms_merge_detections(
    const float* detections,      // Input: (N, 6) array [x1, y1, x2, y2, conf, class_id]
    int num_detections,            // Input: number of detections
    float nms_threshold,           // Input: IoU threshold
    int* output_indices,           // Output: indices of kept detections
    int* num_kept)                 // Output: number of kept detections
{
    if (!detections || !output_indices || !num_kept || num_detections <= 0) {
        return -1;  // Invalid input
    }
    
    // Create vector of indices sorted by confidence (descending)
    std::vector<int> indices(num_detections);
    for (int i = 0; i < num_detections; ++i) {
        indices[i] = i;
    }
    
    std::sort(indices.begin(), indices.end(),
        [detections](int i, int j) {
            return detections[i * 6 + 4] > detections[j * 6 + 4];  // Sort by confidence
        });
    
    std::vector<bool> suppressed(num_detections, false);
    int kept = 0;
    
    for (int i = 0; i < num_detections; ++i) {
        int idx_i = indices[i];
        if (suppressed[idx_i]) continue;
        
        output_indices[kept++] = idx_i;
        
        const float* box_i = &detections[idx_i * 6];
        int class_i = static_cast<int>(box_i[5]);
        
        // Suppress overlapping boxes of the same class
        for (int j = i + 1; j < num_detections; ++j) {
            int idx_j = indices[j];
            if (suppressed[idx_j]) continue;
            
            const float* box_j = &detections[idx_j * 6];
            int class_j = static_cast<int>(box_j[5]);
            
            // Only compare boxes of same class
            if (class_i != class_j) continue;
            
            float iou = calculateIoU(box_i, box_j);
            if (iou > nms_threshold) {
                suppressed[idx_j] = true;
            }
        }
    }
    
    *num_kept = kept;
    return 0;  // Success
}

/**
 * Transform tile detections to frame coordinates and apply NMS
 * 
 * @param tile_detections Array of tile detections [tile_id, x1, y1, x2, y2, conf, class_id] (N x 7)
 * @param num_detections Number of detections
 * @param tile_offsets Array of tile offsets [x_offset, y_offset] for each tile (num_tiles x 2)
 * @param num_tiles Number of tiles
 * @param nms_threshold IoU threshold
 * @param output Array for transformed detections (N x 6) [x1, y1, x2, y2, conf, class_id]
 * @param num_kept Output: number of kept detections
 * @return 0 on success
 */
extern "C"
int nms_merge_tiles(
    const float* tile_detections,  // Input: (N, 7) [tile_id, x1, y1, x2, y2, conf, class_id]
    int num_detections,
    const float* tile_offsets,     // Input: (num_tiles, 2) [x_offset, y_offset]
    int num_tiles,
    float nms_threshold,
    float* output,                 // Output: (N, 6) [x1, y1, x2, y2, conf, class_id]
    int* num_kept)
{
    if (!tile_detections || !tile_offsets || !output || !num_kept) {
        return -1;
    }
    
    // Transform tile coordinates to frame coordinates
    std::vector<float> transformed(num_detections * 6);
    
    for (int i = 0; i < num_detections; ++i) {
        int tile_id = static_cast<int>(tile_detections[i * 7 + 0]);
        
        if (tile_id < 0 || tile_id >= num_tiles) {
            continue;  // Invalid tile ID
        }
        
        float x_offset = tile_offsets[tile_id * 2 + 0];
        float y_offset = tile_offsets[tile_id * 2 + 1];
        
        // Transform coordinates
        transformed[i * 6 + 0] = tile_detections[i * 7 + 1] + x_offset;  // x1
        transformed[i * 6 + 1] = tile_detections[i * 7 + 2] + y_offset;  // y1
        transformed[i * 6 + 2] = tile_detections[i * 7 + 3] + x_offset;  // x2
        transformed[i * 6 + 3] = tile_detections[i * 7 + 4] + y_offset;  // y2
        transformed[i * 6 + 4] = tile_detections[i * 7 + 5];              // conf
        transformed[i * 6 + 5] = tile_detections[i * 7 + 6];              // class_id
    }
    
    // Apply NMS
    std::vector<int> kept_indices(num_detections);
    int kept = 0;
    
    int result = nms_merge_detections(
        transformed.data(),
        num_detections,
        nms_threshold,
        kept_indices.data(),
        &kept);
    
    if (result != 0) {
        return result;
    }
    
    // Copy kept detections to output
    for (int i = 0; i < kept; ++i) {
        int idx = kept_indices[i];
        for (int j = 0; j < 6; ++j) {
            output[i * 6 + j] = transformed[idx * 6 + j];
        }
    }
    
    *num_kept = kept;
    return 0;
}

/**
 * Version information
 */
extern "C"
const char* nms_version() {
    return "1.0.0";
}
