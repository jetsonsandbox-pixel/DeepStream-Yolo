/*
 * GStreamer Plugin for Frame Tiling
 * gstnvtileextract - Extract tiles from input frame for batch inference
 *
 * This plugin sits between decoder and nvinfer in the DeepStream pipeline:
 * decoder → nvtileextract → nvinfer(batch=8) → tracker → ...
 *
 * It takes a single 1920x1080 frame and outputs 8 tiles of 640x640 with overlap
 */

#include <gst/gst.h>
#include <gst/base/gstbasetransform.h>
#include <gst/video/video.h>
#include "nvbufsurface.h"
#include "gstnvdsmeta.h"
#include "nvdsinfer_tiled_config.h"
#include <cuda_runtime.h>

GST_DEBUG_CATEGORY_STATIC (gst_nvtileextract_debug);
#define GST_CAT_DEFAULT gst_nvtileextract_debug

#define GST_TYPE_NVTILEEXTRACT (gst_nvtileextract_get_type())
#define GST_NVTILEEXTRACT(obj) (G_TYPE_CHECK_INSTANCE_CAST((obj),GST_TYPE_NVTILEEXTRACT,GstNvTileExtract))

typedef struct _GstNvTileExtract {
    GstBaseTransform base_transform;
    
    // Tiling configuration
    TileConfig* tile_config;
    gint frame_width;
    gint frame_height;
    gint tile_width;
    gint tile_height;
    gint overlap;
    gint tiles_x;
    gint tiles_y;
    gint total_tiles;
    
    // GPU properties
    guint gpu_id;
    
} GstNvTileExtract;

typedef struct _GstNvTileExtractClass {
    GstBaseTransformClass base_transform_class;
} GstNvTileExtractClass;

GType gst_nvtileextract_get_type (void);

// External CUDA kernel
extern "C" {
    bool launchTileExtractionKernel(
        const void* input_frame,
        void* output_tiles,
        int frame_width,
        int frame_height,
        int tile_width,
        int tile_height,
        int tiles_x,
        int tiles_y,
        int overlap,
        int channels,
        cudaStream_t stream
    );
}

// Properties
enum {
    PROP_0,
    PROP_GPU_ID,
    PROP_FRAME_WIDTH,
    PROP_FRAME_HEIGHT,
    PROP_TILE_WIDTH,
    PROP_TILE_HEIGHT,
    PROP_OVERLAP
};

// Pad templates
static GstStaticPadTemplate sink_template = GST_STATIC_PAD_TEMPLATE (
    "sink",
    GST_PAD_SINK,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS ("video/x-raw(memory:NVMM), "
                     "format = (string) {NV12, RGBA}, "
                     "width = (int) [1,MAX], "
                     "height = (int) [1,MAX], "
                     "framerate = (fraction) [0/1,MAX]")
);

static GstStaticPadTemplate src_template = GST_STATIC_PAD_TEMPLATE (
    "src",
    GST_PAD_SRC,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS ("video/x-raw(memory:NVMM), "
                     "format = (string) {NV12, RGBA}, "
                     "width = (int) [1,MAX], "
                     "height = (int) [1,MAX], "
                     "framerate = (fraction) [0/1,MAX]")
);

#define gst_nvtileextract_parent_class parent_class
G_DEFINE_TYPE (GstNvTileExtract, gst_nvtileextract, GST_TYPE_BASE_TRANSFORM);

static void gst_nvtileextract_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec);
static void gst_nvtileextract_get_property (GObject * object, guint prop_id,
    GValue * value, GParamSpec * pspec);
static gboolean gst_nvtileextract_start (GstBaseTransform * btrans);
static gboolean gst_nvtileextract_stop (GstBaseTransform * btrans);
static GstFlowReturn gst_nvtileextract_transform_ip (GstBaseTransform * btrans,
    GstBuffer * buf);

// Class initialization
static void
gst_nvtileextract_class_init (GstNvTileExtractClass * klass)
{
    GObjectClass *gobject_class = G_OBJECT_CLASS (klass);
    GstElementClass *gstelement_class = GST_ELEMENT_CLASS (klass);
    GstBaseTransformClass *gstbasetransform_class = GST_BASE_TRANSFORM_CLASS (klass);

    gobject_class->set_property = gst_nvtileextract_set_property;
    gobject_class->get_property = gst_nvtileextract_get_property;

    g_object_class_install_property (gobject_class, PROP_GPU_ID,
        g_param_spec_uint ("gpu-id", "GPU ID", "GPU device ID",
            0, G_MAXUINT, 0, (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

    g_object_class_install_property (gobject_class, PROP_FRAME_WIDTH,
        g_param_spec_int ("frame-width", "Frame Width", "Input frame width",
            1, G_MAXINT, 1920, (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

    g_object_class_install_property (gobject_class, PROP_TILE_WIDTH,
        g_param_spec_int ("tile-width", "Tile Width", "Output tile width",
            1, G_MAXINT, 640, (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

    g_object_class_install_property (gobject_class, PROP_OVERLAP,
        g_param_spec_int ("overlap", "Overlap", "Tile overlap in pixels",
            0, G_MAXINT, 96, (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

    gst_element_class_set_static_metadata (gstelement_class,
        "NvTileExtract",
        "Video/Transform",
        "Extract tiles from frame for batch inference",
        "NVIDIA Corporation");

    gst_element_class_add_static_pad_template (gstelement_class, &src_template);
    gst_element_class_add_static_pad_template (gstelement_class, &sink_template);

    gstbasetransform_class->start = GST_DEBUG_FUNCPTR (gst_nvtileextract_start);
    gstbasetransform_class->stop = GST_DEBUG_FUNCPTR (gst_nvtileextract_stop);
    gstbasetransform_class->transform_ip = GST_DEBUG_FUNCPTR (gst_nvtileextract_transform_ip);

    GST_DEBUG_CATEGORY_INIT (gst_nvtileextract_debug, "nvtileextract", 0,
        "nvtileextract plugin");
}

// Instance initialization
static void
gst_nvtileextract_init (GstNvTileExtract * nvtileextract)
{
    nvtileextract->gpu_id = 0;
    nvtileextract->frame_width = 1920;
    nvtileextract->frame_height = 1080;
    nvtileextract->tile_width = 640;
    nvtileextract->tile_height = 640;
    nvtileextract->overlap = 96;
    nvtileextract->tile_config = nullptr;
}

// Property setters/getters
static void
gst_nvtileextract_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec)
{
    GstNvTileExtract *nvtileextract = GST_NVTILEEXTRACT (object);

    switch (prop_id) {
        case PROP_GPU_ID:
            nvtileextract->gpu_id = g_value_get_uint (value);
            break;
        case PROP_FRAME_WIDTH:
            nvtileextract->frame_width = g_value_get_int (value);
            break;
        case PROP_TILE_WIDTH:
            nvtileextract->tile_width = g_value_get_int (value);
            break;
        case PROP_OVERLAP:
            nvtileextract->overlap = g_value_get_int (value);
            break;
        default:
            G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
            break;
    }
}

static void
gst_nvtileextract_get_property (GObject * object, guint prop_id,
    GValue * value, GParamSpec * pspec)
{
    GstNvTileExtract *nvtileextract = GST_NVTILEEXTRACT (object);

    switch (prop_id) {
        case PROP_GPU_ID:
            g_value_set_uint (value, nvtileextract->gpu_id);
            break;
        case PROP_FRAME_WIDTH:
            g_value_set_int (value, nvtileextract->frame_width);
            break;
        case PROP_TILE_WIDTH:
            g_value_set_int (value, nvtileextract->tile_width);
            break;
        case PROP_OVERLAP:
            g_value_set_int (value, nvtileextract->overlap);
            break;
        default:
            G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
            break;
    }
}

// Start - initialize tile configuration
static gboolean
gst_nvtileextract_start (GstBaseTransform * btrans)
{
    GstNvTileExtract *nvtileextract = GST_NVTILEEXTRACT (btrans);

    // Initialize tile configuration
    nvtileextract->tile_config = new TileConfig(
        nvtileextract->frame_width,
        nvtileextract->frame_height,
        nvtileextract->tile_width,
        nvtileextract->overlap
    );

    GST_INFO_OBJECT (nvtileextract, "Initialized tiling: %dx%d → %d tiles (%dx%d grid)",
        nvtileextract->frame_width, nvtileextract->frame_height,
        nvtileextract->tile_config->total_tiles,
        nvtileextract->tile_config->tiles_x,
        nvtileextract->tile_config->tiles_y);

    return TRUE;
}

// Stop - cleanup
static gboolean
gst_nvtileextract_stop (GstBaseTransform * btrans)
{
    GstNvTileExtract *nvtileextract = GST_NVTILEEXTRACT (btrans);

    if (nvtileextract->tile_config) {
        delete nvtileextract->tile_config;
        nvtileextract->tile_config = nullptr;
    }

    return TRUE;
}

// Transform - extract tiles from input buffer
static GstFlowReturn
gst_nvtileextract_transform_ip (GstBaseTransform * btrans, GstBuffer * buf)
{
    GstNvTileExtract *nvtileextract = GST_NVTILEEXTRACT (btrans);

    GST_DEBUG_OBJECT (nvtileextract, "Extracting tiles from frame");

    // Get NvBufSurface from buffer
    GstMapInfo map_info;
    if (!gst_buffer_map (buf, &map_info, GST_MAP_READ)) {
        GST_ERROR_OBJECT (nvtileextract, "Failed to map buffer");
        return GST_FLOW_ERROR;
    }

    NvBufSurface *surf = (NvBufSurface *) map_info.data;

    // TODO: Implement actual tile extraction using CUDA kernel
    // This would involve:
    // 1. Allocating output buffer for 8 tiles
    // 2. Calling launchTileExtractionKernel
    // 3. Attaching tile metadata to buffer
    // 4. Modifying buffer to contain tiles instead of single frame

    GST_LOG_OBJECT (nvtileextract, "Processing frame %dx%d",
        surf->surfaceList[0].width, surf->surfaceList[0].height);

    gst_buffer_unmap (buf, &map_info);

    return GST_FLOW_OK;
}

// Plugin registration
static gboolean
plugin_init (GstPlugin * plugin)
{
    return gst_element_register (plugin, "nvtileextract", GST_RANK_PRIMARY,
        GST_TYPE_NVTILEEXTRACT);
}

GST_PLUGIN_DEFINE (
    GST_VERSION_MAJOR,
    GST_VERSION_MINOR,
    nvdsgst_tileextract,
    "NvTileExtract plugin for frame tiling",
    plugin_init,
    "1.0",
    "Proprietary",
    "GStreamer",
    "http://nvidia.com/"
)
