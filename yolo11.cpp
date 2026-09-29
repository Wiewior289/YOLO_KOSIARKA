/**
 * ============================================================================
 * YOLO11 NCNN Zero-Copy AHardwareBuffer Engine for Autonomous Robotic Mower
 * Platform: Samsung Galaxy S9+ (Samsung Exynos 9810 + Mali-G72 MP18 Bifrost GPU)
 * Actuator MCU: STM32F405 (UART/CAN control interface)
 * Framework: Tencent ncnn (Vulkan GPU Acceleration & AHardwareBuffer Zero-Copy)
 * ============================================================================
 */

#include "yolo11.h"

#include <android/log.h>
#include <algorithm>
#include <cmath>
#include <cstring>

#define TAG "Yolo11_ZeroCopy_MaliG72"
#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG, TAG, __VA_ARGS__)
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, TAG, __VA_ARGS__)
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN, TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)

Yolo11::Yolo11()
    : vkdev(nullptr),
      binaryop_scale(nullptr),
      cached_ahb_pipeline(nullptr),
      cached_ahb_format(0),
      cached_ahb_width(0),
      cached_ahb_height(0),
      cached_ahb_rotate(1),
      target_w(320),
      target_h(320),
      num_classes(80),
      model_loaded(false),
      use_vulkan_compute(true),
      emergency_stop_triggered(false),
      last_top_score(0.0f),
      last_top_class(-1)
{
    std::memset(dfl_distribution_scratchpad, 0, sizeof(dfl_distribution_scratchpad));
}

Yolo11::~Yolo11()
{
    clear_ahb_cache();
    net.clear();
}

void Yolo11::clear_ahb_cache()
{
    if (cached_ahb_pipeline) {
        LOGI("Releasing cached Vulkan ImportAndroidHardwareBufferPipeline...");
        delete cached_ahb_pipeline;
        cached_ahb_pipeline = nullptr;
    }
    cached_ahb_format = 0;
    cached_ahb_width = 0;
    cached_ahb_height = 0;
    cached_ahb_rotate = 1;
}

int Yolo11::load(AAssetManager* mgr, const char* modeltype, int target_size, bool use_gpu)
{
    if (!mgr || !modeltype) {
        LOGE("Invalid AssetManager or modeltype pointer");
        return -1;
    }

    clear_ahb_cache();
    net.clear();

    target_w = target_size;
    target_h = target_size;
    use_vulkan_compute = use_gpu;
    model_loaded = false;
    emergency_stop_triggered = false;
    last_top_score = 0.0f;
    last_top_class = -1;

    // Check Vulkan availability on Exynos 9810 (Mali-G72 MP18)
    int gpu_count = ncnn::get_gpu_count();
    if (use_gpu && gpu_count > 0) {
        vkdev = ncnn::get_gpu_device(0);
        if (!vkdev) {
            LOGW("GPU device 0 returned null, falling back to CPU");
            use_vulkan_compute = false;
        } else {
            const int ahb_ext = vkdev->info.support_VK_ANDROID_external_memory_android_hardware_buffer();
            if (ahb_ext <= 0) {
                LOGW("Vulkan driver lacks VK_ANDROID_external_memory_android_hardware_buffer (ver %d)", ahb_ext);
            } else {
                LOGI("Vulkan AHB zero-copy extension detected: version %d on Mali-G72", ahb_ext);
            }

            net.set_vulkan_device(vkdev);
        }
    } else {
        use_vulkan_compute = false;
        vkdev = nullptr;
    }

    // Configure Bifrost optimizations for Mali-G72 MP18
    net.opt.use_vulkan_compute  = use_vulkan_compute;
    net.opt.use_fp16_packed     = use_vulkan_compute;
    net.opt.use_fp16_storage    = use_vulkan_compute;
    net.opt.use_fp16_arithmetic = use_vulkan_compute;
    net.opt.lightmode           = true;
    net.opt.num_threads         = 4;

    char parampath[256];
    char binpath[256];
    snprintf(parampath, sizeof(parampath), "yolo11%s.param", modeltype);
    snprintf(binpath, sizeof(binpath), "yolo11%s.bin", modeltype);

    LOGI("Loading YOLO11: %s, %s (target_size: %dx%d, Vulkan: %d)",
         parampath, binpath, target_w, target_h, use_vulkan_compute ? 1 : 0);

    int ret_param = net.load_param(mgr, parampath);
    if (ret_param != 0) {
        LOGE("Failed to load param: %s (code %d)", parampath, ret_param);
        return -2;
    }

    int ret_bin = net.load_model(mgr, binpath);
    if (ret_bin != 0) {
        LOGE("Failed to load model: %s (code %d)", binpath, ret_bin);
        return -3;
    }

    // Precompute grid locations and strides for detection heads (strides 8, 16, 32)
    generate_grids_and_stride(target_w, target_h);

    const size_t total_anchors = grid_strides.size();
    proposal_scratchpad.reserve(total_anchors);
    candidates_scratchpad.reserve(256);
    nms_indices_scratchpad.reserve(256);

    model_loaded = true;
    LOGI("YOLO11 loaded successfully. Total anchors: %zu", total_anchors);
    return 0;
}

void Yolo11::generate_grids_and_stride(int target_w, int target_h)
{
    grid_strides.clear();
    const int strides[3] = {8, 16, 32};

    size_t total_count = 0;
    for (int s : strides) {
        int gw = (target_w + s - 1) / s;
        int gh = (target_h + s - 1) / s;
        total_count += static_cast<size_t>(gw * gh);
    }
    grid_strides.reserve(total_count);

    for (int stride : strides) {
        int num_grid_x = target_w / stride;
        int num_grid_y = target_h / stride;
        for (int g1 = 0; g1 < num_grid_y; g1++) {
            for (int g0 = 0; g0 < num_grid_x; g0++) {
                GridAndStride gs;
                gs.grid0 = g0;
                gs.grid1 = g1;
                gs.stride = stride;
                grid_strides.push_back(gs);
            }
        }
    }
}

int Yolo11::detect_ahb(AHardwareBuffer* ahb, float conf_thresh, float nms_thresh,
                       std::vector<Object>& objects, int rotate_type)
{
    if (!model_loaded) {
        LOGE("detect_ahb failed: Model not loaded");
        return -1;
    }

    if (!ahb) {
        LOGE("detect_ahb failed: Null AHardwareBuffer");
        return -2;
    }

    if (!use_vulkan_compute || !vkdev) {
        LOGE("detect_ahb requires active VulkanDevice on Mali-G72");
        return -3;
    }

    // ------------------------------------------------------------------------
    // Step 1: Manage AHardwareBuffer Pipeline Cache with Hardware Rotation
    // ------------------------------------------------------------------------
    AHardwareBuffer_Desc desc;
    AHardwareBuffer_describe(ahb, &desc);

    if (!cached_ahb_pipeline ||
        cached_ahb_format != static_cast<int>(desc.format) ||
        cached_ahb_width  != static_cast<int>(desc.width) ||
        cached_ahb_height != static_cast<int>(desc.height) ||
        cached_ahb_rotate != rotate_type)
    {
        if (cached_ahb_pipeline) {
            delete cached_ahb_pipeline;
            cached_ahb_pipeline = nullptr;
        }

        LOGI("Compiling ImportAndroidHardwareBufferPipeline: fmt=%u (%ux%u) rotate=%d...",
             desc.format, desc.width, desc.height, rotate_type);

        ncnn::VkAndroidHardwareBufferImageAllocator init_alloc(vkdev, ahb);
        cached_ahb_pipeline = new ncnn::ImportAndroidHardwareBufferPipeline(vkdev);

        ncnn::Option opt = net.opt;
        opt.use_vulkan_compute = true;
        opt.use_fp16_storage = false;

        // type_to = 1 (PIXEL_RGB), rotate_from = rotate_type (1: 0 deg, 6: 90 deg, 3: 180 deg, 8: 270 deg)
        // Hardware YCbCr -> RGB, rotation and scaling executed inside Mali-G72 compute shader
        int pipe_ret = cached_ahb_pipeline->create(&init_alloc, 1, rotate_type, target_w, target_h, opt);
        if (pipe_ret != 0) {
            LOGE("Failed to create ImportAndroidHardwareBufferPipeline: %d", pipe_ret);
            delete cached_ahb_pipeline;
            cached_ahb_pipeline = nullptr;
            return -4;
        }

        cached_ahb_format = static_cast<int>(desc.format);
        cached_ahb_width  = static_cast<int>(desc.width);
        cached_ahb_height = static_cast<int>(desc.height);
        cached_ahb_rotate = rotate_type;
        LOGI("ImportAndroidHardwareBufferPipeline compiled and cached successfully!");
    }

    // ------------------------------------------------------------------------
    // Step 2: GPU-side Hardware Import (YCbCr -> RGB + Rotate + Bicubic Scale)
    // ------------------------------------------------------------------------
    ncnn::VkAllocator* blob_allocator = vkdev->acquire_blob_allocator();
    ncnn::VkAllocator* staging_allocator = vkdev->acquire_staging_allocator();

    ncnn::Option opt = net.opt;
    opt.blob_vkallocator = blob_allocator;
    opt.staging_vkallocator = staging_allocator;

    ncnn::VkAndroidHardwareBufferImageAllocator frame_alloc(vkdev, ahb);
    ncnn::VkImageMat src = ncnn::VkImageMat::from_android_hardware_buffer(&frame_alloc);

    ncnn::VkMat dst;
    dst.create(target_w, target_h, 3, (size_t)4u, 1, blob_allocator);

    ncnn::VkCompute cmd(vkdev);
    cmd.record_import_android_hardware_buffer(cached_ahb_pipeline, src, dst);

    // Download the scaled 320x320 RGB buffer to host for rock-solid normalization and extractor feeding
    ncnn::Mat in;
    cmd.record_download(dst, in, opt);
    cmd.submit_and_wait();

    vkdev->reclaim_blob_allocator(blob_allocator);
    vkdev->reclaim_staging_allocator(staging_allocator);

    // Standard fast NEON normalization: divide by 255.0f
    const float norm_vals[3] = {1.0f / 255.0f, 1.0f / 255.0f, 1.0f / 255.0f};
    in.substract_mean_normalize(0, norm_vals);

    // ------------------------------------------------------------------------
    // Step 3: Run YOLO11 GPU Inference on Mali-G72 MP18
    // ------------------------------------------------------------------------
    ncnn::Extractor ex = net.create_extractor();
    ex.input("in0", in);

    ncnn::Mat out;
    int ret_extract = ex.extract("out0", out);
    if (ret_extract != 0) {
        LOGE("ex.extract('out0') failed with code: %d", ret_extract);
        return -5;
    }

    // ------------------------------------------------------------------------
    // Step 4: Parse DFL & Class Logits (Adaptive 2D/3D Tensor Shape Support)
    // ------------------------------------------------------------------------
    proposal_scratchpad.clear();
    decode_dfl_and_filter(out, conf_thresh);

    // ------------------------------------------------------------------------
    // Step 5: Non-Maximum Suppression (NMS) & 3-Zone Safety Classification
    // ------------------------------------------------------------------------
    objects.clear();
    emergency_stop_triggered = false;
    nms_sorted_bboxes(nms_thresh, objects);

    static int s_frame_log = 0;
    if (++s_frame_log % 30 == 0) {
        LOGI("Inference OK: out(dims=%d, w=%d, h=%d, c=%d) | top_score=%.3f (class=%d) | obj_count=%zu",
             out.dims, out.w, out.h, out.c, last_top_score, last_top_class, objects.size());
    }

    return 0;
}

void Yolo11::decode_dfl_and_filter(const ncnn::Mat& out_blob, float conf_thresh)
{
    last_top_score = 0.0f;
    last_top_class = -1;

    if (out_blob.empty()) {
        LOGE("decode_dfl_and_filter: out_blob is empty!");
        return;
    }

    // Normalize out_blob to 2D matrix where rows = anchors (2100) and cols = features (144)
    ncnn::Mat out_flat;
    if (out_blob.dims == 3) {
        if (out_blob.c == 1) out_flat = out_blob.reshape(out_blob.w, out_blob.h);
        else if (out_blob.w == 1) out_flat = out_blob.reshape(out_blob.h, out_blob.c);
        else out_flat = out_blob.reshape(out_blob.w * out_blob.h, out_blob.c);
    } else {
        out_flat = out_blob;
    }

    // If anchors are in columns (e.g. w=2100, h=144), transpose to (w=144, h=2100)
    if (out_flat.w > out_flat.h && out_flat.h >= 68) {
        ncnn::Mat out_t;
        out_t.create(out_flat.h, out_flat.w, (size_t)4u);
        for (int r = 0; r < out_flat.h; r++) {
            const float* src_row = out_flat.row(r);
            for (int c = 0; c < out_flat.w; c++) {
                out_t.row(c)[r] = src_row[c];
            }
        }
        out_flat = out_t;
    }

    const int total_anchors = out_flat.h;
    if (total_anchors <= 0 || out_flat.w < 68) {
        LOGW("decode_dfl_and_filter: invalid shape out_flat(w=%d, h=%d, dims=%d)",
             out_flat.w, out_flat.h, out_flat.dims);
        return;
    }

    const int cls_count = out_flat.w - 64;

    for (int i = 0; i < total_anchors && i < static_cast<int>(grid_strides.size()); i++) {
        const float* feat_ptr = out_flat.row(i);
        const float* cls_ptr = feat_ptr + 64;

        // Find best class score via fast sigmoid
        int best_class_id = -1;
        float best_score = -1.0f;

        for (int c = 0; c < cls_count; c++) {
            float logit = cls_ptr[c];
            float score = 1.0f / (1.0f + std::exp(-logit));
            if (score > best_score) {
                best_score = score;
                best_class_id = c;
            }
        }

        // Track highest raw prediction score in the whole frame for live diagnostics
        if (best_score > last_top_score) {
            last_top_score = best_score;
            last_top_class = best_class_id;
        }

        if (best_score < conf_thresh) {
            continue; // Fast rejection
        }

        // Decode 64 DFL values (4 coordinates x 16 distribution bins)
        float pred_ltrb[4];
        for (int k = 0; k < 4; k++) {
            const float* dfl_reg = feat_ptr + (k * REG_MAX);

            float max_val = dfl_reg[0];
            for (int j = 1; j < REG_MAX; j++) {
                if (dfl_reg[j] > max_val) max_val = dfl_reg[j];
            }

            float sum_exp = 0.0f;
            for (int j = 0; j < REG_MAX; j++) {
                dfl_distribution_scratchpad[j] = std::exp(dfl_reg[j] - max_val);
                sum_exp += dfl_distribution_scratchpad[j];
            }

            const float inv_sum = 1.0f / sum_exp;
            float expected_val = 0.0f;
            for (int j = 0; j < REG_MAX; j++) {
                expected_val += static_cast<float>(j) * (dfl_distribution_scratchpad[j] * inv_sum);
            }
            pred_ltrb[k] = expected_val;
        }

        const float stride = static_cast<float>(grid_strides[i].stride);
        const float grid0  = static_cast<float>(grid_strides[i].grid0);
        const float grid1  = static_cast<float>(grid_strides[i].grid1);

        float x0 = std::max(0.0f, std::min(static_cast<float>(target_w), (grid0 + 0.5f - pred_ltrb[0]) * stride));
        float y0 = std::max(0.0f, std::min(static_cast<float>(target_h), (grid1 + 0.5f - pred_ltrb[1]) * stride));
        float x1 = std::max(0.0f, std::min(static_cast<float>(target_w), (grid0 + 0.5f + pred_ltrb[2]) * stride));
        float y1 = std::max(0.0f, std::min(static_cast<float>(target_h), (grid1 + 0.5f + pred_ltrb[3]) * stride));

        float bw = std::max(0.0f, x1 - x0);
        float bh = std::max(0.0f, y1 - y0);

        if (bw < 1.0f || bh < 1.0f) {
            continue;
        }

        Object obj;
        obj.label = best_class_id;
        obj.prob = best_score;
        obj.x = x0;
        obj.y = y0;
        obj.w = bw;
        obj.h = bh;
        obj.zone_id = ZONE_REACTION;

        proposal_scratchpad.push_back(obj);
    }
}

void Yolo11::nms_sorted_bboxes(float nms_thresh, std::vector<Object>& output)
{
    if (proposal_scratchpad.empty()) {
        return;
    }

    std::sort(proposal_scratchpad.begin(), proposal_scratchpad.end(),
              [](const Object& a, const Object& b) {
                  return a.prob > b.prob;
              });

    candidates_scratchpad.clear();

    const size_t total = proposal_scratchpad.size();
    for (size_t i = 0; i < total; i++) {
        const Object& a = proposal_scratchpad[i];

        bool keep = true;
        for (size_t j = 0; j < candidates_scratchpad.size(); j++) {
            const Object& b = candidates_scratchpad[j];

            if (a.label != b.label) {
                continue;
            }

            float ix1 = std::max(a.x, b.x);
            float iy1 = std::max(a.y, b.y);
            float ix2 = std::min(a.x + a.w, b.x + b.w);
            float iy2 = std::min(a.y + a.h, b.y + b.h);

            float iw = std::max(0.0f, ix2 - ix1);
            float ih = std::max(0.0f, iy2 - iy1);

            float inter_area = iw * ih;
            float union_area = a.w * a.h + b.w * b.h - inter_area;

            if (union_area > 0.0f) {
                float iou = inter_area / union_area;
                if (iou > nms_thresh) {
                    keep = false;
                    break;
                }
            }
        }

        if (keep) {
            candidates_scratchpad.push_back(a);
            if (candidates_scratchpad.size() >= 64) {
                break;
            }
        }
    }

    output.clear();
    emergency_stop_triggered = false;

    for (size_t i = 0; i < candidates_scratchpad.size(); i++) {
        Object obj = candidates_scratchpad[i];
        float bottom_y = obj.y + obj.h;
        float norm_bottom_y = bottom_y / static_cast<float>(target_h);

        if (norm_bottom_y > 0.70f) {
            obj.zone_id = ZONE_CRITICAL;
            emergency_stop_triggered = true;
        } else if (norm_bottom_y >= 0.35f) {
            obj.zone_id = ZONE_REACTION;
        } else {
            obj.zone_id = ZONE_HORIZON;
        }

        output.push_back(obj);
    }
}

int Yolo11::pack_detections_flat(const std::vector<Object>& objects, float* out_buffer, int max_capacity)
{
    if (!out_buffer || max_capacity < 1) {
        return 0;
    }

    int count = static_cast<int>(objects.size());
    const int max_objects = (max_capacity - 1) / 7;
    if (count > max_objects) {
        count = max_objects;
    }

    out_buffer[0] = static_cast<float>(count);

    const float inv_w = 1.0f / static_cast<float>(target_w);
    const float inv_h = 1.0f / static_cast<float>(target_h);

    for (int i = 0; i < count; i++) {
        const int base = 1 + (i * 7);
        out_buffer[base + 0] = static_cast<float>(objects[i].zone_id);
        out_buffer[base + 1] = static_cast<float>(objects[i].label);
        out_buffer[base + 2] = objects[i].prob;
        out_buffer[base + 3] = objects[i].x * inv_w;
        out_buffer[base + 4] = objects[i].y * inv_h;
        out_buffer[base + 5] = objects[i].w * inv_w;
        out_buffer[base + 6] = objects[i].h * inv_h;
    }

    return count;
}
