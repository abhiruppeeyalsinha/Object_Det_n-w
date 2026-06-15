#pragma once

#if defined(DRONE_FORCE_TENSORRT) || \
    (__has_include(<NvInfer.h>) && __has_include(<cuda_fp16.h>) && __has_include(<cuda_runtime.h>))
#define DRONE_HAS_TENSORRT 1
#else
#define DRONE_HAS_TENSORRT 0
#endif

#if DRONE_HAS_TENSORRT
#include <NvInfer.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#endif

#include <opencv2/opencv.hpp>
#include <memory>
#include <string>
#include <vector>
#include "drone_tracker.hpp"


#pragma GCC diagnostic ignored "-Wdeprecated-declarations"


// ── TRT logger ───────────────────────────────────────────────────────

#if DRONE_HAS_TENSORRT
class TRTLogger : public nvinfer1::ILogger
{
public:
    void log(Severity sev, const char *msg) noexcept override;
};

// ── Deleter helpers for TRT objects ─────────────────────────────────

struct TRTDeleter
{
    template <typename T>
    void operator()(T *obj) const
    {
        if (obj)
            obj->destroy();
    }
};

template <typename T>
using TRTUPtr = std::unique_ptr<T, TRTDeleter>;

// ── Session ──────────────────────────────────────────────────────────
class TRTSession
{
public:
    explicit TRTSession(const std::string &engine_path, const Config &cfg);
    ~TRTSession();

    // Run FP16 inference. Returns detections in ROI-local coords.
    std::vector<Detection> inference(const cv::Mat &roi_bgr, float conf_thresh, int orig_w, int orig_h);

private:
    // ── Preprocessing ───────────────────────────────────────────────
    cv::Mat letterbox(const cv::Mat &src,
                      float &scale,
                      int &pad_x,
                      int &pad_y) const;

    // BGR uint8 canvas → float16 CHW blob in h_input_
    void to_fp16_chw(const cv::Mat &canvas);

private:
    // ── Config ──────────────────────────────────────────────────────
    int INPUT_W_;
    int INPUT_H_;
    int MAX_DET_;
    int DET_COLS_;
    TRTOutputLayout OUTPUT_LAYOUT_;

    // ── TRT objects ─────────────────────────────────────────────────
    TRTLogger logger_;

    TRTUPtr<nvinfer1::IRuntime> runtime_;
    TRTUPtr<nvinfer1::ICudaEngine> engine_;
    TRTUPtr<nvinfer1::IExecutionContext> context_;

    // Tensor names
    std::string input_name_;
    std::string output_name_;

    // ── CUDA buffers ────────────────────────────────────────────────
    __half *h_input_ = nullptr;
    __half *h_output_ = nullptr;

    void *d_input_ = nullptr;
    void *d_output_ = nullptr;

    size_t input_bytes_ = 0;
    size_t output_bytes_ = 0;

    cudaStream_t stream_ = nullptr;
};
#else
class TRTSession
{
public:
    explicit TRTSession(const std::string &engine_path, const Config &cfg);
    ~TRTSession();

    std::vector<Detection> inference(const cv::Mat &roi_bgr,
                                     float conf_thresh,
                                     int orig_w,
                                     int orig_h);
};
#endif
