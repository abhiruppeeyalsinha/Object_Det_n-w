// ════════════════════════════════════════════════════════════════════
// trt_session.cpp — TensorRT FP16 inference
// ════════════════════════════════════════════════════════════════════

#include "trt_session.hpp"

#if DRONE_HAS_TENSORRT
#include <NvInferPlugin.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cassert>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>

using namespace std;
using namespace cv;

// ── CUDA error-check macro ────────────────────────────────────────────
#define CUDA_CHECK(call)                                  \
    do                                                    \
    {                                                     \
        cudaError_t err = (call);                         \
        if (err != cudaSuccess)                           \
        {                                                 \
            std::ostringstream oss;                       \
            oss << "[CUDA] " << cudaGetErrorString(err)   \
                << " at " << __FILE__ << ":" << __LINE__; \
            throw std::runtime_error(oss.str());          \
        }                                                 \
    } while (0)

// ════════════════════════════════════════════════════════════════════
// TRTLogger
// ════════════════════════════════════════════════════════════════════
void TRTLogger::log(Severity sev, const char *msg) noexcept
{
    if (sev <= Severity::kWARNING)
        std::cerr << "[TRT] " << msg << "\n";
}

// ════════════════════════════════════════════════════════════════════
// Constructor
// ════════════════════════════════════════════════════════════════════
TRTSession::TRTSession(const std::string &engine_path,
                       const Config &cfg)
    : INPUT_W_(cfg.TRT_INPUT_W),
      INPUT_H_(cfg.TRT_INPUT_H),
      MAX_DET_(cfg.TRT_MAX_DET),
      DET_COLS_(cfg.TRT_DET_COLS),
      OUTPUT_LAYOUT_(cfg.TRT_OUTPUT_LAYOUT)
{
    // ────────────────────────────────────────────────────────────────
    // Load engine
    // ────────────────────────────────────────────────────────────────

    std::ifstream file(engine_path,
                       std::ios::binary | std::ios::ate);

    if (!file.is_open())
    {
        throw std::runtime_error(
            "[TRT] Cannot open engine: " + engine_path);
    }

    std::streamsize size = file.tellg();

    file.seekg(0, std::ios::beg);

    std::vector<char> engine_data(size);

    if (!file.read(engine_data.data(), size))
    {
        throw std::runtime_error(
            "[TRT] Failed to read engine: " + engine_path);
    }

    // ────────────────────────────────────────────────────────────────
    // TensorRT runtime
    // ────────────────────────────────────────────────────────────────

    initLibNvInferPlugins(&logger_, "");

    runtime_.reset(
        nvinfer1::createInferRuntime(logger_));

    if (!runtime_)
    {
        throw std::runtime_error(
            "[TRT] createInferRuntime failed");
    }

    engine_.reset(
        runtime_->deserializeCudaEngine(
            engine_data.data(),
            static_cast<size_t>(size)));

    if (!engine_)
    {
        throw std::runtime_error(
            "[TRT] deserializeCudaEngine failed");
    }

    context_.reset(
        engine_->createExecutionContext());

    if (!context_)
    {
        throw std::runtime_error(
            "[TRT] createExecutionContext failed");
    }

    // ────────────────────────────────────────────────────────────────
    // Resolve tensor names
    // ────────────────────────────────────────────────────────────────

    const int n_io = engine_->getNbIOTensors();

    for (int i = 0; i < n_io; ++i)
    {
        const char *name =
            engine_->getIOTensorName(i);

        auto mode =
            engine_->getTensorIOMode(name);

        if (mode == nvinfer1::TensorIOMode::kINPUT)
            input_name_ = name;
        else
            output_name_ = name;
    }

    if (input_name_.empty() || output_name_.empty())
    {
        throw std::runtime_error(
            "[TRT] Could not find input/output tensors");
    }

    // ────────────────────────────────────────────────────────────────
    // Validate datatypes
    // ────────────────────────────────────────────────────────────────

    auto in_dtype =
        engine_->getTensorDataType(
            input_name_.c_str());

    auto out_dtype =
        engine_->getTensorDataType(
            output_name_.c_str());

    if (in_dtype != nvinfer1::DataType::kHALF)
    {
        throw std::runtime_error(
            "[TRT] Input tensor is not FP16");
    }

    if (out_dtype != nvinfer1::DataType::kHALF)
    {
        std::cerr
            << "[TRT] WARNING: output dtype is not FP16\n";
    }

    // ────────────────────────────────────────────────────────────────
    // Set dynamic input shape
    // ────────────────────────────────────────────────────────────────

    nvinfer1::Dims4 in_dims{
        1,
        3,
        INPUT_H_,
        INPUT_W_};

    if (!context_->setInputShape(
            input_name_.c_str(),
            in_dims))
    {
        throw std::runtime_error(
            "[TRT] setInputShape failed");
    }

    // ────────────────────────────────────────────────────────────────
    // Allocate memory
    // ────────────────────────────────────────────────────────────────

    input_bytes_ =
        1UL * 3 * INPUT_H_ * INPUT_W_ * sizeof(__half);

    output_bytes_ =
        1UL * MAX_DET_ * DET_COLS_ * sizeof(__half);

    CUDA_CHECK(
        cudaMallocHost(&h_input_, input_bytes_));

    CUDA_CHECK(
        cudaMallocHost(&h_output_, output_bytes_));

    CUDA_CHECK(
        cudaMalloc(&d_input_, input_bytes_));

    CUDA_CHECK(
        cudaMalloc(&d_output_, output_bytes_));

    // ────────────────────────────────────────────────────────────────
    // Register tensor addresses
    // ────────────────────────────────────────────────────────────────

    if (!context_->setTensorAddress(
            input_name_.c_str(),
            d_input_))
    {
        throw std::runtime_error(
            "[TRT] setTensorAddress input failed");
    }

    if (!context_->setTensorAddress(
            output_name_.c_str(),
            d_output_))
    {
        throw std::runtime_error(
            "[TRT] setTensorAddress output failed");
    }

    // ────────────────────────────────────────────────────────────────
    // CUDA stream
    // ────────────────────────────────────────────────────────────────

    CUDA_CHECK(
        cudaStreamCreate(&stream_));

    std::cout
        << "[TRT] Engine loaded : "
        << engine_path << "\n"

        << "[TRT] Input tensor : "
        << input_name_
        << " shape=(1,3,"
        << INPUT_H_
        << ","
        << INPUT_W_
        << ")\n"

        << "[TRT] Output tensor : "
        << output_name_
        << " shape=(1,"
        << MAX_DET_
        << ","
        << DET_COLS_
        << ")\n";
}

// ════════════════════════════════════════════════════════════════════
// Destructor
// ════════════════════════════════════════════════════════════════════
TRTSession::~TRTSession()
{
    if (stream_)
        cudaStreamDestroy(stream_);

    if (h_input_)
        cudaFreeHost(h_input_);

    if (h_output_)
        cudaFreeHost(h_output_);

    if (d_input_)
        cudaFree(d_input_);

    if (d_output_)
        cudaFree(d_output_);
}

// ════════════════════════════════════════════════════════════════════
// Letterbox resize
// ════════════════════════════════════════════════════════════════════
Mat TRTSession::letterbox(const Mat &src,
                          float &scale,
                          int &pad_x,
                          int &pad_y) const
{
    const int src_w = src.cols;
    const int src_h = src.rows;

    scale = std::min(
        static_cast<float>(INPUT_W_) / src_w,
        static_cast<float>(INPUT_H_) / src_h);

    const int new_w =
        static_cast<int>(src_w * scale);

    const int new_h =
        static_cast<int>(src_h * scale);

    pad_x = (INPUT_W_ - new_w) / 2;
    pad_y = (INPUT_H_ - new_h) / 2;

    Mat resized;

    resize(src,
           resized,
           {new_w, new_h},
           0,
           0,
           INTER_LINEAR);

    Mat canvas(
        INPUT_H_,
        INPUT_W_,
        CV_8UC3,
        Scalar(114, 114, 114));

    resized.copyTo(
        canvas(Rect(
            pad_x,
            pad_y,
            new_w,
            new_h)));

    return canvas;
}

// ════════════════════════════════════════════════════════════════════
// BGR uint8 → FP16 CHW
// ════════════════════════════════════════════════════════════════════
void TRTSession::to_fp16_chw(const Mat &canvas)
{
    const int H = INPUT_H_;
    const int W = INPUT_W_;

    const int plane = H * W;

    // Planar RGB
    __half *r_ptr = h_input_;
    __half *g_ptr = h_input_ + plane;
    __half *b_ptr = h_input_ + plane * 2;

    for (int row = 0; row < H; ++row)
    {
        const uchar *src =
            canvas.ptr<uchar>(row);

        const int off = row * W;

        for (int col = 0; col < W; ++col)
        {
            const int i = off + col;

            const float b =
                src[col * 3 + 0] / 255.0f;

            const float g =
                src[col * 3 + 1] / 255.0f;

            const float r =
                src[col * 3 + 2] / 255.0f;

            r_ptr[i] = __float2half(r);
            g_ptr[i] = __float2half(g);
            b_ptr[i] = __float2half(b);
        }
    }
}

// ════════════════════════════════════════════════════════════════════
// Inference
// ════════════════════════════════════════════════════════════════════
std::vector<Detection>
TRTSession::inference(const Mat &roi_bgr,
                      float conf_thresh,
                      int orig_w,
                      int orig_h)
{
    // ────────────────────────────────────────────────────────────────
    // 1. Letterbox preprocess
    // ────────────────────────────────────────────────────────────────

    float scale;

    int pad_x;
    int pad_y;

    Mat canvas =
        letterbox(
            roi_bgr,
            scale,
            pad_x,
            pad_y);

    to_fp16_chw(canvas);

    // ────────────────────────────────────────────────────────────────
    // 2. Host → Device
    // ────────────────────────────────────────────────────────────────

    CUDA_CHECK(
        cudaMemcpyAsync(
            d_input_,
            h_input_,
            input_bytes_,
            cudaMemcpyHostToDevice,
            stream_));

    // ────────────────────────────────────────────────────────────────
    // 3. TensorRT inference
    // ────────────────────────────────────────────────────────────────

    if (!context_->enqueueV3(stream_))
    {
        throw std::runtime_error(
            "[TRT] enqueueV3 failed");
    }

    // ────────────────────────────────────────────────────────────────
    // 4. Device → Host
    // ────────────────────────────────────────────────────────────────

    CUDA_CHECK(
        cudaMemcpyAsync(
            h_output_,
            d_output_,
            output_bytes_,
            cudaMemcpyDeviceToHost,
            stream_));

    CUDA_CHECK(
        cudaStreamSynchronize(stream_));

    // ────────────────────────────────────────────────────────────────
    // 5. Parse detections
    // ────────────────────────────────────────────────────────────────

    std::vector<Detection> dets;

    dets.reserve(16);

    const __half *row = h_output_;

    auto to_input_pixels = [this](float value, int axis_size) -> float
    {
        if (value >= 0.0f && value <= 2.0f)
            return value * static_cast<float>(axis_size);

        return value;
    };

    for (int d = 0;
         d < MAX_DET_;
         ++d, row += DET_COLS_)
    {
        Detection det;

        if (OUTPUT_LAYOUT_ == TRTOutputLayout::IR_25200x9)
        {
            if (DET_COLS_ < 6)
                continue;

            const float objectness =
                __half2float(row[4]);

            float best_class_score = 0.0f;
            int best_class = 0;

            for (int c = 5; c < DET_COLS_; ++c)
            {
                const float class_score =
                    __half2float(row[c]);

                if (class_score > best_class_score)
                {
                    best_class_score = class_score;
                    best_class = c - 5;
                }
            }

            const float conf =
                objectness * best_class_score;

            if (conf < conf_thresh)
                continue;

            float raw_cx =
                to_input_pixels(__half2float(row[0]), INPUT_W_);

            float raw_cy =
                to_input_pixels(__half2float(row[1]), INPUT_H_);

            float raw_w =
                to_input_pixels(__half2float(row[2]), INPUT_W_);

            float raw_h =
                to_input_pixels(__half2float(row[3]), INPUT_H_);

            const float cx =
                (raw_cx - pad_x) / scale;

            const float cy =
                (raw_cy - pad_y) / scale;

            const float w =
                raw_w / scale;

            const float h =
                raw_h / scale;

            float x1 =
                std::clamp(cx - w * 0.5f,
                           0.f,
                           static_cast<float>(orig_w - 1));

            float y1 =
                std::clamp(cy - h * 0.5f,
                           0.f,
                           static_cast<float>(orig_h - 1));

            float x2 =
                std::clamp(cx + w * 0.5f,
                           0.f,
                           static_cast<float>(orig_w - 1));

            float y2 =
                std::clamp(cy + h * 0.5f,
                           0.f,
                           static_cast<float>(orig_h - 1));

            det.w = std::max(0.f, x2 - x1);
            det.h = std::max(0.f, y2 - y1);
            det.cx = x1 + det.w * 0.5f;
            det.cy = y1 + det.h * 0.5f;
            det.conf = conf;
            det.cls = best_class;
        }
        else
        {
            const float conf =
                __half2float(row[4]);

            if (conf < conf_thresh)
                continue;

            const float raw_x1 =
                to_input_pixels(__half2float(row[0]), INPUT_W_);

            const float raw_y1 =
                to_input_pixels(__half2float(row[1]), INPUT_H_);

            const float raw_x2 =
                to_input_pixels(__half2float(row[2]), INPUT_W_);

            const float raw_y2 =
                to_input_pixels(__half2float(row[3]), INPUT_H_);

            float x1 =
                (raw_x1 - pad_x) / scale;

            float y1 =
                (raw_y1 - pad_y) / scale;

            float x2 =
                (raw_x2 - pad_x) / scale;

            float y2 =
                (raw_y2 - pad_y) / scale;

            x1 = std::clamp(x1, 0.f, static_cast<float>(orig_w - 1));
            y1 = std::clamp(y1, 0.f, static_cast<float>(orig_h - 1));
            x2 = std::clamp(x2, 0.f, static_cast<float>(orig_w - 1));
            y2 = std::clamp(y2, 0.f, static_cast<float>(orig_h - 1));

            det.w = std::max(0.f, x2 - x1);
            det.h = std::max(0.f, y2 - y1);
            det.cx = x1 + det.w * 0.5f;
            det.cy = y1 + det.h * 0.5f;
            det.conf = conf;
            det.cls = static_cast<int>(__half2float(row[5]));
        }

        if (det.w <= 1.f || det.h <= 1.f)
            continue;

        dets.push_back(det);
    }

    return dets;
}
#else
#include <stdexcept>

TRTSession::TRTSession(const std::string &engine_path,
                       const Config &)
{
    throw std::runtime_error(
        "[TRT] TensorRT/CUDA headers are not available. "
        "Install TensorRT/CUDA or build with their include paths to use AI detection. Engine: " +
        engine_path);
}

TRTSession::~TRTSession() = default;

std::vector<Detection>
TRTSession::inference(const cv::Mat &,
                      float,
                      int,
                      int)
{
    return {};
}
#endif
