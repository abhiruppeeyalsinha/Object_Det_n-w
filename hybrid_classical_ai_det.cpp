// #include "Includes.hpp"
#include <NvInfer.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <chrono>
#include <cmath>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <filesystem>
#include <chrono>
#include <iomanip>
#include <fstream>
#include <algorithm>
#include <cmath>
#include <thread>
#include <mutex>
#include <queue>
#include <condition_variable>
#include <atomic>
#include <cctype>
#include <cstdlib>
#include <limits>
#include <sstream>
#include <unistd.h>
#include <memory>
#include <stdexcept>



using namespace std::chrono;
using namespace std;
using namespace cv;
using namespace nvinfer1;
namespace fs = std::filesystem;

static constexpr int INPUT_W = 640;
static constexpr int INPUT_H = 640;
static constexpr float DEFAULT_CONF_THRESHOLD = 0.25f;
static constexpr float DEFAULT_NMS_THRESHOLD = 0.45f;

#define CUDA_CHECK(call)                                           \
    do                                                             \
    {                                                              \
        cudaError_t err = (call);                                  \
        if (err != cudaSuccess)                                    \
        {                                                          \
            ostringstream cuda_error;                              \
            cuda_error << "CUDA Error: " << cudaGetErrorString(err) \
                       << " at " << __FILE__ << ":" << __LINE__;   \
            throw runtime_error(cuda_error.str());                 \
        }                                                          \
    } while (0)

class Logger : public ILogger
{
public:
    void log(Severity severity, const char* msg) noexcept override
    {
        if (severity <= Severity::kWARNING)
        {
            cerr << "[TrT] " << msg << endl;
        }
    }
};

vector<string> loadClassNames(const string& path)
{
    vector<string> classes;
    ifstream file(path);
    if (!file)
    {
        throw runtime_error("Cannot open class names file: " + path);
    }

    string line;
    while (getline(file, line))
    {
        if (!line.empty() && line.back() == '\r')
        {
            line.pop_back();
        }
        if (!line.empty())
        {
            classes.push_back(line);
        }
    }
    return classes;
}

enum class DetMode
{
    DoG,
    HoughCircles,
    Hybrid,
    FPN,
    AI_Classical_Consensus
};

enum class PriorMode
{
    Size,
    Contrast,
    Center
};

struct TrackerConfig
{
    float CONF_THRESH = 0.5f;
    float HYBRID_LOG_WEIGHT = 0.5f;
    int FPN_LEVELS = 5;
    float DOG_THRESHOLD = -1.0f;
    float DOG_STDDEV_FACTOR = 1.25f;
    float MIN_DOG_THRESHOLD = 8.0f;
    float MAX_DOG_THRESHOLD = 48.0f;
    double MIN_AREA_FRAC = 0.00002;
    double MAX_AREA_FRAC = 0.45;
    double MIN_EXTENT = 0.04;
    double MIN_ASPECT = 0.08;
    double MAX_ASPECT = 12.0;
    double MAX_VERTICAL_POSITION = 0.95;
    int MIN_CONTOUR_AREA = 4;
    int HOUGH_MIN_RADIUS = 0;
    int HOUGH_MAX_RADIUS = 0;
    int HOUGH_MIN_DIST = 0;
    double HOUGH_PARAM1 = 70.0;
    double HOUGH_PARAM2 = 22.0;
};

struct BBox
{
    int x, y, w, h;
    float score = 0.0f;
    int class_id = -1;
    string class_name = "unknown";
    string type = "detection";
    double area = 0.0;
    double aspect = 0.0;
    double extent = 0.0;
    double dog_mean = 0.0;

    int cx() const { return x + w / 2; }
    int cy() const { return y + h / 2; }
    Rect rect() const { return Rect(x, y, w, h); }
};

struct LetterboxInfo
{
    float scale;
    float pad_x;
    float pad_y;
    int resized_w;
    int resized_h;
};

Mat letterbox(const Mat& image, LetterboxInfo& info)
{
    if (image.empty())
    {
        throw invalid_argument("Cannot letterbox an empty image");
    }

    const int original_w = image.cols;
    const int original_h = image.rows;
    if (original_w <= 0 || original_h <= 0)
    {
        throw invalid_argument("Invalid image dimensions for letterbox");
    }

    const float scale = min(static_cast<float>(INPUT_W) / original_w, static_cast<float>(INPUT_H) / original_h);
    int resized_w = static_cast<int>(round(original_w * scale));
    int resized_h = static_cast<int>(round(original_h * scale));

    Mat resized;
    resize(image, resized, Size(resized_w, resized_h), 0, 0, INTER_LINEAR);
    Mat output(INPUT_H, INPUT_W, CV_8UC3, Scalar(114, 114, 114));

    const int pad_x = (INPUT_W - resized_w) / 2;
    const int pad_y = (INPUT_H - resized_h) / 2;

    resized.copyTo(output(Rect(pad_x, pad_y, resized_w, resized_h)));

    info.scale = scale;
    info.pad_x = static_cast<float>(pad_x);
    info.pad_y = static_cast<float>(pad_y);
    info.resized_w = resized_w;
    info.resized_h = resized_h;

    return output;
}

void convertToFP16(const Mat& image, vector<__half>& host_input)
{
    const int total_pixels = INPUT_W * INPUT_H;
    host_input.resize(3 * total_pixels);

    for (int y = 0; y < INPUT_H; ++y)
    {
        for (int x = 0; x < INPUT_W; ++x)
        {
            const Vec3b& pixel = image.at<Vec3b>(y, x);
            const int index = y * INPUT_W + x;

            const float r = static_cast<float>(pixel[2]) / 255.0f;
            const float g = static_cast<float>(pixel[1]) / 255.0f;
            const float b = static_cast<float>(pixel[0]) / 255.0f;

            host_input[0 * total_pixels + index] = __float2half(r);
            host_input[1 * total_pixels + index] = __float2half(g);
            host_input[2 * total_pixels + index] = __float2half(b);
        }
    }
}

class TRTDetector
{
private:
    Logger logger;
    unique_ptr<IRuntime> runtime;
    unique_ptr<ICudaEngine> engine;
    unique_ptr<IExecutionContext> context;

    string input_name;
    string output_name;
    int num_detections = 0;
    int output_channels = 0;
    bool layout_transposed = false; // true if [1, 8400, channels]
    float conf_threshold_ = DEFAULT_CONF_THRESHOLD;
    float nms_threshold_ = DEFAULT_NMS_THRESHOLD;

    void* device_input = nullptr;
    void* device_output = nullptr;
    size_t input_bytes = 0;
    size_t output_bytes = 0;
    vector<__half> host_input;
    vector<__half> host_output;
    cudaStream_t stream = nullptr;

    static bool has_dynamic_dim(const Dims& dims)
    {
        for (int i = 0; i < dims.nbDims; ++i)
        {
            if (dims.d[i] < 0) return true;
        }
        return false;
    }

    static size_t tensor_element_count(const Dims& dims)
    {
        if (dims.nbDims <= 0) throw runtime_error("Tensor has invalid dimensions");

        size_t count = 1;
        for (int i = 0; i < dims.nbDims; ++i)
        {
            if (dims.d[i] <= 0) throw runtime_error("Tensor shape contains unresolved or invalid dimensions");
            count *= static_cast<size_t>(dims.d[i]);
        }
        return count;
    }

public:
    TRTDetector(const string& engine_path, float conf_threshold, float nms_threshold)
        : conf_threshold_(conf_threshold), nms_threshold_(nms_threshold)
    {
        loadEngine(engine_path);
        CUDA_CHECK(cudaMalloc(&device_input, input_bytes));
        CUDA_CHECK(cudaMalloc(&device_output, output_bytes));
        CUDA_CHECK(cudaStreamCreate(&stream));
        if (!context->setTensorAddress(input_name.c_str(), device_input))
        {
            throw runtime_error("Failed to bind TensorRT input tensor: " + input_name);
        }
        if (!context->setTensorAddress(output_name.c_str(), device_output))
        {
            throw runtime_error("Failed to bind TensorRT output tensor: " + output_name);
        }
        host_input.resize(INPUT_W * INPUT_H * 3);
        host_output.resize(num_detections * output_channels);
    }

    ~TRTDetector()
    {
        if (stream) cudaStreamDestroy(stream);
        if (device_input) cudaFree(device_input);
        if (device_output) cudaFree(device_output);
    }

    void loadEngine(const string& engine_path)
    {
        ifstream file(engine_path, ios::binary);
        if (!file) throw runtime_error("Cannot open TrT engine: " + engine_path);

        file.seekg(0, ios::end);
        const streampos end_pos = file.tellg();
        if (end_pos <= 0) throw runtime_error("TensorRT engine file is empty: " + engine_path);
        const size_t size = static_cast<size_t>(end_pos);
        file.seekg(0, ios::beg);
        vector<char> buffer(size);
        file.read(buffer.data(), size);
        if (!file) throw runtime_error("Failed to read TensorRT engine: " + engine_path);

        runtime.reset(createInferRuntime(logger));
        if (!runtime) throw runtime_error("Failed to create TensorRT runtime");

        engine.reset(runtime->deserializeCudaEngine(buffer.data(), size));
        if (!engine) throw runtime_error("Failed to deserialize TensorRT engine");

        context.reset(engine->createExecutionContext());
        if (!context) throw runtime_error("Failed to create TensorRT execution context");

        const int nb_tensors = engine->getNbIOTensors();
        for (int i = 0; i < nb_tensors; ++i)
        {
            const char* tensor_name = engine->getIOTensorName(i);
            if (!tensor_name) continue;

            TensorIOMode mode = engine->getTensorIOMode(tensor_name);
            if (mode == TensorIOMode::kINPUT && input_name.empty())
            {
                input_name = tensor_name;
            }
            else if (mode == TensorIOMode::kOUTPUT && output_name.empty())
            {
                output_name = tensor_name;
            }
        }

        if (input_name.empty() || output_name.empty())
        {
            throw runtime_error("TensorRT engine must expose at least one input and one output tensor");
        }

        if (engine->getTensorDataType(input_name.c_str()) != nvinfer1::DataType::kHALF ||
            engine->getTensorDataType(output_name.c_str()) != nvinfer1::DataType::kHALF)
        {
            throw runtime_error("This detector currently expects FP16 input and FP16 output tensors");
        }

        Dims input_dims = engine->getTensorShape(input_name.c_str());
        if (has_dynamic_dim(input_dims))
        {
            Dims4 fixed_input{1, 3, INPUT_H, INPUT_W};
            if (!context->setInputShape(input_name.c_str(), fixed_input))
            {
                throw runtime_error("Failed to set TensorRT dynamic input shape to 1x3x640x640");
            }
            input_dims = fixed_input;
        }

        if (input_dims.nbDims != 4 || input_dims.d[0] != 1 || input_dims.d[1] != 3 ||
            input_dims.d[2] != INPUT_H || input_dims.d[3] != INPUT_W)
        {
            throw runtime_error("TensorRT input tensor must be NCHW FP16 with shape 1x3x640x640");
        }

        Dims output_dims = context->getTensorShape(output_name.c_str());
        if (has_dynamic_dim(output_dims))
        {
            throw runtime_error("TensorRT output shape is still dynamic after setting input shape");
        }

        if (output_dims.nbDims == 3)
        {
            if (output_dims.d[1] < output_dims.d[2])
            {
                output_channels = output_dims.d[1];
                num_detections = output_dims.d[2];
                layout_transposed = false;
            }
            else
            {
                num_detections = output_dims.d[1];
                output_channels = output_dims.d[2];
                layout_transposed = true;
            }
        }
        else
        {
            throw runtime_error("Unexpected TensorRT output dimensions format.");
        }

        if (output_channels <= 4 || num_detections <= 0)
        {
            throw runtime_error("TensorRT output must contain boxes plus at least one class score");
        }

        input_bytes = tensor_element_count(input_dims) * sizeof(__half);
        output_bytes = tensor_element_count(output_dims) * sizeof(__half);
    }

    vector<BBox> detect(const Mat& frame, const vector<string>& class_names)
    {
        if (frame.empty()) return {};

        Mat bgr_frame;
        if (frame.channels() == 1)
        {
            cvtColor(frame, bgr_frame, COLOR_GRAY2BGR);
        }
        else if (frame.channels() == 4)
        {
            cvtColor(frame, bgr_frame, COLOR_BGRA2BGR);
        }
        else
        {
            bgr_frame = frame;
        }

        LetterboxInfo info;
        Mat input = letterbox(bgr_frame, info);
        convertToFP16(input, host_input);

        CUDA_CHECK(cudaMemcpyAsync(device_input, host_input.data(), input_bytes, cudaMemcpyHostToDevice, stream));

        if (!context->enqueueV3(stream)) throw runtime_error("TrT asynchronous inference failed");

        CUDA_CHECK(cudaMemcpyAsync(host_output.data(), device_output, output_bytes, cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        vector<BBox> detections;
        const int num_classes = output_channels - 4;

        for (int i = 0; i < num_detections; ++i)
        {
            float cx, cy, w, h;
            if (!layout_transposed)
            {
                cx = __half2float(host_output[0 * num_detections + i]);
                cy = __half2float(host_output[1 * num_detections + i]);
                w = __half2float(host_output[2 * num_detections + i]);
                h = __half2float(host_output[3 * num_detections + i]);
            }
            else
            {
                cx = __half2float(host_output[i * output_channels + 0]);
                cy = __half2float(host_output[i * output_channels + 1]);
                w = __half2float(host_output[i * output_channels + 2]);
                h = __half2float(host_output[i * output_channels + 3]);
            }

            float max_class_score = 0.0f;
            int best_class_id = 0;
            for (int c = 0; c < num_classes; ++c)
            {
                float score = 0.0f;
                if (!layout_transposed)
                    score = __half2float(host_output[(4 + c) * num_detections + i]);
                else
                    score = __half2float(host_output[i * output_channels + 4 + c]);

                if (score > max_class_score)
                {
                    max_class_score = score;
                    best_class_id = c;
                }
            }

            if (max_class_score < conf_threshold_) continue;

            float x1 = cx - w * 0.5f;
            float y1 = cy - h * 0.5f;
            float x2 = cx + w * 0.5f;
            float y2 = cy + h * 0.5f;

            x1 = max(0.0f, min((x1 - info.pad_x) / info.scale, static_cast<float>(frame.cols - 1)));
            y1 = max(0.0f, min((y1 - info.pad_y) / info.scale, static_cast<float>(frame.rows - 1)));
            x2 = max(0.0f, min((x2 - info.pad_x) / info.scale, static_cast<float>(frame.cols - 1)));
            y2 = max(0.0f, min((y2 - info.pad_y) / info.scale, static_cast<float>(frame.rows - 1)));

            int width = static_cast<int>(x2 - x1);
            int height = static_cast<int>(y2 - y1);
            if (width <= 1 || height <= 1) continue;

            string c_name = (!class_names.empty() && best_class_id >= 0 &&
                             static_cast<size_t>(best_class_id) < class_names.size()) ? class_names[best_class_id] : "target";

            detections.push_back(BBox{ static_cast<int>(x1), static_cast<int>(y1), width, height, max_class_score, best_class_id, c_name, "ai_target", (double)(width * height), (double)width / height, 1.0, 0.0 });
        }

        if (detections.empty()) return {};

        vector<Rect> rects;
        vector<float> scores;
        rects.reserve(detections.size());
        scores.reserve(detections.size());
        for (const auto& det : detections)
        {
            rects.push_back(det.rect());
            scores.push_back(det.score);
        }

        vector<int> keep;
        dnn::NMSBoxes(rects, scores, conf_threshold_, nms_threshold_, keep);

        vector<BBox> filtered;
        filtered.reserve(keep.size());
        for (int idx : keep)
        {
            if (idx >= 0 && static_cast<size_t>(idx) < detections.size())
            {
                filtered.push_back(detections[static_cast<size_t>(idx)]);
            }
        }
        return filtered;
    }
};

class BlobDetector
{
public:
    BlobDetector(const TrackerConfig& cfg, int roi_w, int roi_h, PriorMode prior = PriorMode::Center, bool debug_mode = false, bool is_video = false, const string& run_dir = "")
        : cfg_(cfg), roi_w_(roi_w), roi_h_(roi_h), prior_(prior),
        debug_mode_(debug_mode), is_video_(is_video), run_dir_(run_dir) {
    }

    Mat blob_mask(const Mat& r32, float thresh = 18.f) const
    {
        Mat pos_clip, neg_clip;
        Mat pos_mask, neg_mask;
        max(r32, 0, pos_clip);
        pos_clip.convertTo(pos_clip, CV_8U);
        threshold(pos_clip, pos_mask, thresh, 255, THRESH_BINARY);

        Mat neg_r32 = -r32;
        max(neg_r32, 0, neg_clip);
        neg_clip.convertTo(neg_clip, CV_8U);
        threshold(neg_clip, neg_mask, thresh, 255, THRESH_BINARY);

        Mat out;
        bitwise_or(pos_mask, neg_mask, out);
        return out;
    }

    std::vector<BBox> apply_nms(std::vector<BBox>& boxes, float iou_threshold = 0.12f) const
    {
        if (boxes.empty()) return {};
        const Rect image_bounds(0, 0, std::max(1, roi_w_), std::max(1, roi_h_));
        sort(boxes.begin(), boxes.end(), [](const BBox& a, const BBox& b) { return a.score > b.score; });
        vector<BBox> result;
        vector<bool> suppressed(boxes.size(), false);

        for (size_t i = 0; i < boxes.size(); ++i)
        {
            if (suppressed[i]) continue;
            result.push_back(boxes[i]);
            for (size_t j = i + 1; j < boxes.size(); ++j)
            {
                if (suppressed[j]) continue;
                Rect a = boxes[i].rect() & image_bounds;
                Rect b = boxes[j].rect() & image_bounds;
                if (a.empty() || b.empty()) continue;
                Rect intersection = a & b;
                float intersection_area = intersection.area();
                float union_area = a.area() + b.area() - intersection_area;
                float iou = (union_area > 0) ? (intersection_area / union_area) : 0.0f;
                if (iou > iou_threshold) suppressed[j] = true;
            }
        }
        return result;
    }

    std::vector<BBox> select_all(const Mat& gray, const Mat& fused_mask, const Mat& dog_conf, float hybrid_w, const string& box_type = "dog_detected") const
    {
        std::vector<std::vector<Point>> contours;
        findContours(fused_mask, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
        const double roi_area = std::max(1.0, static_cast<double>(gray.cols) * gray.rows);
        const double min_area = std::max(static_cast<double>(cfg_.MIN_CONTOUR_AREA), roi_area * cfg_.MIN_AREA_FRAC);
        const double max_rel_percent = cfg_.MAX_AREA_FRAC * 100.0;
        std::vector<BBox> candidates;

        for (auto& cnt : contours)
        {
            double area = contourArea(cnt);
            if (area <= min_area) continue;
            Rect r = boundingRect(cnt);
            if (r.width <= 0 || r.height <= 0 || r.x < 0 || r.y < 0 || r.x + r.width > gray.cols || r.y + r.height > gray.rows) continue;
            if (r.y > gray.rows * cfg_.MAX_VERTICAL_POSITION) continue;

            double aspect = static_cast<double>(r.width) / r.height;
            double extent = area / (r.width * r.height);
            double rel = (area / roi_area) * 100.0;
            if (!(cfg_.MIN_ASPECT < aspect && aspect < cfg_.MAX_ASPECT &&
                  extent > cfg_.MIN_EXTENT && rel < max_rel_percent)) continue;

            double dog_mean = dog_conf.empty() ? 0.0 : mean(dog_conf(r))[0];
            double dog_n = std::min(dog_mean / 50.0, 1.0);
            double score = area * (0.5 + hybrid_w * dog_n) * prior_weight(r, gray.size());
            candidates.push_back(BBox{ r.x, r.y, r.width, r.height, (float)score, -1, "classical_blob", box_type, area, aspect, extent, dog_mean });
        }
        return apply_nms(candidates, 0.18f);
    }

    vector<BBox> detect_dog_all(const Mat& roi) const
    {
        if (roi.empty()) return {};
        Mat gray;
        if (roi.channels() == 3) cvtColor(roi, gray, COLOR_BGR2GRAY);
        else if (roi.channels() == 4) cvtColor(roi, gray, COLOR_BGRA2GRAY);
        else gray = roi.clone();

        Mat g1a, g2a, dog_a;
        GaussianBlur(gray, g1a, { 3, 3 }, 0.8);
        GaussianBlur(gray, g2a, { 7, 7 }, 2.0);
        subtract(g1a, g2a, dog_a, noArray(), CV_16S);

        Mat dog_a32;
        dog_a.convertTo(dog_a32, CV_32F);
        Mat abs_a, dog_conf;
        absdiff(dog_a32, Scalar(0), abs_a);
        dog_conf = abs_a;
        Mat m1 = blob_mask(dog_a32, adaptive_response_threshold(dog_conf));
        cleanup_mask(m1);

        return select_all(gray, m1, dog_conf, 0.0f, "dog_detected");
    }

    vector<BBox> detect_fpn_all(const Mat& img) const
    {
        if (img.empty()) return {};
        Mat gray;
        if (img.channels() == 3) cvtColor(img, gray, COLOR_BGR2GRAY);
        else if (img.channels() == 4) cvtColor(img, gray, COLOR_BGRA2GRAY);
        else gray = img.clone();

        vector<BBox> all_pyramid_candidates;
        const int min_dim = std::max(1, std::min(gray.cols, gray.rows));
        const int levels = std::clamp(cfg_.FPN_LEVELS, 1, 8);
        vector<pair<double, double>> scales;
        scales.reserve(static_cast<size_t>(levels));
        for (int i = 0; i < levels; ++i)
        {
            const double t = (levels == 1) ? 0.0 : static_cast<double>(i) / (levels - 1);
            const double sigma1 = 0.8 + t * std::max(1.5, min_dim / 220.0);
            scales.push_back({ sigma1, sigma1 * 2.0 });
        }

        for (size_t s = 0; s < scales.size(); ++s)
        {
            int k1 = static_cast<int>(scales[s].first * 3) | 1;
            int k2 = static_cast<int>(scales[s].second * 3) | 1;
            if (k1 < 3) k1 = 3;
            if (k2 < k1 + 2) k2 = k1 + 2;

            Mat g1, g2, dog;
            GaussianBlur(gray, g1, Size(k1, k1), scales[s].first);
            GaussianBlur(gray, g2, Size(k2, k2), scales[s].second);
            subtract(g1, g2, dog, noArray(), CV_16S);

            Mat dog32;
            dog.convertTo(dog32, CV_32F);
            Mat abs_d, dog_conf;
            absdiff(dog32, Scalar(0), abs_d);
            dog_conf = abs_d;
            Mat mask = blob_mask(dog32, adaptive_response_threshold(dog_conf));
            cleanup_mask(mask);

            auto scale_boxes = select_all(gray, mask, dog_conf, 0.35f, "fpn_scale_" + to_string(s));
            all_pyramid_candidates.insert(all_pyramid_candidates.end(), scale_boxes.begin(), scale_boxes.end());
        }

        return apply_nms(all_pyramid_candidates, 0.15f);
    }

    vector<BBox> detect_hough_circles(const Mat& img) const
    {
        if (img.empty()) return {};
        Mat gray;
        if (img.channels() == 3) cvtColor(img, gray, COLOR_BGR2GRAY);
        else if (img.channels() == 4) cvtColor(img, gray, COLOR_BGRA2GRAY);
        else gray = img.clone();

        Mat smoothed;
        GaussianBlur(gray, smoothed, Size(5, 5), 1.2);
        vector<Vec3f> circles;
        const int min_dim = std::max(1, std::min(gray.cols, gray.rows));
        int min_radius = (cfg_.HOUGH_MIN_RADIUS > 0) ? cfg_.HOUGH_MIN_RADIUS : std::max(3, min_dim / 180);
        int max_radius = (cfg_.HOUGH_MAX_RADIUS > 0) ? cfg_.HOUGH_MAX_RADIUS : std::max(min_radius + 2, min_dim / 24);
        int min_dist = (cfg_.HOUGH_MIN_DIST > 0) ? cfg_.HOUGH_MIN_DIST : std::max(8, min_dim / 14);
        HoughCircles(smoothed, circles, HOUGH_GRADIENT, 1.2, min_dist,
                     cfg_.HOUGH_PARAM1, cfg_.HOUGH_PARAM2, min_radius, max_radius);

        vector<BBox> boxes;
        for (const auto& c : circles)
        {
            int cx = cvRound(c[0]);
            int cy = cvRound(c[1]);
            int r = cvRound(c[2]);
            int x = max(0, cx - r);
            int y = max(0, cy - r);
            int w = min(img.cols - x, 2 * r);
            int h = min(img.rows - y, 2 * r);
            if (w <= 1 || h <= 1) continue;
            if ((y + h) < (img.rows * cfg_.MAX_VERTICAL_POSITION))
            {
                boxes.push_back(BBox{ x, y, w, h, 0.95f, -1, "circle", "hough_circle", CV_PI * r * r, 1.0, 0.78, 25.0 });
            }
        }
        return apply_nms(boxes, 0.12f);
    }

    vector<BBox> detect_hybrid_all(const Mat& img) const
    {
        auto dog_boxes = detect_fpn_all(img);
        auto hough_boxes = detect_hough_circles(img);
        vector<BBox> combined = dog_boxes;
        for (const auto& hb : hough_boxes)
        {
            bool matched = false;
            for (auto& db : dog_boxes)
            {
                if ((db.rect() & hb.rect()).area() > 0) { matched = true; break; }
            }
            if (!matched) combined.push_back(hb);
        }
        for (auto& b : combined) b.type = "hybrid";
        return apply_nms(combined, 0.15f);
    }

private:
    float adaptive_response_threshold(const Mat& response) const
    {
        if (cfg_.DOG_THRESHOLD > 0.0f) return cfg_.DOG_THRESHOLD;

        Mat abs_response;
        absdiff(response, Scalar(0), abs_response);
        Scalar mean_val, stddev_val;
        meanStdDev(abs_response, mean_val, stddev_val);

        const double raw = mean_val[0] + cfg_.DOG_STDDEV_FACTOR * stddev_val[0];
        return static_cast<float>(std::clamp(raw,
            static_cast<double>(cfg_.MIN_DOG_THRESHOLD),
            static_cast<double>(cfg_.MAX_DOG_THRESHOLD)));
    }

    void cleanup_mask(Mat& mask) const
    {
        if (mask.empty()) return;

        const int min_dim = std::max(1, std::min(mask.cols, mask.rows));
        int kernel_size = (min_dim >= 720) ? 5 : 3;
        Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(kernel_size, kernel_size));
        morphologyEx(mask, mask, MORPH_OPEN, kernel);
        morphologyEx(mask, mask, MORPH_CLOSE, kernel);
    }

    double prior_weight(const Rect& r, Size image_size) const
    {
        const double area = static_cast<double>(r.area());
        const double frame_area = std::max(1.0, static_cast<double>(image_size.width) * image_size.height);

        switch (prior_)
        {
        case PriorMode::Size:
            return 0.75 + std::min(area / (frame_area * 0.03), 1.0) * 0.35;
        case PriorMode::Contrast:
            return 1.0;
        case PriorMode::Center:
        default:
        {
            const double dx = (r.x + r.width * 0.5) - image_size.width * 0.5;
            const double dy = (r.y + r.height * 0.5) - image_size.height * 0.5;
            const double max_dist = std::max(1.0, std::hypot(image_size.width * 0.5, image_size.height * 0.5));
            return 0.75 + (1.0 - std::min(std::hypot(dx, dy) / max_dist, 1.0)) * 0.35;
        }
        }
    }

    TrackerConfig cfg_;
    int roi_w_, roi_h_;
    PriorMode prior_;
    bool debug_mode_, is_video_;
    string run_dir_;
};

DetMode get_detection_mode()
{
    cout << "\n--- Detection Engine ---\n"
        << " [1] Filtered DoG\n"
        << " [2] Optimized Hough Circle Transform\n"
        << " [3] Hybrid (DoG + Hough)\n"
        << " [4] Feature Pyramid Network (Multi-scale DoG FPN)\n"
        << " [5] AI + Classical Spatial Consensus (Strict Hybrid)\n";
    string sel;
    cout << " Select (default 4): ";
    getline(cin, sel);
    if (sel == "1") return DetMode::DoG;
    if (sel == "2") return DetMode::HoughCircles;
    if (sel == "3") return DetMode::Hybrid;
    if (sel == "5") return DetMode::AI_Classical_Consensus;
    return DetMode::FPN;
}

string to_lower_copy(string value)
{
    transform(value.begin(), value.end(), value.begin(),
        [](unsigned char c) { return static_cast<char>(tolower(c)); });
    return value;
}

bool parse_detection_mode(const string& raw, DetMode& mode)
{
    const string value = to_lower_copy(raw);
    if (value == "1" || value == "dog" || value == "dog_filter" || value == "dog-filter")
    {
        mode = DetMode::DoG;
        return true;
    }
    if (value == "2" || value == "hough" || value == "houghcircles" || value == "hough-circles")
    {
        mode = DetMode::HoughCircles;
        return true;
    }
    if (value == "3" || value == "hybrid")
    {
        mode = DetMode::Hybrid;
        return true;
    }
    if (value == "4" || value == "fpn" || value == "pyramid")
    {
        mode = DetMode::FPN;
        return true;
    }
    if (value == "5" || value == "consensus" || value == "ai" || value == "ai-classical" || value == "ai_classical")
    {
        mode = DetMode::AI_Classical_Consensus;
        return true;
    }
    return false;
}

bool parse_prior_mode(const string& raw, PriorMode& prior)
{
    const string value = to_lower_copy(raw);
    if (value == "size")
    {
        prior = PriorMode::Size;
        return true;
    }
    if (value == "contrast")
    {
        prior = PriorMode::Contrast;
        return true;
    }
    if (value == "center" || value == "centre")
    {
        prior = PriorMode::Center;
        return true;
    }
    return false;
}

bool parse_float_arg(const string& raw, float& out)
{
    try
    {
        size_t pos = 0;
        out = stof(raw, &pos);
        return pos == raw.size() && std::isfinite(out);
    }
    catch (...)
    {
        return false;
    }
}

bool parse_int_arg(const string& raw, int& out)
{
    try
    {
        size_t pos = 0;
        out = stoi(raw, &pos);
        return pos == raw.size();
    }
    catch (...)
    {
        return false;
    }
}

bool display_available()
{
#ifdef _WIN32
    return true;
#else
    return getenv("DISPLAY") != nullptr || getenv("WAYLAND_DISPLAY") != nullptr;
#endif
}

bool can_prompt()
{
    return isatty(STDIN_FILENO);
}

bool is_integer_source(const string& value)
{
    return !value.empty() && all_of(value.begin(), value.end(),
        [](unsigned char c) { return isdigit(c); });
}

string mode_to_string(DetMode mode)
{
    switch (mode)
    {
    case DetMode::DoG:
        return "Dog";
    case DetMode::HoughCircles:
        return "Hough";
    case DetMode::Hybrid:
        return "Hybrid";
    case DetMode::AI_Classical_Consensus:
        return "Consensus";
    case DetMode::FPN:
    default:
        return "FPN";
    }
}

void print_usage(const char* exe)
{
    cerr << "Usage:\n"
         << "  " << exe << " <input_image_or_video_or_camera> [options]\n"
         << "  " << exe << " <model.engine> <classes.txt> <input> [options]   (legacy)\n\n"
         << "Options:\n"
         << "  --mode <dog|hough|hybrid|fpn|consensus>\n"
         << "  --engine <model.engine>        Required for consensus mode.\n"
         << "  --classes <classes.txt>        Optional names for consensus mode.\n"
         << "  --conf <value>                 AI confidence threshold. Default: 0.25\n"
         << "  --nms <value>                  AI NMS IoU threshold. Default: 0.45\n"
         << "  --fpn-levels <1..8>            Dynamic FPN scale count. Default: 5\n"
         << "  --dog-threshold <value>        Fixed DoG threshold. Default: adaptive.\n"
         << "  --dog-stddev <value>           Adaptive DoG sensitivity. Default: 1.25\n"
         << "  --hough-min-radius <px>        Override dynamic Hough minimum radius.\n"
         << "  --hough-max-radius <px>        Override dynamic Hough maximum radius.\n"
         << "  --hough-min-dist <px>          Override dynamic Hough center spacing.\n"
         << "  --prior <center|size|contrast> Candidate score prior. Default: center\n"
         << "  --out-dir <dir>                Output root directory. Default: Res\n"
         << "  --display                      Show OpenCV output windows. This is the default when a display exists.\n"
         << "  --no-display                   Disable OpenCV windows for batch/headless runs.\n"
         << "  --verbose                      Print per-frame timing.\n";
}

struct AppOptions
{
    string input_path;
    string engine_path;
    string classes_path;
    string output_root = "Res";
    DetMode mode = DetMode::FPN;
    PriorMode prior = PriorMode::Center;
    TrackerConfig config;
    float ai_conf_threshold = DEFAULT_CONF_THRESHOLD;
    float ai_nms_threshold = DEFAULT_NMS_THRESHOLD;
    bool mode_provided = false;
    bool display = true;
    bool verbose = false;
    bool help_requested = false;
    bool legacy_invocation = false;
};

bool consume_option_value(int& i, int argc, char** argv, string& value)
{
    if (i + 1 >= argc) return false;
    value = argv[++i];
    return true;
}

bool parse_args(int argc, char** argv, AppOptions& options)
{
    if (argc < 2)
    {
        print_usage(argv[0]);
        return false;
    }

    string first_arg = argv[1];
    if (first_arg == "--help" || first_arg == "-h")
    {
        print_usage(argv[0]);
        options.help_requested = true;
        return true;
    }

    int option_start = 2;
    if (argc >= 4 && fs::path(argv[1]).extension() == ".engine")
    {
        options.engine_path = argv[1];
        options.classes_path = argv[2];
        options.input_path = argv[3];
        options.legacy_invocation = true;
        option_start = 4;
    }
    else
    {
        options.input_path = argv[1];
    }

    for (int i = option_start; i < argc; ++i)
    {
        string arg = argv[i];
        string value;
        auto read_value = [&]() -> bool {
            size_t eq = arg.find('=');
            if (eq != string::npos)
            {
                value = arg.substr(eq + 1);
                arg = arg.substr(0, eq);
                return !value.empty();
            }
            return consume_option_value(i, argc, argv, value);
        };

        if (arg == "--help" || arg == "-h")
        {
            print_usage(argv[0]);
            options.help_requested = true;
            return true;
        }
        if (arg == "--no-display")
        {
            options.display = false;
            continue;
        }
        if (arg == "--display")
        {
            options.display = true;
            continue;
        }
        if (arg == "--verbose")
        {
            options.verbose = true;
            continue;
        }
        if (arg == "--mode" || arg.rfind("--mode=", 0) == 0)
        {
            if (!read_value() || !parse_detection_mode(value, options.mode))
            {
                cerr << "[ERROR] Invalid --mode value.\n";
                return false;
            }
            options.mode_provided = true;
            continue;
        }
        if (arg == "--engine" || arg.rfind("--engine=", 0) == 0)
        {
            if (!read_value())
            {
                cerr << "[ERROR] Missing --engine value.\n";
                return false;
            }
            options.engine_path = value;
            continue;
        }
        if (arg == "--classes" || arg.rfind("--classes=", 0) == 0)
        {
            if (!read_value())
            {
                cerr << "[ERROR] Missing --classes value.\n";
                return false;
            }
            options.classes_path = value;
            continue;
        }
        if (arg == "--out-dir" || arg.rfind("--out-dir=", 0) == 0)
        {
            if (!read_value())
            {
                cerr << "[ERROR] Missing --out-dir value.\n";
                return false;
            }
            options.output_root = value;
            continue;
        }
        if (arg == "--prior" || arg.rfind("--prior=", 0) == 0)
        {
            if (!read_value() || !parse_prior_mode(value, options.prior))
            {
                cerr << "[ERROR] Invalid --prior value.\n";
                return false;
            }
            continue;
        }

        float float_value = 0.0f;
        int int_value = 0;
        if (arg == "--conf" || arg.rfind("--conf=", 0) == 0)
        {
            if (!read_value() || !parse_float_arg(value, float_value) || float_value < 0.0f || float_value > 1.0f)
            {
                cerr << "[ERROR] --conf must be between 0 and 1.\n";
                return false;
            }
            options.ai_conf_threshold = float_value;
            continue;
        }
        if (arg == "--nms" || arg.rfind("--nms=", 0) == 0)
        {
            if (!read_value() || !parse_float_arg(value, float_value) || float_value < 0.0f || float_value > 1.0f)
            {
                cerr << "[ERROR] --nms must be between 0 and 1.\n";
                return false;
            }
            options.ai_nms_threshold = float_value;
            continue;
        }
        if (arg == "--dog-threshold" || arg.rfind("--dog-threshold=", 0) == 0)
        {
            if (!read_value() || !parse_float_arg(value, float_value) || float_value <= 0.0f)
            {
                cerr << "[ERROR] --dog-threshold must be a positive number.\n";
                return false;
            }
            options.config.DOG_THRESHOLD = float_value;
            continue;
        }
        if (arg == "--dog-stddev" || arg.rfind("--dog-stddev=", 0) == 0)
        {
            if (!read_value() || !parse_float_arg(value, float_value) || float_value < 0.0f)
            {
                cerr << "[ERROR] --dog-stddev must be a non-negative number.\n";
                return false;
            }
            options.config.DOG_STDDEV_FACTOR = float_value;
            continue;
        }
        if (arg == "--fpn-levels" || arg.rfind("--fpn-levels=", 0) == 0)
        {
            if (!read_value() || !parse_int_arg(value, int_value) || int_value < 1 || int_value > 8)
            {
                cerr << "[ERROR] --fpn-levels must be between 1 and 8.\n";
                return false;
            }
            options.config.FPN_LEVELS = int_value;
            continue;
        }
        if (arg == "--hough-min-radius" || arg.rfind("--hough-min-radius=", 0) == 0)
        {
            if (!read_value() || !parse_int_arg(value, int_value) || int_value <= 0)
            {
                cerr << "[ERROR] --hough-min-radius must be a positive integer.\n";
                return false;
            }
            options.config.HOUGH_MIN_RADIUS = int_value;
            continue;
        }
        if (arg == "--hough-max-radius" || arg.rfind("--hough-max-radius=", 0) == 0)
        {
            if (!read_value() || !parse_int_arg(value, int_value) || int_value <= 0)
            {
                cerr << "[ERROR] --hough-max-radius must be a positive integer.\n";
                return false;
            }
            options.config.HOUGH_MAX_RADIUS = int_value;
            continue;
        }
        if (arg == "--hough-min-dist" || arg.rfind("--hough-min-dist=", 0) == 0)
        {
            if (!read_value() || !parse_int_arg(value, int_value) || int_value <= 0)
            {
                cerr << "[ERROR] --hough-min-dist must be a positive integer.\n";
                return false;
            }
            options.config.HOUGH_MIN_DIST = int_value;
            continue;
        }

        cerr << "[ERROR] Unknown option: " << arg << "\n";
        return false;
    }

    if (options.input_path.empty())
    {
        cerr << "[ERROR] Missing input path.\n";
        return false;
    }
    if (options.config.HOUGH_MIN_RADIUS > 0 && options.config.HOUGH_MAX_RADIUS > 0 &&
        options.config.HOUGH_MAX_RADIUS <= options.config.HOUGH_MIN_RADIUS)
    {
        cerr << "[ERROR] --hough-max-radius must be greater than --hough-min-radius.\n";
        return false;
    }
    return true;
}

string create_run_folder(const string& output_root)
{
    auto t = chrono::system_clock::now();
    auto tt = chrono::system_clock::to_time_t(t);
    tm local_tm{};
    localtime_r(&tt, &local_tm);

    stringstream ss;
    ss << output_root << "/run_" << put_time(&local_tm, "%Y%m%d_%H%M%S");
    string base_dir = ss.str();
    string dir = base_dir;
    int suffix = 1;
    while (fs::exists(dir))
    {
        dir = base_dir + "_" + to_string(suffix++);
    }
    fs::create_directories(dir);
    return dir;
}

float centerDistance(const BBox& a, const BBox& b)
{
    float dx = static_cast<float>(a.cx() - b.cx());
    float dy = static_cast<float>(a.cy() - b.cy());
    return std::sqrt(dx * dx + dy * dy);
}

void process_frame_detections(Mat& frame, BlobDetector& detector, TRTDetector* ai_detector, DetMode det_mode,
    long long& elapsed_ms, int& det_count, vector<BBox>& out_detections, const vector<string>& class_names, bool verbose)
{
    if (frame.empty())
    {
        elapsed_ms = 0;
        det_count = 0;
        out_detections.clear();
        return;
    }

    auto total_start = chrono::high_resolution_clock::now();

    auto cls_start = chrono::high_resolution_clock::now();
    vector<BBox> classical_dets;
    switch (det_mode)
    {
    case DetMode::DoG:
        classical_dets = detector.detect_dog_all(frame);
        break;
    case DetMode::HoughCircles:
        classical_dets = detector.detect_hough_circles(frame);
        break;
    case DetMode::Hybrid:
        classical_dets = detector.detect_hybrid_all(frame);
        break;
    case DetMode::FPN:
        classical_dets = detector.detect_fpn_all(frame);
        break;
    case DetMode::AI_Classical_Consensus:
        classical_dets = detector.detect_fpn_all(frame);
        break;
    }
    auto cls_end = chrono::high_resolution_clock::now();
    long long cls_ms = chrono::duration_cast<chrono::milliseconds>(cls_end - cls_start).count();

    vector<BBox> ai_dets;
    long long ai_ms = 0;

    if (det_mode == DetMode::AI_Classical_Consensus)
    {
        if (ai_detector == nullptr)
        {
            throw runtime_error("Consensus mode requires a TensorRT detector");
        }

        auto ai_start = chrono::high_resolution_clock::now();
        ai_dets = ai_detector->detect(frame, class_names);
        auto ai_end = chrono::high_resolution_clock::now();
        ai_ms = chrono::duration_cast<chrono::milliseconds>(ai_end - ai_start).count();

        if (verbose)
        {
            cout << "[Pipeline Perf] Classical Detector: " << cls_ms << " ms | AI TensorRT Detector: " << ai_ms << " ms" << endl;
        }
    }
    else if (verbose)
    {
        cout << "[Pipeline Perf] Classical Detector: " << cls_ms << " ms" << endl;
    }

    if (det_mode == DetMode::AI_Classical_Consensus)
    {
        out_detections.clear();
        for (const auto& ai_b : ai_dets)
        {
            for (const auto& cls_b : classical_dets)
            {
                Rect intersection = ai_b.rect() & cls_b.rect();
                float inter_area = static_cast<float>(intersection.area());
                float union_area = static_cast<float>(ai_b.rect().area() + cls_b.rect().area() - intersection.area());
                float iou = (union_area > 0) ? (inter_area / union_area) : 0.0f;
                float c_dist = centerDistance(ai_b, cls_b);

                const float dynamic_radius = max(8.0f, max(ai_b.w, ai_b.h) * 0.75f);
                if (iou > 0.02f || inter_area > 0.0f || c_dist < dynamic_radius)
                {
                    BBox consensus_box = ai_b;
                    consensus_box.type = "AI_Classical_Consensus";
                    out_detections.push_back(consensus_box);
                    break;
                }
            }
        }
    }
    else
    {
        out_detections = classical_dets;
    }

    auto total_end = chrono::high_resolution_clock::now();
    elapsed_ms = chrono::duration_cast<chrono::milliseconds>(total_end - total_start).count();
    det_count = (int)out_detections.size();

    Point frame_center(frame.cols / 2, frame.rows / 2);
    drawMarker(frame, frame_center, Scalar(0, 0, 255), MARKER_CROSS, 15, 2);

    for (size_t i = 0; i < out_detections.size(); ++i)
    {
        const auto& b = out_detections[i];
        Scalar color = (b.type == "AI_Classical_Consensus") ? Scalar(255, 0, 255) :
            ((b.type == "ai_target") ? Scalar(0, 255, 0) : Scalar(255, 255, 255));

        Rect box = b.rect() & Rect(0, 0, frame.cols, frame.rows);
        if (box.empty()) continue;

        rectangle(frame, box, color, (b.type == "ai_target" || b.type == "AI_Classical_Consensus") ? 2 : 1);

        Point target_center(box.x + box.width / 2, box.y + box.height / 2);
        line(frame, frame_center, target_center, color, 1, LINE_AA);

        string label = (b.type == "AI_Classical_Consensus") ? (b.class_name + " [Consensus]: " + to_string((int)(b.score * 100)) + "%") :
            ((b.type == "ai_target") ? (b.class_name + ": " + to_string((int)(b.score * 100)) + "%") : ("Cls: " + b.type));
        int baseline = 0;
        Size text_size = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.45, 1, &baseline);
        int label_x = std::clamp(box.x, 0, std::max(0, frame.cols - text_size.width - 2));
        int label_y = (box.y - 5 > text_size.height) ? box.y - 5 : std::min(frame.rows - 2, box.y + box.height + text_size.height + 4);
        putText(frame, label, Point(label_x, label_y), FONT_HERSHEY_SIMPLEX, 0.45, color, 1);
    }
}

struct FrameData
{
    int frame_idx;
    Mat frame;
};

bool open_video_writer(VideoWriter& writer, const string& path, double fps, Size frame_size)
{
    if (frame_size.width <= 0 || frame_size.height <= 0) return false;

    const vector<int> codecs = {
        VideoWriter::fourcc('m', 'p', '4', 'v'),
        VideoWriter::fourcc('a', 'v', 'c', '1'),
        VideoWriter::fourcc('X', 'V', 'I', 'D')
    };

    for (int codec : codecs)
    {
        writer.open(path, codec, fps, frame_size);
        if (writer.isOpened()) return true;
    }
    return false;
}

void write_report_header(ofstream& report, const string& mode_str)
{
    report << "Mode: " << mode_str << "\n";
    report << "\n";
    report << "Detections Debug Log:\n";
    report << "Frame\tID\tX\tY\tW\tH\tScore\tClass\tType\tArea\tAspect\textent\tDoG_Mean\tDist(px)\n";
    report << "------------------------------------------------------------------------------------------------\n";
}

void write_report_columns(ofstream& report)
{
    report << "Frame\tID\tX\tY\tW\tH\tScore\tClass\tType\tArea\tAspect\textent\tDoG_Mean\tDist(px)\n";
    report << "------------------------------------------------------------------------------------------------\n";
}

void write_detection_rows(ofstream& report, int frame_idx, const Mat& frame, const vector<BBox>& detections)
{
    if (!report) return;

    for (size_t i = 0; i < detections.size(); ++i)
    {
        const auto& b = detections[i];
        double dist = sqrt(pow(b.cx() - frame.cols / 2.0, 2) + pow(b.cy() - frame.rows / 2.0, 2));
        report << frame_idx << "\t" << (i + 1) << "\t" << b.x << "\t" << b.y
               << "\t" << b.w << "\t" << b.h << "\t" << fixed << setprecision(2) << b.score
               << "\t" << b.class_name << "\t" << b.type << "\t" << b.area << "\t"
               << b.aspect << "\t" << b.extent << "\t" << b.dog_mean << "\t" << dist << "\n";
    }
}

class ThreadedVideoProcessor
{
public:
    ThreadedVideoProcessor(const string& video_path, BlobDetector& detector, TRTDetector* ai_detector, DetMode mode,
        const vector<string>& class_names, bool display, bool verbose)
        : detector_(detector), ai_detector_(ai_detector), det_mode_(mode), cap_(video_path),
        class_names_(class_names), stop_threads_(false), display_(display), verbose_(verbose) {
    }

    ThreadedVideoProcessor(int camera_index, BlobDetector& detector, TRTDetector* ai_detector, DetMode mode,
        const vector<string>& class_names, bool display, bool verbose)
        : detector_(detector), ai_detector_(ai_detector), det_mode_(mode), cap_(camera_index),
        class_names_(class_names), stop_threads_(false), display_(display), verbose_(verbose) {
    }

    void run(const string& run_dir, string mode_str)
    {
        if (!cap_.isOpened())
        {
            cerr << "[ERROR] Failed to open video stream.\n";
            return;
        }
        double fps = cap_.get(CAP_PROP_FPS);
        if (fps <= 0) fps = 30.0;
        int fw = static_cast<int>(cap_.get(CAP_PROP_FRAME_WIDTH));
        int fh = static_cast<int>(cap_.get(CAP_PROP_FRAME_HEIGHT));
        if (fw <= 0 || fh <= 0)
        {
            Mat probe;
            if (cap_.read(probe) && !probe.empty())
            {
                fw = probe.cols;
                fh = probe.rows;
                cap_.set(CAP_PROP_POS_FRAMES, 0);
            }
        }
        if (fw <= 0 || fh <= 0)
        {
            cerr << "[ERROR] Could not determine video frame size.\n";
            return;
        }

        string out_video_path = run_dir + "/Output_" + mode_str + "_Video.mp4";
        VideoWriter writer;
        if (!open_video_writer(writer, out_video_path, fps, Size(fw, fh)))
        {
            cerr << "[WARN] Could not open video writer. Report will still be saved.\n";
        }

        string txt_path = run_dir + "/video_detection_report.txt";
        ofstream report(txt_path);
        if (!report)
        {
            cerr << "[WARN] Could not create report file: " << txt_path << "\n";
        }
        else
        {
            write_report_header(report, mode_str);
        }
        mutex report_mutex;

        thread producer([this]()
            {
                int idx = 0;
                Mat temp;
                while (!stop_threads_ && cap_.read(temp))
                {
                    if (temp.empty()) break;
                    idx++;
                    {
                        lock_guard<mutex> lock(mailbox_mutex_);
                        latest_frame_ = { idx, temp.clone() };
                        has_new_frame_ = true;
                    }
                    mailbox_cond_.notify_one();
                }
                is_producer_done_ = true;
                mailbox_cond_.notify_all();
            });

        thread consumer([this, &writer, &report, &report_mutex]()
            {
                while (true)
                {
                    FrameData item;
                    {
                        unique_lock<mutex> lock(mailbox_mutex_);
                        mailbox_cond_.wait_for(lock, chrono::milliseconds(50), [this] { return has_new_frame_ || is_producer_done_ || stop_threads_; });
                        if (stop_threads_) break;
                        if (!has_new_frame_ && is_producer_done_) break;
                        if (!has_new_frame_) continue;

                        item = latest_frame_;
                        has_new_frame_ = false;
                    }

                    long long p_ms = 0;
                    int d_count = 0;
                    vector<BBox> detections;
                    try
                    {
                        process_frame_detections(item.frame, detector_, ai_detector_, det_mode_, p_ms, d_count, detections, class_names_, verbose_);
                    }
                    catch (const exception& e)
                    {
                        cerr << "[ERROR] Processing failed on frame " << item.frame_idx << ": " << e.what() << "\n";
                        stop_threads_ = true;
                        mailbox_cond_.notify_all();
                        break;
                    }

                    {
                        lock_guard<mutex> lock(report_mutex);
                        write_detection_rows(report, item.frame_idx, item.frame, detections);
                    }

                    if (writer.isOpened()) writer.write(item.frame);

                    if (display_)
                    {
                        imshow("Hybrid AI + Classical Disp", item.frame);
                        if ((waitKey(1) & 0xFF) == 27)
                        {
                            stop_threads_ = true;
                            mailbox_cond_.notify_all();
                            break;
                        }
                    }
                }
            });

        if (producer.joinable()) producer.join();
        if (consumer.joinable()) consumer.join();
        cap_.release();
        if (writer.isOpened()) writer.release();
        report.close();
        if (display_) destroyAllWindows();
        cout << "\n[DONE] Video processing complete. Saved in: " << run_dir << "\n";
    }

private:
    BlobDetector& detector_;
    TRTDetector* ai_detector_;
    DetMode det_mode_;
    VideoCapture cap_;
    vector<string> class_names_;
    atomic<bool> stop_threads_;
    atomic<bool> is_producer_done_{ false };
    FrameData latest_frame_;
    bool has_new_frame_ = false;
    mutex mailbox_mutex_;
    condition_variable mailbox_cond_;
    bool display_;
    bool verbose_;
};

int main(int argc, char** argv)
{
    AppOptions options;
    if (!parse_args(argc, argv, options))
    {
        return 1;
    }
    if (options.help_requested)
    {
        return 0;
    }

    if (!options.mode_provided)
    {
        if (can_prompt())
        {
            options.mode = get_detection_mode();
        }
        else
        {
            cout << "[INFO] No --mode provided and stdin is non-interactive. Using fpn.\n";
        }
    }

    if (options.display && !display_available())
    {
        cout << "[INFO] No graphical display detected. Running with --no-display behavior.\n";
        options.display = false;
    }

    const bool use_camera = is_integer_source(options.input_path);
    string ext = to_lower_copy(fs::path(options.input_path).extension().string());
    bool is_video = use_camera || (ext == ".mp4" || ext == ".avi" || ext == ".mov" || ext == ".mkv" ||
                                   ext == ".m4v" || ext == ".webm" || ext == ".mpg" || ext == ".mpeg");

    int width = 0;
    int height = 0;
    Mat sample_frame;

    if (is_video)
    {
        VideoCapture test_cap;
        if (use_camera)
        {
            test_cap.open(stoi(options.input_path));
        }
        else
        {
            test_cap.open(options.input_path);
        }
        if (!test_cap.isOpened())
        {
            cerr << "[ERROR] Failed to open video source: " << options.input_path << "\n";
            return 1;
        }
        width = static_cast<int>(test_cap.get(CAP_PROP_FRAME_WIDTH));
        height = static_cast<int>(test_cap.get(CAP_PROP_FRAME_HEIGHT));
        if (width <= 0 || height <= 0)
        {
            Mat probe;
            if (test_cap.read(probe) && !probe.empty())
            {
                width = probe.cols;
                height = probe.rows;
            }
        }
        test_cap.release();
    }
    else
    {
        sample_frame = imread(options.input_path);
        if (sample_frame.empty())
        {
            VideoCapture fallback_cap(options.input_path);
            if (fallback_cap.isOpened())
            {
                is_video = true;
                width = static_cast<int>(fallback_cap.get(CAP_PROP_FRAME_WIDTH));
                height = static_cast<int>(fallback_cap.get(CAP_PROP_FRAME_HEIGHT));
                if (width <= 0 || height <= 0)
                {
                    Mat probe;
                    if (fallback_cap.read(probe) && !probe.empty())
                    {
                        width = probe.cols;
                        height = probe.rows;
                    }
                }
                fallback_cap.release();
            }
            else
            {
                cerr << "[ERROR] Failed to load input as image or video: " << options.input_path << "\n";
                return 1;
            }
        }
        else
        {
            width = sample_frame.cols;
            height = sample_frame.rows;
        }
    }

    if (width <= 0 || height <= 0)
    {
        cerr << "[ERROR] Could not determine input dimensions for: " << options.input_path << "\n";
        return 1;
    }

    if (options.mode == DetMode::AI_Classical_Consensus && options.engine_path.empty())
    {
        cerr << "[ERROR] Consensus mode requires --engine <model.engine>.\n";
        return 1;
    }

    vector<string> class_names;
    if (!options.classes_path.empty())
    {
        try
        {
            class_names = loadClassNames(options.classes_path);
            if (class_names.empty())
            {
                cerr << "[WARN] Class names file is empty. Detections will use 'target'.\n";
            }
        }
        catch (const exception& e)
        {
            cerr << "[ERROR] " << e.what() << "\n";
            return 1;
        }
    }

    unique_ptr<TRTDetector> ai_detector;
    if (options.mode == DetMode::AI_Classical_Consensus)
    {
        try
        {
            ai_detector = make_unique<TRTDetector>(options.engine_path, options.ai_conf_threshold, options.ai_nms_threshold);
        }
        catch (const exception& e)
        {
            cerr << "[ERROR] Failed to initialize TensorRT detector: " << e.what() << "\n";
            return 1;
        }
    }
    else if (options.legacy_invocation && !options.engine_path.empty())
    {
        cout << "[INFO] Engine path was provided but mode is " << mode_to_string(options.mode)
             << "; TensorRT initialization skipped.\n";
    }

    cout << "[INFO] Loaded input dimensions: " << width << "x" << height << "\n";
    cout << "[INFO] Mode: " << mode_to_string(options.mode) << "\n";

    string run_dir;
    try
    {
        run_dir = create_run_folder(options.output_root);
    }
    catch (const fs::filesystem_error& e)
    {
        cerr << "[ERROR] Failed to create output directory under '" << options.output_root << "': " << e.what() << "\n";
        return 1;
    }

    BlobDetector detector(options.config, width, height, options.prior, options.display, is_video, run_dir);
    string mode_str = mode_to_string(options.mode);

    if (!is_video)
    {
        long long processing_ms = 0;
        int det_count = 0;
        vector<BBox> detections;
        try
        {
            process_frame_detections(sample_frame, detector, ai_detector.get(), options.mode,
                processing_ms, det_count, detections, class_names, options.verbose);
        }
        catch (const exception& e)
        {
            cerr << "[ERROR] Processing failed: " << e.what() << "\n";
            return 1;
        }

        string report_path = run_dir + "/detection_report.txt";
        ofstream report(report_path);
        if (!report)
        {
            cerr << "[WARN] Could not create report file: " << report_path << "\n";
        }
        else
        {
            report << "Mode: " << mode_str << "\n";
            report << "Processing Time: " << processing_ms << " ms\n";
            report << "Number of Detections: " << det_count << "\n\n";
            report << "Detections Debug Log:\n";
            write_report_columns(report);
            write_detection_rows(report, 1, sample_frame, detections);
        }

        string out_path = run_dir + "/Output_" + mode_str + "_Result.jpg";
        if (!imwrite(out_path, sample_frame))
        {
            cerr << "[WARN] Failed to save result image: " << out_path << "\n";
        }

        cout << "\n[DONE] Image processing complete. Saved in: " << run_dir << "\n";
        if (options.display)
        {
            imshow("Hybrid Detector Result", sample_frame);
            waitKey(0);
            destroyAllWindows();
        }
    }
    else
    {
        if (options.display) namedWindow("Hybrid AI + Classical Disp", WINDOW_NORMAL);
        if (use_camera)
        {
            ThreadedVideoProcessor processor(stoi(options.input_path), detector, ai_detector.get(), options.mode,
                class_names, options.display, options.verbose);
            processor.run(run_dir, mode_str);
        }
        else
        {
            ThreadedVideoProcessor processor(options.input_path, detector, ai_detector.get(), options.mode,
                class_names, options.display, options.verbose);
            processor.run(run_dir, mode_str);
        }
    }

    return 0;
}



// ./Exe-hybrid_classical_ai_det image.jpg --mode fpn
// ./Exe-hybrid_classical_ai_det video.mp4 --mode hybrid --no-display
// ./Exe-hybrid_classical_ai_det video.mp4 --mode hybrid --display
// ./Exe-hybrid_classical_ai_det image.jpg --mode fpn --display
// ./Exe-hybrid_classical_ai_det image.jpg --mode fpn

