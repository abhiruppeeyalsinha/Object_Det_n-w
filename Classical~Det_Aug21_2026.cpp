// ════════════════════════════════════════════════════════════════════
// image_classical_detector.cpp (Robust & Fully Fixed)
// Multi-Threaded Stream Processor, Step-by-Step Visual Debugger & Rich Reports
// ════════════════════════════════════════════════════════════════════

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

using namespace std;
using namespace cv;
namespace fs = std::filesystem;

enum class DetMode
{
    DoG,
    HoughCircles,
    Hybrid
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
    float DOG_THRESHOLD = -1.0f;
    float DOG_STDDEV_FACTOR = 1.25f;
    float MIN_DOG_THRESHOLD = 8.0f;
    float MAX_DOG_THRESHOLD = 48.0f;
    double MIN_AREA_FRAC = 0.00002;
    double MAX_AREA_FRAC = 0.35;
    double MIN_EXTENT = 0.05;
    double MIN_ASPECT = 0.10;
    double MAX_ASPECT = 10.0;
    double MAX_VERTICAL_POSITION = 0.92;
    int MIN_CONTOUR_AREA = 6;
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
    string type = "detection";
    double area = 0.0;
    double aspect = 0.0;
    double extent = 0.0;
    double dog_mean = 0.0;

    int cx() const { return x + w / 2; }
    int cy() const { return y + h / 2; }
    Rect rect() const { return Rect(x, y, w, h); }
};

class BlobDetector
{
public:
    BlobDetector(const TrackerConfig &cfg, int roi_w, int roi_h, PriorMode prior = PriorMode::Center, bool debug_mode = false, bool is_video = false)
        : cfg_(cfg), roi_w_(roi_w), roi_h_(roi_h), prior_(prior), debug_mode_(debug_mode), is_video_(is_video) {}

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
        sort(boxes.begin(), boxes.end(), [](const BBox& a, const BBox& b) {
            return a.score > b.score;
        });

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

                if (iou > iou_threshold)
                {
                    suppressed[j] = true;
                }
            }
        }
        return result;
    }

    void handle_debug_pause(const string& message) const
    {
        if (!debug_mode_) return;
        
        if (is_video_) {
            // Non-blocking or short wait for video streams to prevent complete lockup
            waitKey(15); 
        } else {
            // Full interactive pause for single images
            cout << message << " Press any key to continue...\n";
            waitKey(0);
        }
    }

    std::vector<BBox> select_all(const Mat& gray,
                                 const Mat& fused_mask,
                                 const Mat& dog_conf,
                                 const Mat& log_conf,
                                 float hybrid_w) const
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
            if (r.width <= 0 || r.height <= 0) continue;
            if (r.x < 0 || r.y < 0 || r.x + r.width > gray.cols || r.y + r.height > gray.rows) continue;
            if (r.y > gray.rows * cfg_.MAX_VERTICAL_POSITION) continue;

            double aspect = static_cast<double>(r.width) / r.height;
            double extent = area / (r.width * r.height);
            double rel = (area / roi_area) * 100.0;

            if (!(cfg_.MIN_ASPECT < aspect && aspect < cfg_.MAX_ASPECT &&
                  extent > cfg_.MIN_EXTENT && rel < max_rel_percent)) continue;

            double dog_mean = dog_conf.empty() ? 0.0 : mean(dog_conf(r))[0];
            double log_mean = log_conf.empty() ? 0.0 : mean(log_conf(r))[0];

            double dog_n = std::min(dog_mean / 50.0, 1.0);
            double log_n = std::min(log_mean / 50.0, 1.0);
            double hyb_conf = (1.0 - hybrid_w) * dog_n + hybrid_w * log_n;
            double score = area * (0.5 + hyb_conf) * prior_weight(r, gray.size());

            candidates.push_back(BBox{r.x, r.y, r.width, r.height, (float)score, "dog_detected", area, aspect, extent, dog_mean});
        }

        auto final_boxes = apply_nms(candidates, 0.18f);

        if (debug_mode_)
        {
            Mat contour_vis;
            cvtColor(gray, contour_vis, COLOR_GRAY2BGR);
            drawContours(contour_vis, contours, -1, Scalar(0, 255, 0), 1);
            for (const auto& b : final_boxes)
            {
                rectangle(contour_vis, b.rect(), Scalar(0, 0, 255), 2);
            }
            namedWindow("Debug: Contour Filtering & NMS", WINDOW_NORMAL);
            imshow("Debug: Contour Filtering & NMS", contour_vis);
            handle_debug_pause("[DEBUG] Contours and NMS results displayed.");
        }

        return final_boxes;
    }

    vector<BBox> detect_dog_all(const Mat &roi) const
    {
        if (roi.empty()) return {};

        Mat gray;
        if (roi.channels() == 3) cvtColor(roi, gray, COLOR_BGR2GRAY);
        else gray = roi.clone();

        if (debug_mode_)
        {
            namedWindow("Debug: Grayscale Input", WINDOW_NORMAL);
            imshow("Debug: Grayscale Input", gray);
        }

        Mat g1a, g2a, dog_a;
        GaussianBlur(gray, g1a, {3, 3}, 0.8);
        GaussianBlur(gray, g2a, {7, 7}, 2.0);
        subtract(g1a, g2a, dog_a, noArray(), CV_16S);

        Mat g1b, g2b, dog_b;
        GaussianBlur(gray, g1b, {5, 5}, 1.5);
        GaussianBlur(gray, g2b, {11, 11}, 3.5);
        subtract(g1b, g2b, dog_b, noArray(), CV_16S);

        Mat dog_a32, dog_b32;
        dog_a.convertTo(dog_a32, CV_32F);
        dog_b.convertTo(dog_b32, CV_32F);

        if (debug_mode_)
        {
            Mat vis_a, vis_b;
            normalize(dog_a32, vis_a, 0, 255, NORM_MINMAX, CV_8U);
            normalize(dog_b32, vis_b, 0, 255, NORM_MINMAX, CV_8U);

            namedWindow("Debug: DoG Scale A (3x3 - 7x7)", WINDOW_NORMAL);
            namedWindow("Debug: DoG Scale B (5x5 - 11x11)", WINDOW_NORMAL);
            imshow("Debug: DoG Scale A (3x3 - 7x7)", vis_a);
            imshow("Debug: DoG Scale B (5x5 - 11x11)", vis_b);
        }

        Mat abs_a, abs_b, dog_conf;
        absdiff(dog_a32, Scalar(0), abs_a);
        absdiff(dog_b32, Scalar(0), abs_b);
        max(abs_a, abs_b, dog_conf);

        const float dog_thresh = adaptive_response_threshold(dog_conf);
        Mat m1 = blob_mask(dog_a32, dog_thresh);
        Mat m2 = blob_mask(dog_b32, dog_thresh);
        Mat combined;
        bitwise_or(m1, m2, combined);
        cleanup_mask(combined);

        if (debug_mode_)
        {
            namedWindow("Debug: Fused Thresholded Mask", WINDOW_NORMAL);
            imshow("Debug: Fused Thresholded Mask", combined);
            handle_debug_pause("[DEBUG] DoG scale-space maps displayed.");
        }

        return select_all(gray, combined, dog_conf, Mat{}, 0.0f);
    }

    vector<BBox> detect_hough_circles(const Mat &img) const
    {
        if (img.empty()) return {};

        Mat gray;
        // FIXED: Passing 'img' instead of uninitialized/empty 'gray' to cvtColor
        if (img.channels() == 3) cvtColor(img, gray, COLOR_BGR2GRAY);
        else gray = img.clone();

        Mat smoothed;
        GaussianBlur(gray, smoothed, Size(5, 5), 1.2);

        if (debug_mode_)
        {
            Mat edges;
            Canny(smoothed, edges, 35, 70);
            namedWindow("Debug: Hough Smoothed Input", WINDOW_NORMAL);
            namedWindow("Debug: Canny Edge Map for Accumulator", WINDOW_NORMAL);
            imshow("Debug: Hough Smoothed Input", smoothed);
            imshow("Debug: Canny Edge Map for Accumulator", edges);
            handle_debug_pause("[DEBUG] Hough pre-processing & edges displayed.");
        }

        vector<Vec3f> circles;
        const int min_dim = std::max(1, std::min(gray.cols, gray.rows));
        int min_radius = (cfg_.HOUGH_MIN_RADIUS > 0) ? cfg_.HOUGH_MIN_RADIUS : std::max(3, min_dim / 180);
        int max_radius = (cfg_.HOUGH_MAX_RADIUS > 0) ? cfg_.HOUGH_MAX_RADIUS : std::max(min_radius + 2, min_dim / 24);
        int min_dist = (cfg_.HOUGH_MIN_DIST > 0) ? cfg_.HOUGH_MIN_DIST : std::max(8, min_dim / 14);

        HoughCircles(smoothed, circles, HOUGH_GRADIENT, 1.2,
                     min_dist, cfg_.HOUGH_PARAM1, cfg_.HOUGH_PARAM2, min_radius, max_radius);

        vector<BBox> boxes;
        for (const auto &c : circles)
        {
            int cx = cvRound(c[0]);
            int cy = cvRound(c[1]);
            int r  = cvRound(c[2]);

            int x = max(0, cx - r);
            int y = max(0, cy - r);
            int w = min(img.cols - x, 2 * r);
            int h = min(img.rows - y, 2 * r);

            if (w <= 1 || h <= 1) continue;

            if ((y + h) < (img.rows * cfg_.MAX_VERTICAL_POSITION))
            {
                double area_approx = CV_PI * r * r;
                boxes.push_back(BBox{x, y, w, h, 0.95f, "hough_circle", area_approx, 1.0, 0.78, 25.0});
            }
        }
        
        auto final_boxes = apply_nms(boxes, 0.12f);

        if (debug_mode_)
        {
            Mat hough_vis;
            cvtColor(gray, hough_vis, COLOR_GRAY2BGR);
            for (const auto& b : final_boxes)
            {
                rectangle(hough_vis, b.rect(), Scalar(255, 0, 0), 2);
            }
            namedWindow("Debug: Hough Circle NMS Output", WINDOW_NORMAL);
            imshow("Debug: Hough Circle NMS Output", hough_vis);
            handle_debug_pause("[DEBUG] Hough circles displayed.");
        }

        return final_boxes;
    }

    vector<BBox> detect_hybrid_all(const Mat &img) const
    {
        auto dog_boxes = detect_dog_all(img);
        auto hough_boxes = detect_hough_circles(img);

        vector<BBox> combined = dog_boxes;
        for (const auto &hb : hough_boxes)
        {
            bool matched = false;
            for (auto &db : dog_boxes)
            {
                if ((db.rect() & hb.rect()).area() > 0)
                {
                    matched = true;
                    break;
                }
            }
            if (!matched) combined.push_back(hb);
        }
        for (auto &b : combined) b.type = "hybrid";
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
    int roi_w_;
    int roi_h_;
    PriorMode prior_;
    bool debug_mode_;
    bool is_video_;
};

DetMode get_detection_mode()
{
    cout << "\n--- Detection Engine ---\n"
         << " [1] Filtered DoG\n"
         << " [2] Optimized Hough Circle Transform\n"
         << " [3] Hybrid (DoG + Hough)\n";

    string sel;
    cout << " Select (default 2): ";
    getline(cin, sel);

    if (sel == "1") return DetMode::DoG;
    if (sel == "3") return DetMode::Hybrid;
    return DetMode::HoughCircles;
}

bool get_debug_mode_choice()
{
    string sel;
    cout << " Enable Step-by-Step Visual Debugger? (y/n, default n): ";
    getline(cin, sel);
    return (sel == "y" || sel == "Y");
}

string to_lower_copy(string value)
{
    transform(value.begin(), value.end(), value.begin(),
              [](unsigned char c) { return static_cast<char>(tolower(c)); });
    return value;
}

bool parse_detection_mode(const string& raw, DetMode& mode)
{
    string value = to_lower_copy(raw);
    if (value == "1" || value == "dog" || value == "dogg" || value == "dog-filter" || value == "dog_filter")
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
    return false;
}

bool parse_prior_mode(const string& raw, PriorMode& prior)
{
    string value = to_lower_copy(raw);
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
    case DetMode::Hybrid:
        return "Hybrid";
    case DetMode::HoughCircles:
    default:
        return "Hough";
    }
}

void print_usage(const char* exe)
{
    cerr << "Usage: " << exe << " <path_to_image_or_video> [options]\n\n"
         << "Options:\n"
         << "  --mode <dog|hough|hybrid>     Detection engine. Prompts in a terminal, defaults to hough otherwise.\n"
         << "  --debug                       Show step-by-step debug windows when a display is available.\n"
         << "  --no-display                  Disable all OpenCV windows for batch/headless runs.\n"
         << "  --out-dir <dir>               Output root directory. Default: Res\n"
         << "  --dog-threshold <value>       Fixed DoG threshold. Default: adaptive.\n"
         << "  --dog-stddev <value>          Adaptive DoG sensitivity. Default: 1.25\n"
         << "  --hough-min-radius <px>       Override dynamic Hough minimum radius.\n"
         << "  --hough-max-radius <px>       Override dynamic Hough maximum radius.\n"
         << "  --hough-min-dist <px>         Override dynamic Hough center spacing.\n"
         << "  --prior <center|size|contrast> Candidate score prior. Default: center\n";
}

struct AppOptions
{
    string input_path;
    string output_root = "Res";
    DetMode mode = DetMode::HoughCircles;
    PriorMode prior = PriorMode::Center;
    TrackerConfig config;
    bool mode_provided = false;
    bool debug = false;
    bool display = true;
    bool help_requested = false;
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

    options.input_path = argv[1];

    for (int i = 2; i < argc; ++i)
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
        if (arg == "--debug")
        {
            options.debug = true;
            continue;
        }
        if (arg == "--no-display")
        {
            options.display = false;
            continue;
        }
        if (arg == "--mode")
        {
            if (!read_value() || !parse_detection_mode(value, options.mode))
            {
                cerr << "[ERROR] Invalid --mode value.\n";
                return false;
            }
            options.mode_provided = true;
            continue;
        }
        if (arg.rfind("--mode=", 0) == 0)
        {
            value = arg.substr(7);
            if (!parse_detection_mode(value, options.mode))
            {
                cerr << "[ERROR] Invalid --mode value.\n";
                return false;
            }
            options.mode_provided = true;
            continue;
        }
        if (arg == "--out-dir" || arg.rfind("--out-dir=", 0) == 0)
        {
            if (arg == "--out-dir")
            {
                if (!consume_option_value(i, argc, argv, value))
                {
                    cerr << "[ERROR] Missing --out-dir value.\n";
                    return false;
                }
            }
            else
            {
                value = arg.substr(10);
            }
            if (value.empty())
            {
                cerr << "[ERROR] Output directory cannot be empty.\n";
                return false;
            }
            options.output_root = value;
            continue;
        }
        if (arg == "--prior" || arg.rfind("--prior=", 0) == 0)
        {
            if (arg == "--prior")
            {
                if (!consume_option_value(i, argc, argv, value))
                {
                    cerr << "[ERROR] Missing --prior value.\n";
                    return false;
                }
            }
            else
            {
                value = arg.substr(8);
            }
            if (!parse_prior_mode(value, options.prior))
            {
                cerr << "[ERROR] Invalid --prior value.\n";
                return false;
            }
            continue;
        }

        float float_value = 0.0f;
        int int_value = 0;
        if (arg == "--dog-threshold" || arg.rfind("--dog-threshold=", 0) == 0)
        {
            if (arg == "--dog-threshold" && !consume_option_value(i, argc, argv, value))
            {
                cerr << "[ERROR] Missing --dog-threshold value.\n";
                return false;
            }
            if (arg.rfind("--dog-threshold=", 0) == 0) value = arg.substr(16);
            if (!parse_float_arg(value, float_value) || float_value <= 0.0f)
            {
                cerr << "[ERROR] --dog-threshold must be a positive number.\n";
                return false;
            }
            options.config.DOG_THRESHOLD = float_value;
            continue;
        }
        if (arg == "--dog-stddev" || arg.rfind("--dog-stddev=", 0) == 0)
        {
            if (arg == "--dog-stddev" && !consume_option_value(i, argc, argv, value))
            {
                cerr << "[ERROR] Missing --dog-stddev value.\n";
                return false;
            }
            if (arg.rfind("--dog-stddev=", 0) == 0) value = arg.substr(13);
            if (!parse_float_arg(value, float_value) || float_value < 0.0f)
            {
                cerr << "[ERROR] --dog-stddev must be a non-negative number.\n";
                return false;
            }
            options.config.DOG_STDDEV_FACTOR = float_value;
            continue;
        }
        if (arg == "--hough-min-radius" || arg.rfind("--hough-min-radius=", 0) == 0)
        {
            if (arg == "--hough-min-radius" && !consume_option_value(i, argc, argv, value))
            {
                cerr << "[ERROR] Missing --hough-min-radius value.\n";
                return false;
            }
            if (arg.rfind("--hough-min-radius=", 0) == 0) value = arg.substr(19);
            if (!parse_int_arg(value, int_value) || int_value <= 0)
            {
                cerr << "[ERROR] --hough-min-radius must be a positive integer.\n";
                return false;
            }
            options.config.HOUGH_MIN_RADIUS = int_value;
            continue;
        }
        if (arg == "--hough-max-radius" || arg.rfind("--hough-max-radius=", 0) == 0)
        {
            if (arg == "--hough-max-radius" && !consume_option_value(i, argc, argv, value))
            {
                cerr << "[ERROR] Missing --hough-max-radius value.\n";
                return false;
            }
            if (arg.rfind("--hough-max-radius=", 0) == 0) value = arg.substr(19);
            if (!parse_int_arg(value, int_value) || int_value <= 0)
            {
                cerr << "[ERROR] --hough-max-radius must be a positive integer.\n";
                return false;
            }
            options.config.HOUGH_MAX_RADIUS = int_value;
            continue;
        }
        if (arg == "--hough-min-dist" || arg.rfind("--hough-min-dist=", 0) == 0)
        {
            if (arg == "--hough-min-dist" && !consume_option_value(i, argc, argv, value))
            {
                cerr << "[ERROR] Missing --hough-min-dist value.\n";
                return false;
            }
            if (arg.rfind("--hough-min-dist=", 0) == 0) value = arg.substr(17);
            if (!parse_int_arg(value, int_value) || int_value <= 0)
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

void process_frame_detections(Mat &frame, const BlobDetector &detector, DetMode det_mode, 
                              long long &elapsed_ms, int &det_count, vector<BBox> &out_detections)
{
    if (frame.empty())
    {
        elapsed_ms = 0;
        det_count = 0;
        out_detections.clear();
        return;
    }

    auto start_time = chrono::high_resolution_clock::now();

    switch (det_mode)
    {
    case DetMode::DoG:
        out_detections = detector.detect_dog_all(frame);
        break;
    case DetMode::HoughCircles:
        out_detections = detector.detect_hough_circles(frame);
        break;
    case DetMode::Hybrid:
        out_detections = detector.detect_hybrid_all(frame);
        break;
    }

    auto end_time = chrono::high_resolution_clock::now();
    elapsed_ms = chrono::duration_cast<chrono::milliseconds>(end_time - start_time).count();
    det_count = out_detections.size();

    Point frame_center(frame.cols / 2, frame.rows / 2);
    drawMarker(frame, frame_center, Scalar(0, 0, 255), MARKER_CROSS, 15, 2);

    for (size_t i = 0; i < out_detections.size(); ++i)
    {
        const auto &b = out_detections[i];
        Scalar color  = Scalar(255, 255, 255);
        Rect box = b.rect() & Rect(0, 0, frame.cols, frame.rows);
        if (box.empty()) continue;

        rectangle(frame, box, color, 1);
        
        Point target_center(box.x + box.width / 2, box.y + box.height / 2);
        line(frame, frame_center, target_center, color, 1, LINE_AA);

        string label = "Target " + to_string(i + 1) + " (X:" + to_string(box.x) + " Y:" + to_string(box.y) + ")";
        int baseline = 0;
        Size text_size = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.45, 1, &baseline);
        int label_x = std::clamp(box.x, 0, std::max(0, frame.cols - text_size.width - 2));
        int label_y = (box.y - 5 > text_size.height) ? box.y - 5 : std::min(frame.rows - 2, box.y + box.height + text_size.height + 4);
        putText(frame, label, Point(label_x, label_y),
                FONT_HERSHEY_SIMPLEX, 0.45, Scalar(0, 255, 255), 1);
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

class ThreadedVideoProcessor
{
public:
    ThreadedVideoProcessor(const string& video_path, BlobDetector& detector, DetMode mode, bool display)
        : detector_(detector), det_mode_(mode), cap_(video_path), stop_threads_(false), display_(display) {}

    ThreadedVideoProcessor(int camera_index, BlobDetector& detector, DetMode mode, bool display)
        : detector_(detector), det_mode_(mode), cap_(camera_index), stop_threads_(false), display_(display) {}

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
            report << "Mode: " << mode_str << "\n";
            report << "Frame\tID\tX\tY\tW\tH\tScore\tType\tArea\tAspect\textent\tDoG_Mean\tDist(px)\n";
            report << "---------------------------------------------------------------------------------------\n";
        }
        mutex report_mutex;

        thread producer([this]() {
            int idx = 0;
            Mat temp;
            while (!stop_threads_ && cap_.read(temp))
            {
                if (temp.empty()) break;
                idx++;
                {
                    unique_lock<mutex> lock(queue_mutex_);
                    queue_cond_.wait(lock, [this] { return stop_threads_ || frame_queue_.size() < max_queue_size_; });
                    if (stop_threads_) break;
                    frame_queue_.push({idx, temp.clone()});
                }
                queue_cond_.notify_one();
            }
            is_producer_done_ = true;
            queue_cond_.notify_all();
        });

        thread consumer([this, &writer, &report, &report_mutex]() {
            while (true)
            {
                FrameData item;
                {
                    unique_lock<mutex> lock(queue_mutex_);
                    queue_cond_.wait(lock, [this] { return stop_threads_ || !frame_queue_.empty() || is_producer_done_; });
                    
                    if (stop_threads_ || (frame_queue_.empty() && is_producer_done_)) break;

                    item = frame_queue_.front();
                    frame_queue_.pop();
                }
                queue_cond_.notify_one();

                long long p_ms = 0;
                int d_count = 0;
                vector<BBox> detections;
                process_frame_detections(item.frame, detector_, det_mode_, p_ms, d_count, detections);

                {
                    lock_guard<mutex> lock(report_mutex);
                    if (report)
                    {
                        for (size_t i = 0; i < detections.size(); ++i)
                        {
                            const auto &b = detections[i];
                            double dist = sqrt(pow(b.cx() - item.frame.cols / 2.0, 2) + pow(b.cy() - item.frame.rows / 2.0, 2));
                            report << item.frame_idx << "\t" << (i + 1) << "\t" << b.x << "\t" << b.y 
                                   << "\t" << b.w << "\t" << b.h << "\t" << fixed << setprecision(2) << b.score << "\t" << b.type 
                                   << "\t" << b.area << "\t" << b.aspect << "\t" << b.extent << "\t" << b.dog_mean << "\t" << dist << "\n";
                        }
                    }
                }

                if (writer.isOpened()) writer.write(item.frame);

                if (display_)
                {
                    imshow("Multi-Threaded Video Stream", item.frame);
                    if ((waitKey(1) & 0xFF) == 27)
                    {
                        stop_threads_ = true;
                        queue_cond_.notify_all();
                        break;
                    }
                }
            }
        });

        producer.join();
        consumer.join();
        cap_.release();
        if (writer.isOpened()) writer.release();
        report.close();
        cout << "\n[DONE] Multi-threaded video processing complete. Saved in: " << run_dir << "\n";
    }

private:
    BlobDetector& detector_;
    DetMode det_mode_;
    VideoCapture cap_;
    atomic<bool> stop_threads_;
    atomic<bool> is_producer_done_{false};
    queue<FrameData> frame_queue_;
    mutex queue_mutex_;
    condition_variable queue_cond_;
    const size_t max_queue_size_ = 10;
    bool display_;
};

int main(int argc, char **argv)
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
            cout << "[INFO] No --mode provided and stdin is non-interactive. Using hough.\n";
        }
    }

    if (can_prompt() && !options.debug)
    {
        options.debug = get_debug_mode_choice();
    }

    if (options.display && !display_available())
    {
        cout << "[INFO] No graphical display detected. Running with --no-display behavior.\n";
        options.display = false;
    }
    if (options.debug && !options.display)
    {
        cout << "[INFO] Debug windows disabled because display output is unavailable.\n";
        options.debug = false;
    }

    string input_path = options.input_path;
    const bool use_camera = is_integer_source(input_path);
    string ext = fs::path(input_path).extension().string();
    ext = to_lower_copy(ext);
    bool is_video = use_camera || (ext == ".mp4" || ext == ".avi" || ext == ".mov" || ext == ".mkv" || ext == ".m4v" ||
                                   ext == ".webm" || ext == ".mpg" || ext == ".mpeg");

    Mat sample_frame;
    int width = 0;
    int height = 0;

    if (is_video)
    {
        VideoCapture test_cap;
        if (use_camera)
        {
            test_cap.open(stoi(input_path));
        }
        else
        {
            test_cap.open(input_path);
        }
        if (!test_cap.isOpened())
        {
            cerr << "[ERROR] Failed to open video source: " << input_path << "\n";
            return -1;
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
        sample_frame = imread(input_path);
        if (sample_frame.empty())
        {
            VideoCapture fallback_cap(input_path);
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
                cerr << "[ERROR] Failed to load input as image or video: " << input_path << "\n";
                return -1;
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
        cerr << "[ERROR] Could not determine input dimensions for: " << input_path << "\n";
        return -1;
    }

    cout << "[INFO] Loaded input dimensions: " << width << "x" << height << "\n";

    TrackerConfig cfg = options.config;
    DetMode det_mode = options.mode;

    BlobDetector detector(cfg, width, height, options.prior, options.debug, is_video);

    string run_dir;
    try
    {
        run_dir = create_run_folder(options.output_root);
    }
    catch (const fs::filesystem_error& e)
    {
        cerr << "[ERROR] Failed to create output directory under '" << options.output_root << "': " << e.what() << "\n";
        return -1;
    }
    string mode_str = mode_to_string(det_mode);

    if (!is_video)
    {
        long long processing_ms = 0;
        int det_count = 0;
        vector<BBox> detections;

        process_frame_detections(sample_frame, detector, det_mode, processing_ms, det_count, detections);

        string txt_path = run_dir + "/detection_report.txt";
        ofstream report(txt_path);
        if (!report)
        {
            cerr << "[WARN] Could not create report file: " << txt_path << "\n";
        }
        else
        {
            report << "Mode: " << mode_str << "\n";
            report << "Processing Time: " << processing_ms << " ms\n";
            report << "Number of Detections: " << det_count << "\n\n";
            report << "Detections Debug Log:\n";
            report << "ID\tX\tY\tW\tH\tScore\tType\tArea\tAspect\textent\tDoG_Mean\tDist(px)\n";
            report << "---------------------------------------------------------------------------------------\n";

            for (size_t i = 0; i < detections.size(); ++i)
            {
                const auto &b = detections[i];
                double dist = sqrt(pow(b.cx() - sample_frame.cols / 2.0, 2) + pow(b.cy() - sample_frame.rows / 2.0, 2));
                report << (i + 1) << "\t" << b.x << "\t" << b.y << "\t" << b.w << "\t" << b.h 
                       << "\t" << fixed << setprecision(2) << b.score << "\t" << b.type 
                       << "\t" << b.area << "\t" << b.aspect << "\t" << b.extent << "\t" << b.dog_mean << "\t" << dist << "\n";
            }
        }
        report.close();

        string out_path = run_dir + "/Output_" + mode_str + "_Result.jpg";
        if (!imwrite(out_path, sample_frame))
        {
            cerr << "[WARN] Failed to save result image: " << out_path << "\n";
        }

        cout << "\n[DONE] Processing complete. Output saved in subfolder: " << run_dir << "\n";
        if (options.display)
        {
            namedWindow("Detector Result", WINDOW_NORMAL);
            imshow("Detector Result", sample_frame);
            waitKey(0);
        }
    }
    else
    {
        if (options.display) namedWindow("Multi-Threaded Video Stream", WINDOW_NORMAL);
        if (use_camera)
        {
            ThreadedVideoProcessor processor(stoi(input_path), detector, det_mode, options.display);
            processor.run(run_dir, mode_str);
        }
        else
        {
            ThreadedVideoProcessor processor(input_path, detector, det_mode, options.display);
            processor.run(run_dir, mode_str);
        }
    }

    return 0;
}
