// ============================================================================
// balloon_detector.cpp
// Production-Ready Balloon Detector - Image & Video Support
//
// Detection methods:
//   1. DoG          - Multi-scale Difference of Gaussian
//   2. Hough        - Hough Circle Transform
//   3. Hybrid       - DoG + Hough fusion/confirmation
//   4. Compare All  - Run all three independently
//
// Designed for:
//   - EO images
//   - IR / thermal grayscale images
//   - Small circular / blob-like targets
//   - Video files (requires FFmpeg)
//
// OpenCV: 4.x (with FFmpeg support)
// C++   : C++17
// ============================================================================

#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <sstream>
#include <string>
#include <vector>

using namespace std;
using namespace cv;
namespace fs = std::filesystem;

// ============================================================================
// Detection Mode
// ============================================================================

enum class DetMode
{
    DoG,
    HoughCircles,
    Hybrid,
    CompareAll
};

// ============================================================================
// Input Type
// ============================================================================

enum class InputType
{
    Image,
    Video,
    Invalid
};

// ============================================================================
// DoG Configuration
// ============================================================================

struct DoGConfig
{
    // Base DoG scales
    double sigma1 = 1.0;
    double sigma2 = 3.0;

    // Optional kernel sizes.
    // Size(0,0) means OpenCV automatically derives kernel size from sigma.
    Size blur1_size = Size(5, 5);
    Size blur2_size = Size(15, 15);

    // Thresholding
    int threshold = 20;
    bool use_adaptive_threshold = true;
    double threshold_std_multiplier = 1.5;

    // Candidate geometry
    double min_area = 15.0;
    double max_area = 5000.0;

    double min_aspect_ratio = 0.45;
    double max_aspect_ratio = 2.20;

    double min_solidity = 0.35;
    double min_circularity = 0.20;

    // Edge enhancement
    bool enhance_edges = true;
    double edge_enhancement_sigma = 0.8;

    // Multi-scale
    vector<double> scale_factors = {
        0.75,
        1.00,
        1.50,
        2.00
    };

    // Minimum number of scales supporting a candidate
    int min_scale_support = 1;

    // Local contrast verification
    bool use_local_contrast = true;
    double min_local_contrast = 0.03;
};

// ============================================================================
// Hough Configuration
// ============================================================================

struct HoughConfig
{
    Size blur_size = Size(9, 9);
    double blur_sigma = 2.0;

    double dp = 1.0;

    // Minimum distance between circle centers.
    // Expressed as fraction of minimum image dimension.
    double min_dist_factor = 8.0;

    double param1 = 100.0;
    double param2 = 25.0;

    int min_radius_percent = 1;
    int max_radius_percent = 15;

    // Absolute safety limits
    int absolute_min_radius = 5;
    int absolute_max_radius = 0; // 0 = no additional limit

    bool use_edge_confidence = true;
    double min_confidence = 0.25;

    double min_circularity = 0.50;

    double max_aspect_ratio = 1.50;

    // Local contrast
    bool use_local_contrast = true;
    double min_local_contrast = 0.03;

    // Hough candidates are independently deduplicated
    double duplicate_iou_threshold = 0.30;
};

// ============================================================================
// Hybrid Configuration
// ============================================================================

struct HybridConfig
{
    // Match DoG and Hough candidates using IoU
    bool merge_by_iou = true;

    double iou_threshold = 0.20;

    // If true, Hough-only detections are retained
    bool retain_hough_only = true;

    // If true, DoG-only detections are retained
    bool retain_dog_only = true;

    // Confidence weighting
    bool use_weighted_scoring = true;

    double hough_weight = 0.60;
    double dog_weight = 0.40;

    // Bonus when both detectors agree
    double agreement_bonus = 0.10;

    // Stronger threshold for detector agreement
    double confirmed_threshold = 0.35;
};

// ============================================================================
// Video Processing Configuration
// ============================================================================

struct VideoConfig
{
    // Frame skip (process every Nth frame)
    int frame_step = 1;

    // Maximum number of frames to process (0 = unlimited)
    int max_frames = 0;

    // Resize factor for video (0.0 = no resize)
    double resize_factor = 0.0;

    // Output video settings
    bool save_output_video = true;
    double output_fps = 30.0;
    Size output_size = Size(0, 0);

    // Display during processing
    bool show_preview = true;
    int preview_delay_ms = 30;

    // Write frame-by-frame results
    bool save_frame_results = true;
};

// ============================================================================
// Global Detection Configuration
// ============================================================================

struct DetectionConfig
{
    float confidence_threshold = 0.20f;

    bool enable_nms = true;
    float nms_iou_threshold = 0.35f;

    DoGConfig dog;
    HoughConfig hough;
    HybridConfig hybrid;
    VideoConfig video;
};

// ============================================================================
// Detection Result Structure
// ============================================================================

struct DetectionResult
{
    int frame_number = 0;
    double timestamp = 0.0;
    vector<BBox> detections;
    double processing_time = 0.0;
};

// ============================================================================
// Bounding Box
// ============================================================================

struct BBox
{
    int x = 0;
    int y = 0;
    int w = 0;
    int h = 0;

    float score = 0.0f;
    float confidence = 0.0f;

    string type = "unknown";

    int radius = 0;
    int id = 0;

    float circularity = 0.0f;

    // Additional verification metrics
    float solidity = 0.0f;
    float aspect_score = 0.0f;
    float contrast_score = 0.0f;
    float edge_score = 0.0f;
    float scale_support = 0.0f;

    int cx() const
    {
        return x + w / 2;
    }

    int cy() const
    {
        return y + h / 2;
    }

    Rect rect() const
    {
        return Rect(x, y, w, h);
    }

    float area() const
    {
        return static_cast<float>(w * h);
    }

    float aspect_ratio() const
    {
        if (h <= 0)
            return 0.0f;

        return static_cast<float>(w) /
               static_cast<float>(h);
    }
};

// ============================================================================
// Utility Functions
// ============================================================================

static double clamp_double(double value, double low, double high)
{
    return std::max(low, std::min(high, value));
}

static float clamp_float(float value, float low, float high)
{
    return std::max(low, std::min(high, value));
}

static int make_odd(int value)
{
    if (value <= 0)
        return 0;

    if ((value % 2) == 0)
        ++value;

    return value;
}

static Size sanitize_kernel(Size size)
{
    if (size.width <= 0 || size.height <= 0)
        return Size(0, 0);

    size.width = make_odd(size.width);
    size.height = make_odd(size.height);

    return size;
}

static string get_file_extension(const string& path)
{
    size_t dot_pos = path.find_last_of('.');
    if (dot_pos == string::npos)
        return "";

    string ext = path.substr(dot_pos + 1);
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    return ext;
}

static InputType detect_input_type(const string& path)
{
    // Check if file exists
    if (!fs::exists(path))
        return InputType::Invalid;

    // Check if it's a directory
    if (fs::is_directory(path))
        return InputType::Invalid;

    // Check extension
    string ext = get_file_extension(path);

    // Common video extensions
    vector<string> video_exts = {
        "mp4", "avi", "mov", "mkv", "flv", "wmv",
        "webm", "m4v", "mpg", "mpeg", "3gp", "ogv"
    };

    // Common image extensions
    vector<string> image_exts = {
        "jpg", "jpeg", "png", "bmp", "tiff", "tif",
        "webp", "gif", "ppm", "pgm", "pbm"
    };

    for (const auto& ve : video_exts)
    {
        if (ext == ve)
            return InputType::Video;
    }

    for (const auto& ie : image_exts)
    {
        if (ext == ie)
            return InputType::Image;
    }

    // Unknown extension - try to open as image first, then video
    return InputType::Image;
}

// ============================================================================
// Balloon Detector
// ============================================================================

class BalloonDetector
{
public:

    explicit BalloonDetector(
        const DetectionConfig& config = DetectionConfig())
        : config_(config)
    {
        sanitize_config();
    }

    // ------------------------------------------------------------------------
    // Main Detection API
    // ------------------------------------------------------------------------

    vector<BBox> detect(
        const Mat& image,
        DetMode mode)
    {
        if (image.empty())
        {
            cerr << "[ERROR] Empty image supplied.\n";
            return {};
        }

        if (image.cols < 16 || image.rows < 16)
        {
            cerr << "[ERROR] Image is too small: "
                 << image.cols << "x"
                 << image.rows << "\n";

            return {};
        }

        try
        {
            prepare_input(image);

            vector<BBox> boxes;

            switch (mode)
            {
                case DetMode::DoG:
                    boxes = detect_dog();
                    break;

                case DetMode::HoughCircles:
                    boxes = detect_hough();
                    break;

                case DetMode::Hybrid:
                    boxes = detect_hybrid();
                    break;

                case DetMode::CompareAll:
                    // CompareAll is handled by main().
                    boxes.clear();
                    break;
            }

            boxes = clip_boxes(boxes);

            boxes = filter_by_confidence(
                boxes,
                config_.confidence_threshold);

            if (config_.enable_nms && boxes.size() > 1)
            {
                boxes = apply_nms(
                    boxes,
                    config_.nms_iou_threshold);
            }

            boxes = verify_detections(boxes);

            // Assign IDs after final filtering.
            for (size_t i = 0; i < boxes.size(); ++i)
            {
                boxes[i].id = static_cast<int>(i + 1);
            }

            return boxes;
        }
        catch (const cv::Exception& e)
        {
            cerr << "[ERROR] OpenCV exception: "
                 << e.what() << "\n";

            return {};
        }
        catch (const exception& e)
        {
            cerr << "[ERROR] Detection exception: "
                 << e.what() << "\n";

            return {};
        }
    }

    DetectionConfig& config()
    {
        return config_;
    }

    const DetectionConfig& config() const
    {
        return config_;
    }

private:

    DetectionConfig config_;

    // Prepared grayscale image
    Mat gray_cache_;

    // Floating point normalized image
    Mat gray_float_;

    // Canny edge image
    Mat edge_cache_;

    // HSV is prepared once for color images.
    Mat hsv_cache_;

    bool color_input_ = false;

    // ------------------------------------------------------------------------
    // Configuration validation
    // ------------------------------------------------------------------------

    void sanitize_config()
    {
        config_.dog.blur1_size =
            sanitize_kernel(config_.dog.blur1_size);

        config_.dog.blur2_size =
            sanitize_kernel(config_.dog.blur2_size);

        config_.hough.blur_size =
            sanitize_kernel(config_.hough.blur_size);

        config_.dog.sigma1 =
            max(0.1, config_.dog.sigma1);

        config_.dog.sigma2 =
            max(config_.dog.sigma1 + 0.1,
                config_.dog.sigma2);

        config_.dog.edge_enhancement_sigma =
            max(0.1,
                config_.dog.edge_enhancement_sigma);

        config_.dog.threshold =
            std::max(1,
                       std::min(254,
                                config_.dog.threshold));

        config_.dog.threshold_std_multiplier =
            max(0.0,
                config_.dog.threshold_std_multiplier);

        config_.dog.min_area =
            max(1.0,
                config_.dog.min_area);

        config_.dog.max_area =
            max(config_.dog.min_area,
                config_.dog.max_area);

        config_.dog.min_aspect_ratio =
            max(0.05,
                config_.dog.min_aspect_ratio);

        config_.dog.max_aspect_ratio =
            max(config_.dog.min_aspect_ratio,
                config_.dog.max_aspect_ratio);

        config_.dog.min_solidity =
            clamp_double(config_.dog.min_solidity,
                         0.0,
                         1.0);

        config_.dog.min_circularity =
            clamp_double(config_.dog.min_circularity,
                         0.0,
                         1.0);

        config_.dog.min_scale_support =
            max(1,
                config_.dog.min_scale_support);

        if (config_.dog.scale_factors.empty())
        {
            config_.dog.scale_factors = {
                0.75,
                1.0,
                1.5,
                2.0
            };
        }

        config_.hough.dp =
            max(0.5,
                config_.hough.dp);

        config_.hough.min_dist_factor =
            max(1.0,
                config_.hough.min_dist_factor);

        config_.hough.param1 =
            max(1.0,
                config_.hough.param1);

        config_.hough.param2 =
            max(1.0,
                config_.hough.param2);

        config_.hough.min_radius_percent =
            max(1,
                config_.hough.min_radius_percent);

        config_.hough.max_radius_percent =
            max(config_.hough.min_radius_percent,
                config_.hough.max_radius_percent);

        config_.hough.absolute_min_radius =
            max(1,
                config_.hough.absolute_min_radius);

        config_.hough.min_confidence =
            clamp_double(config_.hough.min_confidence,
                         0.0,
                         1.0);

        config_.hough.min_circularity =
            clamp_double(config_.hough.min_circularity,
                         0.0,
                         1.0);

        config_.hough.max_aspect_ratio =
            max(1.0,
                config_.hough.max_aspect_ratio);

        config_.hybrid.iou_threshold =
            clamp_double(config_.hybrid.iou_threshold,
                         0.0,
                         1.0);

        config_.hybrid.hough_weight =
            max(0.0,
                config_.hybrid.hough_weight);

        config_.hybrid.dog_weight =
            max(0.0,
                config_.hybrid.dog_weight);

        double weight_sum =
            config_.hybrid.hough_weight +
            config_.hybrid.dog_weight;

        if (weight_sum <= 0.0)
        {
            config_.hybrid.hough_weight = 0.6;
            config_.hybrid.dog_weight = 0.4;
        }
        else
        {
            config_.hybrid.hough_weight /= weight_sum;
            config_.hybrid.dog_weight /= weight_sum;
        }
    }

    // ------------------------------------------------------------------------
    // Prepare image
    // ------------------------------------------------------------------------

    void prepare_input(const Mat& image)
    {
        color_input_ = image.channels() >= 3;

        if (image.channels() == 1)
        {
            gray_cache_ = image;
        }
        else if (image.channels() == 3)
        {
            cvtColor(
                image,
                gray_cache_,
                COLOR_BGR2GRAY);
        }
        else if (image.channels() == 4)
        {
            cvtColor(
                image,
                gray_cache_,
                COLOR_BGRA2GRAY);
        }
        else
        {
            throw runtime_error(
                "Unsupported number of image channels.");
        }

        if (gray_cache_.depth() != CV_8U)
        {
            double min_val = 0.0;
            double max_val = 0.0;

            minMaxLoc(
                gray_cache_,
                &min_val,
                &max_val);

            if (max_val > min_val)
            {
                gray_cache_.convertTo(
                    gray_cache_,
                    CV_8U,
                    255.0 /
                    (max_val - min_val),
                    -min_val *
                    255.0 /
                    (max_val - min_val));
            }
            else
            {
                gray_cache_.convertTo(
                    gray_cache_,
                    CV_8U);
            }
        }

        // Normalize to floating point [0,1].
        gray_cache_.convertTo(
            gray_float_,
            CV_32F,
            1.0 / 255.0);

        // Edge map is calculated ONCE.
        Canny(
            gray_cache_,
            edge_cache_,
            50,
            150);

        // HSV conversion only once.
        if (color_input_)
        {
            if (image.channels() == 3)
            {
                cvtColor(
                    image,
                    hsv_cache_,
                    COLOR_BGR2HSV);
            }
            else if (image.channels() == 4)
            {
                Mat bgr;
                cvtColor(
                    image,
                    bgr,
                    COLOR_BGRA2BGR);

                cvtColor(
                    bgr,
                    hsv_cache_,
                    COLOR_BGR2HSV);
            }
        }
        else
        {
            hsv_cache_.release();
        }
    }

    // ------------------------------------------------------------------------
    // Gaussian helper
    // ------------------------------------------------------------------------

    Mat gaussian(
        const Mat& input,
        double sigma,
        Size configured_size) const
    {
        Mat output;

        Size ksize = configured_size;

        if (ksize.width <= 0 ||
            ksize.height <= 0)
        {
            GaussianBlur(
                input,
                output,
                Size(0, 0),
                sigma,
                sigma,
                BORDER_REPLICATE);
        }
        else
        {
            GaussianBlur(
                input,
                output,
                ksize,
                sigma,
                sigma,
                BORDER_REPLICATE);
        }

        return output;
    }

    // ------------------------------------------------------------------------
    // Edge enhancement
    // ------------------------------------------------------------------------

    Mat enhance_edges(
        const Mat& gray) const
    {
        Mat blurred;
        Mat enhanced;

        GaussianBlur(
            gray,
            blurred,
            Size(0, 0),
            config_.dog.edge_enhancement_sigma,
            config_.dog.edge_enhancement_sigma,
            BORDER_REPLICATE);

        addWeighted(
            gray,
            1.5,
            blurred,
            -0.5,
            0.0,
            enhanced);

        return enhanced;
    }

    // ------------------------------------------------------------------------
    // DoG detector
    // ------------------------------------------------------------------------

    vector<BBox> detect_dog()
    {
        Mat source = gray_float_;

        if (config_.dog.enhance_edges)
        {
            Mat source8;
            source.convertTo(
                source8,
                CV_8U,
                255.0);

            Mat enhanced =
                enhance_edges(source8);

            enhanced.convertTo(
                source,
                CV_32F,
                1.0 / 255.0);
        }

        vector<Mat> responses;

        double global_max = 0.0;

        // ------------------------------------------------------------
        // Generate multi-scale DoG maps.
        // ------------------------------------------------------------

        for (double scale :
             config_.dog.scale_factors)
        {
            if (scale <= 0.0)
                continue;

            double sigma1 =
                config_.dog.sigma1 * scale;

            double sigma2 =
                config_.dog.sigma2 * scale;

            Mat blur1 =
                gaussian(
                    source,
                    sigma1,
                    config_.dog.blur1_size);

            Mat blur2 =
                gaussian(
                    source,
                    sigma2,
                    config_.dog.blur2_size);

            Mat dog;
            subtract(
                blur1,
                blur2,
                dog,
                noArray(),
                CV_32F);

            Mat abs_dog;
            cv::absdiff(
                dog,
                Scalar::all(0),
                abs_dog);

            double local_max = 0.0;
            minMaxLoc(
                abs_dog,
                nullptr,
                &local_max);

            global_max =
                max(global_max,
                    local_max);

            responses.push_back(abs_dog);
        }

        if (responses.empty() ||
            global_max <= 1e-6)
        {
            return {};
        }

        // ------------------------------------------------------------
        // Combine multi-scale responses.
        // ------------------------------------------------------------

        Mat combined =
            Mat::zeros(
                source.size(),
                CV_32F);

        Mat support =
            Mat::zeros(
                source.size(),
                CV_8U);

        for (const Mat& response :
             responses)
        {
            max(
                combined,
                response,
                combined);

            Mat normalized;
            response.convertTo(
                normalized,
                CV_32F,
                255.0 / global_max);

            Mat binary_support;
            threshold(
                normalized,
                binary_support,
                max(5.0,
                    static_cast<double>(
                        config_.dog.threshold) * 0.5),
                1.0,
                THRESH_BINARY);

            Mat support8;
            binary_support.convertTo(
                support8,
                CV_8U);

            add(
                support,
                support8,
                support);
        }

        // ------------------------------------------------------------
        // Normalize combined response.
        // ------------------------------------------------------------

        Mat normalized_combined;

        combined.convertTo(
            normalized_combined,
            CV_8U,
            255.0 / global_max);

        // ------------------------------------------------------------
        // Threshold
        // ------------------------------------------------------------

        double threshold_value =
            static_cast<double>(
                config_.dog.threshold);

        if (config_.dog.use_adaptive_threshold)
        {
            Scalar mean_val;
            Scalar std_val;

            meanStdDev(
                normalized_combined,
                mean_val,
                std_val);

            threshold_value =
                mean_val[0] +
                config_.dog.threshold_std_multiplier *
                std_val[0];

            threshold_value =
                clamp_double(
                    threshold_value,
                    5.0,
                    245.0);
        }

        Mat thresholded;

        threshold(
            normalized_combined,
            thresholded,
            threshold_value,
            255,
            THRESH_BINARY);

        // ------------------------------------------------------------
        // Morphological cleanup
        // ------------------------------------------------------------

        Mat kernel =
            getStructuringElement(
                MORPH_ELLIPSE,
                Size(3, 3));

        morphologyEx(
            thresholded,
            thresholded,
            MORPH_OPEN,
            kernel);

        morphologyEx(
            thresholded,
            thresholded,
            MORPH_CLOSE,
            kernel);

        // ------------------------------------------------------------
        // Find contours
        // ------------------------------------------------------------

        vector<vector<Point>> contours;

        findContours(
            thresholded,
            contours,
            RETR_EXTERNAL,
            CHAIN_APPROX_SIMPLE);

        vector<BBox> boxes;

        for (const auto& contour :
             contours)
        {
            if (contour.size() < 5)
                continue;

            double area =
                contourArea(contour);

            if (area <
                    config_.dog.min_area ||
                area >
                    config_.dog.max_area)
            {
                continue;
            }

            Rect r =
                boundingRect(contour);

            if (r.width < 3 ||
                r.height < 3)
            {
                continue;
            }

            // --------------------------------------------------------
            // Correct aspect ratio.
            // --------------------------------------------------------

            double aspect =
                static_cast<double>(r.width) /
                static_cast<double>(r.height);

            if (aspect <
                    config_.dog.min_aspect_ratio ||
                aspect >
                    config_.dog.max_aspect_ratio)
            {
                continue;
            }

            // --------------------------------------------------------
            // Solidity
            // --------------------------------------------------------

            double bbox_area =
                static_cast<double>(
                    r.area());

            double solidity =
                area /
                max(1.0,
                    bbox_area);

            if (solidity <
                config_.dog.min_solidity)
            {
                continue;
            }

            // --------------------------------------------------------
            // Circularity
            // --------------------------------------------------------

            double perimeter =
                arcLength(
                    contour,
                    true);

            double circularity = 0.0;

            if (perimeter > 1e-6)
            {
                circularity =
                    4.0 *
                    CV_PI *
                    area /
                    (perimeter *
                     perimeter);
            }

            circularity =
                clamp_double(
                    circularity,
                    0.0,
                    1.0);

            if (circularity <
                config_.dog.min_circularity)
            {
                continue;
            }

            // --------------------------------------------------------
            // Aspect score
            // --------------------------------------------------------

            double aspect_deviation =
                abs(aspect - 1.0);

            double aspect_range =
                max(
                    0.001,
                    config_.dog.max_aspect_ratio -
                    1.0);

            double aspect_score =
                1.0 -
                aspect_deviation /
                aspect_range;

            aspect_score =
                clamp_double(
                    aspect_score,
                    0.0,
                    1.0);

            // --------------------------------------------------------
            // Local contrast
            // --------------------------------------------------------

            float contrast_score =
                calculate_local_contrast(
                    r);

            if (config_.dog.use_local_contrast &&
                contrast_score <
                config_.dog.min_local_contrast)
            {
                continue;
            }

            // --------------------------------------------------------
            // Scale support
            // --------------------------------------------------------

            Point center(
                r.x + r.width / 2,
                r.y + r.height / 2);

            int support_value =
                support.at<uchar>(
                    center);

            double scale_support =
                static_cast<double>(
                    support_value) /
                static_cast<double>(
                    max<size_t>(
                        1,
                        responses.size()));

            if (support_value <
                config_.dog.min_scale_support)
            {
                continue;
            }

            // --------------------------------------------------------
            // Response strength
            // --------------------------------------------------------

            Rect safe_r =
                clip_rect(
                    r,
                    gray_cache_.size());

            double response_mean =
                0.0;

            if (safe_r.area() > 0)
            {
                Scalar m =
                    mean(
                        normalized_combined(
                            safe_r));

                response_mean =
                    m[0] / 255.0;
            }

            // --------------------------------------------------------
            // Confidence
            // --------------------------------------------------------

            double confidence =
                0.20 * solidity +
                0.20 * circularity +
                0.15 * aspect_score +
                0.20 * contrast_score +
                0.15 * scale_support +
                0.10 * response_mean;

            confidence =
                clamp_double(
                    confidence,
                    0.0,
                    1.0);

            BBox box;

            box.x = r.x;
            box.y = r.y;
            box.w = r.width;
            box.h = r.height;

            box.score =
                static_cast<float>(
                    confidence);

            box.confidence =
                static_cast<float>(
                    confidence);

            box.type = "dog";

            box.radius =
                max(
                    r.width,
                    r.height) / 2;

            box.circularity =
                static_cast<float>(
                    circularity);

            box.solidity =
                static_cast<float>(
                    solidity);

            box.aspect_score =
                static_cast<float>(
                    aspect_score);

            box.contrast_score =
                contrast_score;

            box.scale_support =
                static_cast<float>(
                    scale_support);

            boxes.push_back(box);
        }

        return boxes;
    }

    // ------------------------------------------------------------------------
    // Hough detector
    // ------------------------------------------------------------------------

    vector<BBox> detect_hough()
    {
        const int image_size =
            min(
                gray_cache_.cols,
                gray_cache_.rows);

        // ------------------------------------------------------------
        // Radius range
        // ------------------------------------------------------------

        int min_radius =
            static_cast<int>(
                image_size *
                config_.hough.min_radius_percent /
                100.0);

        int max_radius =
            static_cast<int>(
                image_size *
                config_.hough.max_radius_percent /
                100.0);

        min_radius =
            max(
                min_radius,
                config_.hough.absolute_min_radius);

        max_radius =
            max(
                max_radius,
                min_radius + 2);

        if (config_.hough.absolute_max_radius > 0)
        {
            max_radius =
                min(
                    max_radius,
                    config_.hough.absolute_max_radius);
        }

        max_radius =
            min(
                max_radius,
                image_size / 2);

        if (max_radius <= min_radius)
            return {};

        // ------------------------------------------------------------
        // Prepare Hough input
        // ------------------------------------------------------------

        Mat blurred;

        GaussianBlur(
            gray_cache_,
            blurred,
            config_.hough.blur_size,
            config_.hough.blur_sigma,
            config_.hough.blur_sigma,
            BORDER_REPLICATE);

        // ------------------------------------------------------------
        // Hough circle detection
        // ------------------------------------------------------------

        vector<Vec3f> circles;

        double min_dist =
            max(
                static_cast<double>(
                    image_size) /
                config_.hough.min_dist_factor,
                5.0);

        HoughCircles(
            blurred,
            circles,
            HOUGH_GRADIENT,
            config_.hough.dp,
            min_dist,
            config_.hough.param1,
            config_.hough.param2,
            min_radius,
            max_radius);

        vector<BBox> boxes;

        for (const Vec3f& c :
             circles)
        {
            int cx =
                cvRound(c[0]);

            int cy =
                cvRound(c[1]);

            int radius =
                cvRound(c[2]);

            if (radius <= 0)
                continue;

            if (cx < 0 ||
                cy < 0 ||
                cx >= gray_cache_.cols ||
                cy >= gray_cache_.rows)
            {
                continue;
            }

            // --------------------------------------------------------
            // Bounding circle rectangle
            // --------------------------------------------------------

            Rect circle_rect(
                cx - radius,
                cy - radius,
                2 * radius,
                2 * radius);

            Rect safe_rect =
                clip_rect(
                    circle_rect,
                    gray_cache_.size());

            if (safe_rect.area() <= 0)
                continue;

            // --------------------------------------------------------
            // Correct geometric aspect ratio.
            // --------------------------------------------------------

            double aspect =
                static_cast<double>(
                    safe_rect.width) /
                max(
                    1,
                    safe_rect.height);

            if (aspect < 1.0 /
                    config_.hough.max_aspect_ratio ||
                aspect >
                    config_.hough.max_aspect_ratio)
            {
                continue;
            }

            // --------------------------------------------------------
            // Edge confidence
            // --------------------------------------------------------

            double edge_score = 1.0;

            if (config_.hough.use_edge_confidence)
            {
                edge_score =
                    check_circle_edges(
                        edge_cache_,
                        cx,
                        cy,
                        radius);
            }

            // --------------------------------------------------------
            // Local contrast
            // --------------------------------------------------------

            float contrast_score =
                calculate_local_contrast(
                    safe_rect);

            if (config_.hough.use_local_contrast &&
                contrast_score <
                config_.hough.min_local_contrast)
            {
                continue;
            }

            // --------------------------------------------------------
            // Circularity approximation
            // --------------------------------------------------------

            double circularity =
                edge_score;

            circularity =
                clamp_double(
                    circularity,
                    0.0,
                    1.0);

            if (circularity <
                config_.hough.min_circularity)
            {
                continue;
            }

            // --------------------------------------------------------
            // Intensity variation.
            // --------------------------------------------------------

            Mat roi =
                gray_cache_(
                    safe_rect);

            Scalar mean_value;
            Scalar std_value;

            meanStdDev(
                roi,
                mean_value,
                std_value);

            if (std_value[0] < 3.0)
                continue;

            double texture_score =
                clamp_double(
                    std_value[0] / 64.0,
                    0.0,
                    1.0);

            // --------------------------------------------------------
            // Confidence
            // --------------------------------------------------------

            double confidence =
                0.35 * edge_score +
                0.30 * contrast_score +
                0.20 * circularity +
                0.15 * texture_score;

            confidence =
                clamp_double(
                    confidence,
                    0.0,
                    1.0);

            if (confidence <
                config_.hough.min_confidence)
            {
                continue;
            }

            BBox box;

            box.x = safe_rect.x;
            box.y = safe_rect.y;
            box.w = safe_rect.width;
            box.h = safe_rect.height;

            box.score =
                static_cast<float>(
                    confidence);

            box.confidence =
                static_cast<float>(
                    confidence);

            box.type = "hough";

            box.radius = radius;

            box.circularity =
                static_cast<float>(
                    circularity);

            box.edge_score =
                static_cast<float>(
                    edge_score);

            box.contrast_score =
                contrast_score;

            boxes.push_back(box);
        }

        if (boxes.size() > 1)
        {
            boxes =
                apply_nms(
                    boxes,
                    static_cast<float>(
                        config_.hough.duplicate_iou_threshold));
        }

        return boxes;
    }

    // ------------------------------------------------------------------------
    // Hybrid detector
    // ------------------------------------------------------------------------

    vector<BBox> detect_hybrid()
    {
        vector<BBox> dog_boxes =
            detect_dog();

        vector<BBox> hough_boxes =
            detect_hough();

        vector<BBox> combined;

        vector<bool>
            dog_matched(
                dog_boxes.size(),
                false);

        vector<bool>
            hough_matched(
                hough_boxes.size(),
                false);

        // ------------------------------------------------------------
        // Match DoG candidates against Hough candidates.
        // ------------------------------------------------------------

        if (config_.hybrid.merge_by_iou)
        {
            sort(
                dog_boxes.begin(),
                dog_boxes.end(),
                [](const BBox& a,
                   const BBox& b)
                {
                    return a.confidence >
                           b.confidence;
                });

            sort(
                hough_boxes.begin(),
                hough_boxes.end(),
                [](const BBox& a,
                   const BBox& b)
                {
                    return a.confidence >
                           b.confidence;
                });

            dog_matched.assign(
                dog_boxes.size(),
                false);

            hough_matched.assign(
                hough_boxes.size(),
                false);

            for (size_t i = 0;
                 i < dog_boxes.size();
                 ++i)
            {
                float best_iou = 0.0f;
                int best_index = -1;

                for (size_t j = 0;
                     j < hough_boxes.size();
                     ++j)
                {
                    if (hough_matched[j])
                        continue;

                    float iou =
                        compute_iou(
                            dog_boxes[i].rect(),
                            hough_boxes[j].rect());

                    if (iou >
                        config_.hybrid.iou_threshold &&
                        iou > best_iou)
                    {
                        best_iou = iou;
                        best_index =
                            static_cast<int>(j);
                    }
                }

                if (best_index >= 0)
                {
                    BBox merged =
                        merge_boxes(
                            dog_boxes[i],
                            hough_boxes[
                                best_index]);

                    double confidence;

                    if (config_.hybrid.use_weighted_scoring)
                    {
                        confidence =
                            config_.hybrid.dog_weight *
                            dog_boxes[i].confidence +
                            config_.hybrid.hough_weight *
                            hough_boxes[
                                best_index].confidence;
                    }
                    else
                    {
                        confidence =
                            max(
                                dog_boxes[i].confidence,
                                hough_boxes[
                                    best_index].confidence);
                    }

                    confidence +=
                        config_.hybrid.agreement_bonus *
                        best_iou;

                    confidence =
                        clamp_double(
                            confidence,
                            0.0,
                            1.0);

                    merged.confidence =
                        static_cast<float>(
                            confidence);

                    merged.score =
                        merged.confidence;

                    merged.type =
                        "hybrid_merged";

                    merged.radius =
                        hough_boxes[
                            best_index].radius;

                    merged.circularity =
                        max(
                            dog_boxes[i].circularity,
                            hough_boxes[
                                best_index].circularity);

                    merged.edge_score =
                        hough_boxes[
                            best_index].edge_score;

                    merged.contrast_score =
                        max(
                            dog_boxes[i].contrast_score,
                            hough_boxes[
                                best_index].contrast_score);

                    dog_matched[i] = true;

                    hough_matched[
                        best_index] = true;

                    combined.push_back(
                        merged);
                }
            }
        }

        // ------------------------------------------------------------
        // DoG-only candidates
        // ------------------------------------------------------------

        if (config_.hybrid.retain_dog_only)
        {
            for (size_t i = 0;
                 i < dog_boxes.size();
                 ++i)
            {
                if (!dog_matched[i])
                {
                    BBox b =
                        dog_boxes[i];

                    b.type =
                        "dog_only";

                    b.confidence *= 0.90f;
                    b.score =
                        b.confidence;

                    combined.push_back(b);
                }
            }
        }

        // ------------------------------------------------------------
        // Hough-only candidates
        // ------------------------------------------------------------

        if (config_.hybrid.retain_hough_only)
        {
            for (size_t j = 0;
                 j < hough_boxes.size();
                 ++j)
            {
                if (!hough_matched[j])
                {
                    BBox b =
                        hough_boxes[j];

                    b.type =
                        "hough_only";

                    b.confidence *= 0.90f;
                    b.score =
                        b.confidence;

                    combined.push_back(b);
                }
            }
        }

        return combined;
    }

    // ------------------------------------------------------------------------
    // Clip bounding boxes
    // ------------------------------------------------------------------------

    vector<BBox> clip_boxes(
        const vector<BBox>& boxes) const
    {
        vector<BBox> result;

        Rect image_rect(
            0,
            0,
            gray_cache_.cols,
            gray_cache_.rows);

        for (const BBox& b :
             boxes)
        {
            Rect clipped =
                b.rect() & image_rect;

            if (clipped.width <= 0 ||
                clipped.height <= 0)
            {
                continue;
            }

            BBox copy = b;

            copy.x = clipped.x;
            copy.y = clipped.y;
            copy.w = clipped.width;
            copy.h = clipped.height;

            result.push_back(copy);
        }

        return result;
    }

    // ------------------------------------------------------------------------
    // Confidence filtering
    // ------------------------------------------------------------------------

    vector<BBox> filter_by_confidence(
        const vector<BBox>& boxes,
        float minimum) const
    {
        vector<BBox> result;

        for (const BBox& b :
             boxes)
        {
            if (b.confidence >= minimum)
                result.push_back(b);
        }

        return result;
    }

    // ------------------------------------------------------------------------
    // IoU
    // ------------------------------------------------------------------------

    float compute_iou(
        const Rect& a,
        const Rect& b) const
    {
        Rect intersection =
            a & b;

        if (intersection.area() <= 0)
            return 0.0f;

        double inter =
            static_cast<double>(
                intersection.area());

        double union_area =
            static_cast<double>(
                a.area()) +
            static_cast<double>(
                b.area()) -
            inter;

        if (union_area <= 0.0)
            return 0.0f;

        return static_cast<float>(
            inter / union_area);
    }

    // ------------------------------------------------------------------------
    // NMS
    // ------------------------------------------------------------------------

    vector<BBox> apply_nms(
        const vector<BBox>& boxes,
        float iou_threshold) const
    {
        if (boxes.empty())
            return {};

        vector<BBox> sorted =
            boxes;

        sort(
            sorted.begin(),
            sorted.end(),
            [](const BBox& a,
               const BBox& b)
            {
                return a.confidence >
                       b.confidence;
            });

        vector<BBox> result;

        vector<bool> suppressed(
            sorted.size(),
            false);

        for (size_t i = 0;
             i < sorted.size();
             ++i)
        {
            if (suppressed[i])
                continue;

            result.push_back(
                sorted[i]);

            for (size_t j = i + 1;
                 j < sorted.size();
                 ++j)
            {
                if (suppressed[j])
                    continue;

                float iou =
                    compute_iou(
                        sorted[i].rect(),
                        sorted[j].rect());

                if (iou >=
                    iou_threshold)
                {
                    suppressed[j] = true;
                }
            }
        }

        return result;
    }

    // ------------------------------------------------------------------------
    // Safe rectangle
    // ------------------------------------------------------------------------

    Rect clip_rect(
        const Rect& r,
        const Size& image_size) const
    {
        Rect image_rect(
            0,
            0,
            image_size.width,
            image_size.height);

        return r & image_rect;
    }

    // ------------------------------------------------------------------------
    // Local contrast
    // ------------------------------------------------------------------------

    float calculate_local_contrast(
        const Rect& candidate) const
    {
        Rect safe =
            clip_rect(
                candidate,
                gray_cache_.size());

        if (safe.width < 4 ||
            safe.height < 4)
        {
            return 0.0f;
        }

        Mat mask =
            Mat::zeros(
                gray_cache_.size(),
                CV_8U);

        Rect center_rect =
            safe;

        int shrink_x =
            max(
                1,
                safe.width / 6);

        int shrink_y =
            max(
                1,
                safe.height / 6);

        center_rect.x += shrink_x;
        center_rect.y += shrink_y;
        center_rect.width -=
            2 * shrink_x;
        center_rect.height -=
            2 * shrink_y;

        if (center_rect.width <= 0 ||
            center_rect.height <= 0)
        {
            return 0.0f;
        }

        rectangle(
            mask,
            center_rect,
            Scalar(255),
            FILLED);

        Scalar center_mean =
            mean(
                gray_float_,
                mask);

        int pad_x =
            max(
                2,
                safe.width / 2);

        int pad_y =
            max(
                2,
                safe.height / 2);

        Rect outer(
            safe.x - pad_x,
            safe.y - pad_y,
            safe.width + 2 * pad_x,
            safe.height + 2 * pad_y);

        outer =
            clip_rect(
                outer,
                gray_cache_.size());

        if (outer.area() <= safe.area())
            return 0.0f;

        Mat outer_mask =
            Mat::zeros(
                gray_cache_.size(),
                CV_8U);

        rectangle(
            outer_mask,
            outer,
            Scalar(255),
            FILLED);

        rectangle(
            outer_mask,
            safe,
            Scalar(0),
            FILLED);

        Scalar surround_mean =
            mean(
                gray_float_,
                outer_mask);

        double difference =
            abs(
                center_mean[0] -
                surround_mean[0]);

        double contrast =
            difference;

        return static_cast<float>(
            clamp_double(
                contrast,
                0.0,
                1.0));
    }

    // ------------------------------------------------------------------------
    // Circle edge confidence
    // ------------------------------------------------------------------------

    double check_circle_edges(
        const Mat& edges,
        int cx,
        int cy,
        int radius) const
    {
        if (edges.empty() ||
            radius <= 0)
        {
            return 0.0;
        }

        const int num_points = 72;

        int edge_hits = 0;

        for (int i = 0;
             i < num_points;
             ++i)
        {
            double angle =
                2.0 *
                CV_PI *
                static_cast<double>(i) /
                static_cast<double>(num_points);

            int px =
                cx +
                cvRound(
                    radius *
                    cos(angle));

            int py =
                cy +
                cvRound(
                    radius *
                    sin(angle));

            bool found = false;

            for (int dy = -2;
                 dy <= 2 && !found;
                 ++dy)
            {
                for (int dx = -2;
                     dx <= 2;
                     ++dx)
                {
                    int nx = px + dx;
                    int ny = py + dy;

                    if (nx < 0 ||
                        ny < 0 ||
                        nx >= edges.cols ||
                        ny >= edges.rows)
                    {
                        continue;
                    }

                    if (edges.at<uchar>(
                            ny,
                            nx) > 0)
                    {
                        found = true;
                        break;
                    }
                }
            }

            if (found)
                ++edge_hits;
        }

        return static_cast<double>(
                   edge_hits) /
               static_cast<double>(
                   num_points);
    }

    // ------------------------------------------------------------------------
    // Verification
    // ------------------------------------------------------------------------

    vector<BBox> verify_detections(
        const vector<BBox>& boxes) const
    {
        vector<BBox> verified;

        for (BBox b :
             boxes)
        {
            Rect safe =
                clip_rect(
                    b.rect(),
                    gray_cache_.size());

            if (safe.width < 5 ||
                safe.height < 5)
            {
                continue;
            }

            float contrast =
                calculate_local_contrast(
                    safe);

            b.contrast_score =
                max(
                    b.contrast_score,
                    contrast);

            bool acceptable =
                b.confidence >=
                    config_.confidence_threshold;

            if (!acceptable)
                continue;

            verified.push_back(b);
        }

        return verified;
    }

public:

    // ------------------------------------------------------------------------
    // Statistics helper
    // ------------------------------------------------------------------------

    static string mode_to_string(
        DetMode mode)
    {
        switch (mode)
        {
            case DetMode::DoG:
                return "DoG";

            case DetMode::HoughCircles:
                return "Hough";

            case DetMode::Hybrid:
                return "Hybrid";

            case DetMode::CompareAll:
                return "CompareAll";
        }

        return "Unknown";
    }
};

// ============================================================================
// Visualization
// ============================================================================

static Scalar detection_color(
    const string& type)
{
    if (type == "dog" ||
        type == "dog_only")
    {
        return Scalar(
            0,
            255,
            0);
    }

    if (type == "hough" ||
        type == "hough_only")
    {
        return Scalar(
            0,
            165,
            255);
    }

    if (type == "hybrid_merged")
    {
        return Scalar(
            255,
            255,
            0);
    }

    return Scalar(
        255,
        255,
        255);
}

// ============================================================================
// Draw detections
// ============================================================================

void draw_detections_clean(
    Mat& image,
    const vector<BBox>& detections,
    DetMode mode,
    double processing_time,
    int frame_number = -1,
    double timestamp = 0.0)
{
    const int center_x =
        image.cols / 2;

    const int center_y =
        image.rows / 2;

    // ------------------------------------------------------------------------
    // Center crosshair
    // ------------------------------------------------------------------------

    Scalar center_color(
        200,
        200,
        200);

    const int cross_size = 25;

    line(
        image,
        Point(
            center_x - cross_size,
            center_y),
        Point(
            center_x + cross_size,
            center_y),
        center_color,
        1);

    line(
        image,
        Point(
            center_x,
            center_y - cross_size),
        Point(
            center_x,
            center_y + cross_size),
        center_color,
        1);

    circle(
        image,
        Point(
            center_x,
            center_y),
        5,
        center_color,
        1);

    putText(
        image,
        "CENTER",
        Point(
            center_x - 25,
            center_y + 20),
        FONT_HERSHEY_SIMPLEX,
        0.4,
        center_color,
        1,
        LINE_AA);

    // ------------------------------------------------------------------------
    // Detections
    // ------------------------------------------------------------------------

    for (const BBox& b :
         detections)
    {
        Scalar color =
            detection_color(
                b.type);

        // Connection line.
        line(
            image,
            Point(
                center_x,
                center_y),
            Point(
                b.cx(),
                b.cy()),
            Scalar(
                255,
                255,
                100),
            1,
            LINE_AA);

        // Bounding box.
        rectangle(
            image,
            b.rect(),
            color,
            2,
            LINE_AA);

        // Hough circle.
        if ((b.type == "hough" ||
             b.type == "hough_only" ||
             b.type == "hybrid_merged") &&
            b.radius > 0)
        {
            circle(
                image,
                Point(
                    b.cx(),
                    b.cy()),
                b.radius,
                color,
                1,
                LINE_AA);
        }

        // Target center.
        const int small_cross = 5;

        line(
            image,
            Point(
                b.cx() - small_cross,
                b.cy()),
            Point(
                b.cx() + small_cross,
                b.cy()),
            Scalar(
                0,
                0,
                255),
            1);

        line(
            image,
            Point(
                b.cx(),
                b.cy() - small_cross),
            Point(
                b.cx(),
                b.cy() + small_cross),
            Scalar(
                0,
                0,
                255),
            1);

        // Angle.
        double angle =
            atan2(
                static_cast<double>(
                    b.cy() - center_y),
                static_cast<double>(
                    b.cx() - center_x))
            *
            180.0 /
            CV_PI;

        // Distance.
        double distance =
            hypot(
                static_cast<double>(
                    b.cx() - center_x),
                static_cast<double>(
                    b.cy() - center_y));

        // Label.
        stringstream label_stream;

        label_stream
            << "#"
            << b.id
            << " "
            << fixed
            << setprecision(0)
            << b.confidence * 100.0
            << "% "
            << static_cast<int>(
                   angle)
            << "deg";

        string label =
            label_stream.str();

        int baseline = 0;

        Size text_size =
            getTextSize(
                label,
                FONT_HERSHEY_SIMPLEX,
                0.5,
                1,
                &baseline);

        int tx = b.x;

        int ty =
            max(
                text_size.height + 5,
                b.y - 8);

        if (tx +
            text_size.width +
            10 >
            image.cols)
        {
            tx =
                max(
                    5,
                    image.cols -
                    text_size.width -
                    10);
        }

        // Label background.
        rectangle(
            image,
            Point(
                tx - 3,
                ty -
                text_size.height -
                3),
            Point(
                tx +
                text_size.width +
                3,
                ty + 3),
            Scalar(
                0,
                0,
                0),
            FILLED);

        rectangle(
            image,
            Point(
                tx - 3,
                ty -
                text_size.height -
                3),
            Point(
                tx +
                text_size.width +
                3,
                ty + 3),
            color,
            1);

        putText(
            image,
            label,
            Point(
                tx,
                ty),
            FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            LINE_AA);

        // Distance label.
        stringstream distance_stream;

        distance_stream
            << static_cast<int>(
                   distance)
            << " px";

        string distance_label =
            distance_stream.str();

        int mid_x =
            (center_x +
             b.cx()) /
            2;

        int mid_y =
            (center_y +
             b.cy()) /
            2;

        mid_x += 8;
        mid_y -= 8;

        if (mid_x > 20 &&
            mid_x <
                image.cols - 20 &&
            mid_y > 20 &&
            mid_y <
                image.rows - 20)
        {
            Size dsize =
                getTextSize(
                    distance_label,
                    FONT_HERSHEY_SIMPLEX,
                    0.35,
                    1,
                    &baseline);

            rectangle(
                image,
                Point(
                    mid_x - 3,
                    mid_y -
                    dsize.height -
                    2),
                Point(
                    mid_x +
                    dsize.width +
                    3,
                    mid_y + 3),
                Scalar(
                    0,
                    0,
                    0),
                FILLED);

            putText(
                image,
                distance_label,
                Point(
                    mid_x,
                    mid_y),
                FONT_HERSHEY_SIMPLEX,
                0.35,
                Scalar(
                    255,
                    255,
                    255),
                1,
                LINE_AA);
        }
    }

    // ------------------------------------------------------------------------
    // Information panel
    // ------------------------------------------------------------------------

    string mode_string =
        BalloonDetector::mode_to_string(
            mode);

    const int panel_width =
        min(
            480,
            image.cols - 10);

    const int panel_height =
        (frame_number >= 0) ? 110 : 82;

    rectangle(
        image,
        Point(
            5,
            5),
        Point(
            panel_width,
            panel_height),
        Scalar(
            0,
            0,
            0),
        FILLED);

    rectangle(
        image,
        Point(
            5,
            5),
        Point(
            panel_width,
            panel_height),
        Scalar(
            255,
            255,
            255),
        1);

    string line1 =
        "Mode: " +
        mode_string +
        " | Detections: " +
        to_string(
            detections.size());

    stringstream time_stream;

    time_stream
        << fixed
        << setprecision(2)
        << processing_time
        << " ms";

    string line2 =
        "Processing: " +
        time_stream.str();

    string line3 =
        "Center: (" +
        to_string(center_x) +
        "," +
        to_string(center_y) +
        ")";

    putText(
        image,
        line1,
        Point(
            15,
            23),
        FONT_HERSHEY_SIMPLEX,
        0.5,
        Scalar(
            255,
            255,
            255),
        1,
        LINE_AA);

    putText(
        image,
        line2,
        Point(
            15,
            43),
        FONT_HERSHEY_SIMPLEX,
        0.5,
        Scalar(
            200,
            200,
            200),
        1,
        LINE_AA);

    putText(
        image,
        line3,
        Point(
            15,
            63),
        FONT_HERSHEY_SIMPLEX,
        0.5,
        Scalar(
            200,
            200,
            200),
        1,
        LINE_AA);

    // Frame info for video.
    if (frame_number >= 0)
    {
        stringstream frame_stream;

        frame_stream
            << "Frame: "
            << frame_number
            << " | Time: "
            << fixed
            << setprecision(2)
            << timestamp
            << "s";

        string frame_info =
            frame_stream.str();

        putText(
            image,
            frame_info,
            Point(
                15,
                83),
            FONT_HERSHEY_SIMPLEX,
            0.5,
            Scalar(
                200,
                200,
                200),
            1,
            LINE_AA);
    }
}

// ============================================================================
// CLI
// ============================================================================

DetMode get_detection_mode()
{
    cout
        << "\n"
        << "============================================================\n"
        << "              BALLOON DETECTION ENGINE\n"
        << "============================================================\n"
        << "\n"
        << "Select detection mode:\n"
        << "\n"
        << "  [1] DoG       - Recommended\n"
        << "  [2] Hough     - Circle based\n"
        << "  [3] Hybrid    - DoG + Hough\n"
        << "  [0] Compare   - Run all modes\n"
        << "\n";

    string selection;

    cout
        << "Select (default 1): ";

    getline(
        cin,
        selection);

    if (selection == "2")
        return DetMode::HoughCircles;

    if (selection == "3")
        return DetMode::Hybrid;

    if (selection == "0")
        return DetMode::CompareAll;

    return DetMode::DoG;
}

// ============================================================================
// Read double safely
// ============================================================================

bool parse_double(
    const string& input,
    double& value)
{
    if (input.empty())
        return false;

    try
    {
        size_t pos = 0;

        double parsed =
            stod(
                input,
                &pos);

        if (pos != input.size())
            return false;

        value = parsed;

        return true;
    }
    catch (...)
    {
        return false;
    }
}

// ============================================================================
// Read float safely
// ============================================================================

bool parse_float(
    const string& input,
    float& value)
{
    if (input.empty())
        return false;

    try
    {
        size_t pos = 0;

        float parsed =
            stof(
                input,
                &pos);

        if (pos != input.size())
            return false;

        value = parsed;

        return true;
    }
    catch (...)
    {
        return false;
    }
}

// ============================================================================
// Read int safely
// ============================================================================

bool parse_int(
    const string& input,
    int& value)
{
    if (input.empty())
        return false;

    try
    {
        size_t pos = 0;

        int parsed =
            stoi(
                input,
                &pos);

        if (pos != input.size())
            return false;

        value = parsed;

        return true;
    }
    catch (...)
    {
        return false;
    }
}

// ============================================================================
// Detection Configuration
// ============================================================================

DetectionConfig get_detection_config()
{
    DetectionConfig config;

    cout
        << "\n"
        << "------------------------------------------------------------\n"
        << " Detection Configuration\n"
        << " Press ENTER to use default values\n"
        << "------------------------------------------------------------\n";

    string input;
    double d;
    float f;
    int i;

    // ------------------------------------------------------------------------
    // DoG
    // ------------------------------------------------------------------------

    cout
        << "\n[DoG]\n";

    cout
        << "Sigma1 ["
        << config.dog.sigma1
        << "]: ";

    getline(
        cin,
        input);

    if (parse_double(input, d))
        config.dog.sigma1 = d;

    cout
        << "Sigma2 ["
        << config.dog.sigma2
        << "]: ";

    getline(
        cin,
        input);

    if (parse_double(input, d))
        config.dog.sigma2 = d;

    cout
        << "Threshold ["
        << config.dog.threshold
        << "]: ";

    getline(
        cin,
        input);

    if (parse_double(input, d))
        config.dog.threshold =
            static_cast<int>(d);

    cout
        << "Adaptive threshold multiplier ["
        << config.dog.threshold_std_multiplier
        << "]: ";

    getline(
        cin,
        input);

    if (parse_double(input, d))
        config.dog.threshold_std_multiplier = d;

    cout
        << "Minimum area ["
        << config.dog.min_area
        << "]: ";

    getline(
        cin,
        input);

    if (parse_double(input, d))
        config.dog.min_area = d;

    cout
        << "Maximum area ["
        << config.dog.max_area
        << "]: ";

    getline(
        cin,
        input);

    if (parse_double(input, d))
        config.dog.max_area = d;

    cout
        << "Minimum solidity ["
        << config.dog.min_solidity
        << "]: ";

    getline(
        cin,
        input);

    if (parse_double(input, d))
        config.dog.min_solidity = d;

    cout
        << "Minimum circularity ["
        << config.dog.min_circularity
        << "]: ";

    getline(
        cin,
        input);

    if (parse_double(input, d))
        config.dog.min_circularity = d;

    // ------------------------------------------------------------------------
    // Hough
    // ------------------------------------------------------------------------

    cout
        << "\n[Hough]\n";

    cout
        << "Param1 ["
        << config.hough.param1
        << "]: ";

    getline(
        cin,
        input);

    if (parse_double(input, d))
        config.hough.param1 = d;

    cout
        << "Param2 ["
        << config.hough.param2
        << "]: ";

    getline(
        cin,
        input);

    if (parse_double(input, d))
        config.hough.param2 = d;

    cout
        << "Minimum radius % ["
        << config.hough.min_radius_percent
        << "]: ";

    getline(
        cin,
        input);

    if (parse_double(input, d))
        config.hough.min_radius_percent =
            static_cast<int>(d);

    cout
        << "Maximum radius % ["
        << config.hough.max_radius_percent
        << "]: ";

    getline(
        cin,
        input);

    if (parse_double(input, d))
        config.hough.max_radius_percent =
            static_cast<int>(d);

    // ------------------------------------------------------------------------
    // Video
    // ------------------------------------------------------------------------

    cout
        << "\n[Video Options]\n";

    cout
        << "Process every Nth frame ["
        << config.video.frame_step
        << "]: ";

    getline(
        cin,
        input);

    if (parse_int(input, i))
        config.video.frame_step = max(1, i);

    cout
        << "Max frames (0=unlimited) ["
        << config.video.max_frames
        << "]: ";

    getline(
        cin,
        input);

    if (parse_int(input, i))
        config.video.max_frames = max(0, i);

    cout
        << "Resize factor (0=no resize) ["
        << config.video.resize_factor
        << "]: ";

    getline(
        cin,
        input);

    if (parse_double(input, d))
        config.video.resize_factor = max(0.0, d);

    cout
        << "Show preview during processing (y/n) [y]: ";

    getline(
        cin,
        input);

    if (!input.empty() &&
        (input == "n" ||
         input == "N"))
    {
        config.video.show_preview = false;
    }

    // ------------------------------------------------------------------------
    // Global
    // ------------------------------------------------------------------------

    cout
        << "\n[Global]\n";

    cout
        << "Confidence threshold ["
        << config.confidence_threshold
        << "]: ";

    getline(
        cin,
        input);

    if (parse_float(input, f))
        config.confidence_threshold = f;

    cout
        << "Enable NMS (y/n) [y]: ";

    getline(
        cin,
        input);

    if (!input.empty() &&
        (input == "n" ||
         input == "N"))
    {
        config.enable_nms = false;
    }

    return config;
}

// ============================================================================
// Run directory
// ============================================================================

string create_run_folder()
{
    auto now =
        chrono::system_clock::now();

    time_t time_now =
        chrono::system_clock::to_time_t(
            now);

    tm local_tm{};

#ifdef _WIN32
    localtime_s(
        &local_tm,
        &time_now);
#else
    localtime_r(
        &time_now,
        &local_tm);
#endif

    stringstream ss;

    ss << "Res/run_"
       << put_time(
              &local_tm,
              "%Y%m%d_%H%M%S");

    string directory =
        ss.str();

    try
    {
        fs::create_directories(
            directory);
    }
    catch (const exception& e)
    {
        cerr
            << "[ERROR] Could not create "
               "output directory: "
            << e.what()
            << "\n";
    }

    return directory;
}

// ============================================================================
// Save Results - Image
// ============================================================================

void save_image_results(
    const Mat& image,
    const vector<BBox>& detections,
    const string& run_directory,
    DetMode mode,
    double processing_time)
{
    string mode_string =
        BalloonDetector::mode_to_string(
            mode);

    // ------------------------------------------------------------------------
    // Image
    // ------------------------------------------------------------------------

    string image_path =
        run_directory +
        "/Output_" +
        mode_string +
        "_Balloon.jpg";

    if (!imwrite(
            image_path,
            image))
    {
        cerr
            << "[ERROR] Failed to save image: "
            << image_path
            << "\n";
    }
    else
    {
        cout
            << "[INFO] Image saved: "
            << image_path
            << "\n";
    }

    // ------------------------------------------------------------------------
    // Text report
    // ------------------------------------------------------------------------

    string text_path =
        run_directory +
        "/detections_" +
        mode_string +
        ".txt";

    ofstream output(
        text_path);

    if (!output.is_open())
    {
        cerr
            << "[ERROR] Failed to create report: "
            << text_path
            << "\n";

        return;
    }

    const int center_x =
        image.cols / 2;

    const int center_y =
        image.rows / 2;

    output
        << "============================================================\n"
        << "BALLOON DETECTION RESULT\n"
        << "============================================================\n";

    output
        << "Mode              : "
        << mode_string
        << "\n";

    output
        << "Image Size        : "
        << image.cols
        << " x "
        << image.rows
        << "\n";

    output
        << "Processing Time   : "
        << fixed
        << setprecision(3)
        << processing_time
        << " ms\n";

    output
        << "Number of Targets : "
        << detections.size()
        << "\n";

    output
        << "\n";

    output
        << "ID\tX\tY\tW\tH\t"
        << "Confidence\tType\t"
        << "Circularity\tContrast\t"
        << "Angle(deg)\tDistance(px)\n";

    output
        << "-------------------------------------------------------------------------------\n";

    for (const BBox& b :
         detections)
    {
        double angle =
            atan2(
                static_cast<double>(
                    b.cy() - center_y),
                static_cast<double>(
                    b.cx() - center_x))
            *
            180.0 /
            CV_PI;

        double distance =
            hypot(
                static_cast<double>(
                    b.cx() - center_x),
                static_cast<double>(
                    b.cy() - center_y));

        output
            << b.id << "\t"
            << b.x << "\t"
            << b.y << "\t"
            << b.w << "\t"
            << b.h << "\t"
            << fixed
            << setprecision(4)
            << b.confidence << "\t"
            << b.type << "\t"
            << b.circularity << "\t"
            << b.contrast_score << "\t"
            << angle << "\t"
            << distance
            << "\n";
    }

    output.close();

    cout
        << "[INFO] Report saved: "
        << text_path
        << "\n";
}

// ============================================================================
// Save Results - Video
// ============================================================================

void save_video_results(
    const string& run_directory,
    DetMode mode,
    const vector<DetectionResult>& results,
    const VideoConfig& video_config)
{
    if (results.empty())
        return;

    string mode_string =
        BalloonDetector::mode_to_string(
            mode);

    // ------------------------------------------------------------------------
    // Summary report
    // ------------------------------------------------------------------------

    string summary_path =
        run_directory +
        "/video_summary_" +
        mode_string +
        ".txt";

    ofstream summary(
        summary_path);

    if (summary.is_open())
    {
        summary
            << "============================================================\n"
            << "VIDEO DETECTION SUMMARY\n"
            << "============================================================\n";

        summary
            << "Mode              : "
            << mode_string
            << "\n";

        summary
            << "Total Frames      : "
            << results.size()
            << "\n";

        // Count total detections
        int total_detections = 0;
        for (const auto& r : results)
        {
            total_detections += r.detections.size();
        }

        summary
            << "Total Detections  : "
            << total_detections
            << "\n";

        summary
            << "Average Time      : "
            << fixed
            << setprecision(3);

        double avg_time = 0.0;
        for (const auto& r : results)
        {
            avg_time += r.processing_time;
        }
        avg_time /= results.size();

        summary
            << avg_time
            << " ms\n";

        summary
            << "\n";
        summary
            << "Frame\tTimestamp\tDetections\n";
        summary
            << "----------------------------------------\n";

        for (const auto& r : results)
        {
            summary
                << r.frame_number << "\t"
                << fixed
                << setprecision(2)
                << r.timestamp << "\t"
                << r.detections.size()
                << "\n";
        }

        summary.close();

        cout
            << "[INFO] Video summary saved: "
            << summary_path
            << "\n";
    }

    // ------------------------------------------------------------------------
    // Per-frame detection details
    // ------------------------------------------------------------------------

    string detailed_path =
        run_directory +
        "/video_detections_" +
        mode_string +
        ".csv";

    ofstream detailed(
        detailed_path);

    if (detailed.is_open())
    {
        detailed
            << "Frame,Timestamp,ID,X,Y,W,H,Confidence,Type,Angle(deg),Distance(px)\n";

        for (const auto& r : results)
        {
            const int center_x = 0;  // Will be set per frame
            const int center_y = 0;

            for (const auto& b : r.detections)
            {
                double angle =
                    atan2(
                        static_cast<double>(
                            b.cy() - center_y),
                        static_cast<double>(
                            b.cx() - center_x))
                    *
                    180.0 /
                    CV_PI;

                double distance =
                    hypot(
                        static_cast<double>(
                            b.cx() - center_x),
                        static_cast<double>(
                            b.cy() - center_y));

                detailed
                    << r.frame_number << ","
                    << fixed
                    << setprecision(2)
                    << r.timestamp << ","
                    << b.id << ","
                    << b.x << ","
                    << b.y << ","
                    << b.w << ","
                    << b.h << ","
                    << b.confidence << ","
                    << b.type << ","
                    << angle << ","
                    << distance
                    << "\n";
            }
        }

        detailed.close();

        cout
            << "[INFO] Detailed detections saved: "
            << detailed_path
            << "\n";
    }
}

// ============================================================================
// Print detections
// ============================================================================

void print_detections(
    const vector<BBox>& detections,
    const Mat& image)
{
    const int center_x =
        image.cols / 2;

    const int center_y =
        image.rows / 2;

    if (detections.empty())
    {
        cout
            << "  No targets detected.\n";

        return;
    }

    cout
        << "  Targets found: "
        << detections.size()
        << "\n";

    for (const BBox& b :
         detections)
    {
        double angle =
            atan2(
                static_cast<double>(
                    b.cy() - center_y),
                static_cast<double>(
                    b.cx() - center_x))
            *
            180.0 /
            CV_PI;

        double distance =
            hypot(
                static_cast<double>(
                    b.cx() - center_x),
                static_cast<double>(
                    b.cy() - center_y));

        cout
            << "    ["
            << b.id
            << "] "
            << "BBox=("
            << b.x
            << ","
            << b.y
            << ","
            << b.w
            << "x"
            << b.h
            << ") "
            << "Conf="
            << fixed
            << setprecision(3)
            << b.confidence
            << " "
            << "Type="
            << b.type
            << " "
            << "Angle="
            << static_cast<int>(
                   angle)
            << " deg "
            << "Distance="
            << static_cast<int>(
                   distance)
            << " px\n";
    }
}

// ============================================================================
// Process Image
// ============================================================================

void process_image(
    const Mat& image,
    const string& image_path,
    BalloonDetector& detector,
    DetMode mode,
    const string& run_directory)
{
    string mode_string =
        BalloonDetector::mode_to_string(
            mode);

    cout
        << "\n"
        << "============================================================\n"
        << "Processing Image: "
        << fs::path(image_path).filename().string()
        << "\n"
        << "Mode: "
        << mode_string
        << "\n"
        << "============================================================\n";

    auto start =
        chrono::steady_clock::now();

    vector<BBox> detections =
        detector.detect(
            image,
            mode);

    auto end =
        chrono::steady_clock::now();

    chrono::duration<double,
                     milli> elapsed =
        end - start;

    double processing_time =
        elapsed.count();

    cout
        << "\n[RESULT]\n"
        << "  Mode        : "
        << mode_string
        << "\n"
        << "  Time        : "
        << fixed
        << setprecision(3)
        << processing_time
        << " ms\n"
        << "  Detections  : "
        << detections.size()
        << "\n";

    print_detections(
        detections,
        image);

    // Visualization.
    Mat output =
        image.clone();

    draw_detections_clean(
        output,
        detections,
        mode,
        processing_time);

    save_image_results(
        output,
        detections,
        run_directory,
        mode,
        processing_time);

    string window_name =
        "Balloon Detector - " +
        mode_string;

    namedWindow(
        window_name,
        WINDOW_NORMAL);

    imshow(
        window_name,
        output);
}

// ============================================================================
// Process Video
// ============================================================================

void process_video(
    const string& video_path,
    BalloonDetector& detector,
    DetMode mode,
    const string& run_directory)
{
    string mode_string =
        BalloonDetector::mode_to_string(
            mode);

    cout
        << "\n"
        << "============================================================\n"
        << "Processing Video: "
        << fs::path(video_path).filename().string()
        << "\n"
        << "Mode: "
        << mode_string
        << "\n"
        << "============================================================\n";

    // ------------------------------------------------------------------------
    // Open video
    // ------------------------------------------------------------------------

    VideoCapture cap(
        video_path);

    if (!cap.isOpened())
    {
        cerr
            << "[ERROR] Could not open video: "
            << video_path
            << "\n";

        return;
    }

    // ------------------------------------------------------------------------
    // Video properties
    // ------------------------------------------------------------------------

    double fps =
        cap.get(
            CAP_PROP_FPS);

    int total_frames =
        static_cast<int>(
            cap.get(
                CAP_PROP_FRAME_COUNT));

    Size frame_size(
        static_cast<int>(
            cap.get(
                CAP_PROP_FRAME_WIDTH)),
        static_cast<int>(
            cap.get(
                CAP_PROP_FRAME_HEIGHT)));

    cout
        << "\n[VIDEO INFO]\n"
        << "  FPS          : "
        << fps
        << "\n"
        << "  Total Frames : "
        << total_frames
        << "\n"
        << "  Size         : "
        << frame_size.width
        << "x"
        << frame_size.height
        << "\n";

    // ------------------------------------------------------------------------
    // Output video writer
    // ------------------------------------------------------------------------

    VideoWriter writer;

    Size output_size = frame_size;

    if (detector.config().video.resize_factor > 0.0)
    {
        output_size.width =
            static_cast<int>(
                frame_size.width *
                detector.config().video.resize_factor);

        output_size.height =
            static_cast<int>(
                frame_size.height *
                detector.config().video.resize_factor);

        output_size.width = max(16, output_size.width);
        output_size.height = max(16, output_size.height);
    }

    string output_video_path =
        run_directory +
        "/Output_" +
        mode_string +
        "_Video.avi";

    if (detector.config().video.save_output_video)
    {
        int fourcc =
            VideoWriter::fourcc(
                'M',
                'J',
                'P',
                'G');

        writer.open(
            output_video_path,
            fourcc,
            detector.config().video.output_fps,
            output_size);

        if (!writer.isOpened())
        {
            cerr
                << "[WARNING] Could not create output video.\n";
        }
        else
        {
            cout
                << "[INFO] Output video: "
                << output_video_path
                << "\n";
        }
    }

    // ------------------------------------------------------------------------
    // Process frames
    // ------------------------------------------------------------------------

    vector<DetectionResult> results;

    Mat frame;
    int frame_count = 0;
    int processed_count = 0;
    int max_frames = detector.config().video.max_frames;
    int frame_step = max(1, detector.config().video.frame_step);

    chrono::steady_clock::time_point total_start =
        chrono::steady_clock::now();

    while (cap.read(frame))
    {
        frame_count++;

        // Skip frames.
        if ((frame_count - 1) % frame_step != 0)
            continue;

        // Check max frames.
        if (max_frames > 0 &&
            processed_count >= max_frames)
        {
            break;
        }

        // Resize if needed.
        Mat process_frame = frame;

        if (detector.config().video.resize_factor > 0.0)
        {
            resize(
                frame,
                process_frame,
                output_size,
                0,
                0,
                INTER_AREA);
        }

        // --------------------------------------------------------------------
        // Detect
        // --------------------------------------------------------------------

        auto start =
            chrono::steady_clock::now();

        vector<BBox> detections =
            detector.detect(
                process_frame,
                mode);

        auto end =
            chrono::steady_clock::now();

        chrono::duration<double,
                         milli> elapsed =
            end - start;

        double processing_time =
            elapsed.count();

        // --------------------------------------------------------------------
        // Store result
        // --------------------------------------------------------------------

        DetectionResult result;

        result.frame_number = frame_count;
        result.timestamp =
            static_cast<double>(
                frame_count) /
            fps;

        result.detections = detections;
        result.processing_time = processing_time;

        results.push_back(result);

        // --------------------------------------------------------------------
        // Visualization
        // --------------------------------------------------------------------

        Mat display_frame =
            process_frame.clone();

        draw_detections_clean(
            display_frame,
            detections,
            mode,
            processing_time,
            frame_count,
            result.timestamp);

        // --------------------------------------------------------------------
        // Write output video
        // --------------------------------------------------------------------

        if (writer.isOpened())
        {
            writer.write(
                display_frame);
        }

        // --------------------------------------------------------------------
        // Display
        // --------------------------------------------------------------------

        if (detector.config().video.show_preview)
        {
            string window_name =
                "Balloon Detector - " +
                mode_string +
                " (Frame " +
                to_string(frame_count) +
                ")";

            namedWindow(
                window_name,
                WINDOW_NORMAL);

            imshow(
                window_name,
                display_frame);

            int key =
                waitKey(
                    detector.config().video.preview_delay_ms);

            // ESC to exit.
            if (key == 27)
            {
                cout
                    << "[INFO] User interrupted.\n";

                break;
            }
        }

        // --------------------------------------------------------------------
        // Progress
        // --------------------------------------------------------------------

        processed_count++;

        if (processed_count % 10 == 0)
        {
            cout
                << "\r[PROGRESS] Frame "
                << frame_count
                << "/"
                << total_frames
                << " ("
                << processed_count
                << " processed)    "
                << flush;
        }
    }

    chrono::steady_clock::time_point total_end =
        chrono::steady_clock::now();

    chrono::duration<double>
        total_time =
        total_end - total_start;

    cout
        << "\n[INFO] Processing complete.\n"
        << "  Frames processed: "
        << processed_count
        << "\n"
        << "  Total time      : "
        << fixed
        << setprecision(2)
        << total_time.count()
        << "s\n";

    // ------------------------------------------------------------------------
    // Save results
    // ------------------------------------------------------------------------

    save_video_results(
        run_directory,
        mode,
        results,
        detector.config().video);

    // Release resources.
    cap.release();

    if (writer.isOpened())
    {
        writer.release();
        cout
            << "[INFO] Output video saved: "
            << output_video_path
            << "\n";
    }

    cout
        << "[INFO] Video processing complete.\n";
}

// ============================================================================
// Main
// ============================================================================

int main(
    int argc,
    char** argv)
{
    cout
        << "\n"
        << "============================================================\n"
        << "          ADVANCED BALLOON DETECTOR\n";
    cout
        << "          Image / Video Support (FFmpeg)\n";
    cout
        << "          DoG / Hough / Hybrid\n"
        << "============================================================\n";

    // ------------------------------------------------------------------------
    // Argument check
    // ------------------------------------------------------------------------

    if (argc < 2)
    {
        cerr
            << "\nUsage:\n"
            << "  "
            << argv[0]
            << " <input_path>\n\n"
            << "Supported formats:\n"
            << "  Images  : jpg, jpeg, png, bmp, tiff, webp, gif, ppm, pgm, pbm\n"
            << "  Videos  : mp4, avi, mov, mkv, flv, wmv, webm, m4v, mpg, mpeg, 3gp, ogv\n"
            << "\n"
            << "Example:\n"
            << "  "
            << argv[0]
            << " balloon.jpg\n"
            << "  "
            << argv[0]
            << " video.mp4\n\n";

        return EXIT_FAILURE;
    }

    string input_path =
        argv[1];

    // ------------------------------------------------------------------------
    // Detect input type
    // ------------------------------------------------------------------------

    if (!fs::exists(input_path))
    {
        cerr
            << "[ERROR] File or directory does not exist:\n"
            << input_path
            << "\n";

        return EXIT_FAILURE;
    }

    InputType input_type =
        detect_input_type(input_path);

    if (input_type == InputType::Invalid)
    {
        cerr
            << "[ERROR] Unsupported file type:\n"
            << input_path
            << "\n";

        return EXIT_FAILURE;
    }

    string type_string =
        (input_type == InputType::Image) ?
        "Image" : "Video";

    cout
        << "\n[INFO] Input type: "
        << type_string
        << "\n";

    // ------------------------------------------------------------------------
    // Select mode
    // ------------------------------------------------------------------------

    DetMode mode =
        get_detection_mode();

    // ------------------------------------------------------------------------
    // Configuration
    // ------------------------------------------------------------------------

    DetectionConfig config =
        get_detection_config();

    // ------------------------------------------------------------------------
    // Detector
    // ------------------------------------------------------------------------

    BalloonDetector detector(
        config);

    // ------------------------------------------------------------------------
    // Output folder
    // ------------------------------------------------------------------------

    string run_directory =
        create_run_folder();

    cout
        << "\n[INFO] Output directory: "
        << run_directory
        << "\n";

    // ------------------------------------------------------------------------
    // Process based on input type
    // ------------------------------------------------------------------------

    if (input_type == InputType::Image)
    {
        // --------------------------------------------------------------------
        // Load image
        // --------------------------------------------------------------------

        Mat image =
            imread(
                input_path,
                IMREAD_UNCHANGED);

        if (image.empty())
        {
            cerr
                << "[ERROR] Could not load image:\n"
                << input_path
                << "\n";

            return EXIT_FAILURE;
        }

        cout
            << "\n[INFO] Input Image\n"
            << "  File     : "
            << fs::path(
                   input_path)
                   .filename()
                   .string()
            << "\n"
            << "  Size     : "
            << image.cols
            << " x "
            << image.rows
            << "\n"
            << "  Channels : "
            << image.channels()
            << "\n";

        // --------------------------------------------------------------------
        // Process
        // --------------------------------------------------------------------

        if (mode == DetMode::CompareAll)
        {
            process_image(
                image,
                input_path,
                detector,
                DetMode::DoG,
                run_directory);

            process_image(
                image,
                input_path,
                detector,
                DetMode::HoughCircles,
                run_directory);

            process_image(
                image,
                input_path,
                detector,
                DetMode::Hybrid,
                run_directory);
        }
        else
        {
            process_image(
                image,
                input_path,
                detector,
                mode,
                run_directory);
        }
    }
    else  // Video
    {
        process_video(
            input_path,
            detector,
            mode,
            run_directory);
    }

    // ------------------------------------------------------------------------
    // Wait for display
    // ------------------------------------------------------------------------

    cout
        << "\n"
        << "============================================================\n"
        << "Processing complete.\n"
        << "Results saved in:\n"
        << run_directory
        << "\n"
        << "Press any key in an OpenCV window to exit.\n"
        << "============================================================\n";

    if (config.video.show_preview ||
        input_type == InputType::Image)
    {
        waitKey(0);
    }

    destroyAllWindows();

    return EXIT_SUCCESS;
}
