// // Classical OpenCV detector + automatic YOLO annotation exporter.
// // Output dataset layout per run:
// //   run_.../images/frame_000001.jpg
// //   run_.../labels/frame_000001.txt
// // Each label line uses YOLO detection format:
// //   <class_id> <x_center_norm> <y_center_norm> <width_norm> <height_norm>
// //
// #include <opencv2/opencv.hpp>

// #include <algorithm>
// #include <atomic>
// #include <chrono>
// #include <cmath>
// #include <condition_variable>
// #include <cstdlib>
// #include <filesystem>
// #include <fstream>
// #include <iomanip>
// #include <iostream>
// #include <mutex>
// #include <queue>
// #include <sstream>
// #include <string>
// #include <thread>
// #include <vector>

// #ifdef _WIN32
// #include <io.h>
// #else
// #include <unistd.h>
// #endif

// using namespace std;
// using namespace cv;
// namespace fs = std::filesystem;

// enum class DetMode
// {
//     DoG,
//     HoughCircles,
//     Hybrid
// };

// enum class PriorMode
// {
//     Size,
//     Contrast,
//     Center
// };

// struct TrackerConfig
// {
//     float CONF_THRESH = 0.5f;
//     float HYBRID_LOG_WEIGHT = 0.5f;
//     float DOG_THRESHOLD = -1.0f;
//     float DOG_STDDEV_FACTOR = 1.25f;
//     float MIN_DOG_THRESHOLD = 8.0f;
//     float MAX_DOG_THRESHOLD = 48.0f;
//     double MIN_AREA_FRAC = 0.00002;
//     double MAX_AREA_FRAC = 0.35;
//     double MIN_EXTENT = 0.05;
//     double MIN_ASPECT = 0.10;
//     double MAX_ASPECT = 10.0;
//     double MAX_VERTICAL_POSITION = 0.92;
//     int MIN_CONTOUR_AREA = 6;
//     int HOUGH_MIN_RADIUS = 0;
//     int HOUGH_MAX_RADIUS = 0;
//     int HOUGH_MIN_DIST = 0;
//     double HOUGH_PARAM1 = 70.0;
//     double HOUGH_PARAM2 = 22.0;
// };

// struct BBox
// {
//     int x, y, w, h;
//     float score = 0.0f;
//     string type = "detection";
//     double area = 0.0;
//     double aspect = 0.0;
//     double extent = 0.0;
//     double dog_mean = 0.0;
//     bool center_protected = false;
//     bool center_held = false;          // true only when -C reuses last center target during a short detector miss
//     double center_distance_px = -1.0;

//     int cx() const { return x + w / 2; }
//     int cy() const { return y + h / 2; }
//     Rect rect() const { return Rect(x, y, w, h); }
// };

// class BlobDetector
// {
// public:
//     BlobDetector(const TrackerConfig& cfg, int roi_w, int roi_h, PriorMode prior = PriorMode::Center, bool debug_mode = false, bool is_video = false)
//         : cfg_(cfg), roi_w_(roi_w), roi_h_(roi_h), prior_(prior), debug_mode_(debug_mode), is_video_(is_video) {
//     }

//     Mat blob_mask(const Mat& r32, float thresh = 18.f) const
//     {
//         Mat pos_clip, neg_clip;
//         Mat pos_mask, neg_mask;

//         max(r32, 0, pos_clip);
//         pos_clip.convertTo(pos_clip, CV_8U);
//         threshold(pos_clip, pos_mask, thresh, 255, THRESH_BINARY);

//         Mat neg_r32 = -r32;
//         max(neg_r32, 0, neg_clip);
//         neg_clip.convertTo(neg_clip, CV_8U);
//         threshold(neg_clip, neg_mask, thresh, 255, THRESH_BINARY);

//         Mat out;
//         bitwise_or(pos_mask, neg_mask, out);
//         return out;
//     }

//     std::vector<BBox> apply_nms(std::vector<BBox>& boxes, float iou_threshold = 0.12f) const
//     {
//         if (boxes.empty()) return {};

//         const Rect image_bounds(0, 0, std::max(1, roi_w_), std::max(1, roi_h_));
//         sort(boxes.begin(), boxes.end(), [](const BBox& a, const BBox& b) {
//             return a.score > b.score;
//             });

//         vector<BBox> result;
//         vector<bool> suppressed(boxes.size(), false);

//         for (size_t i = 0; i < boxes.size(); ++i)
//         {
//             if (suppressed[i]) continue;
//             result.push_back(boxes[i]);

//             for (size_t j = i + 1; j < boxes.size(); ++j)
//             {
//                 if (suppressed[j]) continue;

//                 Rect a = boxes[i].rect() & image_bounds;
//                 Rect b = boxes[j].rect() & image_bounds;
//                 if (a.empty() || b.empty()) continue;

//                 Rect intersection = a & b;
//                 float intersection_area = intersection.area();
//                 float union_area = a.area() + b.area() - intersection_area;
//                 float iou = (union_area > 0) ? (intersection_area / union_area) : 0.0f;

//                 if (iou > iou_threshold)
//                 {
//                     suppressed[j] = true;
//                 }
//             }
//         }
//         return result;
//     }

//     void handle_debug_pause(const string& message) const
//     {
//         if (!debug_mode_) return;

//         if (is_video_) {
//             // Non-blocking or short wait for video streams to prevent complete lockup
//             waitKey(15);
//         }
//         else {
//             // Full interactive pause for single images
//             cout << message << " Press any key to continue...\n";
//             waitKey(0);
//         }
//     }

//     std::vector<BBox> select_all(const Mat& gray,
//         const Mat& fused_mask,
//         const Mat& dog_conf,
//         const Mat& log_conf,
//         float hybrid_w) const
//     {
//         std::vector<std::vector<Point>> contours;
//         findContours(fused_mask, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

//         const double roi_area = std::max(1.0, static_cast<double>(gray.cols) * gray.rows);
//         const double min_area = std::max(static_cast<double>(cfg_.MIN_CONTOUR_AREA), roi_area * cfg_.MIN_AREA_FRAC);
//         const double max_rel_percent = cfg_.MAX_AREA_FRAC * 100.0;
//         std::vector<BBox> candidates;

//         for (auto& cnt : contours)
//         {
//             double area = contourArea(cnt);
//             if (area <= min_area) continue;

//             Rect r = boundingRect(cnt);
//             if (r.width <= 0 || r.height <= 0) continue;
//             if (r.x < 0 || r.y < 0 || r.x + r.width > gray.cols || r.y + r.height > gray.rows) continue;
//             if (r.y > gray.rows * cfg_.MAX_VERTICAL_POSITION) continue;

//             double aspect = static_cast<double>(r.width) / r.height;
//             double extent = area / (r.width * r.height);
//             double rel = (area / roi_area) * 100.0;

//             if (!(cfg_.MIN_ASPECT < aspect && aspect < cfg_.MAX_ASPECT &&
//                 extent > cfg_.MIN_EXTENT && rel < max_rel_percent)) continue;

//             double dog_mean = dog_conf.empty() ? 0.0 : mean(dog_conf(r))[0];
//             double log_mean = log_conf.empty() ? 0.0 : mean(log_conf(r))[0];

//             double dog_n = std::min(dog_mean / 50.0, 1.0);
//             double log_n = std::min(log_mean / 50.0, 1.0);
//             double hyb_conf = (1.0 - hybrid_w) * dog_n + hybrid_w * log_n;
//             double score = area * (0.5 + hyb_conf) * prior_weight(r, gray.size());

//             candidates.push_back(BBox{ r.x, r.y, r.width, r.height, (float)score, "dog_detected", area, aspect, extent, dog_mean });
//         }

//         auto final_boxes = apply_nms(candidates, 0.18f);

//         if (debug_mode_)
//         {
//             Mat contour_vis;
//             cvtColor(gray, contour_vis, COLOR_GRAY2BGR);
//             drawContours(contour_vis, contours, -1, Scalar(0, 255, 0), 1);
//             for (const auto& b : final_boxes)
//             {
//                 // rectangle(contour_vis, b.rect(), Scalar(0, 0, 255), 2);
//             }
//             namedWindow("Debug: Contour Filtering & NMS", WINDOW_NORMAL);
//             imshow("Debug: Contour Filtering & NMS", contour_vis);
//             handle_debug_pause("[DEBUG] Contours and NMS results displayed.");
//         }

//         return final_boxes;
//     }

//     vector<BBox> detect_dog_all(const Mat& roi) const
//     {
//         if (roi.empty()) return {};

//         Mat gray;
//         if (roi.channels() == 3) cvtColor(roi, gray, COLOR_BGR2GRAY);
//         else gray = roi.clone();

//         if (debug_mode_)
//         {
//             namedWindow("Debug: Grayscale Input", WINDOW_NORMAL);
//             imshow("Debug: Grayscale Input", gray);
//         }

//         Mat g1a, g2a, dog_a;
//         GaussianBlur(gray, g1a, { 3, 3 }, 0.8);
//         GaussianBlur(gray, g2a, { 7, 7 }, 2.0);
//         subtract(g1a, g2a, dog_a, noArray(), CV_16S);

//         Mat g1b, g2b, dog_b;
//         GaussianBlur(gray, g1b, { 5, 5 }, 1.5);
//         GaussianBlur(gray, g2b, { 11, 11 }, 3.5);
//         subtract(g1b, g2b, dog_b, noArray(), CV_16S);

//         Mat dog_a32, dog_b32;
//         dog_a.convertTo(dog_a32, CV_32F);
//         dog_b.convertTo(dog_b32, CV_32F);

//         if (debug_mode_)
//         {
//             Mat vis_a, vis_b;
//             normalize(dog_a32, vis_a, 0, 255, NORM_MINMAX, CV_8U);
//             normalize(dog_b32, vis_b, 0, 255, NORM_MINMAX, CV_8U);

//             namedWindow("Debug: DoG Scale A (3x3 - 7x7)", WINDOW_NORMAL);
//             namedWindow("Debug: DoG Scale B (5x5 - 11x11)", WINDOW_NORMAL);
//             imshow("Debug: DoG Scale A (3x3 - 7x7)", vis_a);
//             imshow("Debug: DoG Scale B (5x5 - 11x11)", vis_b);
//         }

//         Mat abs_a, abs_b, dog_conf;
//         absdiff(dog_a32, Scalar(0), abs_a);
//         absdiff(dog_b32, Scalar(0), abs_b);
//         max(abs_a, abs_b, dog_conf);

//         const float dog_thresh = adaptive_response_threshold(dog_conf);
//         Mat m1 = blob_mask(dog_a32, dog_thresh);
//         Mat m2 = blob_mask(dog_b32, dog_thresh);
//         Mat combined;
//         bitwise_or(m1, m2, combined);
//         cleanup_mask(combined);

//         if (debug_mode_)
//         {
//             namedWindow("Debug: Fused Thresholded Mask", WINDOW_NORMAL);
//             imshow("Debug: Fused Thresholded Mask", combined);
//             handle_debug_pause("[DEBUG] DoG scale-space maps displayed.");
//         }

//         return select_all(gray, combined, dog_conf, Mat{}, 0.0f);
//     }

//     vector<BBox> detect_hough_circles(const Mat& img) const
//     {
//         if (img.empty()) return {};

//         Mat gray;
//         // FIXED: Passing 'img' instead of uninitialized/empty 'gray' to cvtColor
//         if (img.channels() == 3) cvtColor(img, gray, COLOR_BGR2GRAY);
//         else gray = img.clone();

//         Mat smoothed;
//         GaussianBlur(gray, smoothed, Size(5, 5), 1.2);

//         if (debug_mode_)
//         {
//             Mat edges;
//             Canny(smoothed, edges, 35, 70);
//             namedWindow("Debug: Hough Smoothed Input", WINDOW_NORMAL);
//             namedWindow("Debug: Canny Edge Map for Accumulator", WINDOW_NORMAL);
//             imshow("Debug: Hough Smoothed Input", smoothed);
//             imshow("Debug: Canny Edge Map for Accumulator", edges);
//             handle_debug_pause("[DEBUG] Hough pre-processing & edges displayed.");
//         }

//         vector<Vec3f> circles;
//         const int min_dim = std::max(1, std::min(gray.cols, gray.rows));
//         int min_radius = (cfg_.HOUGH_MIN_RADIUS > 0) ? cfg_.HOUGH_MIN_RADIUS : std::max(3, min_dim / 180);
//         int max_radius = (cfg_.HOUGH_MAX_RADIUS > 0) ? cfg_.HOUGH_MAX_RADIUS : std::max(min_radius + 2, min_dim / 24);
//         int min_dist = (cfg_.HOUGH_MIN_DIST > 0) ? cfg_.HOUGH_MIN_DIST : std::max(8, min_dim / 14);

//         HoughCircles(smoothed, circles, HOUGH_GRADIENT, 1.2,
//             min_dist, cfg_.HOUGH_PARAM1, cfg_.HOUGH_PARAM2, min_radius, max_radius);

//         vector<BBox> boxes;
//         for (const auto& c : circles)
//         {
//             int cx = cvRound(c[0]);
//             int cy = cvRound(c[1]);
//             int r = cvRound(c[2]);

//             int x = max(0, cx - r);
//             int y = max(0, cy - r);
//             int w = min(img.cols - x, 2 * r);
//             int h = min(img.rows - y, 2 * r);

//             if (w <= 1 || h <= 1) continue;

//             if ((y + h) < (img.rows * cfg_.MAX_VERTICAL_POSITION))
//             {
//                 double area_approx = CV_PI * r * r;
//                 boxes.push_back(BBox{ x, y, w, h, 0.95f, "hough_circle", area_approx, 1.0, 0.78, 25.0 });
//             }
//         }

//         auto final_boxes = apply_nms(boxes, 0.12f);

//         if (debug_mode_)
//         {
//             Mat hough_vis;
//             cvtColor(gray, hough_vis, COLOR_GRAY2BGR);
//             for (const auto& b : final_boxes)
//             {
//                 rectangle(hough_vis, b.rect(), Scalar(255, 0, 0), 2);
//             }
//             namedWindow("Debug: Hough Circle NMS Output", WINDOW_NORMAL);
//             imshow("Debug: Hough Circle NMS Output", hough_vis);
//             handle_debug_pause("[DEBUG] Hough circles displayed.");
//         }

//         return final_boxes;
//     }

//     vector<BBox> detect_hybrid_all(const Mat& img) const
//     {
//         auto dog_boxes = detect_dog_all(img);
//         auto hough_boxes = detect_hough_circles(img);

//         vector<BBox> combined = dog_boxes;
//         for (const auto& hb : hough_boxes)
//         {
//             bool matched = false;
//             for (auto& db : dog_boxes)
//             {
//                 if ((db.rect() & hb.rect()).area() > 0)
//                 {
//                     matched = true;
//                     break;
//                 }
//             }
//             if (!matched) combined.push_back(hb);
//         }
//         for (auto& b : combined) b.type = "hybrid";
//         return apply_nms(combined, 0.15f);
//     }

// private:
//     float adaptive_response_threshold(const Mat& response) const
//     {
//         if (cfg_.DOG_THRESHOLD > 0.0f) return cfg_.DOG_THRESHOLD;

//         Mat abs_response;
//         absdiff(response, Scalar(0), abs_response);

//         Scalar mean_val, stddev_val;
//         meanStdDev(abs_response, mean_val, stddev_val);

//         const double raw = mean_val[0] + cfg_.DOG_STDDEV_FACTOR * stddev_val[0];
//         return static_cast<float>(std::clamp(raw,
//             static_cast<double>(cfg_.MIN_DOG_THRESHOLD),
//             static_cast<double>(cfg_.MAX_DOG_THRESHOLD)));
//     }

//     void cleanup_mask(Mat& mask) const
//     {
//         if (mask.empty()) return;

//         const int min_dim = std::max(1, std::min(mask.cols, mask.rows));
//         int kernel_size = (min_dim >= 720) ? 5 : 3;
//         Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(kernel_size, kernel_size));
//         morphologyEx(mask, mask, MORPH_OPEN, kernel);
//         morphologyEx(mask, mask, MORPH_CLOSE, kernel);
//     }

//     double prior_weight(const Rect& r, Size image_size) const
//     {
//         const double area = static_cast<double>(r.area());
//         const double frame_area = std::max(1.0, static_cast<double>(image_size.width) * image_size.height);

//         switch (prior_)
//         {
//         case PriorMode::Size:
//             return 0.75 + std::min(area / (frame_area * 0.03), 1.0) * 0.35;
//         case PriorMode::Contrast:
//             return 1.0;
//         case PriorMode::Center:
//         default:
//         {
//             const double dx = (r.x + r.width * 0.5) - image_size.width * 0.5;
//             const double dy = (r.y + r.height * 0.5) - image_size.height * 0.5;
//             const double max_dist = std::max(1.0, std::hypot(image_size.width * 0.5, image_size.height * 0.5));
//             return 0.75 + (1.0 - std::min(std::hypot(dx, dy) / max_dist, 1.0)) * 0.35;
//         }
//         }
//     }

//     TrackerConfig cfg_;
//     int roi_w_;
//     int roi_h_;
//     PriorMode prior_;
//     bool debug_mode_;
//     bool is_video_;
// };

// DetMode get_detection_mode()
// {
//     cout << "\n--- Detection Engine ---\n"
//         << " [1] Filtered DoG\n"
//         << " [2] Optimized Hough Circle Transform\n"
//         << " [3] Hybrid (DoG + Hough)\n";

//     string sel;
//     cout << " Select (default 2): ";
//     getline(cin, sel);

//     if (sel == "1") return DetMode::DoG;
//     if (sel == "3") return DetMode::Hybrid;
//     return DetMode::HoughCircles;
// }

// bool get_debug_mode_choice()
// {
//     string sel;
//     cout << " Enable Step-by-Step Visual Debugger? (y/n, default n): ";
//     getline(cin, sel);
//     return (sel == "y" || sel == "Y");
// }

// string to_lower_copy(string value)
// {
//     transform(value.begin(), value.end(), value.begin(),
//         [](unsigned char c) { return static_cast<char>(tolower(c)); });
//     return value;
// }

// bool parse_detection_mode(const string& raw, DetMode& mode)
// {
//     string value = to_lower_copy(raw);
//     if (value == "1" || value == "dog" || value == "dogg" || value == "dog-filter" || value == "dog_filter")
//     {
//         mode = DetMode::DoG;
//         return true;
//     }
//     if (value == "2" || value == "hough" || value == "houghcircles" || value == "hough-circles")
//     {
//         mode = DetMode::HoughCircles;
//         return true;
//     }
//     if (value == "3" || value == "hybrid")
//     {
//         mode = DetMode::Hybrid;
//         return true;
//     }
//     return false;
// }

// bool parse_prior_mode(const string& raw, PriorMode& prior)
// {
//     string value = to_lower_copy(raw);
//     if (value == "size")
//     {
//         prior = PriorMode::Size;
//         return true;
//     }
//     if (value == "contrast")
//     {
//         prior = PriorMode::Contrast;
//         return true;
//     }
//     if (value == "center" || value == "centre")
//     {
//         prior = PriorMode::Center;
//         return true;
//     }
//     return false;
// }

// bool parse_float_arg(const string& raw, float& out)
// {
//     try
//     {
//         size_t pos = 0;
//         out = stof(raw, &pos);
//         return pos == raw.size() && std::isfinite(out);
//     }
//     catch (...)
//     {
//         return false;
//     }
// }

// bool parse_int_arg(const string& raw, int& out)
// {
//     try
//     {
//         size_t pos = 0;
//         out = stoi(raw, &pos);
//         return pos == raw.size();
//     }
//     catch (...)
//     {
//         return false;
//     }
// }

// bool display_available()
// {
// #ifdef _WIN32
//     return true;
// #else
//     return getenv("DISPLAY") != nullptr || getenv("WAYLAND_DISPLAY") != nullptr;
// #endif
// }

// bool can_prompt()
// {
// #ifdef _WIN32
//     return _isatty(_fileno(stdin)) != 0;
// #else
//     return isatty(STDIN_FILENO) != 0;
// #endif
// }

// bool is_integer_source(const string& value)
// {
//     return !value.empty() && all_of(value.begin(), value.end(),
//         [](unsigned char c) { return isdigit(c); });
// }

// string mode_to_string(DetMode mode)
// {
//     switch (mode)
//     {
//     case DetMode::DoG:
//         return "Dog";
//     case DetMode::Hybrid:
//         return "Hybrid";
//     case DetMode::HoughCircles:
//     default:
//         return "Hough";
//     }
// }

// void print_usage(const char* exe)
// {
//     cerr << "Usage: " << exe << " <path_to_image_or_video> [options]\n\n"
//         << "Options:\n"
//         << "  --mode <dog|hough|hybrid>     Detection engine. Prompts in a terminal, defaults to hough otherwise.\n"
//         << "  --debug                       Show step-by-step debug windows when a display is available.\n"
//         << "  --no-display                  Disable all OpenCV windows for batch/headless runs.\n"
//         << "  -C [N], -c [N], --center [N]  CENTER-ONLY mode: keep/output N nearest detections to frame center. Default N=1.\n"
//         << "  --static-coord-frames <N>     In -C video mode, suppress an exact X/Y repeated for N consecutive frames. Default: 10.\n"
//         << "  --out-dir <dir>               Output root directory. Default: Res\n"
//         << "  --class-id <N>                YOLO class ID (0,1,2,...). If omitted, prompts before processing.\n"
//         << "  --dog-threshold <value>       Fixed DoG threshold. Default: adaptive.\n"
//         << "  --dog-stddev <value>          Adaptive DoG sensitivity. Default: 1.25\n"
//         << "  --hough-min-radius <px>       Override dynamic Hough minimum radius.\n"
//         << "  --hough-max-radius <px>       Override dynamic Hough maximum radius.\n"
//         << "  --hough-min-dist <px>         Override dynamic Hough center spacing.\n"
//         << "  --prior <center|size|contrast> Candidate score prior. Default: center\n";
// }

// struct AppOptions
// {
//     string input_path;
//     string output_root = "Res";
//     DetMode mode = DetMode::HoughCircles;
//     PriorMode prior = PriorMode::Center;
//     TrackerConfig config;
//     bool mode_provided = false;
//     bool debug = false;
//     bool display = true;
//     bool protect_center_target = false;
//     int center_target_count = 1;   // -C alone => 1; -C N => keep N nearest targets
//     int static_coord_frames = 10;  // suppress exact repeated X/Y after this many consecutive video frames in -C mode
//     int class_id = -1;              // YOLO class ID. Prompted if not supplied on CLI.
//     bool class_id_provided = false;
//     bool help_requested = false;
// };

// bool consume_option_value(int& i, int argc, char** argv, string& value)
// {
//     if (i + 1 >= argc) return false;
//     value = argv[++i];
//     return true;
// }

// bool parse_args(int argc, char** argv, AppOptions& options)
// {
//     if (argc < 2)
//     {
//         print_usage(argv[0]);
//         return false;
//     }

//     string first_arg = argv[1];
//     if (first_arg == "--help" || first_arg == "-h")
//     {
//         print_usage(argv[0]);
//         options.help_requested = true;
//         return true;
//     }

//     options.input_path = argv[1];

//     for (int i = 2; i < argc; ++i)
//     {
//         string arg = argv[i];
//         string value;
//         auto read_value = [&]() -> bool {
//             size_t eq = arg.find('=');
//             if (eq != string::npos)
//             {
//                 value = arg.substr(eq + 1);
//                 arg = arg.substr(0, eq);
//                 return !value.empty();
//             }
//             return consume_option_value(i, argc, argv, value);
//             };

//         if (arg == "--help" || arg == "-h")
//         {
//             print_usage(argv[0]);
//             options.help_requested = true;
//             return true;
//         }
//         if (arg == "--debug")
//         {
//             options.debug = true;
//             continue;
//         }
//         if (arg == "--no-display")
//         {
//             options.display = false;
//             continue;
//         }
//         if (arg == "-C" || arg == "-c" || arg == "--center" || arg == "--protect-center")
//         {
//             options.protect_center_target = true;
//             options.center_target_count = 1; // preserve legacy/default -C behavior

//             // Optional positive integer immediately after -C selects how many
//             // nearest-to-center targets are kept. Do not consume another flag.
//             if (i + 1 < argc)
//             {
//                 string maybe_count = argv[i + 1];
//                 int requested_count = 0;

//                 if (parse_int_arg(maybe_count, requested_count))
//                 {
//                     if (requested_count <= 0)
//                     {
//                         cerr << "[ERROR] -C target count must be a positive integer (1, 2, 3, ...).\n";
//                         return false;
//                     }
//                     options.center_target_count = requested_count;
//                     ++i;
//                 }
//                 else if (!maybe_count.empty() && maybe_count[0] != '-')
//                 {
//                     cerr << "[ERROR] Invalid -C target count: " << maybe_count
//                         << ". Use -C or -C <positive integer>.\n";
//                     return false;
//                 }
//             }
//             continue;
//         }
//         if (arg.rfind("-C=", 0) == 0 || arg.rfind("-c=", 0) == 0 ||
//             arg.rfind("--center=", 0) == 0 || arg.rfind("--protect-center=", 0) == 0)
//         {
//             options.protect_center_target = true;

//             size_t eq = arg.find('=');
//             string count_text = (eq == string::npos) ? string{} : arg.substr(eq + 1);
//             int requested_count = 0;
//             if (!parse_int_arg(count_text, requested_count) || requested_count <= 0)
//             {
//                 cerr << "[ERROR] -C target count must be a positive integer (1, 2, 3, ...).\n";
//                 return false;
//             }

//             options.center_target_count = requested_count;
//             continue;
//         }
//         if (arg == "--static-coord-frames" || arg.rfind("--static-coord-frames=", 0) == 0)
//         {
//             if (arg == "--static-coord-frames")
//             {
//                 if (!consume_option_value(i, argc, argv, value))
//                 {
//                     cerr << "[ERROR] Missing --static-coord-frames value.\n";
//                     return false;
//                 }
//             }
//             else
//             {
//                 value = arg.substr(22);
//             }

//             int requested_frames = 0;
//             if (!parse_int_arg(value, requested_frames) || requested_frames < 2)
//             {
//                 cerr << "[ERROR] --static-coord-frames must be an integer >= 2.\n";
//                 return false;
//             }
//             options.static_coord_frames = requested_frames;
//             continue;
//         }
//         if (arg == "--mode")
//         {
//             if (!read_value() || !parse_detection_mode(value, options.mode))
//             {
//                 cerr << "[ERROR] Invalid --mode value.\n";
//                 return false;
//             }
//             options.mode_provided = true;
//             continue;
//         }
//         if (arg.rfind("--mode=", 0) == 0)
//         {
//             value = arg.substr(7);
//             if (!parse_detection_mode(value, options.mode))
//             {
//                 cerr << "[ERROR] Invalid --mode value.\n";
//                 return false;
//             }
//             options.mode_provided = true;
//             continue;
//         }
//         if (arg == "--out-dir" || arg.rfind("--out-dir=", 0) == 0)
//         {
//             if (arg == "--out-dir")
//             {
//                 if (!consume_option_value(i, argc, argv, value))
//                 {
//                     cerr << "[ERROR] Missing --out-dir value.\n";
//                     return false;
//                 }
//             }
//             else
//             {
//                 value = arg.substr(10);
//             }
//             if (value.empty())
//             {
//                 cerr << "[ERROR] Output directory cannot be empty.\n";
//                 return false;
//             }
//             options.output_root = value;
//             continue;
//         }
//         if (arg == "--prior" || arg.rfind("--prior=", 0) == 0)
//         {
//             if (arg == "--prior")
//             {
//                 if (!consume_option_value(i, argc, argv, value))
//                 {
//                     cerr << "[ERROR] Missing --prior value.\n";
//                     return false;
//                 }
//             }
//             else
//             {
//                 value = arg.substr(8);
//             }
//             if (!parse_prior_mode(value, options.prior))
//             {
//                 cerr << "[ERROR] Invalid --prior value.\n";
//                 return false;
//             }
//             continue;
//         }

//         if (arg == "--class-id" || arg.rfind("--class-id=", 0) == 0)
//         {
//             if (arg == "--class-id")
//             {
//                 if (!consume_option_value(i, argc, argv, value))
//                 {
//                     cerr << "[ERROR] Missing --class-id value.\n";
//                     return false;
//                 }
//             }
//             else
//             {
//                 value = arg.substr(11);
//             }

//             int requested_class_id = -1;
//             if (!parse_int_arg(value, requested_class_id) || requested_class_id < 0)
//             {
//                 cerr << "[ERROR] --class-id must be an integer >= 0.\n";
//                 return false;
//             }
//             options.class_id = requested_class_id;
//             options.class_id_provided = true;
//             continue;
//         }

//         float float_value = 0.0f;
//         int int_value = 0;
//         if (arg == "--dog-threshold" || arg.rfind("--dog-threshold=", 0) == 0)
//         {
//             if (arg == "--dog-threshold" && !consume_option_value(i, argc, argv, value))
//             {
//                 cerr << "[ERROR] Missing --dog-threshold value.\n";
//                 return false;
//             }
//             if (arg.rfind("--dog-threshold=", 0) == 0) value = arg.substr(16);
//             if (!parse_float_arg(value, float_value) || float_value <= 0.0f)
//             {
//                 cerr << "[ERROR] --dog-threshold must be a positive number.\n";
//                 return false;
//             }
//             options.config.DOG_THRESHOLD = float_value;
//             continue;
//         }
//         if (arg == "--dog-stddev" || arg.rfind("--dog-stddev=", 0) == 0)
//         {
//             if (arg == "--dog-stddev" && !consume_option_value(i, argc, argv, value))
//             {
//                 cerr << "[ERROR] Missing --dog-stddev value.\n";
//                 return false;
//             }
//             if (arg.rfind("--dog-stddev=", 0) == 0) value = arg.substr(13);
//             if (!parse_float_arg(value, float_value) || float_value < 0.0f)
//             {
//                 cerr << "[ERROR] --dog-stddev must be a non-negative number.\n";
//                 return false;
//             }
//             options.config.DOG_STDDEV_FACTOR = float_value;
//             continue;
//         }
//         if (arg == "--hough-min-radius" || arg.rfind("--hough-min-radius=", 0) == 0)
//         {
//             if (arg == "--hough-min-radius" && !consume_option_value(i, argc, argv, value))
//             {
//                 cerr << "[ERROR] Missing --hough-min-radius value.\n";
//                 return false;
//             }
//             if (arg.rfind("--hough-min-radius=", 0) == 0) value = arg.substr(19);
//             if (!parse_int_arg(value, int_value) || int_value <= 0)
//             {
//                 cerr << "[ERROR] --hough-min-radius must be a positive integer.\n";
//                 return false;
//             }
//             options.config.HOUGH_MIN_RADIUS = int_value;
//             continue;
//         }
//         if (arg == "--hough-max-radius" || arg.rfind("--hough-max-radius=", 0) == 0)
//         {
//             if (arg == "--hough-max-radius" && !consume_option_value(i, argc, argv, value))
//             {
//                 cerr << "[ERROR] Missing --hough-max-radius value.\n";
//                 return false;
//             }
//             if (arg.rfind("--hough-max-radius=", 0) == 0) value = arg.substr(19);
//             if (!parse_int_arg(value, int_value) || int_value <= 0)
//             {
//                 cerr << "[ERROR] --hough-max-radius must be a positive integer.\n";
//                 return false;
//             }
//             options.config.HOUGH_MAX_RADIUS = int_value;
//             continue;
//         }
//         if (arg == "--hough-min-dist" || arg.rfind("--hough-min-dist=", 0) == 0)
//         {
//             if (arg == "--hough-min-dist" && !consume_option_value(i, argc, argv, value))
//             {
//                 cerr << "[ERROR] Missing --hough-min-dist value.\n";
//                 return false;
//             }
//             if (arg.rfind("--hough-min-dist=", 0) == 0) value = arg.substr(17);
//             if (!parse_int_arg(value, int_value) || int_value <= 0)
//             {
//                 cerr << "[ERROR] --hough-min-dist must be a positive integer.\n";
//                 return false;
//             }
//             options.config.HOUGH_MIN_DIST = int_value;
//             continue;
//         }

//         cerr << "[ERROR] Unknown option: " << arg << "\n";
//         return false;
//     }

//     if (options.config.HOUGH_MIN_RADIUS > 0 && options.config.HOUGH_MAX_RADIUS > 0 &&
//         options.config.HOUGH_MAX_RADIUS <= options.config.HOUGH_MIN_RADIUS)
//     {
//         cerr << "[ERROR] --hough-max-radius must be greater than --hough-min-radius.\n";
//         return false;
//     }

//     return true;
// }

// bool prompt_for_class_id(int& class_id)
// {
//     while (true)
//     {
//         cout << "\n--- YOLO Annotation Setup ---\n";
//         cout << " Enter class ID (0, 1, 2, ...): ";

//         string value;
//         if (!getline(cin, value))
//         {
//             cerr << "[ERROR] Could not read class ID from stdin.\n";
//             return false;
//         }

//         int parsed = -1;
//         if (parse_int_arg(value, parsed) && parsed >= 0)
//         {
//             class_id = parsed;
//             return true;
//         }

//         cerr << "[ERROR] Class ID must be a non-negative integer.\n";
//     }
// }

// struct YoloDatasetPaths
// {
//     string images_dir;
//     string labels_dir;
// };

// YoloDatasetPaths create_yolo_dataset_folders(const string& run_dir)
// {
//     YoloDatasetPaths paths;
//     paths.images_dir = (fs::path(run_dir) / "images").string();
//     paths.labels_dir = (fs::path(run_dir) / "labels").string();
//     fs::create_directories(paths.images_dir);
//     fs::create_directories(paths.labels_dir);
//     return paths;
// }

// string zero_padded_index(long long index, int width = 6)
// {
//     ostringstream ss;
//     ss << setw(width) << setfill('0') << index;
//     return ss.str();
// }

// // Save one clean training image plus its matching YOLO label file.
// // YOLO detection format per line:
// //   class_id x_center_norm y_center_norm width_norm height_norm
// // Coordinates are normalized to [0,1] using the full saved image dimensions.
// // CENTER-HOLD detections are intentionally skipped because they are stale boxes
// // reused from previous frames rather than fresh detector observations.
// bool save_yolo_sample(const Mat& clean_frame,
//     const vector<BBox>& detections,
//     int class_id,
//     const YoloDatasetPaths& paths,
//     const string& stem,
//     int& labels_written)
// {
//     labels_written = 0;
//     if (clean_frame.empty() || class_id < 0) return false;

//     ostringstream label_text;
//     label_text << fixed << setprecision(6);

//     const Rect image_bounds(0, 0, clean_frame.cols, clean_frame.rows);
//     for (const auto& b : detections)
//     {
//         if (b.center_held) continue;

//         Rect box = b.rect() & image_bounds;
//         if (box.width <= 0 || box.height <= 0) continue;

//         const double x_center = (box.x + box.width * 0.5) / static_cast<double>(clean_frame.cols);
//         const double y_center = (box.y + box.height * 0.5) / static_cast<double>(clean_frame.rows);
//         const double width = box.width / static_cast<double>(clean_frame.cols);
//         const double height = box.height / static_cast<double>(clean_frame.rows);

//         label_text << class_id << " "
//             << std::clamp(x_center, 0.0, 1.0) << " "
//             << std::clamp(y_center, 0.0, 1.0) << " "
//             << std::clamp(width, 0.0, 1.0) << " "
//             << std::clamp(height, 0.0, 1.0) << "\n";
//         ++labels_written;
//     }

//     // User requested detected-target images, so frames with no fresh labels are
//     // not added to the training set.
//     if (labels_written == 0) return false;

//     const string image_path = (fs::path(paths.images_dir) / (stem + ".jpg")).string();
//     const string label_path = (fs::path(paths.labels_dir) / (stem + ".txt")).string();

//     if (!imwrite(image_path, clean_frame))
//     {
//         cerr << "[WARN] Failed to save YOLO training image: " << image_path << "\n";
//         labels_written = 0;
//         return false;
//     }

//     ofstream label_file(label_path, ios::out | ios::trunc);
//     if (!label_file)
//     {
//         cerr << "[WARN] Failed to save YOLO label file: " << label_path << "\n";
//         std::error_code ec;
//         fs::remove(image_path, ec); // avoid an unmatched image if label creation fails
//         labels_written = 0;
//         return false;
//     }

//     label_file << label_text.str();
//     return true;
// }

// string create_run_folder(const string& output_root)
// {
//     auto t = chrono::system_clock::now();
//     auto tt = chrono::system_clock::to_time_t(t);
//     tm local_tm{};
// #ifdef _WIN32
//     localtime_s(&local_tm, &tt);
// #else
//     localtime_r(&tt, &local_tm);
// #endif

//     stringstream ss;
//     ss << output_root << "/run_" << put_time(&local_tm,  "%Y-%m-%d_%I-%M-%S_%p");
//     string base_dir = ss.str();
//     string dir = base_dir;
//     int suffix = 1;
//     while (fs::exists(dir))
//     {
//         dir = base_dir + "_" + to_string(suffix++);
//     }
//     fs::create_directories(dir);
//     return dir;
// }

// void drawCornerBox(Mat& frame, const Rect& box, const Scalar& color, int thickness = 1, int cornerLength = 25)
// {
//     if (frame.empty() || box.width <= 0 || box.height <= 0) return;

//     // OpenCV Rect is [x, x + width) x [y, y + height), so the last
//     // drawable pixel is width - 1 / height - 1.
//     const int x1 = box.x;
//     const int y1 = box.y;
//     const int x2 = box.x + box.width - 1;
//     const int y2 = box.y + box.height - 1;

//     // Keep the corner arms proportional on small detections while
//     // allowing a larger HUD-style corner on normal detections.
//     const int lenX = std::max(1, std::min(cornerLength, box.width / 3));
//     const int lenY = std::max(1, std::min(cornerLength, box.height / 3));

//     // Every L faces inward toward the detected object.

//     // Top-left: ┌
//     line(frame, Point(x1, y1), Point(x1 + lenX, y1), color, thickness, LINE_AA);
//     line(frame, Point(x1, y1), Point(x1, y1 + lenY), color, thickness, LINE_AA);

//     // Top-right: ┐
//     line(frame, Point(x2, y1), Point(x2 - lenX, y1), color, thickness, LINE_AA);
//     line(frame, Point(x2, y1), Point(x2, y1 + lenY), color, thickness, LINE_AA);

//     // Bottom-left: └
//     line(frame, Point(x1, y2), Point(x1 + lenX, y2), color, thickness, LINE_AA);
//     line(frame, Point(x1, y2), Point(x1, y2 - lenY), color, thickness, LINE_AA);

//     // Bottom-right: ┘
//     line(frame, Point(x2, y2), Point(x2 - lenX, y2), color, thickness, LINE_AA);
//     line(frame, Point(x2, y2), Point(x2, y2 - lenY), color, thickness, LINE_AA);
// }

// // Temporal filter used only for video/camera processing.
// // A detection is considered stationary when it keeps returning in almost the
// // same location for several frames. Stationary detections are then ignored
// // until they move again.
// class StationaryTargetFilter
// {
// public:
//     StationaryTargetFilter(
//         int stable_frames = 6,
//         double position_tolerance_px = 3.0,
//         int size_tolerance_px = 4,
//         double match_distance_px = 12.0,
//         int max_missed_frames = 3)
//         : stable_frames_(std::max(2, stable_frames)),
//         position_tolerance_px_(std::max(0.0, position_tolerance_px)),
//         size_tolerance_px_(std::max(0, size_tolerance_px)),
//         match_distance_px_(std::max(1.0, match_distance_px)),
//         max_missed_frames_(std::max(0, max_missed_frames))
//     {
//     }

//     vector<BBox> filter(const vector<BBox>& detections, int protected_detection_index = -1)
//     {
//         vector<BBox> moving_detections;
//         moving_detections.reserve(detections.size());

//         // Prevent two detections in the same frame from being assigned to the
//         // same temporal track.
//         vector<bool> track_used(tracks_.size(), false);

//         for (size_t detection_index = 0; detection_index < detections.size(); ++detection_index)
//         {
//             const auto& detection = detections[detection_index];
//             const bool protected_from_stationary_filter =
//                 static_cast<int>(detection_index) == protected_detection_index;
//             const Rect current_box = detection.rect();
//             int best_track = -1;
//             double best_distance = 1e30;

//             for (size_t t = 0; t < tracks_.size(); ++t)
//             {
//                 if (track_used[t]) continue;

//                 const Rect& previous_box = tracks_[t].last_box;
//                 const double distance = center_distance(previous_box, current_box);

//                 // Allow a slightly larger association radius for larger boxes,
//                 // but keep a fixed minimum for small Hough detections.
//                 const double dynamic_match_distance = std::max(
//                     match_distance_px_,
//                     0.35 * static_cast<double>(std::max(previous_box.width, previous_box.height)));

//                 const int max_size_change = std::max(
//                     10,
//                     static_cast<int>(0.50 * std::max(previous_box.width, previous_box.height)));

//                 const bool size_is_reasonable =
//                     std::abs(current_box.width - previous_box.width) <= max_size_change &&
//                     std::abs(current_box.height - previous_box.height) <= max_size_change;

//                 if (size_is_reasonable && distance <= dynamic_match_distance && distance < best_distance)
//                 {
//                     best_distance = distance;
//                     best_track = static_cast<int>(t);
//                 }
//             }

//             bool stationary = false;

//             if (best_track >= 0)
//             {
//                 Track& track = tracks_[best_track];
//                 track_used[best_track] = true;
//                 track.missed_frames = 0;
//                 track.last_box = current_box;

//                 track.history.push_back(current_box);
//                 if (track.history.size() > static_cast<size_t>(stable_frames_))
//                 {
//                     track.history.erase(track.history.begin());
//                 }

//                 stationary = is_stationary(track.history);
//                 track.stationary = stationary;
//             }
//             else
//             {
//                 Track new_track;
//                 new_track.last_box = current_box;
//                 new_track.history.push_back(current_box);
//                 tracks_.push_back(new_track);
//                 track_used.push_back(true);
//             }

//             // Normally, only moving / not-yet-proven-stationary targets continue.
//             // With -C, the current detection nearest the frame center is protected
//             // and remains visible/reported even if its temporal track is stationary.
//             if (!stationary || protected_from_stationary_filter)
//             {
//                 moving_detections.push_back(detection);
//             }
//         }

//         // Age tracks that did not appear in this frame. A short grace period
//         // tolerates detector flicker without forgetting a stationary false target.
//         for (size_t t = 0; t < tracks_.size(); ++t)
//         {
//             if (!track_used[t])
//             {
//                 tracks_[t].missed_frames++;
//             }
//         }

//         for (size_t t = tracks_.size(); t-- > 0; )
//         {
//             if (tracks_[t].missed_frames > max_missed_frames_)
//             {
//                 tracks_.erase(tracks_.begin() + static_cast<long>(t));
//             }
//         }

//         return moving_detections;
//     }

//     void reset()
//     {
//         tracks_.clear();
//     }

// private:
//     struct Track
//     {
//         Rect last_box;
//         vector<Rect> history;
//         int missed_frames = 0;
//         bool stationary = false;
//     };

//     static Point2d center_of(const Rect& box)
//     {
//         return Point2d(
//             box.x + box.width * 0.5,
//             box.y + box.height * 0.5);
//     }

//     static double center_distance(const Rect& a, const Rect& b)
//     {
//         const Point2d ca = center_of(a);
//         const Point2d cb = center_of(b);
//         return std::hypot(ca.x - cb.x, ca.y - cb.y);
//     }

//     bool is_stationary(const vector<Rect>& history) const
//     {
//         if (history.size() < static_cast<size_t>(stable_frames_))
//         {
//             return false;
//         }

//         double min_cx = 1e30;
//         double max_cx = -1e30;
//         double min_cy = 1e30;
//         double max_cy = -1e30;
//         int min_w = 1000000000;
//         int max_w = 0;
//         int min_h = 1000000000;
//         int max_h = 0;

//         for (const Rect& box : history)
//         {
//             const Point2d c = center_of(box);
//             min_cx = std::min(min_cx, c.x);
//             max_cx = std::max(max_cx, c.x);
//             min_cy = std::min(min_cy, c.y);
//             max_cy = std::max(max_cy, c.y);
//             min_w = std::min(min_w, box.width);
//             max_w = std::max(max_w, box.width);
//             min_h = std::min(min_h, box.height);
//             max_h = std::max(max_h, box.height);
//         }

//         const Point2d first_center = center_of(history.front());
//         const Point2d last_center = center_of(history.back());

//         // Require the target to remain in the same small region, not merely
//         // move slowly in one direction. This helps preserve genuinely moving
//         // targets whose position accumulates across the frame.
//         const double first_to_last = std::hypot(
//             last_center.x - first_center.x,
//             last_center.y - first_center.y);

//         const bool position_stable =
//             first_to_last <= position_tolerance_px_ &&
//             (max_cx - min_cx) <= position_tolerance_px_ * 2.0 &&
//             (max_cy - min_cy) <= position_tolerance_px_ * 2.0;

//         const bool size_stable =
//             (max_w - min_w) <= size_tolerance_px_ * 2 &&
//             (max_h - min_h) <= size_tolerance_px_ * 2;

//         return position_stable && size_stable;
//     }

//     vector<Track> tracks_;
//     int stable_frames_;
//     double position_tolerance_px_;
//     int size_tolerance_px_;
//     double match_distance_px_;
//     int max_missed_frames_;
// };


// // Exact repeated-coordinate suppression used for -C video/camera mode.
// // A detector result is considered static only when its CENTER X and Y are
// // exactly identical for stable_frames consecutive processed frames.
// // Target IDs are intentionally ignored because -C re-ranks targets every frame.
// class RepeatedCoordinateFilter
// {
// public:
//     explicit RepeatedCoordinateFilter(int stable_frames = 10)
//         : stable_frames_(std::max(2, stable_frames))
//     {
//     }

//     vector<BBox> filter(const vector<BBox>& detections)
//     {
//         ++frame_number_;
//         for (auto& track : tracks_)
//         {
//             track.seen_this_frame = false;
//         }

//         vector<BBox> kept;
//         kept.reserve(detections.size());

//         for (const auto& detection : detections)
//         {
//             const int x = detection.cx();
//             const int y = detection.cy();

//             int matched_track = -1;
//             for (size_t t = 0; t < tracks_.size(); ++t)
//             {
//                 if (!tracks_[t].seen_this_frame && tracks_[t].x == x && tracks_[t].y == y)
//                 {
//                     matched_track = static_cast<int>(t);
//                     break;
//                 }
//             }

//             int consecutive = 1;
//             if (matched_track >= 0)
//             {
//                 Track& track = tracks_[matched_track];
//                 if (track.last_seen_frame == frame_number_ - 1)
//                     track.consecutive_frames++;
//                 else
//                     track.consecutive_frames = 1;

//                 track.last_seen_frame = frame_number_;
//                 track.seen_this_frame = true;
//                 consecutive = track.consecutive_frames;
//             }
//             else
//             {
//                 Track track;
//                 track.x = x;
//                 track.y = y;
//                 track.consecutive_frames = 1;
//                 track.last_seen_frame = frame_number_;
//                 track.seen_this_frame = true;
//                 tracks_.push_back(track);
//             }

//             const bool suppress = consecutive >= stable_frames_;
//             if (!suppress)
//             {
//                 kept.push_back(detection);
//             }
//             else if (consecutive == stable_frames_)
//             {
//                 cout << "[STATIC-COORD] Suppressing repeated target X=" << x
//                     << " Y=" << y
//                     << " after " << consecutive << " consecutive identical frames.\n";
//             }
//         }

//         // Exact-repeat counting is consecutive. If a coordinate is absent for
//         // even one frame, forget that streak so a later reappearance starts at 1.
//         for (size_t t = tracks_.size(); t-- > 0; )
//         {
//             if (!tracks_[t].seen_this_frame)
//             {
//                 tracks_.erase(tracks_.begin() + static_cast<long>(t));
//             }
//         }

//         return kept;
//     }

//     void reset()
//     {
//         tracks_.clear();
//         frame_number_ = 0;
//     }

//     int stable_frames() const { return stable_frames_; }

// private:
//     struct Track
//     {
//         int x = 0;
//         int y = 0;
//         int consecutive_frames = 0;
//         long long last_seen_frame = 0;
//         bool seen_this_frame = false;
//     };

//     vector<Track> tracks_;
//     int stable_frames_ = 10;
//     long long frame_number_ = 0;
// };

// // -C nearest-N center-target selection/persistence.
// // -C alone keeps the single nearest detection (legacy/default behavior).
// // -C N keeps up to N detections with the smallest Euclidean center distance.
// // All non-selected detections are discarded before drawing/reporting/downstream X/Y use.
// // If the detector returns zero detections for a very short gap, the previously
// // selected center targets are held for up to max_hold_frames frames and marked
// // center_held=true so they are distinguishable from fresh detector results.
// class CenterTargetKeeper
// {
// public:
//     explicit CenterTargetKeeper(int max_hold_frames = 3)
//         : max_hold_frames_(std::max(0, max_hold_frames))
//     {
//     }

//     int select_nearest_n(vector<BBox>& detections, Size frame_size, int requested_count)
//     {
//         // Clear any stale per-frame display state first.
//         for (auto& b : detections)
//         {
//             b.center_protected = false;
//             b.center_held = false;
//             b.center_distance_px = -1.0;
//         }

//         if (detections.empty())
//         {
//             return 0;
//         }

//         const int keep_count = std::min(
//             std::max(1, requested_count),
//             static_cast<int>(detections.size()));

//         const Point2d frame_center(frame_size.width * 0.5, frame_size.height * 0.5);

//         struct RankedDetection
//         {
//             double distance;
//             size_t index;
//         };

//         vector<RankedDetection> ranked;
//         ranked.reserve(detections.size());

//         for (size_t i = 0; i < detections.size(); ++i)
//         {
//             const double dx = static_cast<double>(detections[i].cx()) - frame_center.x;
//             const double dy = static_cast<double>(detections[i].cy()) - frame_center.y;
//             ranked.push_back({ std::hypot(dx, dy), i });
//         }

//         std::sort(ranked.begin(), ranked.end(),
//             [](const RankedDetection& a, const RankedDetection& b)
//             {
//                 if (a.distance != b.distance) return a.distance < b.distance;
//                 return a.index < b.index;
//             });

//         vector<BBox> selected;
//         selected.reserve(keep_count);

//         for (int rank = 0; rank < keep_count; ++rank)
//         {
//             BBox b = detections[ranked[rank].index];
//             b.center_protected = true;
//             b.center_held = false;
//             b.center_distance_px = ranked[rank].distance;
//             selected.push_back(b);
//         }

//         // Keep selected output ordered nearest -> farthest. Therefore Target 1
//         // is always the closest-to-center target, Target 2 the next closest, etc.
//         detections = selected;

//         last_center_targets_ = selected;
//         missed_frames_ = 0;
//         return static_cast<int>(detections.size());
//     }

//     // Called only when the detector returned zero detections in this frame.
//     // This bridges short detector flicker. It reuses only the last selected -C
//     // targets; it never introduces unrelated/non-nearest detections.
//     bool append_short_hold(vector<BBox>& detections, Size frame_size)
//     {
//         if (last_center_targets_.empty() || missed_frames_ >= max_hold_frames_)
//         {
//             if (!last_center_targets_.empty())
//             {
//                 ++missed_frames_;
//             }
//             return false;
//         }

//         ++missed_frames_;
//         const Point2d frame_center(frame_size.width * 0.5, frame_size.height * 0.5);

//         for (const auto& previous : last_center_targets_)
//         {
//             BBox held = previous;
//             held.center_protected = true;
//             held.center_held = true;
//             held.type = "center_hold";

//             const double dx = static_cast<double>(held.cx()) - frame_center.x;
//             const double dy = static_cast<double>(held.cy()) - frame_center.y;
//             held.center_distance_px = std::hypot(dx, dy);

//             detections.push_back(held);
//         }

//         // Preserve nearest -> farthest ordering for held targets too.
//         std::sort(detections.begin(), detections.end(),
//             [](const BBox& a, const BBox& b)
//             {
//                 return a.center_distance_px < b.center_distance_px;
//             });

//         return !detections.empty();
//     }

//     void reset()
//     {
//         last_center_targets_.clear();
//         missed_frames_ = 0;
//     }

// private:
//     vector<BBox> last_center_targets_;
//     int missed_frames_ = 0;
//     int max_hold_frames_ = 3;
// };

// void process_frame_detections(Mat& frame, const BlobDetector& detector, DetMode det_mode,
//     long long& elapsed_ms, int& det_count, vector<BBox>& out_detections,
//     StationaryTargetFilter* stationary_filter = nullptr,
//     CenterTargetKeeper* center_keeper = nullptr,
//     bool protect_center_target = false,
//     int center_target_count = 1,
//     RepeatedCoordinateFilter* repeated_coordinate_filter = nullptr)
// {
//     if (frame.empty())
//     {
//         elapsed_ms = 0;
//         det_count = 0;
//         out_detections.clear();
//         return;
//     }

//     auto start_time = chrono::high_resolution_clock::now();

//     switch (det_mode)
//     {
//     case DetMode::DoG:
//         out_detections = detector.detect_dog_all(frame);
//         break;
//     case DetMode::HoughCircles:
//         out_detections = detector.detect_hough_circles(frame);
//         break;
//     case DetMode::Hybrid:
//         out_detections = detector.detect_hybrid_all(frame);
//         break;
//     }

//     const bool detector_returned_any = !out_detections.empty();

//     // -C is an EXCLUSIVE nearest-N output mode for every input type.
//     // For video/camera, first remove exact X/Y coordinates that have remained
//     // unchanged for many consecutive frames. This happens BEFORE nearest-N
//     // ranking, so suppressed static detections do not consume -C target slots.
//     if (protect_center_target && center_keeper != nullptr)
//     {
//         if (repeated_coordinate_filter != nullptr)
//         {
//             // Call every frame, including empty detector frames, so an exact-X/Y
//             // streak is truly consecutive and resets immediately after a miss.
//             out_detections = repeated_coordinate_filter->filter(out_detections);
//         }

//         // The detector may internally produce many candidates, but only the
//         // requested N non-static candidates nearest to frame center survive.
//         if (!out_detections.empty())
//         {
//             center_keeper->select_nearest_n(
//                 out_detections, frame.size(), center_target_count);
//         }
//         else if (!detector_returned_any)
//         {
//             // Genuine detector miss: bridge only this short flicker gap.
//             center_keeper->append_short_hold(
//                 out_detections, frame.size());
//         }
//         else
//         {
//             // The detector did return candidates, but every candidate was
//             // rejected as a repeated static coordinate. Do NOT resurrect the
//             // previous center targets through CENTER-HOLD.
//             center_keeper->reset();
//         }

//         // -C still bypasses the broader stationary box/size filter; only the
//         // exact repeated-coordinate rule above is applied in this mode.
//     }
//     else if (stationary_filter != nullptr)
//     {
//         // Normal video/camera mode (no -C): preserve stationary-object suppression.
//         out_detections = stationary_filter->filter(out_detections, -1);
//     }

//     auto end_time = chrono::high_resolution_clock::now();
//     elapsed_ms = chrono::duration_cast<chrono::milliseconds>(end_time - start_time).count();
//     det_count = out_detections.size();

//     Point frame_center(frame.cols / 2, frame.rows / 2);
//     // drawMarker(frame, frame_center, Scalar(0, 0, 255), MARKER_CROSS, 15, 2);

//     for (size_t i = 0; i < out_detections.size(); ++i)
//     {
//         const auto& b = out_detections[i];
//         const bool center_protected = protect_center_target && b.center_protected;
//         const bool center_held = center_protected && b.center_held;

//         // White = ordinary detection. Green = current/held -C center target.
//         // Scalar color = center_protected ? Scalar(0, 255, 0) : Scalar(255, 255, 255);

//         Scalar color = Scalar(255, 255, 255);
//         // Match runClassicalEODetection() geometry logic:
//         // 1) clip the detector box to the valid frame, then
//         // 2) derive the object coordinate from the clipped box center.
//         Rect box = Rect(b.x, b.y, b.w, b.h) & Rect(0, 0, frame.cols, frame.rows);
//         if (box.width <= 0 || box.height <= 0) continue;

//         Point target_center(
//             box.x + box.width / 2,
//             box.y + box.height / 2);

//         drawCornerBox(frame, box, color, center_protected ? 1 : 1, 25);
//         line(frame, frame_center, target_center, color, center_protected ? 1 : 2, LINE_AA);

//         if (center_protected)
//         {
//             // Extra marker makes each selected -C nearest target unmistakable in live/saved video.
//             // drawMarker(frame, target_center, color, MARKER_DIAMOND, 14, 2, LINE_AA);
//             ;
//         }

//         string label;
//         if (center_protected)
//         {
//             ostringstream label_ss;
//             label_ss << (center_held ? "CENTER-HOLD " : "CENTER-PROTECTED ")
//                 << "Target " << (i + 1)
//                 << " D:" << fixed << setprecision(1) << b.center_distance_px << "px"
//                 << " (X:" << target_center.x << " Y:" << target_center.y << ")";
//             label = label_ss.str();
//         }
//         else
//         {
//             label = "Target " + to_string(i + 1) +
//                 " (X:" + to_string(target_center.x) +
//                 " Y:" + to_string(target_center.y) + ")";
//         }

//         int baseline = 0;
//         Size text_size = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.45, 1, &baseline);
//         int label_x = std::clamp(box.x, 0, std::max(0, frame.cols - text_size.width - 2));
//         int label_y = (box.y - 5 > text_size.height) ? box.y - 5 : std::min(frame.rows - 2, box.y + box.height + text_size.height + 4);
//         putText(frame, label, Point(label_x, label_y),
//             FONT_HERSHEY_SIMPLEX, 0.45, center_protected ? color : Scalar(0, 255, 255), center_protected ? 1 : 2);
//     }
// }

// struct FrameData
// {
//     int frame_idx;
//     Mat frame;
// };

// bool open_video_writer(VideoWriter& writer, const string& path, double fps, Size frame_size)
// {
//     if (frame_size.width <= 0 || frame_size.height <= 0) return false;

//     const vector<int> codecs = {
//         VideoWriter::fourcc('m', 'p', '4', 'v'),
//         VideoWriter::fourcc('a', 'v', 'c', '1'),
//         VideoWriter::fourcc('X', 'V', 'I', 'D')
//     };

//     for (int codec : codecs)
//     {
//         writer.open(path, codec, fps, frame_size);
//         if (writer.isOpened()) return true;
//     }

//     return false;
// }

// class ThreadedVideoProcessor
// {
// public:
//     ThreadedVideoProcessor(const string& video_path, BlobDetector& detector, DetMode mode, bool display,
//         bool protect_center_target, int center_target_count, int static_coord_frames)
//         : detector_(detector), det_mode_(mode), cap_(video_path), stop_threads_(false),
//         repeated_coordinate_filter_(static_coord_frames),
//         protect_center_target_(protect_center_target), center_target_count_(std::max(1, center_target_count)), display_(display) {
//     }

//     ThreadedVideoProcessor(int camera_index, BlobDetector& detector, DetMode mode, bool display,
//         bool protect_center_target, int center_target_count, int static_coord_frames)
//         : detector_(detector), det_mode_(mode), cap_(camera_index), stop_threads_(false),
//         repeated_coordinate_filter_(static_coord_frames),
//         protect_center_target_(protect_center_target), center_target_count_(std::max(1, center_target_count)), display_(display) {
//     }

//     void run(const string& run_dir, string mode_str, int class_id)
//     {
//         if (!cap_.isOpened())
//         {
//             cerr << "[ERROR] Failed to open video stream.\n";
//             return;
//         }

//         double fps = cap_.get(CAP_PROP_FPS);
//         if (fps <= 0) fps = 30.0;
//         int fw = static_cast<int>(cap_.get(CAP_PROP_FRAME_WIDTH));
//         int fh = static_cast<int>(cap_.get(CAP_PROP_FRAME_HEIGHT));
//         if (fw <= 0 || fh <= 0)
//         {
//             Mat probe;
//             if (cap_.read(probe) && !probe.empty())
//             {
//                 fw = probe.cols;
//                 fh = probe.rows;
//                 cap_.set(CAP_PROP_POS_FRAMES, 0);
//             }
//         }
//         if (fw <= 0 || fh <= 0)
//         {
//             cerr << "[ERROR] Could not determine video frame size.\n";
//             return;
//         }

//         YoloDatasetPaths yolo_paths;
//         try
//         {
//             yolo_paths = create_yolo_dataset_folders(run_dir);
//         }
//         catch (const fs::filesystem_error& e)
//         {
//             cerr << "[ERROR] Failed to create YOLO dataset folders: " << e.what() << "\n";
//             return;
//         }

//         long long saved_yolo_images = 0;
//         long long saved_yolo_boxes = 0;

//         string out_video_path = run_dir + "/Output_" + mode_str + "_Video.mp4";
//         VideoWriter writer;
//         if (!open_video_writer(writer, out_video_path, fps, Size(fw, fh)))
//         {
//             cerr << "[WARN] Could not open video writer. Report will still be saved.\n";
//         }

//         string txt_path = run_dir + "/video_detection_report.txt";
//         ofstream report(txt_path);

//         // Testing coordinate log: one CSV row per currently output target.
//         // In -C mode this means only the selected nearest-N targets are logged.
//         string coord_path = run_dir + "/target_coordinates.csv";
//         ofstream coord_log(coord_path);
//         if (!coord_log)
//         {
//             cerr << "[WARN] Could not create coordinate log: " << coord_path << "\n";
//         }
//         else
//         {
//             coord_log << "Frame,TargetID,X,Y,CenterProtected,CenterHeld,CenterDistancePx,Type\n";
//         }

//         if (!report)
//         {
//             cerr << "[WARN] Could not create report file: " << txt_path << "\n";
//         }
//         else
//         {
//             report << "Mode: " << mode_str << "\n";
//             report << "YOLO Class ID: " << class_id << "\n";
//             report << "YOLO Images Dir: " << yolo_paths.images_dir << "\n";
//             report << "YOLO Labels Dir: " << yolo_paths.labels_dir << "\n";
//             report << "CenterOnly(-C): " << (protect_center_target_ ? "ON" : "OFF") << "\n";
//             if (protect_center_target_)
//             {
//                 report << "CenterTargetCount: " << center_target_count_ << "\n";
//                 report << "StaticCoordinateConsecutiveFrames: " << repeated_coordinate_filter_.stable_frames() << "\n";
//             }
//             report << "Frame\tID\tX\tY\tW\tH\tScore\tType\tArea\tAspect\textent\tDoG_Mean\tDist(px)\tCenterProtected\tCenterHeld\tCenterDist(px)\n";
//             report << "---------------------------------------------------------------------------------------\n";
//         }
//         mutex report_mutex;

//         thread producer([this]() {
//             int idx = 0;
//             Mat temp;
//             while (!stop_threads_ && cap_.read(temp))
//             {
//                 if (temp.empty()) break;
//                 idx++;
//                 {
//                     unique_lock<mutex> lock(queue_mutex_);
//                     queue_cond_.wait(lock, [this] { return stop_threads_ || frame_queue_.size() < max_queue_size_; });
//                     if (stop_threads_) break;
//                     frame_queue_.push({ idx, temp.clone() });
//                 }
//                 queue_cond_.notify_one();
//             }
//             is_producer_done_ = true;
//             queue_cond_.notify_all();
//             });

//         thread consumer([this, &writer, &report, &coord_log, &report_mutex, &yolo_paths, &saved_yolo_images, &saved_yolo_boxes, class_id]() {
//             while (true)
//             {
//                 FrameData item;
//                 {
//                     unique_lock<mutex> lock(queue_mutex_);
//                     queue_cond_.wait(lock, [this] { return stop_threads_ || !frame_queue_.empty() || is_producer_done_; });

//                     if (stop_threads_ || (frame_queue_.empty() && is_producer_done_)) break;

//                     item = frame_queue_.front();
//                     frame_queue_.pop();
//                 }
//                 queue_cond_.notify_one();

//                 Mat clean_frame = item.frame.clone();
//                 long long p_ms = 0;
//                 int d_count = 0;
//                 vector<BBox> detections;
//                 process_frame_detections(item.frame, detector_, det_mode_, p_ms, d_count, detections,
//                     &stationary_filter_, &center_keeper_, protect_center_target_, center_target_count_,
//                     protect_center_target_ ? &repeated_coordinate_filter_ : nullptr);

//                 int frame_labels = 0;
//                 const string sample_stem = "frame_" + zero_padded_index(item.frame_idx);
//                 if (save_yolo_sample(clean_frame, detections, class_id, yolo_paths, sample_stem, frame_labels))
//                 {
//                     ++saved_yolo_images;
//                     saved_yolo_boxes += frame_labels;
//                     cout << "[YOLO] Saved " << sample_stem << ".jpg with " << frame_labels << " label(s).\n";
//                 }

//                 {
//                     lock_guard<mutex> lock(report_mutex);
//                     for (size_t i = 0; i < detections.size(); ++i)
//                     {
//                         const auto& b = detections[i];
//                         Rect report_box = Rect(b.x, b.y, b.w, b.h) & Rect(0, 0, item.frame.cols, item.frame.rows);
//                         if (report_box.width <= 0 || report_box.height <= 0) continue;
//                         Point report_center(
//                             report_box.x + report_box.width / 2,
//                             report_box.y + report_box.height / 2);
//                         double dist = sqrt(pow(report_center.x - item.frame.cols / 2.0, 2) + pow(report_center.y - item.frame.rows / 2.0, 2));

//                         // Print the target coordinate immediately for testing.
//                         cout << "[COORD] Frame " << item.frame_idx
//                             << " Target " << (i + 1)
//                             << " X=" << report_center.x
//                             << " Y=" << report_center.y;
//                         if (b.center_protected)
//                         {
//                             cout << " CenterDist=" << fixed << setprecision(2) << b.center_distance_px << "px";
//                         }
//                         if (b.center_held)
//                         {
//                             cout << " [HELD]";
//                         }
//                         cout << "\n";

//                         // Save a compact machine-readable coordinate log as CSV.
//                         if (coord_log)
//                         {
//                             coord_log << item.frame_idx << "," << (i + 1) << ","
//                                 << report_center.x << "," << report_center.y << ","
//                                 << (b.center_protected ? 1 : 0) << ","
//                                 << (b.center_held ? 1 : 0) << ","
//                                 << fixed << setprecision(2)
//                                 << (b.center_protected ? b.center_distance_px : -1.0) << ","
//                                 << b.type << "\n";
//                         }

//                         if (report)
//                         {
//                             report << item.frame_idx << "\t" << (i + 1) << "\t" << report_center.x << "\t" << report_center.y
//                                 << "\t" << report_box.width << "\t" << report_box.height << "\t" << fixed << setprecision(2) << b.score << "\t" << b.type
//                                 << "\t" << b.area << "\t" << b.aspect << "\t" << b.extent << "\t" << b.dog_mean << "\t" << dist
//                                 << "\t" << (b.center_protected ? 1 : 0)
//                                 << "\t" << (b.center_held ? 1 : 0)
//                                 << "\t" << (b.center_protected ? b.center_distance_px : -1.0) << "\n";
//                         }
//                     }
//                 }

//                 if (writer.isOpened()) writer.write(item.frame);

//                 if (display_)
//                 {
//                     imshow("Multi-Threaded Video Stream", item.frame);
//                     if ((waitKey(1) & 0xFF) == 27)
//                     {
//                         stop_threads_ = true;
//                         queue_cond_.notify_all();
//                         break;
//                     }
//                 }
//             }
//             });

//         producer.join();
//         consumer.join();
//         cap_.release();
//         if (writer.isOpened()) writer.release();
//         report.close();
//         coord_log.close();
//         cout << "[INFO] Target coordinates saved to: " << coord_path << "\n";
//         cout << "[YOLO] Dataset images: " << yolo_paths.images_dir << "\n";
//         cout << "[YOLO] Dataset labels: " << yolo_paths.labels_dir << "\n";
//         cout << "[YOLO] Saved " << saved_yolo_images << " image(s) and " << saved_yolo_boxes << " box label(s).\n";
//         cout << "\n[DONE] Multi-threaded video processing complete. Saved in: " << run_dir << "\n";
//     }

// private:
//     BlobDetector& detector_;
//     DetMode det_mode_;
//     VideoCapture cap_;
//     atomic<bool> stop_threads_;
//     atomic<bool> is_producer_done_{ false };
//     queue<FrameData> frame_queue_;
//     mutex queue_mutex_;
//     condition_variable queue_cond_;
//     const size_t max_queue_size_ = 10;

//     // Persistent state across video frames. Defaults:
//     // 6 stable frames, <=3 px center drift, small size jitter.
//     StationaryTargetFilter stationary_filter_;
//     CenterTargetKeeper center_keeper_{ 3 };
//     RepeatedCoordinateFilter repeated_coordinate_filter_{ 10 };
//     bool protect_center_target_;
//     int center_target_count_ = 1;
//     bool display_;
// };

// int main(int argc, char** argv)
// {
//     AppOptions options;
//     if (!parse_args(argc, argv, options))
//     {
//         return 1;
//     }
//     if (options.help_requested)
//     {
//         return 0;
//     }

//     if (!options.class_id_provided)
//     {
//         if (!can_prompt())
//         {
//             cerr << "[ERROR] No interactive terminal is available. Provide --class-id <N>.\n";
//             return 1;
//         }
//         if (!prompt_for_class_id(options.class_id))
//         {
//             return 1;
//         }
//     }

//     cout << "[INFO] YOLO annotation class ID: " << options.class_id << "\n";

//     if (!options.mode_provided)
//     {
//         if (can_prompt())
//         {
//             options.mode = get_detection_mode();
//         }
//         else
//         {
//             cout << "[INFO] No --mode provided and stdin is non-interactive. Using hough.\n";
//         }
//     }

//     if (can_prompt() && !options.debug)
//     {
//         options.debug = get_debug_mode_choice();
//     }

//     if (options.display && !display_available())
//     {
//         cout << "[INFO] No graphical display detected. Running with --no-display behavior.\n";
//         options.display = false;
//     }
//     if (options.debug && !options.display)
//     {
//         cout << "[INFO] Debug windows disabled because display output is unavailable.\n";
//         options.debug = false;
//     }

//     string input_path = options.input_path;
//     const bool use_camera = is_integer_source(input_path);
//     string ext = fs::path(input_path).extension().string();
//     ext = to_lower_copy(ext);
//     bool is_video = use_camera || (ext == ".mp4" || ext == ".avi" || ext == ".mov" || ext == ".mkv" || ext == ".m4v" ||
//         ext == ".webm" || ext == ".mpg" || ext == ".mpeg");

//     Mat sample_frame;
//     int width = 0;
//     int height = 0;

//     if (is_video)
//     {
//         VideoCapture test_cap;
//         if (use_camera)
//         {
//             test_cap.open(stoi(input_path));
//         }
//         else
//         {
//             test_cap.open(input_path);
//         }
//         if (!test_cap.isOpened())
//         {
//             cerr << "[ERROR] Failed to open video source: " << input_path << "\n";
//             return -1;
//         }
//         width = static_cast<int>(test_cap.get(CAP_PROP_FRAME_WIDTH));
//         height = static_cast<int>(test_cap.get(CAP_PROP_FRAME_HEIGHT));
//         if (width <= 0 || height <= 0)
//         {
//             Mat probe;
//             if (test_cap.read(probe) && !probe.empty())
//             {
//                 width = probe.cols;
//                 height = probe.rows;
//             }
//         }
//         test_cap.release();
//     }
//     else
//     {
//         sample_frame = imread(input_path);
//         if (sample_frame.empty())
//         {
//             VideoCapture fallback_cap(input_path);
//             if (fallback_cap.isOpened())
//             {
//                 is_video = true;
//                 width = static_cast<int>(fallback_cap.get(CAP_PROP_FRAME_WIDTH));
//                 height = static_cast<int>(fallback_cap.get(CAP_PROP_FRAME_HEIGHT));
//                 if (width <= 0 || height <= 0)
//                 {
//                     Mat probe;
//                     if (fallback_cap.read(probe) && !probe.empty())
//                     {
//                         width = probe.cols;
//                         height = probe.rows;
//                     }
//                 }
//                 fallback_cap.release();
//             }
//             else
//             {
//                 cerr << "[ERROR] Failed to load input as image or video: " << input_path << "\n";
//                 return -1;
//             }
//         }
//         else
//         {
//             width = sample_frame.cols;
//             height = sample_frame.rows;
//         }
//     }

//     if (width <= 0 || height <= 0)
//     {
//         cerr << "[ERROR] Could not determine input dimensions for: " << input_path << "\n";
//         return -1;
//     }

//     cout << "[INFO] Loaded input dimensions: " << width << "x" << height << "\n";
//     cout << "[INFO] Center-target protection (-C): "
//         << (options.protect_center_target ? "ENABLED" : "DISABLED") << "\n";
//     if (options.protect_center_target)
//     {
//         cout << "[INFO] -C nearest-target count: " << options.center_target_count << "\n";
//         cout << "[INFO] -C CENTER-ONLY mode: keeping ONLY the " << options.center_target_count
//             << " nearest-to-center detection(s), ordered nearest -> farthest.\n";
//         cout << "[INFO] All farther detections are discarded before drawing/reporting/downstream X/Y output.\n";
//         cout << "[INFO] Selected center target(s) bypass stationary suppression.\n";
//         cout << "[INFO] Exact repeated-coordinate suppression: " << options.static_coord_frames
//             << " consecutive identical X/Y frames.\n";
//         cout << "[INFO] Static-coordinate suppression runs before -C nearest-target selection.\n";
//         cout << "[INFO] Short detector flicker hold: up to 3 frames (CENTER-HOLD).\n";
//     }

//     TrackerConfig cfg = options.config;
//     DetMode det_mode = options.mode;

//     BlobDetector detector(cfg, width, height, options.prior, options.debug, is_video);

//     string run_dir;
//     try
//     {
//         run_dir = create_run_folder(options.output_root);
//     }
//     catch (const fs::filesystem_error& e)
//     {
//         cerr << "[ERROR] Failed to create output directory under '" << options.output_root << "': " << e.what() << "\n";
//         return -1;
//     }
//     string mode_str = mode_to_string(det_mode);

//     YoloDatasetPaths image_yolo_paths;
//     if (!is_video)
//     {
//         try
//         {
//             image_yolo_paths = create_yolo_dataset_folders(run_dir);
//         }
//         catch (const fs::filesystem_error& e)
//         {
//             cerr << "[ERROR] Failed to create YOLO dataset folders: " << e.what() << "\n";
//             return -1;
//         }
//     }

//     if (!is_video)
//     {
//         Mat clean_sample_frame = sample_frame.clone();
//         long long processing_ms = 0;
//         int det_count = 0;
//         vector<BBox> detections;
//         CenterTargetKeeper image_center_keeper(0);

//         process_frame_detections(
//             sample_frame, detector, det_mode, processing_ms, det_count, detections,
//             nullptr,
//             options.protect_center_target ? &image_center_keeper : nullptr,
//             options.protect_center_target, options.center_target_count);

//         int image_labels_written = 0;
//         const string sample_stem = "image_000001";
//         const bool yolo_saved = save_yolo_sample(
//             clean_sample_frame, detections, options.class_id,
//             image_yolo_paths, sample_stem, image_labels_written);

//         if (yolo_saved)
//         {
//             cout << "[YOLO] Saved " << sample_stem << ".jpg with "
//                 << image_labels_written << " label(s).\n";
//         }
//         else
//         {
//             cout << "[YOLO] No fresh detections; no training sample was written.\n";
//         }

//         string txt_path = run_dir + "/detection_report.txt";
//         ofstream report(txt_path);

//         string coord_path = run_dir + "/target_coordinates.csv";
//         ofstream coord_log(coord_path);
//         if (!coord_log)
//         {
//             cerr << "[WARN] Could not create coordinate log: " << coord_path << "\n";
//         }
//         else
//         {
//             coord_log << "Frame,TargetID,X,Y,CenterProtected,CenterHeld,CenterDistancePx,Type\n";
//         }

//         if (!report)
//         {
//             cerr << "[WARN] Could not create report file: " << txt_path << "\n";
//         }
//         else
//         {
//             report << "Mode: " << mode_str << "\n";
//             report << "YOLO Class ID: " << options.class_id << "\n";
//             report << "YOLO Images Dir: " << image_yolo_paths.images_dir << "\n";
//             report << "YOLO Labels Dir: " << image_yolo_paths.labels_dir << "\n";
//             report << "Processing Time: " << processing_ms << " ms\n";
//             report << "Number of Detections: " << det_count << "\n\n";
//             report << "Detections Debug Log:\n";
//             report << "ID\tX\tY\tW\tH\tScore\tType\tArea\tAspect\textent\tDoG_Mean\tDist(px)\n";
//             report << "---------------------------------------------------------------------------------------\n";

//         }

//         for (size_t i = 0; i < detections.size(); ++i)
//         {
//             const auto& b = detections[i];
//             Rect report_box = Rect(b.x, b.y, b.w, b.h) & Rect(0, 0, sample_frame.cols, sample_frame.rows);
//             if (report_box.width <= 0 || report_box.height <= 0) continue;
//             Point report_center(
//                 report_box.x + report_box.width / 2,
//                 report_box.y + report_box.height / 2);
//             double dist = sqrt(pow(report_center.x - sample_frame.cols / 2.0, 2) + pow(report_center.y - sample_frame.rows / 2.0, 2));

//             cout << "[COORD] Image Target " << (i + 1)
//                 << " X=" << report_center.x
//                 << " Y=" << report_center.y;
//             if (b.center_protected)
//             {
//                 cout << " CenterDist=" << fixed << setprecision(2) << b.center_distance_px << "px";
//             }
//             cout << "\n";

//             if (coord_log)
//             {
//                 coord_log << 1 << "," << (i + 1) << ","
//                     << report_center.x << "," << report_center.y << ","
//                     << (b.center_protected ? 1 : 0) << ","
//                     << (b.center_held ? 1 : 0) << ","
//                     << fixed << setprecision(2)
//                     << (b.center_protected ? b.center_distance_px : -1.0) << ","
//                     << b.type << "\n";
//             }

//             if (report)
//             {
//                 report << (i + 1) << "\t" << report_center.x << "\t" << report_center.y << "\t" << report_box.width << "\t" << report_box.height
//                     << "\t" << fixed << setprecision(2) << b.score << "\t" << b.type
//                     << "\t" << b.area << "\t" << b.aspect << "\t" << b.extent << "\t" << b.dog_mean << "\t" << dist << "\n";
//             }
//         }
//         report.close();
//         coord_log.close();
//         cout << "[INFO] Target coordinates saved to: " << coord_path << "\n";
//         cout << "[YOLO] Dataset images: " << image_yolo_paths.images_dir << "\n";
//         cout << "[YOLO] Dataset labels: " << image_yolo_paths.labels_dir << "\n";

//         string out_path = run_dir + "/Output_" + mode_str + "_Result.jpg";
//         if (!imwrite(out_path, sample_frame))
//         {
//             cerr << "[WARN] Failed to save result image: " << out_path << "\n";
//         }

//         cout << "\n[DONE] Processing complete. Output saved in subfolder: " << run_dir << "\n";
//         if (options.display)
//         {
//             namedWindow("Detector Result", WINDOW_NORMAL);
//             imshow("Detector Result", sample_frame);
//             waitKey(0);
//         }
//     }
//     else
//     {
//         if (options.display) namedWindow("Multi-Threaded Video Stream", WINDOW_NORMAL);
//         if (use_camera)
//         {
//             ThreadedVideoProcessor processor(stoi(input_path), detector, det_mode, options.display,
//                 options.protect_center_target, options.center_target_count, options.static_coord_frames);
//             processor.run(run_dir, mode_str, options.class_id);
//         }
//         else
//         {
//             ThreadedVideoProcessor processor(input_path, detector, det_mode, options.display,
//                 options.protect_center_target, options.center_target_count, options.static_coord_frames);
//             processor.run(run_dir, mode_str, options.class_id);
//         }
//     }

//     return 0;
// }



// ./Exe-classical_annotation_exporter /home/abhirup/Desktop/Development/Classical_Det/EOTS_TONBO/Det/IR_Vids/IR_11-40-14.514.avi --mode dog --class-id 1 -C 1


// =============================@2==============================




// Classical OpenCV detector + ultra-dynamic automatic YOLO annotation exporter.
// Training-ready dataset layout per run:
//   run_.../dataset/data.yaml
//   run_.../dataset/images/train/<sample>.jpg|png
//   run_.../dataset/images/val/<sample>.jpg|png
//   run_.../dataset/labels/train/<sample>.txt
//   run_.../dataset/labels/val/<sample>.txt
// Optional annotated previews are kept separate from clean training images.
// Each label line uses YOLO detection format:
//   <class_id> <x_center_norm> <y_center_norm> <width_norm> <height_norm>
//
#include <opencv2/opencv.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <queue>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#ifdef _WIN32
#include <io.h>
#else
#include <unistd.h>
#endif

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
    bool center_protected = false;
    bool center_held = false;          // true only when -C reuses last center target during a short detector miss
    double center_distance_px = -1.0;

    int cx() const { return x + w / 2; }
    int cy() const { return y + h / 2; }
    Rect rect() const { return Rect(x, y, w, h); }
};

class BlobDetector
{
public:
    BlobDetector(const TrackerConfig& cfg, int roi_w, int roi_h, PriorMode prior = PriorMode::Center, bool debug_mode = false, bool is_video = false)
        : cfg_(cfg), roi_w_(roi_w), roi_h_(roi_h), prior_(prior), debug_mode_(debug_mode), is_video_(is_video) {
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
        }
        else {
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

            candidates.push_back(BBox{ r.x, r.y, r.width, r.height, (float)score, "dog_detected", area, aspect, extent, dog_mean });
        }

        auto final_boxes = apply_nms(candidates, 0.18f);

        if (debug_mode_)
        {
            Mat contour_vis;
            cvtColor(gray, contour_vis, COLOR_GRAY2BGR);
            drawContours(contour_vis, contours, -1, Scalar(0, 255, 0), 1);
            for (const auto& b : final_boxes)
            {
                // rectangle(contour_vis, b.rect(), Scalar(0, 0, 255), 2);
            }
            namedWindow("Debug: Contour Filtering & NMS", WINDOW_NORMAL);
            imshow("Debug: Contour Filtering & NMS", contour_vis);
            handle_debug_pause("[DEBUG] Contours and NMS results displayed.");
        }

        return final_boxes;
    }

    vector<BBox> detect_dog_all(const Mat& roi) const
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
        GaussianBlur(gray, g1a, { 3, 3 }, 0.8);
        GaussianBlur(gray, g2a, { 7, 7 }, 2.0);
        subtract(g1a, g2a, dog_a, noArray(), CV_16S);

        Mat g1b, g2b, dog_b;
        GaussianBlur(gray, g1b, { 5, 5 }, 1.5);
        GaussianBlur(gray, g2b, { 11, 11 }, 3.5);
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

    vector<BBox> detect_hough_circles(const Mat& img) const
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
                double area_approx = CV_PI * r * r;
                boxes.push_back(BBox{ x, y, w, h, 0.95f, "hough_circle", area_approx, 1.0, 0.78, 25.0 });
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

    vector<BBox> detect_hybrid_all(const Mat& img) const
    {
        auto dog_boxes = detect_dog_all(img);
        auto hough_boxes = detect_hough_circles(img);

        vector<BBox> combined = dog_boxes;
        for (const auto& hb : hough_boxes)
        {
            bool matched = false;
            for (auto& db : dog_boxes)
            {
                if ((db.rect() & hb.rect()).area() > 0)
                {
                    matched = true;
                    break;
                }
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
#ifdef _WIN32
    return _isatty(_fileno(stdin)) != 0;
#else
    return isatty(STDIN_FILENO) != 0;
#endif
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
        << "  -C [N], -c [N], --center [N]  CENTER-ONLY mode: keep/output N nearest detections to frame center. Default N=1.\n"
        << "  --static-coord-frames <N>     In -C video mode, suppress an exact X/Y repeated for N consecutive frames. Default: 10.\n"
        << "  --out-dir <dir>               Output root directory. Default: Res\n"
        << "  --class-id <N>                YOLO class ID (0,1,2,...). If omitted, prompts before processing.\n"
        << "  --class-name <name>           Class name written to data.yaml. Default: target\n"
        << "  --frame-step <N>              Export every Nth video frame; detection still runs every frame. Default: 1\n"
        << "  --start-frame <N>             Do not export before this 1-based video frame. Default: 1\n"
        << "  --max-samples <N>             Maximum saved samples. 0 = unlimited. Default: 0\n"
        << "  --val-percent <0..100>        Deterministic validation percentage. Default: 20\n"
        << "  --include-empty               Save sampled frames with zero boxes as negative examples.\n"
        << "  --min-box-px <N>              Reject boxes narrower/shorter than N pixels. Default: 2\n"
        << "  --min-box-area-frac <value>   Minimum box/image area fraction [0,1]. Default: 0\n"
        << "  --max-box-area-frac <value>   Maximum box/image area fraction (0,1]. Default: 1\n"
        << "  --image-format <jpg|png>      Training image format. Default: jpg\n"
        << "  --jpeg-quality <1..100>       JPEG quality. Default: 95\n"
        << "  --save-preview                Save annotated preview images separately.\n"
        << "  --sample-prefix <text>        Export filename prefix. Default: frame\n"
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
    bool protect_center_target = false;
    int center_target_count = 1;   // -C alone => 1; -C N => keep N nearest targets
    int static_coord_frames = 10;  // suppress exact repeated X/Y after this many consecutive video frames in -C mode
    int class_id = -1;              // YOLO class ID. Prompted if not supplied on CLI.
    bool class_id_provided = false;
    string class_name = "target";
    int frame_step = 1;
    long long start_frame = 1;
    long long max_samples = 0;       // 0 = unlimited
    int val_percent = 20;
    bool include_empty = false;
    int min_box_px = 2;
    double min_box_area_frac = 0.0;
    double max_box_area_frac = 1.0;
    string image_format = "jpg";
    int jpeg_quality = 95;
    bool save_preview = false;
    string sample_prefix = "frame";
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
        if (arg == "-C" || arg == "-c" || arg == "--center" || arg == "--protect-center")
        {
            options.protect_center_target = true;
            options.center_target_count = 1; // preserve legacy/default -C behavior

            // Optional positive integer immediately after -C selects how many
            // nearest-to-center targets are kept. Do not consume another flag.
            if (i + 1 < argc)
            {
                string maybe_count = argv[i + 1];
                int requested_count = 0;

                if (parse_int_arg(maybe_count, requested_count))
                {
                    if (requested_count <= 0)
                    {
                        cerr << "[ERROR] -C target count must be a positive integer (1, 2, 3, ...).\n";
                        return false;
                    }
                    options.center_target_count = requested_count;
                    ++i;
                }
                else if (!maybe_count.empty() && maybe_count[0] != '-')
                {
                    cerr << "[ERROR] Invalid -C target count: " << maybe_count
                        << ". Use -C or -C <positive integer>.\n";
                    return false;
                }
            }
            continue;
        }
        if (arg.rfind("-C=", 0) == 0 || arg.rfind("-c=", 0) == 0 ||
            arg.rfind("--center=", 0) == 0 || arg.rfind("--protect-center=", 0) == 0)
        {
            options.protect_center_target = true;

            size_t eq = arg.find('=');
            string count_text = (eq == string::npos) ? string{} : arg.substr(eq + 1);
            int requested_count = 0;
            if (!parse_int_arg(count_text, requested_count) || requested_count <= 0)
            {
                cerr << "[ERROR] -C target count must be a positive integer (1, 2, 3, ...).\n";
                return false;
            }

            options.center_target_count = requested_count;
            continue;
        }
        if (arg == "--static-coord-frames" || arg.rfind("--static-coord-frames=", 0) == 0)
        {
            if (arg == "--static-coord-frames")
            {
                if (!consume_option_value(i, argc, argv, value))
                {
                    cerr << "[ERROR] Missing --static-coord-frames value.\n";
                    return false;
                }
            }
            else
            {
                value = arg.substr(22);
            }

            int requested_frames = 0;
            if (!parse_int_arg(value, requested_frames) || requested_frames < 2)
            {
                cerr << "[ERROR] --static-coord-frames must be an integer >= 2.\n";
                return false;
            }
            options.static_coord_frames = requested_frames;
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

        if (arg == "--class-id" || arg.rfind("--class-id=", 0) == 0)
        {
            if (arg == "--class-id")
            {
                if (!consume_option_value(i, argc, argv, value))
                {
                    cerr << "[ERROR] Missing --class-id value.\n";
                    return false;
                }
            }
            else
            {
                value = arg.substr(11);
            }

            int requested_class_id = -1;
            if (!parse_int_arg(value, requested_class_id) || requested_class_id < 0)
            {
                cerr << "[ERROR] --class-id must be an integer >= 0.\n";
                return false;
            }
            options.class_id = requested_class_id;
            options.class_id_provided = true;
            continue;
        }

        if (arg == "--class-name" || arg.rfind("--class-name=", 0) == 0)
        {
            if (arg == "--class-name")
            {
                if (!consume_option_value(i, argc, argv, value))
                {
                    cerr << "[ERROR] Missing --class-name value.\n";
                    return false;
                }
            }
            else value = arg.substr(13);
            if (value.empty())
            {
                cerr << "[ERROR] --class-name cannot be empty.\n";
                return false;
            }
            options.class_name = value;
            continue;
        }
        if (arg == "--include-empty")
        {
            options.include_empty = true;
            continue;
        }
        if (arg == "--save-preview")
        {
            options.save_preview = true;
            continue;
        }
        if (arg == "--frame-step" || arg.rfind("--frame-step=", 0) == 0)
        {
            if (arg == "--frame-step")
            {
                if (!consume_option_value(i, argc, argv, value)) { cerr << "[ERROR] Missing --frame-step value.\n"; return false; }
            }
            else value = arg.substr(13);
            int parsed = 0;
            if (!parse_int_arg(value, parsed) || parsed < 1) { cerr << "[ERROR] --frame-step must be >= 1.\n"; return false; }
            options.frame_step = parsed;
            continue;
        }
        if (arg == "--start-frame" || arg.rfind("--start-frame=", 0) == 0)
        {
            if (arg == "--start-frame")
            {
                if (!consume_option_value(i, argc, argv, value)) { cerr << "[ERROR] Missing --start-frame value.\n"; return false; }
            }
            else value = arg.substr(14);
            int parsed = 0;
            if (!parse_int_arg(value, parsed) || parsed < 1) { cerr << "[ERROR] --start-frame must be >= 1.\n"; return false; }
            options.start_frame = parsed;
            continue;
        }
        if (arg == "--max-samples" || arg.rfind("--max-samples=", 0) == 0)
        {
            if (arg == "--max-samples")
            {
                if (!consume_option_value(i, argc, argv, value)) { cerr << "[ERROR] Missing --max-samples value.\n"; return false; }
            }
            else value = arg.substr(14);
            int parsed = 0;
            if (!parse_int_arg(value, parsed) || parsed < 0) { cerr << "[ERROR] --max-samples must be >= 0.\n"; return false; }
            options.max_samples = parsed;
            continue;
        }
        if (arg == "--val-percent" || arg.rfind("--val-percent=", 0) == 0)
        {
            if (arg == "--val-percent")
            {
                if (!consume_option_value(i, argc, argv, value)) { cerr << "[ERROR] Missing --val-percent value.\n"; return false; }
            }
            else value = arg.substr(14);
            int parsed = 0;
            if (!parse_int_arg(value, parsed) || parsed < 0 || parsed > 100) { cerr << "[ERROR] --val-percent must be 0..100.\n"; return false; }
            options.val_percent = parsed;
            continue;
        }
        if (arg == "--min-box-px" || arg.rfind("--min-box-px=", 0) == 0)
        {
            if (arg == "--min-box-px")
            {
                if (!consume_option_value(i, argc, argv, value)) { cerr << "[ERROR] Missing --min-box-px value.\n"; return false; }
            }
            else value = arg.substr(13);
            int parsed = 0;
            if (!parse_int_arg(value, parsed) || parsed < 1) { cerr << "[ERROR] --min-box-px must be >= 1.\n"; return false; }
            options.min_box_px = parsed;
            continue;
        }
        if (arg == "--min-box-area-frac" || arg.rfind("--min-box-area-frac=", 0) == 0)
        {
            if (arg == "--min-box-area-frac")
            {
                if (!consume_option_value(i, argc, argv, value)) { cerr << "[ERROR] Missing --min-box-area-frac value.\n"; return false; }
            }
            else value = arg.substr(20);
            float parsed = 0.0f;
            if (!parse_float_arg(value, parsed) || parsed < 0.0f || parsed > 1.0f) { cerr << "[ERROR] --min-box-area-frac must be in [0,1].\n"; return false; }
            options.min_box_area_frac = parsed;
            continue;
        }
        if (arg == "--max-box-area-frac" || arg.rfind("--max-box-area-frac=", 0) == 0)
        {
            if (arg == "--max-box-area-frac")
            {
                if (!consume_option_value(i, argc, argv, value)) { cerr << "[ERROR] Missing --max-box-area-frac value.\n"; return false; }
            }
            else value = arg.substr(20);
            float parsed = 0.0f;
            if (!parse_float_arg(value, parsed) || parsed <= 0.0f || parsed > 1.0f) { cerr << "[ERROR] --max-box-area-frac must be in (0,1].\n"; return false; }
            options.max_box_area_frac = parsed;
            continue;
        }
        if (arg == "--image-format" || arg.rfind("--image-format=", 0) == 0)
        {
            if (arg == "--image-format")
            {
                if (!consume_option_value(i, argc, argv, value)) { cerr << "[ERROR] Missing --image-format value.\n"; return false; }
            }
            else value = arg.substr(15);
            value = to_lower_copy(value);
            if (value != "jpg" && value != "jpeg" && value != "png") { cerr << "[ERROR] --image-format must be jpg or png.\n"; return false; }
            options.image_format = (value == "jpeg") ? "jpg" : value;
            continue;
        }
        if (arg == "--jpeg-quality" || arg.rfind("--jpeg-quality=", 0) == 0)
        {
            if (arg == "--jpeg-quality")
            {
                if (!consume_option_value(i, argc, argv, value)) { cerr << "[ERROR] Missing --jpeg-quality value.\n"; return false; }
            }
            else value = arg.substr(15);
            int parsed = 0;
            if (!parse_int_arg(value, parsed) || parsed < 1 || parsed > 100) { cerr << "[ERROR] --jpeg-quality must be 1..100.\n"; return false; }
            options.jpeg_quality = parsed;
            continue;
        }
        if (arg == "--sample-prefix" || arg.rfind("--sample-prefix=", 0) == 0)
        {
            if (arg == "--sample-prefix")
            {
                if (!consume_option_value(i, argc, argv, value)) { cerr << "[ERROR] Missing --sample-prefix value.\n"; return false; }
            }
            else value = arg.substr(16);
            if (value.empty()) { cerr << "[ERROR] --sample-prefix cannot be empty.\n"; return false; }
            for (char& c : value)
            {
                const unsigned char uc = static_cast<unsigned char>(c);
                if (!(std::isalnum(uc) || c == '_' || c == '-')) c = '_';
            }
            options.sample_prefix = value;
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
    if (options.min_box_area_frac > options.max_box_area_frac)
    {
        cerr << "[ERROR] --min-box-area-frac cannot exceed --max-box-area-frac.\n";
        return false;
    }

    return true;
}

bool prompt_for_class_id(int& class_id)
{
    while (true)
    {
        cout << "\n--- YOLO Annotation Setup ---\n";
        cout << " Enter class ID (0, 1, 2, ...): ";

        string value;
        if (!getline(cin, value))
        {
            cerr << "[ERROR] Could not read class ID from stdin.\n";
            return false;
        }

        int parsed = -1;
        if (parse_int_arg(value, parsed) && parsed >= 0)
        {
            class_id = parsed;
            return true;
        }

        cerr << "[ERROR] Class ID must be a non-negative integer.\n";
    }
}

struct YoloDatasetPaths
{
    string root_dir;
    string train_images_dir;
    string val_images_dir;
    string train_labels_dir;
    string val_labels_dir;
    string train_previews_dir;
    string val_previews_dir;
    string yaml_path;
};

YoloDatasetPaths create_yolo_dataset_folders(const string& run_dir, bool save_preview)
{
    YoloDatasetPaths paths;
    paths.root_dir = (fs::path(run_dir) / "dataset").string();
    paths.train_images_dir = (fs::path(paths.root_dir) / "images" / "train").string();
    paths.val_images_dir = (fs::path(paths.root_dir) / "images" / "val").string();
    paths.train_labels_dir = (fs::path(paths.root_dir) / "labels" / "train").string();
    paths.val_labels_dir = (fs::path(paths.root_dir) / "labels" / "val").string();
    paths.train_previews_dir = (fs::path(paths.root_dir) / "previews" / "train").string();
    paths.val_previews_dir = (fs::path(paths.root_dir) / "previews" / "val").string();
    paths.yaml_path = (fs::path(paths.root_dir) / "data.yaml").string();

    fs::create_directories(paths.train_images_dir);
    fs::create_directories(paths.val_images_dir);
    fs::create_directories(paths.train_labels_dir);
    fs::create_directories(paths.val_labels_dir);
    if (save_preview)
    {
        fs::create_directories(paths.train_previews_dir);
        fs::create_directories(paths.val_previews_dir);
    }
    return paths;
}

string yaml_single_quote(const string& text)
{
    string out;
    for (char c : text)
    {
        if (c == '\'') out += "''";
        else out += c;
    }
    return "'" + out + "'";
}

bool write_yolo_data_yaml(const YoloDatasetPaths& paths, int class_id, const string& class_name, int val_percent)
{
    ofstream yaml(paths.yaml_path, ios::out | ios::trunc);
    if (!yaml)
    {
        cerr << "[WARN] Could not write YOLO data.yaml: " << paths.yaml_path << "\n";
        return false;
    }
    yaml << "# Auto-generated YOLO dataset configuration\n";
    const string absolute_root = fs::absolute(paths.root_dir).generic_string();
    yaml << "path: " << yaml_single_quote(absolute_root) << "\n";
    yaml << "train: images/train\n";
    yaml << "val: " << (val_percent > 0 ? "images/val" : "images/train") << "\n";
    yaml << "names:\n";
    for (int i = 0; i <= class_id; ++i)
    {
        const string name = (i == class_id) ? class_name : ("class_" + to_string(i));
        yaml << "  " << i << ": " << yaml_single_quote(name) << "\n";
    }
    return true;
}

string zero_padded_index(long long index, int width = 6)
{
    ostringstream ss;
    ss << setw(width) << setfill('0') << index;
    return ss.str();
}

string choose_yolo_split(long long sample_key, int val_percent)
{
    if (sample_key <= 0) return "train";
    if (val_percent <= 0) return "train";
    if (val_percent >= 100) return "val";
    uint64_t x = static_cast<uint64_t>(sample_key) + 0x9e3779b97f4a7c15ULL;
    x ^= x >> 30; x *= 0xbf58476d1ce4e5b9ULL;
    x ^= x >> 27; x *= 0x94d049bb133111ebULL;
    x ^= x >> 31;
    return static_cast<int>(x % 100ULL) < val_percent ? "val" : "train";
}

bool should_export_frame(long long frame_idx, int frame_step, long long start_frame,
    long long saved_samples, long long max_samples)
{
    if (frame_idx < start_frame) return false;
    if (max_samples > 0 && saved_samples >= max_samples) return false;
    return ((frame_idx - start_frame) % std::max(1, frame_step)) == 0;
}

struct YoloSaveResult
{
    bool saved = false;
    int labels_written = 0;
    string split;
    string image_path;
    string label_path;
};

YoloSaveResult save_yolo_sample(const Mat& clean_frame,
    const Mat& annotated_frame,
    const vector<BBox>& detections,
    const AppOptions& options,
    const YoloDatasetPaths& paths,
    const string& stem,
    long long sample_key)
{
    YoloSaveResult result;
    if (clean_frame.empty() || options.class_id < 0) return result;

    ostringstream label_text;
    label_text << fixed << setprecision(6);
    const Rect image_bounds(0, 0, clean_frame.cols, clean_frame.rows);
    const double image_area = std::max(1.0, static_cast<double>(clean_frame.cols) * clean_frame.rows);

    for (const auto& b : detections)
    {
        if (b.center_held) continue;
        Rect box = b.rect() & image_bounds;
        if (box.width < options.min_box_px || box.height < options.min_box_px) continue;

        const double area_frac = static_cast<double>(box.area()) / image_area;
        if (area_frac < options.min_box_area_frac || area_frac > options.max_box_area_frac) continue;

        const double x_center = (box.x + box.width * 0.5) / static_cast<double>(clean_frame.cols);
        const double y_center = (box.y + box.height * 0.5) / static_cast<double>(clean_frame.rows);
        const double width = box.width / static_cast<double>(clean_frame.cols);
        const double height = box.height / static_cast<double>(clean_frame.rows);

        label_text << options.class_id << " "
            << std::clamp(x_center, 0.0, 1.0) << " "
            << std::clamp(y_center, 0.0, 1.0) << " "
            << std::clamp(width, 0.0, 1.0) << " "
            << std::clamp(height, 0.0, 1.0) << "\n";
        ++result.labels_written;
    }

    if (result.labels_written == 0 && !options.include_empty) return result;

    result.split = choose_yolo_split(sample_key, options.val_percent);
    const bool is_val = result.split == "val";
    const string& image_dir = is_val ? paths.val_images_dir : paths.train_images_dir;
    const string& label_dir = is_val ? paths.val_labels_dir : paths.train_labels_dir;
    const string& preview_dir = is_val ? paths.val_previews_dir : paths.train_previews_dir;

    const string extension = "." + options.image_format;
    result.image_path = (fs::path(image_dir) / (stem + extension)).string();
    result.label_path = (fs::path(label_dir) / (stem + ".txt")).string();

    vector<int> image_params;
    if (options.image_format == "jpg") image_params = { IMWRITE_JPEG_QUALITY, options.jpeg_quality };

    if (!imwrite(result.image_path, clean_frame, image_params))
    {
        cerr << "[WARN] Failed to save YOLO training image: " << result.image_path << "\n";
        result.labels_written = 0;
        return result;
    }

    ofstream label_file(result.label_path, ios::out | ios::trunc);
    if (!label_file)
    {
        cerr << "[WARN] Failed to save YOLO label file: " << result.label_path << "\n";
        std::error_code ec;
        fs::remove(result.image_path, ec);
        result.labels_written = 0;
        return result;
    }
    label_file << label_text.str();
    label_file.close();

    if (options.save_preview && !annotated_frame.empty())
    {
        const string preview_path = (fs::path(preview_dir) / (stem + extension)).string();
        if (!imwrite(preview_path, annotated_frame, image_params))
            cerr << "[WARN] Could not save preview image: " << preview_path << "\n";
    }

    result.saved = true;
    return result;
}

bool write_dataset_summary(const YoloDatasetPaths& paths,
    long long train_images, long long val_images,
    long long train_boxes, long long val_boxes,
    const AppOptions& options)
{
    ofstream out((fs::path(paths.root_dir) / "dataset_summary.txt").string(), ios::out | ios::trunc);
    if (!out) return false;
    out << "YOLO Dataset Summary\n";
    out << "ClassID: " << options.class_id << "\n";
    out << "ClassName: " << options.class_name << "\n";
    out << "TrainImages: " << train_images << "\n";
    out << "ValidationImages: " << val_images << "\n";
    out << "TrainBoxes: " << train_boxes << "\n";
    out << "ValidationBoxes: " << val_boxes << "\n";
    out << "FrameStep: " << options.frame_step << "\n";
    out << "StartFrame: " << options.start_frame << "\n";
    out << "ValidationPercent: " << options.val_percent << "\n";
    out << "IncludeEmpty: " << (options.include_empty ? "yes" : "no") << "\n";
    out << "MinBoxPx: " << options.min_box_px << "\n";
    out << "MinBoxAreaFraction: " << options.min_box_area_frac << "\n";
    out << "MaxBoxAreaFraction: " << options.max_box_area_frac << "\n";
    out << "ImageFormat: " << options.image_format << "\n";
    return true;
}

string create_run_folder(const string& output_root)
{
    auto t = chrono::system_clock::now();
    auto tt = chrono::system_clock::to_time_t(t);
    tm local_tm{};
#ifdef _WIN32
    localtime_s(&local_tm, &tt);
#else
    localtime_r(&tt, &local_tm);
#endif

    stringstream ss;
    ss << output_root << "/run_" << put_time(&local_tm,  "%Y-%m-%d_%I-%M-%S_%p");
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

void drawCornerBox(Mat& frame, const Rect& box, const Scalar& color, int thickness = 1, int cornerLength = 25)
{
    if (frame.empty() || box.width <= 0 || box.height <= 0) return;

    // OpenCV Rect is [x, x + width) x [y, y + height), so the last
    // drawable pixel is width - 1 / height - 1.
    const int x1 = box.x;
    const int y1 = box.y;
    const int x2 = box.x + box.width - 1;
    const int y2 = box.y + box.height - 1;

    // Keep the corner arms proportional on small detections while
    // allowing a larger HUD-style corner on normal detections.
    const int lenX = std::max(1, std::min(cornerLength, box.width / 3));
    const int lenY = std::max(1, std::min(cornerLength, box.height / 3));

    // Every L faces inward toward the detected object.

    // Top-left: ┌
    line(frame, Point(x1, y1), Point(x1 + lenX, y1), color, thickness, LINE_AA);
    line(frame, Point(x1, y1), Point(x1, y1 + lenY), color, thickness, LINE_AA);

    // Top-right: ┐
    line(frame, Point(x2, y1), Point(x2 - lenX, y1), color, thickness, LINE_AA);
    line(frame, Point(x2, y1), Point(x2, y1 + lenY), color, thickness, LINE_AA);

    // Bottom-left: └
    line(frame, Point(x1, y2), Point(x1 + lenX, y2), color, thickness, LINE_AA);
    line(frame, Point(x1, y2), Point(x1, y2 - lenY), color, thickness, LINE_AA);

    // Bottom-right: ┘
    line(frame, Point(x2, y2), Point(x2 - lenX, y2), color, thickness, LINE_AA);
    line(frame, Point(x2, y2), Point(x2, y2 - lenY), color, thickness, LINE_AA);
}

// Temporal filter used only for video/camera processing.
// A detection is considered stationary when it keeps returning in almost the
// same location for several frames. Stationary detections are then ignored
// until they move again.
class StationaryTargetFilter
{
public:
    StationaryTargetFilter(
        int stable_frames = 6,
        double position_tolerance_px = 3.0,
        int size_tolerance_px = 4,
        double match_distance_px = 12.0,
        int max_missed_frames = 3)
        : stable_frames_(std::max(2, stable_frames)),
        position_tolerance_px_(std::max(0.0, position_tolerance_px)),
        size_tolerance_px_(std::max(0, size_tolerance_px)),
        match_distance_px_(std::max(1.0, match_distance_px)),
        max_missed_frames_(std::max(0, max_missed_frames))
    {
    }

    vector<BBox> filter(const vector<BBox>& detections, int protected_detection_index = -1)
    {
        vector<BBox> moving_detections;
        moving_detections.reserve(detections.size());

        // Prevent two detections in the same frame from being assigned to the
        // same temporal track.
        vector<bool> track_used(tracks_.size(), false);

        for (size_t detection_index = 0; detection_index < detections.size(); ++detection_index)
        {
            const auto& detection = detections[detection_index];
            const bool protected_from_stationary_filter =
                static_cast<int>(detection_index) == protected_detection_index;
            const Rect current_box = detection.rect();
            int best_track = -1;
            double best_distance = 1e30;

            for (size_t t = 0; t < tracks_.size(); ++t)
            {
                if (track_used[t]) continue;

                const Rect& previous_box = tracks_[t].last_box;
                const double distance = center_distance(previous_box, current_box);

                // Allow a slightly larger association radius for larger boxes,
                // but keep a fixed minimum for small Hough detections.
                const double dynamic_match_distance = std::max(
                    match_distance_px_,
                    0.35 * static_cast<double>(std::max(previous_box.width, previous_box.height)));

                const int max_size_change = std::max(
                    10,
                    static_cast<int>(0.50 * std::max(previous_box.width, previous_box.height)));

                const bool size_is_reasonable =
                    std::abs(current_box.width - previous_box.width) <= max_size_change &&
                    std::abs(current_box.height - previous_box.height) <= max_size_change;

                if (size_is_reasonable && distance <= dynamic_match_distance && distance < best_distance)
                {
                    best_distance = distance;
                    best_track = static_cast<int>(t);
                }
            }

            bool stationary = false;

            if (best_track >= 0)
            {
                Track& track = tracks_[best_track];
                track_used[best_track] = true;
                track.missed_frames = 0;
                track.last_box = current_box;

                track.history.push_back(current_box);
                if (track.history.size() > static_cast<size_t>(stable_frames_))
                {
                    track.history.erase(track.history.begin());
                }

                stationary = is_stationary(track.history);
                track.stationary = stationary;
            }
            else
            {
                Track new_track;
                new_track.last_box = current_box;
                new_track.history.push_back(current_box);
                tracks_.push_back(new_track);
                track_used.push_back(true);
            }

            // Normally, only moving / not-yet-proven-stationary targets continue.
            // With -C, the current detection nearest the frame center is protected
            // and remains visible/reported even if its temporal track is stationary.
            if (!stationary || protected_from_stationary_filter)
            {
                moving_detections.push_back(detection);
            }
        }

        // Age tracks that did not appear in this frame. A short grace period
        // tolerates detector flicker without forgetting a stationary false target.
        for (size_t t = 0; t < tracks_.size(); ++t)
        {
            if (!track_used[t])
            {
                tracks_[t].missed_frames++;
            }
        }

        for (size_t t = tracks_.size(); t-- > 0; )
        {
            if (tracks_[t].missed_frames > max_missed_frames_)
            {
                tracks_.erase(tracks_.begin() + static_cast<long>(t));
            }
        }

        return moving_detections;
    }

    void reset()
    {
        tracks_.clear();
    }

private:
    struct Track
    {
        Rect last_box;
        vector<Rect> history;
        int missed_frames = 0;
        bool stationary = false;
    };

    static Point2d center_of(const Rect& box)
    {
        return Point2d(
            box.x + box.width * 0.5,
            box.y + box.height * 0.5);
    }

    static double center_distance(const Rect& a, const Rect& b)
    {
        const Point2d ca = center_of(a);
        const Point2d cb = center_of(b);
        return std::hypot(ca.x - cb.x, ca.y - cb.y);
    }

    bool is_stationary(const vector<Rect>& history) const
    {
        if (history.size() < static_cast<size_t>(stable_frames_))
        {
            return false;
        }

        double min_cx = 1e30;
        double max_cx = -1e30;
        double min_cy = 1e30;
        double max_cy = -1e30;
        int min_w = 1000000000;
        int max_w = 0;
        int min_h = 1000000000;
        int max_h = 0;

        for (const Rect& box : history)
        {
            const Point2d c = center_of(box);
            min_cx = std::min(min_cx, c.x);
            max_cx = std::max(max_cx, c.x);
            min_cy = std::min(min_cy, c.y);
            max_cy = std::max(max_cy, c.y);
            min_w = std::min(min_w, box.width);
            max_w = std::max(max_w, box.width);
            min_h = std::min(min_h, box.height);
            max_h = std::max(max_h, box.height);
        }

        const Point2d first_center = center_of(history.front());
        const Point2d last_center = center_of(history.back());

        // Require the target to remain in the same small region, not merely
        // move slowly in one direction. This helps preserve genuinely moving
        // targets whose position accumulates across the frame.
        const double first_to_last = std::hypot(
            last_center.x - first_center.x,
            last_center.y - first_center.y);

        const bool position_stable =
            first_to_last <= position_tolerance_px_ &&
            (max_cx - min_cx) <= position_tolerance_px_ * 2.0 &&
            (max_cy - min_cy) <= position_tolerance_px_ * 2.0;

        const bool size_stable =
            (max_w - min_w) <= size_tolerance_px_ * 2 &&
            (max_h - min_h) <= size_tolerance_px_ * 2;

        return position_stable && size_stable;
    }

    vector<Track> tracks_;
    int stable_frames_;
    double position_tolerance_px_;
    int size_tolerance_px_;
    double match_distance_px_;
    int max_missed_frames_;
};


// Exact repeated-coordinate suppression used for -C video/camera mode.
// A detector result is considered static only when its CENTER X and Y are
// exactly identical for stable_frames consecutive processed frames.
// Target IDs are intentionally ignored because -C re-ranks targets every frame.
class RepeatedCoordinateFilter
{
public:
    explicit RepeatedCoordinateFilter(int stable_frames = 10)
        : stable_frames_(std::max(2, stable_frames))
    {
    }

    vector<BBox> filter(const vector<BBox>& detections)
    {
        ++frame_number_;
        for (auto& track : tracks_)
        {
            track.seen_this_frame = false;
        }

        vector<BBox> kept;
        kept.reserve(detections.size());

        for (const auto& detection : detections)
        {
            const int x = detection.cx();
            const int y = detection.cy();

            int matched_track = -1;
            for (size_t t = 0; t < tracks_.size(); ++t)
            {
                if (!tracks_[t].seen_this_frame && tracks_[t].x == x && tracks_[t].y == y)
                {
                    matched_track = static_cast<int>(t);
                    break;
                }
            }

            int consecutive = 1;
            if (matched_track >= 0)
            {
                Track& track = tracks_[matched_track];
                if (track.last_seen_frame == frame_number_ - 1)
                    track.consecutive_frames++;
                else
                    track.consecutive_frames = 1;

                track.last_seen_frame = frame_number_;
                track.seen_this_frame = true;
                consecutive = track.consecutive_frames;
            }
            else
            {
                Track track;
                track.x = x;
                track.y = y;
                track.consecutive_frames = 1;
                track.last_seen_frame = frame_number_;
                track.seen_this_frame = true;
                tracks_.push_back(track);
            }

            const bool suppress = consecutive >= stable_frames_;
            if (!suppress)
            {
                kept.push_back(detection);
            }
            else if (consecutive == stable_frames_)
            {
                cout << "[STATIC-COORD] Suppressing repeated target X=" << x
                    << " Y=" << y
                    << " after " << consecutive << " consecutive identical frames.\n";
            }
        }

        // Exact-repeat counting is consecutive. If a coordinate is absent for
        // even one frame, forget that streak so a later reappearance starts at 1.
        for (size_t t = tracks_.size(); t-- > 0; )
        {
            if (!tracks_[t].seen_this_frame)
            {
                tracks_.erase(tracks_.begin() + static_cast<long>(t));
            }
        }

        return kept;
    }

    void reset()
    {
        tracks_.clear();
        frame_number_ = 0;
    }

    int stable_frames() const { return stable_frames_; }

private:
    struct Track
    {
        int x = 0;
        int y = 0;
        int consecutive_frames = 0;
        long long last_seen_frame = 0;
        bool seen_this_frame = false;
    };

    vector<Track> tracks_;
    int stable_frames_ = 10;
    long long frame_number_ = 0;
};

// -C nearest-N center-target selection/persistence.
// -C alone keeps the single nearest detection (legacy/default behavior).
// -C N keeps up to N detections with the smallest Euclidean center distance.
// All non-selected detections are discarded before drawing/reporting/downstream X/Y use.
// If the detector returns zero detections for a very short gap, the previously
// selected center targets are held for up to max_hold_frames frames and marked
// center_held=true so they are distinguishable from fresh detector results.
class CenterTargetKeeper
{
public:
    explicit CenterTargetKeeper(int max_hold_frames = 3)
        : max_hold_frames_(std::max(0, max_hold_frames))
    {
    }

    int select_nearest_n(vector<BBox>& detections, Size frame_size, int requested_count)
    {
        // Clear any stale per-frame display state first.
        for (auto& b : detections)
        {
            b.center_protected = false;
            b.center_held = false;
            b.center_distance_px = -1.0;
        }

        if (detections.empty())
        {
            return 0;
        }

        const int keep_count = std::min(
            std::max(1, requested_count),
            static_cast<int>(detections.size()));

        const Point2d frame_center(frame_size.width * 0.5, frame_size.height * 0.5);

        struct RankedDetection
        {
            double distance;
            size_t index;
        };

        vector<RankedDetection> ranked;
        ranked.reserve(detections.size());

        for (size_t i = 0; i < detections.size(); ++i)
        {
            const double dx = static_cast<double>(detections[i].cx()) - frame_center.x;
            const double dy = static_cast<double>(detections[i].cy()) - frame_center.y;
            ranked.push_back({ std::hypot(dx, dy), i });
        }

        std::sort(ranked.begin(), ranked.end(),
            [](const RankedDetection& a, const RankedDetection& b)
            {
                if (a.distance != b.distance) return a.distance < b.distance;
                return a.index < b.index;
            });

        vector<BBox> selected;
        selected.reserve(keep_count);

        for (int rank = 0; rank < keep_count; ++rank)
        {
            BBox b = detections[ranked[rank].index];
            b.center_protected = true;
            b.center_held = false;
            b.center_distance_px = ranked[rank].distance;
            selected.push_back(b);
        }

        // Keep selected output ordered nearest -> farthest. Therefore Target 1
        // is always the closest-to-center target, Target 2 the next closest, etc.
        detections = selected;

        last_center_targets_ = selected;
        missed_frames_ = 0;
        return static_cast<int>(detections.size());
    }

    // Called only when the detector returned zero detections in this frame.
    // This bridges short detector flicker. It reuses only the last selected -C
    // targets; it never introduces unrelated/non-nearest detections.
    bool append_short_hold(vector<BBox>& detections, Size frame_size)
    {
        if (last_center_targets_.empty() || missed_frames_ >= max_hold_frames_)
        {
            if (!last_center_targets_.empty())
            {
                ++missed_frames_;
            }
            return false;
        }

        ++missed_frames_;
        const Point2d frame_center(frame_size.width * 0.5, frame_size.height * 0.5);

        for (const auto& previous : last_center_targets_)
        {
            BBox held = previous;
            held.center_protected = true;
            held.center_held = true;
            held.type = "center_hold";

            const double dx = static_cast<double>(held.cx()) - frame_center.x;
            const double dy = static_cast<double>(held.cy()) - frame_center.y;
            held.center_distance_px = std::hypot(dx, dy);

            detections.push_back(held);
        }

        // Preserve nearest -> farthest ordering for held targets too.
        std::sort(detections.begin(), detections.end(),
            [](const BBox& a, const BBox& b)
            {
                return a.center_distance_px < b.center_distance_px;
            });

        return !detections.empty();
    }

    void reset()
    {
        last_center_targets_.clear();
        missed_frames_ = 0;
    }

private:
    vector<BBox> last_center_targets_;
    int missed_frames_ = 0;
    int max_hold_frames_ = 3;
};

void process_frame_detections(Mat& frame, const BlobDetector& detector, DetMode det_mode,
    long long& elapsed_ms, int& det_count, vector<BBox>& out_detections,
    StationaryTargetFilter* stationary_filter = nullptr,
    CenterTargetKeeper* center_keeper = nullptr,
    bool protect_center_target = false,
    int center_target_count = 1,
    RepeatedCoordinateFilter* repeated_coordinate_filter = nullptr)
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

    const bool detector_returned_any = !out_detections.empty();

    // -C is an EXCLUSIVE nearest-N output mode for every input type.
    // For video/camera, first remove exact X/Y coordinates that have remained
    // unchanged for many consecutive frames. This happens BEFORE nearest-N
    // ranking, so suppressed static detections do not consume -C target slots.
    if (protect_center_target && center_keeper != nullptr)
    {
        if (repeated_coordinate_filter != nullptr)
        {
            // Call every frame, including empty detector frames, so an exact-X/Y
            // streak is truly consecutive and resets immediately after a miss.
            out_detections = repeated_coordinate_filter->filter(out_detections);
        }

        // The detector may internally produce many candidates, but only the
        // requested N non-static candidates nearest to frame center survive.
        if (!out_detections.empty())
        {
            center_keeper->select_nearest_n(
                out_detections, frame.size(), center_target_count);
        }
        else if (!detector_returned_any)
        {
            // Genuine detector miss: bridge only this short flicker gap.
            center_keeper->append_short_hold(
                out_detections, frame.size());
        }
        else
        {
            // The detector did return candidates, but every candidate was
            // rejected as a repeated static coordinate. Do NOT resurrect the
            // previous center targets through CENTER-HOLD.
            center_keeper->reset();
        }

        // -C still bypasses the broader stationary box/size filter; only the
        // exact repeated-coordinate rule above is applied in this mode.
    }
    else if (stationary_filter != nullptr)
    {
        // Normal video/camera mode (no -C): preserve stationary-object suppression.
        out_detections = stationary_filter->filter(out_detections, -1);
    }

    auto end_time = chrono::high_resolution_clock::now();
    elapsed_ms = chrono::duration_cast<chrono::milliseconds>(end_time - start_time).count();
    det_count = out_detections.size();

    Point frame_center(frame.cols / 2, frame.rows / 2);
    // drawMarker(frame, frame_center, Scalar(0, 0, 255), MARKER_CROSS, 15, 2);

    for (size_t i = 0; i < out_detections.size(); ++i)
    {
        const auto& b = out_detections[i];
        const bool center_protected = protect_center_target && b.center_protected;
        const bool center_held = center_protected && b.center_held;

        // White = ordinary detection. Green = current/held -C center target.
        // Scalar color = center_protected ? Scalar(0, 255, 0) : Scalar(255, 255, 255);

        Scalar color = Scalar(255, 255, 255);
        // Match runClassicalEODetection() geometry logic:
        // 1) clip the detector box to the valid frame, then
        // 2) derive the object coordinate from the clipped box center.
        Rect box = Rect(b.x, b.y, b.w, b.h) & Rect(0, 0, frame.cols, frame.rows);
        if (box.width <= 0 || box.height <= 0) continue;

        Point target_center(
            box.x + box.width / 2,
            box.y + box.height / 2);

        drawCornerBox(frame, box, color, center_protected ? 1 : 1, 25);
        line(frame, frame_center, target_center, color, center_protected ? 1 : 2, LINE_AA);

        if (center_protected)
        {
            // Extra marker makes each selected -C nearest target unmistakable in live/saved video.
            // drawMarker(frame, target_center, color, MARKER_DIAMOND, 14, 2, LINE_AA);
            ;
        }

        string label;
        if (center_protected)
        {
            ostringstream label_ss;
            label_ss << (center_held ? "CENTER-HOLD " : "CENTER-PROTECTED ")
                << "Target " << (i + 1)
                << " D:" << fixed << setprecision(1) << b.center_distance_px << "px"
                << " (X:" << target_center.x << " Y:" << target_center.y << ")";
            label = label_ss.str();
        }
        else
        {
            label = "Target " + to_string(i + 1) +
                " (X:" + to_string(target_center.x) +
                " Y:" + to_string(target_center.y) + ")";
        }

        int baseline = 0;
        Size text_size = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.45, 1, &baseline);
        int label_x = std::clamp(box.x, 0, std::max(0, frame.cols - text_size.width - 2));
        int label_y = (box.y - 5 > text_size.height) ? box.y - 5 : std::min(frame.rows - 2, box.y + box.height + text_size.height + 4);
        putText(frame, label, Point(label_x, label_y),
            FONT_HERSHEY_SIMPLEX, 0.45, center_protected ? color : Scalar(0, 255, 255), center_protected ? 1 : 2);
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
    ThreadedVideoProcessor(const string& video_path, BlobDetector& detector, DetMode mode, bool display,
        bool protect_center_target, int center_target_count, int static_coord_frames)
        : detector_(detector), det_mode_(mode), cap_(video_path), stop_threads_(false),
        repeated_coordinate_filter_(static_coord_frames),
        protect_center_target_(protect_center_target), center_target_count_(std::max(1, center_target_count)), display_(display) {
    }

    ThreadedVideoProcessor(int camera_index, BlobDetector& detector, DetMode mode, bool display,
        bool protect_center_target, int center_target_count, int static_coord_frames)
        : detector_(detector), det_mode_(mode), cap_(camera_index), stop_threads_(false),
        repeated_coordinate_filter_(static_coord_frames),
        protect_center_target_(protect_center_target), center_target_count_(std::max(1, center_target_count)), display_(display) {
    }

    void run(const string& run_dir, string mode_str, const AppOptions& options)
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

        YoloDatasetPaths yolo_paths;
        try
        {
            yolo_paths = create_yolo_dataset_folders(run_dir, options.save_preview);
        }
        catch (const fs::filesystem_error& e)
        {
            cerr << "[ERROR] Failed to create YOLO dataset folders: " << e.what() << "\n";
            return;
        }

        write_yolo_data_yaml(yolo_paths, options.class_id, options.class_name, options.val_percent);

        long long saved_yolo_images = 0;
        long long saved_yolo_boxes = 0;
        long long train_yolo_images = 0;
        long long val_yolo_images = 0;
        long long train_yolo_boxes = 0;
        long long val_yolo_boxes = 0;

        string out_video_path = run_dir + "/Output_" + mode_str + "_Video.mp4";
        VideoWriter writer;
        if (!open_video_writer(writer, out_video_path, fps, Size(fw, fh)))
        {
            cerr << "[WARN] Could not open video writer. Report will still be saved.\n";
        }

        string txt_path = run_dir + "/video_detection_report.txt";
        ofstream report(txt_path);

        // Testing coordinate log: one CSV row per currently output target.
        // In -C mode this means only the selected nearest-N targets are logged.
        string coord_path = run_dir + "/target_coordinates.csv";
        ofstream coord_log(coord_path);
        if (!coord_log)
        {
            cerr << "[WARN] Could not create coordinate log: " << coord_path << "\n";
        }
        else
        {
            coord_log << "Frame,TargetID,X,Y,CenterProtected,CenterHeld,CenterDistancePx,Type\n";
        }

        if (!report)
        {
            cerr << "[WARN] Could not create report file: " << txt_path << "\n";
        }
        else
        {
            report << "Mode: " << mode_str << "\n";
            report << "YOLO Class ID: " << options.class_id << "\n";
            report << "YOLO Class Name: " << options.class_name << "\n";
            report << "YOLO Dataset Root: " << yolo_paths.root_dir << "\n";
            report << "YOLO Frame Step: " << options.frame_step << "\n";
            report << "YOLO Validation Percent: " << options.val_percent << "\n";
            report << "CenterOnly(-C): " << (protect_center_target_ ? "ON" : "OFF") << "\n";
            if (protect_center_target_)
            {
                report << "CenterTargetCount: " << center_target_count_ << "\n";
                report << "StaticCoordinateConsecutiveFrames: " << repeated_coordinate_filter_.stable_frames() << "\n";
            }
            report << "Frame\tID\tX\tY\tW\tH\tScore\tType\tArea\tAspect\textent\tDoG_Mean\tDist(px)\tCenterProtected\tCenterHeld\tCenterDist(px)\n";
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
                    frame_queue_.push({ idx, temp.clone() });
                }
                queue_cond_.notify_one();
            }
            is_producer_done_ = true;
            queue_cond_.notify_all();
            });

        thread consumer([this, &writer, &report, &coord_log, &report_mutex, &yolo_paths, &saved_yolo_images, &saved_yolo_boxes, &train_yolo_images, &val_yolo_images, &train_yolo_boxes, &val_yolo_boxes, &options]() {
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

                Mat clean_frame = item.frame.clone();
                long long p_ms = 0;
                int d_count = 0;
                vector<BBox> detections;
                process_frame_detections(item.frame, detector_, det_mode_, p_ms, d_count, detections,
                    &stationary_filter_, &center_keeper_, protect_center_target_, center_target_count_,
                    protect_center_target_ ? &repeated_coordinate_filter_ : nullptr);

                if (should_export_frame(item.frame_idx, options.frame_step, options.start_frame,
                    saved_yolo_images, options.max_samples))
                {
                    const string sample_stem = options.sample_prefix + "_" + zero_padded_index(item.frame_idx);
                    YoloSaveResult yolo_result = save_yolo_sample(
                        clean_frame, item.frame, detections, options, yolo_paths, sample_stem, item.frame_idx);
                    if (yolo_result.saved)
                    {
                        ++saved_yolo_images;
                        saved_yolo_boxes += yolo_result.labels_written;
                        if (yolo_result.split == "val")
                        {
                            ++val_yolo_images;
                            val_yolo_boxes += yolo_result.labels_written;
                        }
                        else
                        {
                            ++train_yolo_images;
                            train_yolo_boxes += yolo_result.labels_written;
                        }
                        cout << "[YOLO] Saved " << yolo_result.split << "/" << sample_stem
                            << "." << options.image_format << " with " << yolo_result.labels_written << " label(s).\n";
                    }
                }

                {
                    lock_guard<mutex> lock(report_mutex);
                    for (size_t i = 0; i < detections.size(); ++i)
                    {
                        const auto& b = detections[i];
                        Rect report_box = Rect(b.x, b.y, b.w, b.h) & Rect(0, 0, item.frame.cols, item.frame.rows);
                        if (report_box.width <= 0 || report_box.height <= 0) continue;
                        Point report_center(
                            report_box.x + report_box.width / 2,
                            report_box.y + report_box.height / 2);
                        double dist = sqrt(pow(report_center.x - item.frame.cols / 2.0, 2) + pow(report_center.y - item.frame.rows / 2.0, 2));

                        // Print the target coordinate immediately for testing.
                        cout << "[COORD] Frame " << item.frame_idx
                            << " Target " << (i + 1)
                            << " X=" << report_center.x
                            << " Y=" << report_center.y;
                        if (b.center_protected)
                        {
                            cout << " CenterDist=" << fixed << setprecision(2) << b.center_distance_px << "px";
                        }
                        if (b.center_held)
                        {
                            cout << " [HELD]";
                        }
                        cout << "\n";

                        // Save a compact machine-readable coordinate log as CSV.
                        if (coord_log)
                        {
                            coord_log << item.frame_idx << "," << (i + 1) << ","
                                << report_center.x << "," << report_center.y << ","
                                << (b.center_protected ? 1 : 0) << ","
                                << (b.center_held ? 1 : 0) << ","
                                << fixed << setprecision(2)
                                << (b.center_protected ? b.center_distance_px : -1.0) << ","
                                << b.type << "\n";
                        }

                        if (report)
                        {
                            report << item.frame_idx << "\t" << (i + 1) << "\t" << report_center.x << "\t" << report_center.y
                                << "\t" << report_box.width << "\t" << report_box.height << "\t" << fixed << setprecision(2) << b.score << "\t" << b.type
                                << "\t" << b.area << "\t" << b.aspect << "\t" << b.extent << "\t" << b.dog_mean << "\t" << dist
                                << "\t" << (b.center_protected ? 1 : 0)
                                << "\t" << (b.center_held ? 1 : 0)
                                << "\t" << (b.center_protected ? b.center_distance_px : -1.0) << "\n";
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
        coord_log.close();
        write_dataset_summary(yolo_paths, train_yolo_images, val_yolo_images,
            train_yolo_boxes, val_yolo_boxes, options);
        cout << "[INFO] Target coordinates saved to: " << coord_path << "\n";
        cout << "[YOLO] Dataset root: " << yolo_paths.root_dir << "\n";
        cout << "[YOLO] data.yaml: " << yolo_paths.yaml_path << "\n";
        cout << "[YOLO] Train: " << train_yolo_images << " image(s), " << train_yolo_boxes << " box(es).\n";
        cout << "[YOLO] Val: " << val_yolo_images << " image(s), " << val_yolo_boxes << " box(es).\n";
        cout << "[YOLO] Total: " << saved_yolo_images << " image(s), " << saved_yolo_boxes << " box(es).\n";
        cout << "\n[DONE] Multi-threaded video processing complete. Saved in: " << run_dir << "\n";
    }

private:
    BlobDetector& detector_;
    DetMode det_mode_;
    VideoCapture cap_;
    atomic<bool> stop_threads_;
    atomic<bool> is_producer_done_{ false };
    queue<FrameData> frame_queue_;
    mutex queue_mutex_;
    condition_variable queue_cond_;
    const size_t max_queue_size_ = 10;

    // Persistent state across video frames. Defaults:
    // 6 stable frames, <=3 px center drift, small size jitter.
    StationaryTargetFilter stationary_filter_;
    CenterTargetKeeper center_keeper_{ 3 };
    RepeatedCoordinateFilter repeated_coordinate_filter_{ 10 };
    bool protect_center_target_;
    int center_target_count_ = 1;
    bool display_;
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

    if (!options.class_id_provided)
    {
        if (!can_prompt())
        {
            cerr << "[ERROR] No interactive terminal is available. Provide --class-id <N>.\n";
            return 1;
        }
        if (!prompt_for_class_id(options.class_id))
        {
            return 1;
        }
    }

    cout << "[INFO] YOLO annotation class ID: " << options.class_id << "\n";
    cout << "[INFO] YOLO class name: " << options.class_name << "\n";
    cout << "[INFO] Dataset export: frame-step=" << options.frame_step
        << ", start-frame=" << options.start_frame
        << ", val=" << options.val_percent << "%"
        << ", format=" << options.image_format
        << ", include-empty=" << (options.include_empty ? "yes" : "no") << "\n";

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
    cout << "[INFO] Center-target protection (-C): "
        << (options.protect_center_target ? "ENABLED" : "DISABLED") << "\n";
    if (options.protect_center_target)
    {
        cout << "[INFO] -C nearest-target count: " << options.center_target_count << "\n";
        cout << "[INFO] -C CENTER-ONLY mode: keeping ONLY the " << options.center_target_count
            << " nearest-to-center detection(s), ordered nearest -> farthest.\n";
        cout << "[INFO] All farther detections are discarded before drawing/reporting/downstream X/Y output.\n";
        cout << "[INFO] Selected center target(s) bypass stationary suppression.\n";
        cout << "[INFO] Exact repeated-coordinate suppression: " << options.static_coord_frames
            << " consecutive identical X/Y frames.\n";
        cout << "[INFO] Static-coordinate suppression runs before -C nearest-target selection.\n";
        cout << "[INFO] Short detector flicker hold: up to 3 frames (CENTER-HOLD).\n";
    }

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

    YoloDatasetPaths image_yolo_paths;
    if (!is_video)
    {
        try
        {
            image_yolo_paths = create_yolo_dataset_folders(run_dir, options.save_preview);
            write_yolo_data_yaml(image_yolo_paths, options.class_id, options.class_name, options.val_percent);
        }
        catch (const fs::filesystem_error& e)
        {
            cerr << "[ERROR] Failed to create YOLO dataset folders: " << e.what() << "\n";
            return -1;
        }
    }

    if (!is_video)
    {
        Mat clean_sample_frame = sample_frame.clone();
        long long processing_ms = 0;
        int det_count = 0;
        vector<BBox> detections;
        CenterTargetKeeper image_center_keeper(0);

        process_frame_detections(
            sample_frame, detector, det_mode, processing_ms, det_count, detections,
            nullptr,
            options.protect_center_target ? &image_center_keeper : nullptr,
            options.protect_center_target, options.center_target_count);

        const string sample_stem = options.sample_prefix + "_" + zero_padded_index(1);
        YoloSaveResult image_yolo_result = save_yolo_sample(
            clean_sample_frame, sample_frame, detections, options,
            image_yolo_paths, sample_stem, 0);

        if (image_yolo_result.saved)
        {
            cout << "[YOLO] Saved " << image_yolo_result.split << "/" << sample_stem << "."
                << options.image_format << " with " << image_yolo_result.labels_written << " label(s).\n";
        }
        else
        {
            cout << "[YOLO] No valid labels; "
                << (options.include_empty ? "sample save failed." : "no training sample was written.") << "\n";
        }

        string txt_path = run_dir + "/detection_report.txt";
        ofstream report(txt_path);

        string coord_path = run_dir + "/target_coordinates.csv";
        ofstream coord_log(coord_path);
        if (!coord_log)
        {
            cerr << "[WARN] Could not create coordinate log: " << coord_path << "\n";
        }
        else
        {
            coord_log << "Frame,TargetID,X,Y,CenterProtected,CenterHeld,CenterDistancePx,Type\n";
        }

        if (!report)
        {
            cerr << "[WARN] Could not create report file: " << txt_path << "\n";
        }
        else
        {
            report << "Mode: " << mode_str << "\n";
            report << "YOLO Class ID: " << options.class_id << "\n";
            report << "YOLO Class Name: " << options.class_name << "\n";
            report << "YOLO Dataset Root: " << image_yolo_paths.root_dir << "\n";
            report << "YOLO Split: " << (image_yolo_result.saved ? image_yolo_result.split : "none") << "\n";
            report << "Processing Time: " << processing_ms << " ms\n";
            report << "Number of Detections: " << det_count << "\n\n";
            report << "Detections Debug Log:\n";
            report << "ID\tX\tY\tW\tH\tScore\tType\tArea\tAspect\textent\tDoG_Mean\tDist(px)\n";
            report << "---------------------------------------------------------------------------------------\n";

        }

        for (size_t i = 0; i < detections.size(); ++i)
        {
            const auto& b = detections[i];
            Rect report_box = Rect(b.x, b.y, b.w, b.h) & Rect(0, 0, sample_frame.cols, sample_frame.rows);
            if (report_box.width <= 0 || report_box.height <= 0) continue;
            Point report_center(
                report_box.x + report_box.width / 2,
                report_box.y + report_box.height / 2);
            double dist = sqrt(pow(report_center.x - sample_frame.cols / 2.0, 2) + pow(report_center.y - sample_frame.rows / 2.0, 2));

            cout << "[COORD] Image Target " << (i + 1)
                << " X=" << report_center.x
                << " Y=" << report_center.y;
            if (b.center_protected)
            {
                cout << " CenterDist=" << fixed << setprecision(2) << b.center_distance_px << "px";
            }
            cout << "\n";

            if (coord_log)
            {
                coord_log << 1 << "," << (i + 1) << ","
                    << report_center.x << "," << report_center.y << ","
                    << (b.center_protected ? 1 : 0) << ","
                    << (b.center_held ? 1 : 0) << ","
                    << fixed << setprecision(2)
                    << (b.center_protected ? b.center_distance_px : -1.0) << ","
                    << b.type << "\n";
            }

            if (report)
            {
                report << (i + 1) << "\t" << report_center.x << "\t" << report_center.y << "\t" << report_box.width << "\t" << report_box.height
                    << "\t" << fixed << setprecision(2) << b.score << "\t" << b.type
                    << "\t" << b.area << "\t" << b.aspect << "\t" << b.extent << "\t" << b.dog_mean << "\t" << dist << "\n";
            }
        }
        report.close();
        coord_log.close();
        const bool single_is_val = image_yolo_result.saved && image_yolo_result.split == "val";
        write_dataset_summary(image_yolo_paths,
            image_yolo_result.saved && !single_is_val ? 1 : 0,
            image_yolo_result.saved && single_is_val ? 1 : 0,
            image_yolo_result.saved && !single_is_val ? image_yolo_result.labels_written : 0,
            image_yolo_result.saved && single_is_val ? image_yolo_result.labels_written : 0,
            options);
        cout << "[INFO] Target coordinates saved to: " << coord_path << "\n";
        cout << "[YOLO] Dataset root: " << image_yolo_paths.root_dir << "\n";
        cout << "[YOLO] data.yaml: " << image_yolo_paths.yaml_path << "\n";

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
            ThreadedVideoProcessor processor(stoi(input_path), detector, det_mode, options.display,
                options.protect_center_target, options.center_target_count, options.static_coord_frames);
            processor.run(run_dir, mode_str, options);
        }
        else
        {
            ThreadedVideoProcessor processor(input_path, detector, det_mode, options.display,
                options.protect_center_target, options.center_target_count, options.static_coord_frames);
            processor.run(run_dir, mode_str, options);
        }
    }

    return 0;
}


// ./Exe-classical_annotation_exporter /home/abhirup/Desktop/Development/Classical_Det/EOTS_TONBO/Det/IR_Vids/EO_11-40-14.514.avi --mode hybrid     --class-id 0     --class-name target     --frame-step 5     --val-percent 20     --min-box-px 5     --min-box-area-frac 0.00001     --max-box-area-frac 0.30     --image-format jpg     --jpeg-quality 95     --save-preview     -C 3

// Major additions:

// Asks for class ID before starting if --class-id isn't supplied.
// Optional --class-name target.
// Automatically creates a YOLO-ready dataset/images/train, images/val, labels/train, and labels/val structure.
// Automatically generates data.yaml.
// Deterministic automatic train/validation splitting with --val-percent.
// --frame-step N to control video dataset density while still running detection on every frame.
// --start-frame N and --max-samples N.
// --include-empty for negative/background training samples.
// Dynamic box filtering with --min-box-px, --min-box-area-frac, and --max-box-area-frac.
// JPG/PNG selection with --image-format.
// Adjustable JPEG quality.
// --save-preview stores annotated images separately, so training images stay clean.
// Custom filename prefixes with --sample-prefix.
// Automatically creates dataset_summary.txt.
// CENTER-HOLD boxes are still excluded from ground-truth labels because they are persisted boxes rather than fresh detections.
// YOLO labels remain normalized:
// class_id x_center y_center width height