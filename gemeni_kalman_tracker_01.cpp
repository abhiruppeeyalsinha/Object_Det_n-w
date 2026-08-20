// ════════════════════════════════════════════════════════════════════
// image_classical_detector.cpp (Updated with Cleared State, px/s Velocity, & Vector Drawing)
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
#include <map>

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

struct TrackedTarget
{
    int id;
    BBox box;
    float vx = 0.0f;
    float vy = 0.0f;
    float error_cov = 0.0f;
};

// ════════════════════════════════════════════════════════════════════
// OpenCV Kalman Filter Wrapper for Target Tracking
// ════════════════════════════════════════════════════════════════════
class TargetTracker
{
public:
    int track_id;
    cv::KalmanFilter kf;
    int age;          
    int hits;         
    BBox last_box;    

    TargetTracker(int id, const BBox& init_box) : track_id(id), age(0), hits(1), last_box(init_box)
    {
        kf.init(4, 2, 0);
        
        kf.transitionMatrix = (Mat_<float>(4, 4) << 1, 0, 1, 0,
                                                   0, 1, 0, 1,
                                                   0, 0, 1, 0,
                                                   0, 0, 0, 1);
        
        kf.measurementMatrix = (Mat_<float>(2, 4) << 1, 0, 0, 0,
                                                     0, 1, 0, 0);
        
        setIdentity(kf.processNoiseCov, Scalar::all(1e-2));
        setIdentity(kf.measurementNoiseCov, Scalar::all(1e-1));
        setIdentity(kf.errorCovPost, Scalar::all(1.0));

        kf.statePost.at<float>(0) = static_cast<float>(init_box.cx());
        kf.statePost.at<float>(1) = static_cast<float>(init_box.cy());
        kf.statePost.at<float>(2) = 0;
        kf.statePost.at<float>(3) = 0;
    }

    BBox predict()
    {
        Mat prediction = kf.predict();
        float pred_cx = prediction.at<float>(0);
        float pred_cy = prediction.at<float>(1);

        int w = last_box.w;
        int h = last_box.h;
        int x = static_cast<int>(pred_cx - w / 2.0f);
        int y = static_cast<int>(pred_cy - h / 2.0f);

        BBox predicted_box = last_box;
        predicted_box.x = x;
        predicted_box.y = y;
        return predicted_box;
    }

    void update(const BBox& measurement)
    {
        Mat_<float> meas(2, 1);
        meas(0) = static_cast<float>(measurement.cx());
        meas(1) = static_cast<float>(measurement.cy());

        kf.correct(meas);
        
        last_box = measurement;
        age = 0;
        hits++;
    }

    float get_vx() const { return kf.statePost.at<float>(2); }
    float get_vy() const { return kf.statePost.at<float>(3); }
    float get_error_norm() const { return static_cast<float>(norm(kf.errorCovPost, NORM_L2)); }
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

                Rect intersection = boxes[i].rect() & boxes[j].rect();
                float intersection_area = intersection.area();
                float union_area = boxes[i].rect().area() + boxes[j].rect().area() - intersection_area;
                float iou = (union_area > 0) ? (intersection_area / union_area) : 0.0f;

                if (iou > iou_threshold)
                {
                    suppressed[j] = true;
                }
            }
        }
        return result;
    }

    std::vector<BBox> select_all(const Mat& gray,
                                 const Mat& fused_mask,
                                 const Mat& dog_conf,
                                 const Mat& log_conf,
                                 float hybrid_w) const
    {
        std::vector<std::vector<Point>> contours;
        findContours(fused_mask, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

        const double roi_area = static_cast<double>(roi_w_) * roi_h_;
        std::vector<BBox> candidates;

        for (auto& cnt : contours)
        {
            double area = contourArea(cnt);
            if (area <= 5) continue;

            Rect r = boundingRect(cnt);
            if (r.width <= 0 || r.height <= 0) continue;
            if (r.x < 0 || r.y < 0 || r.x + r.width > gray.cols || r.y + r.height > gray.rows) continue;
            if (r.y > gray.rows * 0.88) continue; 

            double aspect = static_cast<double>(r.width) / r.height;
            double extent = area / (r.width * r.height);
            double rel = (area / roi_area) * 100.0;

            if (!(0.12 < aspect && aspect < 8.0 && extent > 0.06 && rel < 35.0)) continue;

            double dog_mean = dog_conf.empty() ? 0.0 : mean(dog_conf(r))[0];
            double log_mean = log_conf.empty() ? 0.0 : mean(log_conf(r))[0];

            double dog_n = std::min(dog_mean / 50.0, 1.0);
            double log_n = std::min(log_mean / 50.0, 1.0);
            double hyb_conf = (1.0 - hybrid_w) * dog_n + hybrid_w * log_n;
            double score = area * (0.5 + hyb_conf);

            candidates.push_back(BBox{r.x, r.y, r.width, r.height, (float)score, "detected", area, aspect, extent, dog_mean});
        }

        return apply_nms(candidates, 0.18f);
    }

    vector<BBox> detect_dog_all(const Mat &roi, Mat &out_conf_map, Mat &out_mask) const
    {
        if (roi.empty()) {
            out_conf_map = Mat();
            out_mask = Mat();
            return {};
        }

        Mat gray;
        if (roi.channels() == 3) cvtColor(roi, gray, COLOR_BGR2GRAY);
        else gray = roi.clone();

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

        Mat abs_a, abs_b;
        absdiff(dog_a32, Scalar(0), abs_a);
        absdiff(dog_b32, Scalar(0), abs_b);
        max(abs_a, abs_b, out_conf_map);

        Mat m1 = blob_mask(dog_a32, 18.f);
        Mat m2 = blob_mask(dog_b32, 18.f);
        bitwise_or(m1, m2, out_mask);

        return select_all(gray, out_mask, out_conf_map, Mat{}, 0.0f);
    }

    vector<BBox> detect_dog_all(const Mat &roi) const
    {
        Mat conf, mask;
        return detect_dog_all(roi, conf, mask);
    }

    vector<BBox> detect_hough_circles(const Mat &img, Mat &out_conf_map, Mat &out_mask) const
    {
        if (img.empty()) {
            out_conf_map = Mat();
            out_mask = Mat();
            return {};
        }

        Mat gray;
        if (img.channels() == 3) cvtColor(img, gray, COLOR_BGR2GRAY);
        else gray = img.clone();

        Mat smoothed;
        GaussianBlur(gray, smoothed, Size(5, 5), 1.2);

        vector<Vec3f> circles;
        HoughCircles(smoothed, circles, HOUGH_GRADIENT, 1.2,
                     gray.rows / 12, 70, 22, 8, 28);

        out_conf_map = Mat::zeros(gray.size(), CV_32F);
        out_mask = Mat::zeros(gray.size(), CV_8U);

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

            if ((y + h) < (img.rows * 0.88))
            {
                Rect r_box(x, y, w, h);
                out_conf_map(r_box).setTo(Scalar(25.0f));
                out_mask(r_box).setTo(Scalar(255));

                double area_approx = CV_PI * r * r;
                boxes.push_back(BBox{x, y, w, h, 0.95f, "hough_circle", area_approx, 1.0, 0.78, 25.0});
            }
        }
        
        return apply_nms(boxes, 0.12f);
    }

    vector<BBox> detect_hough_circles(const Mat &img) const
    {
        Mat conf, mask;
        return detect_hough_circles(img, conf, mask);
    }

    vector<BBox> detect_hybrid_all(const Mat &img) const
    {
        Mat gray = (img.channels() == 3) ? Mat() : img.clone();
        if (img.channels() == 3) cvtColor(img, gray, COLOR_BGR2GRAY);

        Mat dog_conf, dog_mask;
        detect_dog_all(img, dog_conf, dog_mask);

        Mat hough_conf, hough_mask;
        detect_hough_circles(img, hough_conf, hough_mask);

        Mat combined_mask;
        bitwise_or(dog_mask, hough_mask, combined_mask);

        return select_all(gray, combined_mask, dog_conf, hough_conf, cfg_.HYBRID_LOG_WEIGHT);
    }

private:
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
         << " [3] Hybrid (DoG + Hough Conf Fusion)\n";

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

string create_run_folder()
{
    auto t = chrono::system_clock::now();
    auto tt = chrono::system_clock::to_time_t(t);
    stringstream ss;
    ss << "Res/run_" << put_time(localtime(&tt), "%Y%m%d_%H%M%S");
    string dir = ss.str();
    fs::create_directories(dir);
    return dir;
}

static vector<TargetTracker> active_trackers;
static int next_track_id = 1;

void update_kalman_trackers(const vector<BBox>& detections, vector<TrackedTarget>& out_tracked_results, float fps)
{
    out_tracked_results.clear();

    vector<BBox> predicted_boxes;
    for (auto& tracker : active_trackers)
    {
        predicted_boxes.push_back(tracker.predict());
    }

    vector<bool> det_matched(detections.size(), false);
    vector<bool> tracker_matched(active_trackers.size(), false);

    for (size_t t = 0; t < active_trackers.size(); ++t)
    {
        double min_dist = 60.0; 
        int best_det_idx = -1;

        for (size_t d = 0; d < detections.size(); ++d)
        {
            if (det_matched[d]) continue;

            double dist = sqrt(pow(active_trackers[t].last_box.cx() - detections[d].cx(), 2) +
                               pow(active_trackers[t].last_box.cy() - detections[d].cy(), 2));
            if (dist < min_dist)
            {
                min_dist = dist;
                best_det_idx = static_cast<int>(d);
            }
        }

        if (best_det_idx != -1)
        {
            active_trackers[t].update(detections[best_det_idx]);
            det_matched[best_det_idx] = true;
            tracker_matched[t] = true;
            out_tracked_results.push_back({
                active_trackers[t].track_id, 
                active_trackers[t].last_box, 
                active_trackers[t].get_vx() * fps, 
                active_trackers[t].get_vy() * fps, 
                active_trackers[t].get_error_norm()
            });
        }
    }

    for (size_t t = 0; t < active_trackers.size(); ++t)
    {
        if (!tracker_matched[t])
        {
            active_trackers[t].age++;
            if (active_trackers[t].age < 10)
            {
                out_tracked_results.push_back({
                    active_trackers[t].track_id, 
                    active_trackers[t].last_box, 
                    active_trackers[t].get_vx() * fps, 
                    active_trackers[t].get_vy() * fps, 
                    active_trackers[t].get_error_norm()
                });
            }
        }
    }

    active_trackers.erase(
        remove_if(active_trackers.begin(), active_trackers.end(), [](const TargetTracker& tr) { return tr.age >= 10; }),
        active_trackers.end()
    );

    for (size_t d = 0; d < detections.size(); ++d)
    {
        if (!det_matched[d])
        {
            TargetTracker new_tr(next_track_id++, detections[d]);
            active_trackers.push_back(new_tr);
            out_tracked_results.push_back({
                new_tr.track_id, 
                detections[d], 
                new_tr.get_vx() * fps, 
                new_tr.get_vy() * fps, 
                new_tr.get_error_norm()
            });
        }
    }
}

void process_frame_detections(Mat &frame, const BlobDetector &detector, DetMode det_mode, 
                              long long &elapsed_ms, int &det_count, vector<TrackedTarget> &out_tracked_results, bool is_video_stream, float fps)
{
    auto start_time = chrono::high_resolution_clock::now();

    vector<BBox> raw_detections;
    switch (det_mode)
    {
    case DetMode::DoG:
        raw_detections = detector.detect_dog_all(frame);
        break;
    case DetMode::HoughCircles:
        raw_detections = detector.detect_hough_circles(frame);
        break;
    case DetMode::Hybrid:
        raw_detections = detector.detect_hybrid_all(frame);
        break;
    }

    if (is_video_stream)
    {
        update_kalman_trackers(raw_detections, out_tracked_results, fps);
    }
    else
    {
        out_tracked_results.clear();
        for (size_t i = 0; i < raw_detections.size(); ++i)
        {
            out_tracked_results.push_back({static_cast<int>(i + 1), raw_detections[i], 0.0f, 0.0f, 0.0f});
        }
    }

    auto end_time = chrono::high_resolution_clock::now();
    elapsed_ms = chrono::duration_cast<chrono::milliseconds>(end_time - start_time).count();
    det_count = out_tracked_results.size();

    Point frame_center(frame.cols / 2, frame.rows / 2);
    drawMarker(frame, frame_center, Scalar(0, 0, 255), MARKER_CROSS, 15, 2);

    for (const auto &item : out_tracked_results)
    {
        int id = item.id;
        const auto &b = item.box;
        Scalar color = Scalar(255, 255, 255);
        
        rectangle(frame, Rect(b.x, b.y, b.w, b.h), color, 1);
        
        Point target_center(b.cx(), b.cy());
        line(frame, frame_center, target_center, color, 1, LINE_AA);

        // Draw velocity vector arrow (scaled down factor for visual clarity)
        Point velocity_end(target_center.x + static_cast<int>(item.vx / fps * 5), 
                           target_center.y + static_cast<int>(item.vy / fps * 5));
        arrowedLine(frame, target_center, velocity_end, Scalar(0, 255, 0), 1, LINE_AA, 0, 0.3);

        string label = "ID " + to_string(id) + " (Vx:" + to_string((int)item.vx) + "px/s)";
        putText(frame, label, Point(b.x, max(0, b.y - 5)),
                FONT_HERSHEY_SIMPLEX, 0.45, Scalar(0, 255, 255), 1);
    }
}

struct FrameData
{
    int frame_idx;
    Mat frame;
};

class ThreadedVideoProcessor
{
public:
    ThreadedVideoProcessor(const string& video_path, BlobDetector& detector, DetMode mode)
        : detector_(detector), det_mode_(mode), cap_(video_path), stop_threads_(false) {}

    void run(const string& run_dir, string mode_str)
    {
        // CRITICAL FIX: Reset trackers to avoid ID leaks between runs
        active_trackers.clear();
        next_track_id = 1;

        if (!cap_.isOpened())
        {
            cerr << "[ERROR] Failed to open video stream.\n";
            return;
        }

        double fps = cap_.get(CAP_PROP_FPS);
        if (fps <= 0) fps = 30.0;
        int fw = static_cast<int>(cap_.get(CAP_PROP_FRAME_WIDTH));
        int fh = static_cast<int>(cap_.get(CAP_PROP_FRAME_HEIGHT));

        string out_video_path = run_dir + "/Output_" + mode_str + "_Video.avi";
        VideoWriter writer(out_video_path, VideoWriter::fourcc('M','J','P','G'), fps, Size(fw, fh));

        string txt_path = run_dir + "/video_detection_report.txt";
        ofstream report(txt_path);
        report << "Mode: " << mode_str << " (Kalman Tracked with px/s Velocity)\n";
        report << "Frame\tID\tX\tY\tW\tH\tScore\tType\tArea\tAspect\textent\tDoG_Mean\tDist(px)\tVx(px/s)\tVy(px/s)\tError_Cov\n";
        report << "----------------------------------------------------------------------------------------------------------------------\n";
        mutex report_mutex;

        thread producer([this]() {
            int idx = 0;
            Mat temp;
            while (cap_.read(temp))
            {
                if (temp.empty()) break;
                idx++;
                
                if (idx % 2 != 0) continue;

                {
                    unique_lock<mutex> lock(queue_mutex_);
                    queue_cond_.wait(lock, [this] { return frame_queue_.size() < max_queue_size_; });
                    frame_queue_.push({idx, temp.clone()});
                }
                queue_cond_.notify_one();
            }
            is_producer_done_ = true;
            queue_cond_.notify_all();
        });

        thread consumer([this, &writer, &report, &report_mutex, fps]() {
            while (true)
            {
                FrameData item;
                {
                    unique_lock<mutex> lock(queue_mutex_);
                    queue_cond_.wait(lock, [this] { return !frame_queue_.empty() || is_producer_done_; });
                    
                    if (frame_queue_.empty() && is_producer_done_) break;

                    item = frame_queue_.front();
                    frame_queue_.pop();
                }
                queue_cond_.notify_one();

                long long p_ms = 0;
                int d_count = 0;
                vector<TrackedTarget> tracked_results;
                process_frame_detections(item.frame, detector_, det_mode_, p_ms, d_count, tracked_results, true, static_cast<float>(fps));

                {
                    lock_guard<mutex> lock(report_mutex);
                    for (const auto &res : tracked_results)
                    {
                        const auto &b = res.box;
                        double dist = sqrt(pow(b.cx() - item.frame.cols / 2.0, 2) + pow(b.cy() - item.frame.rows / 2.0, 2));
                        report << item.frame_idx << "\t" << res.id << "\t" << b.x << "\t" << b.y 
                               << "\t" << b.w << "\t" << b.h << "\t" << fixed << setprecision(2) << b.score << "\t" << b.type 
                               << "\t" << b.area << "\t" << b.aspect << "\t" << b.extent << "\t" << b.dog_mean << "\t" << dist 
                               << "\t" << res.vx << "\t" << res.vy << "\t" << res.error_cov << "\n";
                    }
                }

                if (writer.isOpened()) writer.write(item.frame);

                imshow("Multi-Threaded Video Stream with Kalman Tracking", item.frame);
                if ((waitKey(1) & 0xFF) == 27) break;
            }
        });

        producer.join();
        consumer.join();
        cap_.release();
        if (writer.isOpened()) writer.release();
        report.close();
        cout << "\n[DONE] Kalman-tracked video processing complete. Report saved in: " << run_dir << "\n";
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
};

int main(int argc, char **argv)
{
    if (argc < 2)
    {
        cerr << "[ERROR] Usage: ./Exe-gemeni <path_to_image_or_video>\n";
        return -1;
    }

    string input_path = argv[1];
    string ext = fs::path(input_path).extension().string();
    transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    bool is_video = (ext == ".mp4" || ext == ".avi" || ext == ".mov" || ext == ".mkv" || ext == ".m4v");

    Mat sample_frame;
    int width = 0;
    int height = 0;
    double fps = 30.0;

    if (is_video)
    {
        VideoCapture test_cap(input_path);
        if (!test_cap.isOpened())
        {
            cerr << "[ERROR] Failed to open video file: " << input_path << "\n";
            return -1;
        }
        width = static_cast<int>(test_cap.get(CAP_PROP_FRAME_WIDTH));
        height = static_cast<int>(test_cap.get(CAP_PROP_FRAME_HEIGHT));
        double f_val = test_cap.get(CAP_PROP_FPS);
        if (f_val > 0) fps = f_val;
        test_cap.release();
    }
    else
    {
        sample_frame = imread(input_path);
        if (sample_frame.empty())
        {
            cerr << "[ERROR] Failed to load image input from: " << input_path << "\n";
            return -1;
        }
        width = sample_frame.cols;
        height = sample_frame.rows;
    }

    cout << "[INFO] Loaded input dimensions: " << width << "x" << height << "\n";

    TrackerConfig cfg;
    DetMode det_mode = get_detection_mode();
    bool debug_mode = get_debug_mode_choice();

    BlobDetector detector(cfg, width, height, PriorMode::Center, debug_mode, is_video);

    string run_dir = create_run_folder();
    string mode_str = (det_mode == DetMode::DoG) ? "Dog" : (det_mode == DetMode::HoughCircles) ? "Hough" : "Hybrid";

    if (!is_video)
    {
        long long processing_ms = 0;
        int det_count = 0;
        vector<TrackedTarget> tracked_results;

        process_frame_detections(sample_frame, detector, det_mode, processing_ms, det_count, tracked_results, false, static_cast<float>(fps));

        string txt_path = run_dir + "/detection_report.txt";
        ofstream report(txt_path);
        report << "Mode: " << mode_str << "\n";
        report << "Processing Time: " << processing_ms << " ms\n";
        report << "Number of Detections: " << det_count << "\n\n";
        report << "Detections Debug Log:\n";
        report << "ID\tX\tY\tW\tH\tScore\tType\tArea\tAspect\textent\tDoG_Mean\tDist(px)\tVx(px/s)\tVy(px/s)\tError_Cov\n";
        report << "----------------------------------------------------------------------------------------------------------------------\n";

        for (const auto &res : tracked_results)
        {
            const auto &b = res.box;
            double dist = sqrt(pow(b.cx() - sample_frame.cols / 2.0, 2) + pow(b.cy() - sample_frame.rows / 2.0, 2));
            report << res.id << "\t" << b.x << "\t" << b.y << "\t" << b.w << "\t" << b.h 
                   << "\t" << fixed << setprecision(2) << b.score << "\t" << b.type 
                   << "\t" << b.area << "\t" << b.aspect << "\t" << b.extent << "\t" << b.dog_mean << "\t" << dist 
                   << "\t" << res.vx << "\t" << res.vy << "\t" << res.error_cov << "\n";
        }
        report.close();

        string out_path = run_dir + "/Output_" + mode_str + "_Result.jpg";
        imwrite(out_path, sample_frame);

        cout << "\n[DONE] Processing complete. Output saved in subfolder: " << run_dir << "\n";
        namedWindow("Detector Result", WINDOW_NORMAL);
        imshow("Detector Result", sample_frame);
        waitKey(0);
    }
    else
    {
        namedWindow("Multi-Threaded Video Stream with Kalman Tracking", WINDOW_NORMAL);
        ThreadedVideoProcessor processor(input_path, detector, det_mode);
        processor.run(run_dir, mode_str);
    }

    return 0;
}