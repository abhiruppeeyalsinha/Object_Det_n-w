//auto_drone_tracker.hpp

#pragma once
#include "drone_tracker.hpp"
#include "drone_kalman.hpp"
#include "detectors.hpp"
#include "frame_reader.hpp"
#include "trt_session.hpp"
#include "trackers.hpp"
#include "target_manager.hpp"
#include "utils.hpp"

#include <memory>
#include <optional>
#include <string>
#include <chrono>
#include <atomic>

using namespace std;
using namespace cv;

enum class OperatingMode
{
    AUTO,
    MANUAL
};

class AutoDroneTracker
{
public:
    AutoDroneTracker(
        const string& source,
        const string& engine_path,
        TRTOutputLayout trt_output_layout,
        float conf_thresh = 0.40f);

    void run();

private:
    // ─────────────────────────────────────────────────────────────
    // Performance parameters
    // ─────────────────────────────────────────────────────────────

    OperatingMode op_mode_;

    bool manual_target_selected_ = false;
    bool drawing_box_ = false;

    cv::Point manual_start_;
    cv::Point manual_end_;

    OperatingMode get_operating_mode();

    bool manual_select_target(cv::Mat& frame);
    static void mouse_callback(int event, int x, int y, int flags, void* userdata);

    uint64_t total_frames_ = 0;
    uint64_t dropped_frames_ = 0;
    uint64_t decode_failures_ = 0;

    double fps_ = 0.0;
    double latency_ms_ = 0.0;

    chrono::steady_clock::time_point fps_timer_;
    int fps_counter_ = 0;

    // ─────────────────────────────────────────────────────────────
    // Setup wizard
    // ─────────────────────────────────────────────────────────────

    pair<int, int> get_user_dimensions();

    DetMode get_detection_mode();

    TrackAlgo get_tracking_algorithm();

    PriorMode get_priority_mode();

    // ─────────────────────────────────────────────────────────────
    // Detection
    // ─────────────────────────────────────────────────────────────

    optional<BBox> detect(const Mat& roi);
    vector<BBox> detect_all(const Mat& roi);

    optional<BBox> AI_Det(const Mat& roi);
    vector<BBox> AI_Dets(const Mat& roi);

    // ─────────────────────────────────────────────────────────────
    // Acquire & re-acquire
    // ─────────────────────────────────────────────────────────────

    bool acquire(const Mat& roi);

    bool reacquire(const Mat& roi);

    // ─────────────────────────────────────────────────────────────
    // Kalman feed
    // ─────────────────────────────────────────────────────────────

    void kalman_feed(const BBox& b);

    // ─────────────────────────────────────────────────────────────
    // Clamp bbox
    // ─────────────────────────────────────────────────────────────

    BBox clamp(int x, int y, int w, int h) const;

    // ─────────────────────────────────────────────────────────────
    // Performance Draw
    // ─────────────────────────────────────────────────────────────

    void draw_performance_stats(Mat& frame);

    // ─────────────────────────────────────────────────────────────
    // Config & objects
    // ─────────────────────────────────────────────────────────────

    Config cfg_;

    string source_;

    int W_ = 0;
    int H_ = 0;

    DetMode det_mode_;
    TrackAlgo track_algo_;
    PriorMode prior_mode_;

    int roi_x_ = 0;
    int roi_y_ = 0;
    int roi_w_ = 0;
    int roi_h_ = 0;

    string run_dir_;

    // ─────────────────────────────────────────────────────────────
    // Initial frame cache
    // ─────────────────────────────────────────────────────────────

    optional<Mat> first_frame_;

    // ─────────────────────────────────────────────────────────────
    // Tracking state
    // ─────────────────────────────────────────────────────────────

    optional<BBox> bbox_;

    optional<int> AI_clss;

    optional<float> AI_conf;

    int frames_lost_ = 0;

    int frame_idx_ = 0;

    DroneKalman kalman_;

    // ─────────────────────────────────────────────────────────────
    // Components
    // ─────────────────────────────────────────────────────────────

    unique_ptr<BlobDetector> detector_;

    unique_ptr<TrackCascade> cascade_;

    unique_ptr<TargetManager> target_manager_;

    unique_ptr<TRTSession> trt_;

    unique_ptr<FrameReader> reader_;

    VideoWriter out_vid_;
};
