// ════════════════════════════════════════════════════════════════════
// auto_drone_tracker.cpp
// FFmpeg FrameReader + DoG/LoG/Hybrid/AI Pipeline
// ════════════════════════════════════════════════════════════════════

#include "auto_drone_tracker.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <tuple>

using namespace std;
using namespace cv;

// ════════════════════════════════════════════════════════════════════
// Constructor
// ════════════════════════════════════════════════════════════════════

AutoDroneTracker::AutoDroneTracker(const string &source,
                                   const string &engine_path,
                                   TRTOutputLayout trt_output_layout,
                                   float conf_thresh)

    : source_(source)
{
    cfg_.CONF_THRESH = conf_thresh;
    cfg_.DET_EVERY_N = max(1, cfg_.DET_EVERY_N);
    cfg_.TRT_OUTPUT_LAYOUT = trt_output_layout;

    if (cfg_.TRT_OUTPUT_LAYOUT == TRTOutputLayout::IR_25200x9)
    {
        cfg_.TRT_MAX_DET = 25200;
        cfg_.TRT_DET_COLS = 9;
    }
    else
    {
        cfg_.TRT_MAX_DET = 300;
        cfg_.TRT_DET_COLS = 6;
    }

    // ════════════════════════════════════════════════════════════════
    // STEP-1 : ROI SIZE
    // ════════════════════════════════════════════════════════════════

    auto [rw, rh] = get_user_dimensions();

    roi_w_ = rw;
    roi_h_ = rh;

    op_mode_ = get_operating_mode();

    if (op_mode_ == OperatingMode::AUTO)
        det_mode_ = get_detection_mode();
    else
        det_mode_ = DetMode::DoG;

    // ════════════════════════════════════════════════════════════════
    // STEP-3 : TRACKER
    // ════════════════════════════════════════════════════════════════

    track_algo_ = get_tracking_algorithm();

    prior_mode_ = get_priority_mode();

    // ════════════════════════════════════════════════════════════════
    // TRT INIT
    // ════════════════════════════════════════════════════════════════

    if (det_mode_ == DetMode::AI)
    {
        cout << "\n[INFO] Loading TensorRT Engine...\n";
        cout << "[INFO] TensorRT engine: " << engine_path << "\n";
        cout << "[INFO] TensorRT output: (1,"
             << cfg_.TRT_MAX_DET
             << ","
             << cfg_.TRT_DET_COLS
             << ")\n";

        trt_ =
            make_unique<TRTSession>(
                engine_path,
                cfg_);
    }

    // ════════════════════════════════════════════════════════════════
    // OUTPUT DIRECTORY
    // ════════════════════════════════════════════════════════════════

    run_dir_ = create_run_folder();

    // ════════════════════════════════════════════════════════════════
    // FFmpeg Frame Reader
    // ════════════════════════════════════════════════════════════════

    cout << "\n[INFO] Opening stream using FFmpeg...\n";

    reader_ = make_unique<FrameReader>(source_);

    if (!reader_->open())
    {
        throw runtime_error(
            "[ERROR] Failed to open stream");
    }

    first_frame_ = reader_->read();

    if (!first_frame_.has_value() ||
        first_frame_->empty())
    {
        throw runtime_error(
            "[ERROR] Failed to read first frame");
    }

    W_ = first_frame_->cols;
    H_ = first_frame_->rows;

    cout << "[INFO] Source frame size: "
         << W_
         << "x"
         << H_
         << "\n";

    // ════════════════════════════════════════════════════════════════
    // SAFE ROI
    // ════════════════════════════════════════════════════════════════

    roi_w_ = min(roi_w_, W_);
    roi_h_ = min(roi_h_, H_);

    roi_x_ = max(0, (W_ - roi_w_) / 2);
    roi_y_ = max(0, (H_ - roi_h_) / 2);

    // ════════════════════════════════════════════════════════════════
    // DETECTOR
    // ════════════════════════════════════════════════════════════════

    detector_ =
        make_unique<BlobDetector>(
            cfg_,
            roi_w_,
            roi_h_,
            prior_mode_);

    // ════════════════════════════════════════════════════════════════
    // TRACK CASCADE
    // ════════════════════════════════════════════════════════════════

    cascade_ =
        make_unique<TrackCascade>(
            cfg_,
            track_algo_,
            roi_w_,
            roi_h_,
            roi_x_,
            roi_y_);

    target_manager_ =
        make_unique<TargetManager>(
            cfg_,
            track_algo_,
            roi_w_,
            roi_h_,
            roi_x_,
            roi_y_);

    // ════════════════════════════════════════════════════════════════
    // OUTPUT VIDEO
    // ════════════════════════════════════════════════════════════════

    string det_name;

    switch (det_mode_)
    {
    case DetMode::DoG:
        det_name = "DoG";
        break;

    case DetMode::LoG:
        det_name = "LoG";
        break;

    case DetMode::Hybrid:
        det_name = "Hybrid";
        break;

    case DetMode::AI:
        det_name = "AI";
        break;
    }

    string track_name;

    switch (track_algo_)
    {
    case TrackAlgo::Centroid:
        track_name = "Centroid";
        break;

    case TrackAlgo::Edge:
        track_name = "Edge";
        break;

    case TrackAlgo::Phase:
        track_name = "Phase";
        break;

    case TrackAlgo::Template:
        track_name = "Template";
        break;

    case TrackAlgo::OpticalFlow:
        track_name = "OpticalFlow";
        break;
    }

    string out_path =
        run_dir_ +
        "/Output_" +
        det_name +
        "_" +
        track_name +
        ".mp4";

    int fourcc =
        VideoWriter::fourcc('a', 'v', 'c', '1');

    out_vid_.open(
        out_path,
        fourcc,
        cfg_.SOURCE_FPS,
        Size(W_, H_));

    if (!out_vid_.isOpened())
    {
        cerr
            << "[WARN] Failed to create output video\n";
    }

    cout << "\n[INFO] System Ready\n";
}

// ════════════════════════════════════════════════════════════════════
// OPERATING MODE
// ════════════════════════════════════════════════════════════════════

OperatingMode AutoDroneTracker::get_operating_mode()
{
    cout
        << "\n--- Step 2: Operating Mode ---\n"
        << " [1] Auto detection\n"
        << " [2] Manual target selection\n";

    string sel;

    cout << " Select (default 1): ";
    getline(cin, sel);

    if (sel == "2")
        return OperatingMode::MANUAL;

    return OperatingMode::AUTO;
}

void AutoDroneTracker::mouse_callback(
    int event,
    int x,
    int y,
    int,
    void *userdata)
{
    auto *self =
        static_cast<AutoDroneTracker *>(userdata);

    if (!self)
        return;

    if (event == EVENT_LBUTTONDOWN)
    {
        self->drawing_box_ = true;
        self->manual_target_selected_ = false;
        self->manual_start_ = {x, y};
        self->manual_end_ = {x, y};
    }
    else if (event == EVENT_MOUSEMOVE &&
             self->drawing_box_)
    {
        self->manual_end_ = {x, y};
    }
    else if (event == EVENT_LBUTTONUP &&
             self->drawing_box_)
    {
        self->drawing_box_ = false;
        self->manual_end_ = {x, y};
        self->manual_target_selected_ = true;
    }
}

bool AutoDroneTracker::manual_select_target(Mat &frame)
{
    manual_target_selected_ = false;
    drawing_box_ = false;
    manual_start_ = {};
    manual_end_ = {};

    const string win = "Manual Target Selection";

    namedWindow(win, WINDOW_NORMAL);
    setMouseCallback(win, AutoDroneTracker::mouse_callback, this);

    cout
        << "\n[MANUAL] Drag a box around the target, then release mouse.\n"
        << "[MANUAL] Press ESC to cancel.\n";

    while (!manual_target_selected_)
    {
        Mat preview = frame.clone();

        if (drawing_box_ ||
            manual_start_ != manual_end_)
        {
            Rect r(manual_start_, manual_end_);
            r &= Rect(0, 0, preview.cols, preview.rows);

            if (r.area() > 0)
                rectangle(preview, r, Scalar(0, 255, 255), 2);
        }

        imshow(win, preview);

        int key = waitKey(20) & 0xFF;

        if (key == 27)
        {
            setMouseCallback(win, nullptr, nullptr);
            destroyWindow(win);
            return false;
        }
    }

    setMouseCallback(win, nullptr, nullptr);
    destroyWindow(win);

    Rect selected(manual_start_, manual_end_);
    selected &= Rect(0, 0, frame.cols, frame.rows);

    return selected.width > 3 &&
           selected.height > 3;
}

// ════════════════════════════════════════════════════════════════════
// ROI SELECTION
// ════════════════════════════════════════════════════════════════════

pair<int, int>
AutoDroneTracker::get_user_dimensions()
{
    cout
        << "\n--- Step 1: ROI Selection ---\n";

    ifstream file(
        "/home/abhirupsinha/Desktop/P207/roi_config.json");

    if (!file.is_open())
    {
        cerr
            << "[WARN] roi_config.json missing\n";

        return {300, 300};
    }

    stringstream buffer;
    buffer << file.rdbuf();

    const string content = buffer.str();

    regex entry_re(
        R"json("([^"]+)"\s*:\s*\{[^{}]*"width"\s*:\s*([0-9]+)\s*,[^{}]*"height"\s*:\s*([0-9]+)[^{}]*\})json");

    vector<tuple<string, int, int>> options;

    for (sregex_iterator it(content.begin(), content.end(), entry_re), end;
         it != end;
         ++it)
    {
        const smatch &m = *it;

        options.emplace_back(
            m[1].str(),
            stoi(m[2].str()),
            stoi(m[3].str()));
    }

    if (options.empty())
    {
        cerr
            << "[WARN] roi_config.json has no valid ROI entries\n";

        return {300, 300};
    }

    for (const auto &[key, width, height] : options)
    {
        cout
            << " ["
            << key
            << "] "
            << width
            << " x "
            << height
            << "\n";
    }

    string id;

    cout << "Select ROI ID : ";

    getline(cin, id);

    if (id.empty())
        id = "1";

    for (const auto &[key, width, height] : options)
    {
        if (key == id)
            return {width, height};
    }

    if (options.size() == 1)
    {
        const auto &[key, width, height] = options.front();

        cerr
            << "[WARN] Invalid selection, using ROI "
            << key
            << "\n";

        return {width, height};
    }

    {
        cerr
            << "[WARN] Invalid selection\n";

        return {300, 300};
    }
}

// ════════════════════════════════════════════════════════════════════
// DET MODE
// ════════════════════════════════════════════════════════════════════

DetMode AutoDroneTracker::get_detection_mode()
{
    cout
        << "\n--- Step 2: Detection Engine ---\n"
        << " [1] Classic DoG\n"
        << " [2] True LoG\n"
        << " [3] Hybrid DoG + LoG\n"
        << " [4] AI Det\n";

    string sel;

    cout << " Select (default 1): ";

    getline(cin, sel);

    if (sel == "2")
        return DetMode::LoG;

    if (sel == "3")
        return DetMode::Hybrid;

    if (sel == "4")
        return DetMode::AI;

    return DetMode::DoG;
}

// ════════════════════════════════════════════════════════════════════
// TRACK MODE
// ════════════════════════════════════════════════════════════════════

TrackAlgo
AutoDroneTracker::get_tracking_algorithm()
{
    cout
        << "\n--- Step 3: Classical Tracking Algorithm ---\n"
        << " [1] Centroid hold\n"
        << " [2] Edge density\n"
        << " [3] Phase correlation\n"
        << " [4] Template matching\n"
        << " [5] Optical flow\n";

    string sel;

    cout << " Select (default 4): ";

    getline(cin, sel);

    if (sel == "1")
        return TrackAlgo::Centroid;

    if (sel == "2")
        return TrackAlgo::Edge;

    if (sel == "3")
        return TrackAlgo::Phase;

    if (sel == "5")
        return TrackAlgo::OpticalFlow;

    return TrackAlgo::Template;
}

// ════════════════════════════════════════════════════════════════════
// PRIORITY MODE
// ════════════════════════════════════════════════════════════════════

PriorMode
AutoDroneTracker::get_priority_mode()
{
    cout
        << "\n--- Step 4: Multi-Target Priority ---\n"
        << " Active target priority is fixed by ROI entry order.\n"
        << " Priority-1 is always the oldest surviving target.\n\n"
        << " Detector tie-breaker for classic detectors:\n"
        << " [1] Nearest to centre\n"
        << " [2] Largest area\n"
        << " [3] Highest contrast\n";

    string sel;

    cout << " Select (default 1): ";

    getline(cin, sel);

    if (sel == "2")
        return PriorMode::Size;

    if (sel == "3")
        return PriorMode::Contrast;

    return PriorMode::Center;
}

// ════════════════════════════════════════════════════════════════════
// CLAMP
// ════════════════════════════════════════════════════════════════════

BBox AutoDroneTracker::clamp(
    int x,
    int y,
    int w,
    int h) const
{
    x = std::clamp(x, 0, roi_w_ - 1);
    y = std::clamp(y, 0, roi_h_ - 1);

    w = std::clamp(w, 1, roi_w_ - x);
    h = std::clamp(h, 1, roi_h_ - y);

    return {x, y, w, h};
}

// ════════════════════════════════════════════════════════════════════
// KALMAN FEED
// ════════════════════════════════════════════════════════════════════

void AutoDroneTracker::kalman_feed(
    const BBox &b)
{
    float cx =
        roi_x_ + b.cx();

    float cy =
        roi_y_ + b.cy();

    if (!kalman_.initialized)
    {
        kalman_.init(
            cx,
            cy,
            b.w,
            b.h);
    }
    else
    {
        kalman_.correct(
            cx,
            cy,
            b.w,
            b.h);
    }
}

// ════════════════════════════════════════════════════════════════════
// AI DET
// ════════════════════════════════════════════════════════════════════

optional<BBox>
AutoDroneTracker::AI_Det(
    const Mat &roi)
{
    if (!trt_)
        return nullopt;

    auto dets =
        trt_->inference(
            roi,
            cfg_.CONF_THRESH,
            roi.cols,
            roi.rows);

    if (dets.empty())
        return nullopt;

    auto best =
        max_element(
            dets.begin(),
            dets.end(),
            [](const Detection &a,
               const Detection &b)
            {
                return a.conf < b.conf;
            });

    int x =
        static_cast<int>(
            best->cx - best->w / 2.f);

    int y =
        static_cast<int>(
            best->cy - best->h / 2.f);

    AI_clss = best->cls;
    AI_conf = best->conf;

    return clamp(
        x,
        y,
        static_cast<int>(best->w),
        static_cast<int>(best->h));
}

vector<BBox>
AutoDroneTracker::AI_Dets(
    const Mat &roi)
{
    vector<BBox> boxes;

    if (!trt_)
        return boxes;

    auto dets =
        trt_->inference(
            roi,
            cfg_.CONF_THRESH,
            roi.cols,
            roi.rows);

    boxes.reserve(dets.size());

    for (const Detection &det : dets)
    {
        int x =
            static_cast<int>(
                det.cx - det.w / 2.f);

        int y =
            static_cast<int>(
                det.cy - det.h / 2.f);

        boxes.push_back(
            clamp(
                x,
                y,
                static_cast<int>(det.w),
                static_cast<int>(det.h)));
    }

    return boxes;
}

// ════════════════════════════════════════════════════════════════════
// DETECT
// ════════════════════════════════════════════════════════════════════

optional<BBox>
AutoDroneTracker::detect(
    const Mat &roi)
{
    switch (det_mode_)
    {
    case DetMode::DoG:
        return detector_->detect_dog(roi);

    case DetMode::LoG:
        return detector_->detect_log(roi);

    case DetMode::Hybrid:
        return detector_->detect_hybrid(roi);

    case DetMode::AI:
        return AI_Det(roi);
    }

    return nullopt;
}

vector<BBox>
AutoDroneTracker::detect_all(
    const Mat &roi)
{
    switch (det_mode_)
    {
    case DetMode::DoG:
        return detector_->detect_dog_all(roi);

    case DetMode::LoG:
        return detector_->detect_log_all(roi);

    case DetMode::Hybrid:
        return detector_->detect_hybrid_all(roi);

    case DetMode::AI:
        return AI_Dets(roi);
    }

    return {};
}

// ════════════════════════════════════════════════════════════════════
// ACQUIRE
// ════════════════════════════════════════════════════════════════════

bool AutoDroneTracker::acquire(
    const Mat &roi)
{
    auto det = detect(roi);

    if (!det)
        return false;

    bbox_ = *det;
    kalman_feed(*bbox_);

    cascade_->on_detection(
        roi,
        *bbox_);

    frames_lost_ = 0;

    return true;
}

// ════════════════════════════════════════════════════════════════════
// REACQUIRE
// ════════════════════════════════════════════════════════════════════

bool AutoDroneTracker::reacquire(
    const Mat &roi)
{
    if (!bbox_)
        return false;

    BBox b = *bbox_;

    bool ok =
        cascade_->reacquire(
            roi,
            b,
            kalman_,
            frames_lost_);

    if (!ok)
        return false;

    bbox_ = b;

    kalman_feed(b);

    frames_lost_ = 0;

    return true;
}

// ════════════════════════════════════════════════════════════════════
// MAIN LOOP
// ════════════════════════════════════════════════════════════════════

void AutoDroneTracker::run()
{
    cout
        << "\n[INFO] Tracking Started\n"
        << "Press ESC to exit\n\n";

    static const vector<string> class_names =
        {
            "Bird",
            "Drone",
            "Plane"};

    namedWindow(
        "Drone Tracker",
        WINDOW_NORMAL);

    while (true)
    {
        // ════════════════════════════════════════════════════════════
        // READ FRAME USING FFMPEG
        // ════════════════════════════════════════════════════════════

        optional<Mat> maybe_frame;

        if (first_frame_.has_value())
        {
            maybe_frame = first_frame_;
            first_frame_.reset();
        }
        else
        {
            maybe_frame = reader_->read();
        }

        if (!maybe_frame.has_value())
        {
            cerr
                << "[INFO] Stream ended\n";

            break;
        }

        Mat frame =
            maybe_frame.value();

        if (frame.empty())
            continue;

        ++frame_idx_;

        // ════════════════════════════════════════════════════════════
        // UPDATE FRAME SIZE
        // ════════════════════════════════════════════════════════════

        W_ = frame.cols;
        H_ = frame.rows;

        roi_w_ = min(roi_w_, W_);
        roi_h_ = min(roi_h_, H_);

        roi_x_ =
            max(0, (W_ - roi_w_) / 2);

        roi_y_ =
            max(0, (H_ - roi_h_) / 2);

        Rect roi_rect(
            roi_x_,
            roi_y_,
            roi_w_,
            roi_h_);

        roi_rect &=
            Rect(0, 0, W_, H_);

        if (roi_rect.width <= 0 ||
            roi_rect.height <= 0)
        {
            cerr
                << "[ERROR] Invalid ROI\n";

            continue;
        }

        Mat roi =
            frame(roi_rect);

        vector<BBox> detections;

        try
        {
            if (op_mode_ == OperatingMode::MANUAL &&
                !manual_target_selected_)
            {
                if (manual_select_target(frame))
                {
                    Rect selected(manual_start_, manual_end_);
                    selected &= Rect(0, 0, W_, H_);

                    Rect local =
                        selected & roi_rect;

                    if (local.area() > 0)
                    {
                        detections.push_back(
                            clamp(
                                local.x - roi_x_,
                                local.y - roi_y_,
                                local.width,
                                local.height));
                    }
                }
            }
            else if (op_mode_ == OperatingMode::AUTO &&
                     frame_idx_ % cfg_.DET_EVERY_N == 0)
            {
                detections =
                    detect_all(roi);
            }
        }
        catch (const exception &e)
        {
            cerr
                << "[DETECT ERROR] "
                << e.what()
                << "\n";
        }

        if (target_manager_)
        {
            target_manager_->update(
                roi,
                detections,
                frame_idx_);

            const TargetTrack *primary =
                target_manager_->primary_target();

            for (const TargetTrack &target : target_manager_->targets())
            {
                const bool is_primary =
                    primary &&
                    target.id == primary->id;

                Scalar color =
                    is_primary
                        ? Scalar(0, 255, 0)
                        : Scalar(255, 255, 255);

                string label =
                    "T" +
                    to_string(target.id) +
                    " P" +
                    to_string(target.priority);

                if (target.state == TrackState::LOST)
                    label += " LOST";

                draw_box(
                    frame,
                    label,
                    target.bbox.x,
                    target.bbox.y,
                    target.bbox.w,
                    target.bbox.h,
                    "",
                    color,
                    roi_x_,
                    roi_y_,
                    W_,
                    H_);
            }

            if (primary)
            {
                draw_crosshair(
                    frame,
                    roi_x_ + primary->bbox.cx(),
                    roi_y_ + primary->bbox.cy());
            }
        }

        // ════════════════════════════════════════════════════════════
        // DRAW ROI
        // ════════════════════════════════════════════════════════════

        rectangle(
            frame,
            roi_rect,
            Scalar(255, 0, 0),
            1);

        // ════════════════════════════════════════════════════════════
        // SHOW
        // ════════════════════════════════════════════════════════════

        imshow(
            "Drone Tracker",
            frame);

        if (out_vid_.isOpened())
        {
            out_vid_.write(frame);
        }

        int key =
            waitKey(1) & 0xFF;

        if (key == 27)
            break;
    }

    // ════════════════════════════════════════════════════════════════
    // CLEANUP
    // ════════════════════════════════════════════════════════════════

    reader_->close();

    if (out_vid_.isOpened())
        out_vid_.release();

    destroyAllWindows();

    cout
        << "\n[DONE] Output Saved : "
        << run_dir_
        << "\n";
}
