//drone_tracker.hpp
#pragma once

#include <opencv2/opencv.hpp>
#include <array>
#include <deque>
#include <optional>
#include <string>
#include <vector>

// ── Enums ────────────────────────────────────────────────────────────
enum class DetMode
{
    DoG,
    LoG,
    Hybrid,
    AI
};
enum class TrackAlgo
{
    Centroid,
    Edge,
    Phase,
    Template,
    OpticalFlow
};
enum class PriorMode
{
    Center,
    Size,
    Contrast
};

enum class TRTOutputLayout
{
    EO_300x6,
    IR_25200x9
};

// ── Constants ─────────────────────────────
struct Config
{
    // Tracker
    int MAX_LOST_FRAMES = 45;
    float TMATCH_THRESH = 0.52f;
    float TMATCH_THRESH_TIGHT = 0.68f;
    int TEMPLATE_POOL_SZ = 6;
    float OF_QUALITY = 0.25f;
    int OF_MIN_PTS = 3;
    int SEARCH_PAD_BASE = 35;
    int SEARCH_PAD_GROW = 3;
    int DET_EVERY_N = 1;
    float HYBRID_LOG_WEIGHT = 0.5f; // 0 = pure DoG, 1 = pure LoG

    //  AI
    int TRT_INPUT_W = 640;
    int TRT_INPUT_H = 640;
    int TRT_MAX_DET = 300;
    int TRT_DET_COLS = 6;
    TRTOutputLayout TRT_OUTPUT_LAYOUT = TRTOutputLayout::EO_300x6;
    float CONF_THRESH = 0.50f;

    // Frame pipeline
    double SOURCE_FPS = 30.0; // filled by probe_fps()
    int FRAME_QUEUE = 1;    // lossless frame buffer depth
};

// ── Detection result (ROI-local coords) ──────────────────────────────
struct BBox
{
    int x{0}, y{0}, w{0}, h{0};
    bool valid() const { return w > 0 && h > 0; }
    int cx() const { return x + w / 2; }
    int cy() const { return y + h / 2; }
    cv::Rect toRect() const { return {x, y, w, h}; }
};
// ── AI Detection (ROI-local coords) ────────────────────────────────
struct Detection
{
    float cx{}, cy{}, w{}, h{};
    float conf{};
    int cls{};
};
// ── Cascade result ───────────────────────────────────────────────────
struct TrackResult
{
    BBox bbox;
    std::string label;
    std::string info;
    cv::Scalar color{0, 255, 0};
};
