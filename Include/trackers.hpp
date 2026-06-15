// trackers.hpp
#pragma once
// ════════════════════════════════════════════════════════════════════
// trackers.hpp — template pool, optical-flow, 4-tier cascade
// ════════════════════════════════════════════════════════════════════

#include "drone_tracker.hpp"
#include "drone_kalman.hpp"
#include <deque>
#include <optional>

// ════════════════════════════════════════════════════════════════════
// TemplatePool — rolling bank of diverse appearance patches
// ════════════════════════════════════════════════════════════════════

class TemplatePool
{
public:
    explicit TemplatePool(int max_sz = 6);
    // Add patch if visually diverse from the newest entry
    void add(const cv::Mat &roi, const cv::Rect &rect);
    void clear();
    bool empty() const { return pool_.empty(); }
    const cv::Mat &front() const { return pool_.front(); }
    // Match all templates against *region* (which is already a sub-rect
    // of roi, at offset sx/sy within roi).
    // Returns(loc_in_roi, best_score, tmpl_w, tmpl_h) or nullopt.
    struct MatchResult
    {
        cv::Point loc;
        float score;
        int tw, th;
    };
    std::optional<MatchResult> match(const cv::Mat &roi, cv::Rect search_rect) const;
    // Full-ROI match variant
    std::optional<MatchResult> match_full(const cv::Mat &roi) const;

private:
    int max_sz_;
    std::deque<cv::Mat> pool_;
};
// ════════════════════════════════════════════════════════════════════
// OpticalFlowTracker — sparse LK with corner seeding
// ════════════════════════════════════════════════════════════════════
class OpticalFlowTracker
{
public:
    explicit OpticalFlowTracker(float quality = 0.25f, int min_pts = 3);

    // Seed tracking points inside bbox
    void seed(const cv::Mat &gray_roi, const BBox &bbox);
    void reset();

    // Track from previous to current gray frame.
    // Returns updated BBox or nullopt on failure.
    std::optional<BBox> track(const cv::Mat &gray_roi, const BBox &prev);

private:
    float quality_;
    int min_pts_;
    cv::Mat prev_gray_;
    std::vector<cv::Point2f> prev_pts_;
};

// ════════════════════════════════════════════════════════════════════
// TrackCascade — 4-tier tracker (mirrors Python _track_cascade)
// ════════════════════════════════════════════════════════════════════
class TrackCascade
{
public:
    TrackCascade(const Config &cfg, TrackAlgo algo, int roi_w, int roi_h, int roi_x, int roi_y);
    // ── Called on every successful detection ─────────────────────────
    void on_detection(const cv::Mat &roi, const BBox &bbox);
    // ── Called when detector misses ─────────────────────────────────
    // Returns TrackResult or nullopt when all tiers fail.
    std::optional<TrackResult> track(const cv::Mat &roi, BBox &current_bbox, DroneKalman &kalman, int frames_lost);
    // ── Kalman-windowed re-acquisition ───────────────────────────────
    bool reacquire(const cv::Mat &roi, BBox &bbox_out, DroneKalman &kalman, int frames_lost);
    void reset();
    TemplatePool &pool() { return pool_; }
    OpticalFlowTracker &of() { return of_; }

private:
    BBox clamp(int x, int y, int w, int h) const;
    void kalman_feed(DroneKalman &k, const BBox &b) const;
    const Config &cfg_;
    TrackAlgo algo_;
    int roi_w_, roi_h_, roi_x_, roi_y_;
    TemplatePool pool_;
    OpticalFlowTracker of_;
};