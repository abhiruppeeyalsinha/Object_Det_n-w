// ════════════════════════════════════════════════════════════════════
// trackers.cpp
// ════════════════════════════════════════════════════════════════════

#include "trackers.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <numeric>
#include <optional>

// ════════════════════════════════════════════════════════════════════
// TemplatePool
// ════════════════════════════════════════════════════════════════════

TemplatePool::TemplatePool(int max_sz) : max_sz_(max_sz) {}
void TemplatePool::clear() { pool_.clear(); }
void TemplatePool::add(const cv::Mat &roi, const cv::Rect &rect)
{
    if (rect.width <= 0 || rect.height <= 0)
        return;
    cv::Rect safe = rect & cv::Rect(0, 0, roi.cols, roi.rows);
    if (safe.area() == 0)
        return;
    cv::Mat patch = roi(safe).clone();
    if (!pool_.empty())
    {
        cv::Mat ref;
        cv::resize(pool_.front(), ref, patch.size(), 0, 0, cv::INTER_NEAREST);
        if (ref.size() == patch.size() && ref.type() == patch.type())
        {
            cv::Mat res;
            cv::matchTemplate(patch, ref, res, cv::TM_CCOEFF_NORMED);
            if (res.at<float>(0, 0) > 0.93f)
                return; // near-duplicate — skip
        }
    }
    pool_.push_front(patch);
    while (static_cast<int>(pool_.size()) > max_sz_)
        pool_.pop_back();
}
std::optional<TemplatePool::MatchResult> TemplatePool::match(const cv::Mat &roi, cv::Rect search_rect) const
{
    if (pool_.empty())
        return std::nullopt;
    // Clamp search rect to ROI bounds
    search_rect &= cv::Rect(0, 0, roi.cols, roi.rows);
    if (search_rect.width < 4 || search_rect.height < 4)
        return std::nullopt;
    cv::Mat region = roi(search_rect);
    float best_val = -1.f;
    cv::Point best_loc;
    int best_tw = 0, best_th = 0;
    for (const auto &tmpl : pool_)
    {
        if (region.rows < tmpl.rows || region.cols < tmpl.cols)
            continue;
        cv::Mat res;
        cv::matchTemplate(region, tmpl, res, cv::TM_CCOEFF_NORMED);
        double val;
        cv::Point loc;
        cv::minMaxLoc(res, nullptr, &val, nullptr, &loc);
        if (static_cast<float>(val) > best_val)
        {
            best_val = static_cast<float>(val);
            // loc is relative to region; shift back to full ROI coords
            best_loc = loc + cv::Point(search_rect.x, search_rect.y);
            best_tw = tmpl.cols;
            best_th = tmpl.rows;
        }
    }
    if (best_val < 0.f)
        return std::nullopt;
    return MatchResult{best_loc, best_val, best_tw, best_th};
}
std::optional<TemplatePool::MatchResult> TemplatePool::match_full(const cv::Mat &roi) const { return match(roi, {0, 0, roi.cols, roi.rows}); }

// ════════════════════════════════════════════════════════════════════
// OpticalFlowTracker
// ════════════════════════════════════════════════════════════════════
OpticalFlowTracker::OpticalFlowTracker(float quality, int min_pts) : quality_(quality), min_pts_(min_pts) {}
void OpticalFlowTracker::reset()
{
    prev_gray_.release();
    prev_pts_.clear();
}
void OpticalFlowTracker::seed(const cv::Mat &gray_roi, const BBox &bbox)
{
    cv::Rect r{bbox.x, bbox.y, bbox.w, bbox.h};
    r &= cv::Rect(0, 0, gray_roi.cols, gray_roi.rows);
    if (r.area() == 0)
        return;
    cv::Mat patch = gray_roi(r);
    std::vector<cv::Point2f> pts;
    cv::goodFeaturesToTrack(patch, pts, 40, quality_, 3, cv::noArray(), 5);
    if (pts.empty())
        return;

    // Shift from patch coords to ROI coords
    for (auto &p : pts)
    {
        p.x += r.x;
        p.y += r.y;
    }
    prev_pts_ = std::move(pts);
    gray_roi.copyTo(prev_gray_);
}
std::optional<BBox> OpticalFlowTracker::track(const cv::Mat &gray_roi, const BBox &prev)
{
    if (prev_gray_.empty() || prev_pts_.empty())
        return std::nullopt;
    if (prev_gray_.size() != gray_roi.size())
        return std::nullopt;
    std::vector<cv::Point2f> next_pts;
    std::vector<uchar> status;
    std::vector<float> err;
    cv::calcOpticalFlowPyrLK(prev_gray_, gray_roi, prev_pts_, next_pts, status, err, {21, 21}, 3,
                             cv::TermCriteria(cv::TermCriteria::EPS | cv::TermCriteria::COUNT, 20, 0.01));

    // Collect good points
    std::vector<float> dxs, dys;
    std::vector<cv::Point2f> good_new;
    for (size_t i = 0; i < status.size(); ++i)
    {
        if (!status[i])
            continue;
        dxs.push_back(next_pts[i].x - prev_pts_[i].x);
        dys.push_back(next_pts[i].y - prev_pts_[i].y);
        good_new.push_back(next_pts[i]);
    }
    if (static_cast<int>(dxs.size()) < min_pts_)
        return std::nullopt;

    // Median displacement
    auto median = [](std::vector<float> &v) -> float
    { size_t n = v.size() / 2; std::nth_element(v.begin(), 
        v.begin() + n, v.end()); return v[n]; };
    float dx = median(dxs);
    float dy = median(dys);
    prev_pts_ = good_new;
    gray_roi.copyTo(prev_gray_);
    return BBox{static_cast<int>(prev.x + dx), static_cast<int>(prev.y + dy), prev.w, prev.h};
}
// ════════════════════════════════════════════════════════════════════
// TrackCascade
// ════════════════════════════════════════════════════════════════════
TrackCascade::TrackCascade(const Config &cfg, TrackAlgo algo, int roi_w, int roi_h, int roi_x, int roi_y) : cfg_(cfg),
                                                                                                            algo_(algo), roi_w_(roi_w), roi_h_(roi_h), roi_x_(roi_x), roi_y_(roi_y), pool_(cfg.TEMPLATE_POOL_SZ), of_(cfg.OF_QUALITY, cfg.OF_MIN_PTS) {}

BBox TrackCascade::clamp(int x, int y, int w, int h) const
{
    x = std::clamp(x, 0, roi_w_ - 1);
    y = std::clamp(y, 0, roi_h_ - 1);
    w = std::max(4, std::min(w, roi_w_ - x));
    h = std::max(4, std::min(h, roi_h_ - y));
    return {x, y, w, h};
}
void TrackCascade::kalman_feed(DroneKalman &k, const BBox &b) const
{
    float cx = roi_x_ + b.x + b.w / 2.f;
    float cy = roi_y_ + b.y + b.h / 2.f;
    if (!k.initialized)
        k.init(cx, cy, b.w, b.h);
    else
        k.correct(cx, cy, b.w, b.h);
}
void TrackCascade::on_detection(const cv::Mat &roi, const BBox &bbox)
{
    pool_.add(roi, bbox.toRect());
    cv::Mat gray;
    cv::cvtColor(roi, gray, cv::COLOR_BGR2GRAY);
    of_.seed(gray, bbox);
}
void TrackCascade::reset()
{
    pool_.clear();
    of_.reset();
}
// ── 4-tier cascade ────────────────────────────────────────────────────
std::optional<TrackResult> TrackCascade::track(const cv::Mat &roi, BBox &bbox, DroneKalman &kalman, int frames_lost)
{
    cv::Mat gray;
    cv::cvtColor(roi, gray, cv::COLOR_BGR2GRAY);

    // ── Tier-1: Optical flow ─────────────────────────────────────────
    auto of_result = of_.track(gray, bbox);
    if (of_result)
    {
        BBox ob = clamp(of_result->x, of_result->y, of_result->w, of_result->h);
        // Sanity-check against Kalman: drift must be plausible
        if (kalman.initialized)
        {
            BBox pb = kalman.predicted_roi_bbox(roi_x_, roi_y_);
            double drift = std::hypot(ob.cx() - (pb.x + pb.w / 2.0), ob.cy() - (pb.y + pb.h / 2.0));
            if (drift < std::max(ob.w, ob.h) * 2.5)
            {
                bbox = ob;
                kalman_feed(kalman, ob);
                pool_.add(roi, ob.toRect());
                return TrackResult{ob, "OF", "", {0, 200, 255}};
            }
        }
    }
    // ── Tier-2: Template pool inside Kalman window ───────────────────
    if (kalman.initialized)
    {
        BBox pb = kalman.predicted_roi_bbox(roi_x_, roi_y_);
        int pad = cfg_.SEARCH_PAD_BASE + frames_lost * cfg_.SEARCH_PAD_GROW;
        cv::Rect search{pb.x - pad, pb.y - pad, pb.w + 2 * pad, pb.h + 2 * pad};
        auto mr = pool_.match(roi, search);
        if (mr && mr->score >= cfg_.TMATCH_THRESH)
        {
            BBox nb = clamp(mr->loc.x, mr->loc.y, mr->tw, mr->th);
            bbox = nb;
            kalman_feed(kalman, nb);
            pool_.add(roi, nb.toRect());
            of_.seed(gray, nb);
            char info[32];
            std::snprintf(info, sizeof(info), "S:%.2f", mr->score);
            return TrackResult{nb, "TMATCH", info, {0, 255, 120}};
        }
    }
    // ── Tier-3a: Edge density ────────────────────────────────────────
    if (algo_ == TrackAlgo::Edge)
    {
        cv::Rect r = bbox.toRect() & cv::Rect(0, 0, roi.cols, roi.rows);
        if (r.area() > 0)
        {
            cv::Mat patch = roi(r);
            cv::Mat patch_gray, edges;
            cv::cvtColor(patch, patch_gray, cv::COLOR_BGR2GRAY);
            cv::Canny(patch_gray, edges, 50, 150);
            double density = static_cast<double>(cv::countNonZero(edges)) / (edges.total() + 1e-9);
            if (density > 0.008)
            {
                kalman_feed(kalman, bbox);
                char info[32];
                std::snprintf(info, sizeof(info), "D:%.2f", density);
                return TrackResult{bbox, "EDGE", info, {0, 255, 255}};
            }
        }
    }

    // ── Tier-3b: Phase correlation ───────────────────────────────────

    else if (algo_ == TrackAlgo::Phase)
    {
        if (!pool_.empty())
        {
            const cv::Mat &prev_patch = pool_.front();
            cv::Rect r = bbox.toRect() & cv::Rect(0, 0, roi.cols, roi.rows);
            if (r.area() > 0)
            {
                cv::Mat curr_patch = roi(r);
                cv::Mat p1_gray, p2_gray, p1f, p2f;
                cv::cvtColor(prev_patch, p1_gray, cv::COLOR_BGR2GRAY);
                cv::cvtColor(curr_patch, p2_gray, cv::COLOR_BGR2GRAY);
                if (p1_gray.size() != p2_gray.size())
                    cv::resize(p2_gray, p2_gray, p1_gray.size());
                p1_gray.convertTo(p1f, CV_32F);
                p2_gray.convertTo(p2f, CV_32F);
                // cv::Point2d shift;
                // double resp = cv::phaseCorrelate(p1f, p2f, cv::noArray(), &shift);

                double resp = 0.0;
                cv::Point2d shift = cv::phaseCorrelate(p1f, p2f, cv::noArray(), &resp);
                if (resp > 0.04)
                {
                    BBox nb = clamp(static_cast<int>(bbox.x + shift.x), static_cast<int>(bbox.y + shift.y), bbox.w, bbox.h);
                    bbox = nb;
                    kalman_feed(kalman, nb);
                    char info[32];
                    std::snprintf(info, sizeof(info), "R:%.2f", resp);
                    return TrackResult{nb, "PHASE", info, {255, 0, 255}};
                }
            }
        }
    }
    // ── Tier-3c: Centroid / OpticalFlow modes fall through to Tier-4 ─
    // ── Tier-4: Kalman extrapolation ─────────────────────────────────
    if (kalman.initialized)
    {
        BBox pb = kalman.predicted_roi_bbox(roi_x_, roi_y_);
        pb = clamp(pb.x, pb.y, pb.w, pb.h);
        bbox = pb;
        char info[32];
        std::snprintf(info, sizeof(info), "F:%d", frames_lost);
        return TrackResult{pb, "PREDICT", info, {255, 140, 0}};
    }
    return std::nullopt;
} // ── Re-acquisition (Kalman-windowed template search) ──────────────────
bool TrackCascade::reacquire(const cv::Mat &roi, BBox &bbox_out, DroneKalman &kalman, int frames_lost)

{
    if (!kalman.initialized)
        return false;
    BBox pb = kalman.predicted_roi_bbox(roi_x_, roi_y_);
    int pad = cfg_.SEARCH_PAD_BASE + frames_lost * cfg_.SEARCH_PAD_GROW;
    cv::Rect search{pb.x - pad, pb.y - pad, pb.w + 2 * pad, pb.h + 2 * pad};
    auto mr = pool_.match(roi, search);
    if (!mr || mr->score < cfg_.TMATCH_THRESH)
        return false;
    BBox nb = clamp(mr->loc.x, mr->loc.y, mr->tw, mr->th);
    bbox_out = nb;
    pool_.add(roi, nb.toRect());
    kalman_feed(kalman, nb);
    cv::Mat gray;
    cv::cvtColor(roi, gray, cv::COLOR_BGR2GRAY);
    of_.seed(gray, nb);
    return true;
}
