// target_manager.hpp
#pragma once

#include "drone_tracker.hpp"
#include "trackers.hpp"

#include <vector>
#include <memory>
#include <optional>

enum class TrackState
{
    DETECTED,
    TRACKING,
    LOST,
    DEAD
};

struct TargetTrack
{
    int id = -1;

    int priority = -1;

    TrackState state = TrackState::DETECTED;

    BBox bbox;

    uint64_t first_seen_frame = 0;

    uint64_t last_seen_frame = 0;

    int lost_frames = 0;

    float confidence = 0.0f;

    DroneKalman kalman;

    std::unique_ptr<TrackCascade> cascade;

    std::vector<cv::Point> history;

    int age = 0;

    uint64_t entry_order = 0;
};

class TargetManager
{
public:
    TargetManager(const Config &cfg,
                  TrackAlgo algo,
                  int roi_w,
                  int roi_h,
                  int roi_x,
                  int roi_y);

    void update(const cv::Mat &roi,
                const std::vector<BBox> &detections,
                uint64_t frame_id);

    const std::vector<TargetTrack> &targets() const;

    const TargetTrack *primary_target() const;

private:
    void create_new_target(const BBox &det,
                           uint64_t frame_id);

    void remove_dead_targets();

    void recompute_priorities();

    float iou(const BBox &a,
              const BBox &b) const;

private:
    const Config &cfg_;

    TrackAlgo algo_;

    int roi_w_;
    int roi_h_;
    int roi_x_;
    int roi_y_;

    std::vector<TargetTrack> tracks_;

    int next_id_ = 0;

    uint64_t global_entry_counter_ = 0;
};
