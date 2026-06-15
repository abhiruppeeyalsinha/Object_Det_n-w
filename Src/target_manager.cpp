#include "target_manager.hpp"

#include <algorithm>

TargetManager::TargetManager(const Config &cfg,
                             TrackAlgo algo,
                             int roi_w,
                             int roi_h,
                             int roi_x,
                             int roi_y)
    : cfg_(cfg),
      algo_(algo),
      roi_w_(roi_w),
      roi_h_(roi_h),
      roi_x_(roi_x),
      roi_y_(roi_y)
{
}

void TargetManager::update(const cv::Mat &roi,
                           const std::vector<BBox> &detections,
                           uint64_t frame_id)
{
    std::vector<bool> matched(tracks_.size(), false);

    for (const BBox &det : detections)
    {
        int best_idx = -1;
        float best_iou = 0.0f;

        for (size_t i = 0; i < tracks_.size(); ++i)
        {
            if (matched[i])
                continue;

            const float score = iou(tracks_[i].bbox, det);

            if (score > best_iou)
            {
                best_iou = score;
                best_idx = static_cast<int>(i);
            }
        }

        if (best_idx >= 0 && best_iou >= 0.10f)
        {
            TargetTrack &track = tracks_[best_idx];

            matched[best_idx] = true;
            track.bbox = det;
            track.state = TrackState::TRACKING;
            track.last_seen_frame = frame_id;
            track.lost_frames = 0;
            track.age++;
            track.confidence = best_iou;

            if (!track.kalman.initialized)
                track.kalman.init(roi_x_ + det.cx(), roi_y_ + det.cy(), det.w, det.h);
            else
                track.kalman.correct(roi_x_ + det.cx(), roi_y_ + det.cy(), det.w, det.h);

            track.history.push_back({det.cx(), det.cy()});

            if (track.cascade)
                track.cascade->on_detection(roi, det);
        }
        else
        {
            create_new_target(det, frame_id);

            TargetTrack &track = tracks_.back();

            track.kalman.init(roi_x_ + det.cx(), roi_y_ + det.cy(), det.w, det.h);
            track.history.push_back({det.cx(), det.cy()});

            if (track.cascade)
                track.cascade->on_detection(roi, det);

            matched.push_back(true);
        }
    }

    for (size_t i = 0; i < tracks_.size(); ++i)
    {
        if (i < matched.size() && matched[i])
            continue;

        TargetTrack &track = tracks_[i];

        track.lost_frames++;
        track.age++;
        track.state = TrackState::LOST;

        if (track.cascade && track.kalman.initialized)
        {
            BBox current = track.bbox;
            auto result = track.cascade->track(roi,
                                               current,
                                               track.kalman,
                                               track.lost_frames);

            if (result)
            {
                track.bbox = result->bbox;
                track.state = TrackState::TRACKING;
                track.history.push_back({track.bbox.cx(), track.bbox.cy()});
            }
        }
    }

    remove_dead_targets();
    recompute_priorities();
}

const std::vector<TargetTrack> &TargetManager::targets() const
{
    return tracks_;
}

const TargetTrack *TargetManager::primary_target() const
{
    if (tracks_.empty())
        return nullptr;

    return &*std::min_element(
        tracks_.begin(),
        tracks_.end(),
        [](const TargetTrack &a,
           const TargetTrack &b)
        {
            return a.priority < b.priority;
        });
}

void TargetManager::create_new_target(
    const BBox &det,
    uint64_t frame_id)
{
    TargetTrack t;

    t.id = next_id_++;
    t.priority = static_cast<int>(tracks_.size()) + 1;
    t.state = TrackState::DETECTED;
    t.bbox = det;
    t.first_seen_frame = frame_id;
    t.last_seen_frame = frame_id;
    t.entry_order = global_entry_counter_++;
    t.lost_frames = 0;
    t.cascade = std::make_unique<TrackCascade>(cfg_, algo_, roi_w_, roi_h_, roi_x_, roi_y_);

    tracks_.push_back(std::move(t));
}

void TargetManager::remove_dead_targets()
{
    for (auto &t : tracks_)
    {
        if (t.lost_frames > cfg_.MAX_LOST_FRAMES)
        {
            t.state = TrackState::DEAD;
        }
    }

    tracks_.erase(
        remove_if(
            tracks_.begin(),
            tracks_.end(),
            [](const TargetTrack &t)
            {
                return t.state == TrackState::DEAD;
            }),
        tracks_.end());
}

void TargetManager::recompute_priorities()
{
    sort(tracks_.begin(),
         tracks_.end(),
         [](const TargetTrack &a,
            const TargetTrack &b)
         {
             return a.entry_order <
                    b.entry_order;
         });

    for (size_t i = 0;
         i < tracks_.size();
         ++i)
    {
        tracks_[i].priority =
            static_cast<int>(i) + 1;
    }
}

float TargetManager::iou(const BBox &a,
                         const BBox &b) const
{
    const int x1 = std::max(a.x, b.x);
    const int y1 = std::max(a.y, b.y);
    const int x2 = std::min(a.x + a.w, b.x + b.w);
    const int y2 = std::min(a.y + a.h, b.y + b.h);

    const int inter_w = std::max(0, x2 - x1);
    const int inter_h = std::max(0, y2 - y1);
    const int inter_area = inter_w * inter_h;

    const int union_area =
        a.w * a.h +
        b.w * b.h -
        inter_area;

    if (union_area <= 0)
        return 0.0f;

    return static_cast<float>(inter_area) /
           static_cast<float>(union_area);
}
