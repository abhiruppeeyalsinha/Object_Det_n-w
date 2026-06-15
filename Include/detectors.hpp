
//detectors.hpp
#pragma once
// ════════════════════════════════════════════════════════════════════
// detectors.hpp — DoG / LoG / Hybrid / AI detectors
// ════════════════════════════════════════════════════════════════════
#include "drone_tracker.hpp"
#include <optional>
#include <vector>

using namespace std;
// ════════════════════════════════════════════════════════════════════
class BlobDetector
{
public:
    explicit BlobDetector(const Config &cfg, int roi_w, int roi_h, PriorMode prior);

    // Returns best BBox in ROI-local coords, or nullopt
    std::optional<BBox> detect_dog(const cv::Mat &roi);
    std::optional<BBox> detect_log(const cv::Mat &roi);
    std::optional<BBox> detect_hybrid(const cv::Mat &roi);

    std::vector<BBox> detect_dog_all(const cv::Mat &roi);
    std::vector<BBox> detect_log_all(const cv::Mat &roi);
    std::vector<BBox> detect_hybrid_all(const cv::Mat &roi);

private:
    // ── helpers ───────────────────────────────────────────────────── // Both-polarity threshold → binary mask
    static cv::Mat blob_mask(const cv::Mat &response32, float thresh = 18.f);
    // Contour filter + priority selection
    std::optional<BBox> select_best(const cv::Mat &gray, const cv::Mat &fused_mask, const cv::Mat &dog_conf,
                                    // may be empty
                                    const cv::Mat &log_conf, // may be empty
                                    float hybrid_w) const;
    std::vector<BBox> select_all(const cv::Mat &gray,
                                 const cv::Mat &fused_mask,
                                 const cv::Mat &dog_conf,
                                 const cv::Mat &log_conf,
                                 float hybrid_w) const;
    const Config &cfg_;
    int roi_w_;
    int roi_h_;
    PriorMode prior_;
}; 
