
//drone_kalman.hpp
#pragma once
#include <opencv2/video/tracking.hpp>
#include "drone_tracker.hpp"
class DroneKalman
{
public:
    DroneKalman();
    void reset();
    // Seed the filter at first detection
    void init(float cx, float cy, float w, float h);

    // Call once per frame — advances the prediction step 
    // Returns (cx, cy, w, h) in global frame coords
    std::array<float, 4> predict();
    
    // Feed a confirmed measurement
    void correct(float cx, float cy, float w, float h); 
    
    // Convert current Kalman prediction to ROI-local BBox
    BBox predicted_roi_bbox(int roi_x, int roi_y);
    bool initialized = false;

private:
    void _build();
    cv::KalmanFilter kf_;
};