// ════════════════════════════════════════════════════════════════════
// drone_kalman.cpp
// ════════════════════════════════════════════════════════════════════

#include "drone_kalman.hpp"
#include <algorithm>
#include <cmath>
using namespace cv;

DroneKalman::DroneKalman() { _build(); }
void DroneKalman::reset() { _build(); }
// ── build / rebuild the internal KalmanFilter ───────────────────
void DroneKalman::_build()
{ // 6 state dims, 4 measurement dims
    kf_.init(6, 4, 0, CV_32F);
    const float dt = 1.0f;
    kf_.transitionMatrix = (Mat_<float>(6, 6)
                                << 1,
                            0, dt, 0, 0, 0,
                            0, 1, 0, dt, 0, 0,
                            0, 0, 1, 0, 0, 0,
                            0, 0, 0, 1, 0, 0,
                            0, 0, 0, 0, 1, 0,
                            0, 0, 0, 0, 0, 1);
    // Measurement matrix H — picks out cx, cy, w, h
    kf_.measurementMatrix = (Mat_<float>(4, 6) << 1, 0, 0, 0, 0, 0,
                             0, 1, 0, 0, 0, 0,
                             0, 0, 0, 0, 1, 0,
                             0, 0, 0, 0, 0, 1);
    // Process noise Q
    setIdentity(kf_.processNoiseCov, Scalar(1e-2));
    // Measurement noise R
    setIdentity(kf_.measurementNoiseCov,
                Scalar(5e-2)); // Initial covariance P
    setIdentity(kf_.errorCovPost, Scalar(1.0));
    // Zero initial state
    kf_.statePost = Mat::zeros(6, 1, CV_32F);
    initialized = false;
}
void DroneKalman::init(float cx, float cy, float w, float h)
{
    kf_.statePost.at<float>(0) = cx;
    kf_.statePost.at<float>(1) = cy;
    kf_.statePost.at<float>(2) = 0.f; // vx
    kf_.statePost.at<float>(3) = 0.f; // vy
    kf_.statePost.at<float>(4) = w;
    kf_.statePost.at<float>(5) = h;
    initialized = true;
}
std::array<float, 4> DroneKalman::predict()
{
    Mat p = kf_.predict();
    return {p.at<float>(0), p.at<float>(1), p.at<float>(4), p.at<float>(5)};
}
void DroneKalman::correct(float cx, float cy, float w, float h)
{
    Mat meas(4, 1, CV_32F);
    meas.at<float>(0) = cx;
    meas.at<float>(1) = cy;
    meas.at<float>(2) = w;
    meas.at<float>(3) = h;
    kf_.correct(meas);
}
BBox DroneKalman::predicted_roi_bbox(int roi_x, int roi_y)
{
    auto [cx, cy, w, h] = predict();
    BBox b;
    b.x = static_cast<int>(cx - roi_x - w / 2.f);
    b.y = static_cast<int>(cy - roi_y - h / 2.f);
    b.w = std::max(static_cast<int>(w), 8);
    b.h = std::max(static_cast<int>(h), 8);
    return b;
}

