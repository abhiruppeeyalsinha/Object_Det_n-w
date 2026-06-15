// utils.hpp
#pragma once
// ════════════════════════════════════════════════════════════════════
// utils.hpp — UI drawing helpers & run-folder creation
// ════════════════════════════════════════════════════════════════════

#include "drone_tracker.hpp"
#include <string>

// Draw a labelled bounding box (ROI-local coords → global frame coords)
void draw_box(cv::Mat& frame, const std::string& label,
    int x, int y, int w, int h,
    const std::string& info,
    cv::Scalar color, int roi_x, int roi_y, int frame_w, int frame_h);

// Draw a crosshair at a global pixel position
void draw_crosshair(cv::Mat& frame, int gx, int gy);

// Createres / <DD - MM - YYYY> / Run_<unix_ts> / and return its path
std::string create_run_folder();
