// ════════════════════════════════════════════════════════════════════
// utils.cpp
// ════════════════════════════════════════════════════════════════════

#include "utils.hpp"
#include <algorithm>
#include <chrono>
#include <ctime>
#include <iomanip>
#include <sstream>
#include <sys/stat.h>
#include <filesystem>

namespace fs = std::filesystem;
using namespace cv;

// ── draw_box ─────────────────────────────────────────────────────────

void draw_box(Mat &frame, const std::string &label, int x,
              int y, int w, int h, const std::string &info,
              Scalar color,
              int roi_x, int roi_y, int frame_w, int frame_h)
{
    // Convert ROI-local → global frame coords
    int gx = std::clamp(x + roi_x, 0, frame_w - 1);
    int gy = std::clamp(y + roi_y, 0, frame_h - 1);
    int gw = std::min(w, frame_w - gx);
    int gh = std::min(h, frame_h - gy);
    rectangle(frame, {gx, gy}, {gx + gw, gy + gh}, color, 1);
    std::string tag = info.empty() ? label : label + ": " + info;
    int text_y = std::max(12, gy - 10);

    // putText(frame, tag, {gx, gy - 10}, FONT_HERSHEY_SIMPLEX, 0.4, {0, 0, 0}, 2);
    putText(frame, tag, {gx, text_y}, FONT_HERSHEY_SIMPLEX, 0.4, color, 1);

    int bb_cx = gx + gw / 2;
    int bb_cy = gy + gh / 2;

    int frame_cx = frame_w / 2;
    int frame_cy = frame_h / 2;
    line(frame, {bb_cx, bb_cy}, {frame_cx, frame_cy}, color, 1, LINE_AA);

} // ── draw_crosshair ───────────────────────────────────────────────────

void draw_crosshair(Mat &frame, int gx, int gy) { drawMarker(frame, {gx, gy}, {0, 100, 255}, MARKER_CROSS, 16, 1); }

// ── create_run_folder ────────────────────────────────────────────────
std::string create_run_folder()
{
    auto now = std::chrono::system_clock::now();
    auto tt = std::chrono::system_clock::to_time_t(now);
    // auto epoch = static_cast<long long>(tt);

    std::tm tm_buf{};
    localtime_r(&tt, &tm_buf);

    std::ostringstream date_ss;
    date_ss << std::put_time(&tm_buf, "%d-%m-%Y");

    std::string base = std::string("res/") + date_ss.str();

    fs::create_directories(base);
    int run = 1;
    for (auto &e : fs::directory_iterator(base))
        if (e.is_directory() && e.path().filename().string().rfind("Run~", 0) == 0)
            run++;

    char buf[16];
    std::snprintf(buf, sizeof(buf), "Run~%02d", run);
    std::string path = base + "/" + buf;
    fs::create_directories(path);
    return path;
}
