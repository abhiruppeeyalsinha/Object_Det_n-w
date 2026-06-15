// frame_packet.hpp
#pragma once
#include <opencv2/opencv.hpp>
#include <cstdint>

enum class CameraType
{
    IR,
    RGB
};

struct FramePacket
{
    cv::Mat frame;
    uint64_t frame_id = 0;
    uint64_t timestamp_us = 0;
    CameraType cam_type = CameraType::RGB;
    bool valid = false;
};

struct SyncedFrame
{
    FramePacket ir;
    FramePacket rgb;
    uint64_t sync_delta_us = 0;
};
