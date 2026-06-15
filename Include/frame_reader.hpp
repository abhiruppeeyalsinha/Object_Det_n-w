//frame_reader.hpp
#pragma once

#include <opencv2/opencv.hpp>
#include <optional>
#include <string>

extern "C"
{
#include "libavcodec/avcodec.h"
#include "libavformat/avformat.h"
#include "libavutil/pixfmt.h"
#include "libswscale/swscale.h"
#include "libavutil/imgutils.h"
}

// ─────────────────────────────────────────────────────────────
// FrameReader 
// ─────────────────────────────────────────────────────────────
class FrameReader {
public:
    explicit FrameReader(const std::string& source);
    ~FrameReader();

    // Open RTSP video stream
    bool open();

    // Read one frame (returns nullopt on failure)
    std::optional<cv::Mat> read();

    // Close and cleanup
    void close();

private:
    std::string source_;

    AVFormatContext* fmt_ctx_ = nullptr;
    AVCodecContext*  codec_ctx_ = nullptr;
    const AVCodec*   codec_ = nullptr;

    AVFrame* frame_ = nullptr;
    AVFrame* frame_bgr_ = nullptr;
    AVPacket* packet_ = nullptr;

    SwsContext* sws_ctx_ = nullptr;
    uint8_t* buffer_ = nullptr;

    int video_stream_idx_ = -1;

    int width_ = 0;
    int height_ = 0;
}; 