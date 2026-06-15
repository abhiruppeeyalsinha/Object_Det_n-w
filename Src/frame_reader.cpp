// ════════════════════════════════════════════════════════════════════
// frame_reader.cpp
// Ultra-Low Latency FFmpeg Reader
// Production-grade RTSP/UDP/TCP reader
// ════════════════════════════════════════════════════════════════════

#include "frame_reader.hpp"

#include <iostream>
#include <thread>
#include <chrono>

using namespace std;
using namespace cv;

namespace
{
bool is_live_source(const std::string &source)
{
    return source.rfind("rtsp://", 0) == 0 ||
           source.rfind("rtmp://", 0) == 0 ||
           source.rfind("udp://", 0) == 0 ||
           source.rfind("tcp://", 0) == 0 ||
           source.rfind("http://", 0) == 0 ||
           source.rfind("https://", 0) == 0;
}
}

// ════════════════════════════════════════════════════════════════════
// Constructor
// ════════════════════════════════════════════════════════════════════
FrameReader::FrameReader(const std::string &source)
    : source_(source)
{
    avformat_network_init();
}

// ════════════════════════════════════════════════════════════════════
// Destructor
// ════════════════════════════════════════════════════════════════════
FrameReader::~FrameReader()
{
    close();
}

// ════════════════════════════════════════════════════════════════════
// Open Stream
// ════════════════════════════════════════════════════════════════════
bool FrameReader::open()
{
    close();

    std::cout << "\n[FrameReader] Opening stream...\n";

    AVDictionary *opts = nullptr;

    // ════════════════════════════════════════════════════════════════
    // ULTRA LOW LATENCY SETTINGS
    // ════════════════════════════════════════════════════════════════

    // No internal buffering
    av_dict_set(&opts, "fflags", "nobuffer+fastseek+discardcorrupt", 0);

    // Low delay decode
    av_dict_set(&opts, "flags", "low_delay", 0);

    // RTSP transport
    av_dict_set(&opts, "rtsp_transport", "tcp", 0);

    // Minimal probe time
    av_dict_set(&opts, "probesize", "32768", 0);

    // Minimal analyze duration
    av_dict_set(&opts, "analyzeduration", "0", 0);

    // Reduce delay
    av_dict_set(&opts, "max_delay", "0", 0);

    // Small socket buffer
    av_dict_set(&opts, "buffer_size", "102400", 0);

    // Connection timeout
    av_dict_set(&opts, "stimeout", "3000000", 0);

    // Real-time stream
    av_dict_set(&opts, "reorder_queue_size", "0", 0);

    // ════════════════════════════════════════════════════════════════
    // OPEN INPUT
    // ════════════════════════════════════════════════════════════════

    int ret =
        avformat_open_input(
            &fmt_ctx_,
            source_.c_str(),
            nullptr,
            &opts);

    av_dict_free(&opts);

    if (ret < 0)
    {
        char errbuf[256];

        av_strerror(ret, errbuf, sizeof(errbuf));

        std::cerr
            << "[FrameReader] Failed to open stream: "
            << errbuf
            << std::endl;

        return false;
    }

    // ════════════════════════════════════════════════════════════════
    // STREAM INFO
    // ════════════════════════════════════════════════════════════════

    if (avformat_find_stream_info(fmt_ctx_, nullptr) < 0)
    {
        cerr << "frame_reader.cpp" << endl;
        std::cerr
            << "[FrameReader] Failed stream info"
            << std::endl;

        close();
        return false;
    }

    // ════════════════════════════════════════════════════════════════
    // FIND VIDEO STREAM
    // ════════════════════════════════════════════════════════════════

    for (unsigned int i = 0;
         i < fmt_ctx_->nb_streams;
         ++i)
    {
        if (fmt_ctx_->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO)
        {
            video_stream_idx_ = i;
            break;
        }
    }

    if (video_stream_idx_ == -1)
    {
        std::cerr
            << "[FrameReader] No video stream found"
            << std::endl;

        close();
        return false;
    }

    // ════════════════════════════════════════════════════════════════
    // DECODER SETUP
    // ════════════════════════════════════════════════════════════════

    AVCodecParameters *codecpar =
        fmt_ctx_->streams[video_stream_idx_]->codecpar;

    codec_ =
        avcodec_find_decoder(codecpar->codec_id);

    if (!codec_)
    {
        std::cerr
            << "[FrameReader] Decoder not found"
            << std::endl;

        close();
        return false;
    }

    codec_ctx_ =
        avcodec_alloc_context3(codec_);

    if (!codec_ctx_)
    {
        std::cerr
            << "[FrameReader] avcodec_alloc_context3 failed"
            << std::endl;

        close();
        return false;
    }

    if (avcodec_parameters_to_context(codec_ctx_, codecpar) < 0)
    {
        std::cerr
            << "[FrameReader] parameters_to_context failed"
            << std::endl;

        close();
        return false;
    }

    // ════════════════════════════════════════════════════════════════
    // LOW LATENCY DECODER CONFIG
    // ════════════════════════════════════════════════════════════════

    codec_ctx_->thread_count = 1;

    codec_ctx_->thread_type = FF_THREAD_SLICE;

    codec_ctx_->flags |= AV_CODEC_FLAG_LOW_DELAY;

    codec_ctx_->flags2 |= AV_CODEC_FLAG2_FAST;

    codec_ctx_->skip_frame = AVDISCARD_DEFAULT;

    codec_ctx_->skip_loop_filter = AVDISCARD_NONREF;

    codec_ctx_->skip_idct = AVDISCARD_NONREF;

    // ════════════════════════════════════════════════════════════════
    // OPEN DECODER
    // ════════════════════════════════════════════════════════════════

    if (avcodec_open2(codec_ctx_, codec_, nullptr) < 0)
    {
        std::cerr
            << "[FrameReader] Failed codec open"
            << std::endl;

        close();
        return false;
    }

    width_ = codec_ctx_->width;

    height_ = codec_ctx_->height;

    // ════════════════════════════════════════════════════════════════
    // ALLOCATE FRAMES
    // ════════════════════════════════════════════════════════════════

    frame_ = av_frame_alloc();

    frame_bgr_ = av_frame_alloc();

    packet_ = av_packet_alloc();

    if (!frame_ || !frame_bgr_ || !packet_)
    {
        std::cerr
            << "[FrameReader] Frame allocation failed"
            << std::endl;

        close();
        return false;
    }

    // ════════════════════════════════════════════════════════════════
    // BGR BUFFER
    // ════════════════════════════════════════════════════════════════

    int num_bytes =
        av_image_get_buffer_size(
            AV_PIX_FMT_BGR24,
            width_,
            height_,
            1);

    buffer_ =
        static_cast<uint8_t *>(
            av_malloc(num_bytes));

    if (!buffer_)
    {
        std::cerr
            << "[FrameReader] Buffer allocation failed"
            << std::endl;

        close();
        return false;
    }

    av_image_fill_arrays(
        frame_bgr_->data,
        frame_bgr_->linesize,
        buffer_,
        AV_PIX_FMT_BGR24,
        width_,
        height_,
        1);

    // ════════════════════════════════════════════════════════════════
    // SWS CONTEXT
    // ════════════════════════════════════════════════════════════════

    sws_ctx_ =
        sws_getContext(
            width_,
            height_,
            codec_ctx_->pix_fmt,
            width_,
            height_,
            AV_PIX_FMT_BGR24,
            SWS_FAST_BILINEAR,
            nullptr,
            nullptr,
            nullptr);

    if (!sws_ctx_)
    {
        std::cerr
            << "[FrameReader] sws_getContext failed"
            << std::endl;

        close();
        return false;
    }

    // ════════════════════════════════════════════════════════════════
    // STREAM FPS
    // ════════════════════════════════════════════════════════════════

    AVRational fps =
        fmt_ctx_->streams[video_stream_idx_]->avg_frame_rate;

    if (fps.den != 0)
    {
        double stream_fps =
            av_q2d(fps);

        std::cout
            << "[FrameReader] FPS: "
            << stream_fps
            << std::endl;
    }

    std::cout
        << "[FrameReader] Opened successfully: "
        << width_
        << "x"
        << height_
        << std::endl;

    return true;
}

// ════════════════════════════════════════════════════════════════════
// Read Frame
// ════════════════════════════════════════════════════════════════════
std::optional<cv::Mat> FrameReader::read()
{
    if (!fmt_ctx_)
        return std::nullopt;

    const bool live_source = is_live_source(source_);

    while (true)
    {
        int ret =
            av_read_frame(fmt_ctx_, packet_);

        // ─────────────────────────────────────────────
        // READ FAILED
        // ─────────────────────────────────────────────
        if (ret < 0)
        {
            // Network jitter / timeout
            if (ret == AVERROR(EAGAIN))
            {
                std::this_thread::sleep_for(
                    std::chrono::milliseconds(1));

                continue;
            }

            // RTSP temporary disconnect
            if (ret == AVERROR_EOF)
            {
                if (!live_source)
                    return std::nullopt;

                std::cerr
                    << "[FrameReader] Stream timeout/reconnect..."
                    << std::endl;

                std::this_thread::sleep_for(
                    std::chrono::milliseconds(100));

                continue;
            }

            char errbuf[256];

            av_strerror(ret, errbuf, sizeof(errbuf));

            std::cerr
                << "[FrameReader] av_read_frame error: "
                << errbuf
                << std::endl;

            if (!live_source)
                return std::nullopt;

            std::this_thread::sleep_for(
                std::chrono::milliseconds(10));

            continue;
        }

        // ─────────────────────────────────────────────
        // Ignore non-video packets
        // ─────────────────────────────────────────────
        if (packet_->stream_index != video_stream_idx_)
        {
            av_packet_unref(packet_);
            continue;
        }

        // ─────────────────────────────────────────────
        // Send packet
        // ─────────────────────────────────────────────
        ret =
            avcodec_send_packet(
                codec_ctx_,
                packet_);

        av_packet_unref(packet_);

        if (ret < 0)
            continue;

        // ─────────────────────────────────────────────
        // Receive frame
        // ─────────────────────────────────────────────
        ret =
            avcodec_receive_frame(
                codec_ctx_,
                frame_);

        if (ret == AVERROR(EAGAIN))
            continue;

        if (ret == AVERROR_EOF)
            continue;

        if (ret < 0)
            continue;

        // ─────────────────────────────────────────────
        // Convert YUV → BGR
        // ─────────────────────────────────────────────
        sws_scale(
            sws_ctx_,
            frame_->data,
            frame_->linesize,
            0,
            height_,
            frame_bgr_->data,
            frame_bgr_->linesize);

        cv::Mat img(
            height_,
            width_,
            CV_8UC3,
            frame_bgr_->data[0],
            frame_bgr_->linesize[0]);

        return img.clone();
    }
}

// ════════════════════════════════════════════════════════════════════
// Close
// ════════════════════════════════════════════════════════════════════
void FrameReader::close()
{
    if (buffer_)
    {
        av_free(buffer_);
        buffer_ = nullptr;
    }

    if (frame_)
    {
        av_frame_free(&frame_);
        frame_ = nullptr;
    }

    if (frame_bgr_)
    {
        av_frame_free(&frame_bgr_);
        frame_bgr_ = nullptr;
    }

    if (packet_)
    {
        av_packet_free(&packet_);
        packet_ = nullptr;
    }

    if (codec_ctx_)
    {
        avcodec_free_context(&codec_ctx_);
        codec_ctx_ = nullptr;
    }

    if (fmt_ctx_)
    {
        avformat_close_input(&fmt_ctx_);
        fmt_ctx_ = nullptr;
    }

    if (sws_ctx_)
    {
        sws_freeContext(sws_ctx_);
        sws_ctx_ = nullptr;
    }

    video_stream_idx_ = -1;

    width_ = 0;
    height_ = 0;

    std::cout
        << "[FrameReader] Closed"
        << std::endl;
}
