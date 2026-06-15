#include "camera_stream.hpp"

#include <iostream>
#include <chrono>

using namespace std;

CameraStream::CameraStream(const std::string& source,
    CameraType type,
    FrameQueue<FramePacket>& queue)
    : source_(source),
    type_(type),
    queue_(queue)
{
}

CameraStream::~CameraStream()
{
    stop();
}

bool CameraStream::start()
{
    if (running_)
        return true;

    reader_ = std::make_unique<FrameReader>(source_);


    if (!reader_->open())
    {
        cerr << "[CameraStream] Failed to open source\n";
        return false;
    }

    running_ = true;

    worker_ = std::thread(&CameraStream::run, this);

    return true;
}

void CameraStream::stop()
{
    running_ = false;

    if (worker_.joinable())
        worker_.join();

    if (reader_)
    {
        reader_->close();
        reader_.reset();
    }
}

void CameraStream::run()
{
    cout << "[CameraStream] Using FFmpeg FrameReader\n";

    while (running_)
    {
        auto frame_opt = reader_->read();

        if (!frame_opt.has_value())
        {
            std::this_thread::sleep_for(
                std::chrono::milliseconds(5));

            continue;
        }

        FramePacket pkt;

        pkt.frame = frame_opt.value();

        pkt.frame_id = frame_counter_++;

        pkt.cam_type = type_;

        pkt.valid = true;

        pkt.timestamp_us =
            std::chrono::duration_cast<
            std::chrono::microseconds>(
                std::chrono::steady_clock::now()
                .time_since_epoch())
            .count();

        queue_.push(pkt);
    }
}
