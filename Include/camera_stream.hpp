//camera_stream.hpp

#pragma once

#include "frame_packet.hpp"
#include "frame_queue.hpp"
#include "frame_reader.hpp"

#include <thread>
#include <atomic>
#include <memory>

class CameraStream
{
public:

    CameraStream(const std::string& source,
                 CameraType type,
                 FrameQueue<FramePacket>& queue);

    ~CameraStream();

    bool start();

    void stop();

private:

    void run();

private:

    std::string source_;

    CameraType type_;

    FrameQueue<FramePacket>& queue_;

    std::thread worker_;

    std::atomic<bool> running_{false};

    std::unique_ptr<FrameReader> reader_;

    uint64_t frame_counter_ = 0;
};