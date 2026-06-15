
// sync_manager.hpp
#pragma once

#include "frame_packet.hpp"
#include "frame_queue.hpp"

#include <thread>
#include <atomic>

class SyncManager
{
public:

    SyncManager(FrameQueue<FramePacket>& ir_q,
                FrameQueue<FramePacket>& rgb_q,
                FrameQueue<SyncedFrame>& out_q);

    void start();
    void stop();

private:

    void run();

private:

    FrameQueue<FramePacket>& ir_queue_;
    FrameQueue<FramePacket>& rgb_queue_;
    FrameQueue<SyncedFrame>& synced_queue_;

    std::thread worker_;
    std::atomic<bool> running_{false};

    // const int64_t MAX_DELTA_US = 300000;
}; 
