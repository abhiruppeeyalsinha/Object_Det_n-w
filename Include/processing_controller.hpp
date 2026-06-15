// processing_controller.hpp
#pragma once

#include "frame_packet.hpp"
#include "frame_queue.hpp"

#include <thread>
#include <atomic>
#include <mutex>

enum class PipelineMode
{
    Idle,
    Active
};

class ProcessingController
{
public:

    ProcessingController(FrameQueue<SyncedFrame>& q);

    void start();
    void stop();

    void setMode(PipelineMode m);

   
    bool getLatest(SyncedFrame& out);

private:

    void run();

private:

    FrameQueue<SyncedFrame>& queue_;

    std::atomic<bool> running_{false};
    std::atomic<PipelineMode> mode_{PipelineMode::Idle};

    std::thread worker_;

  
    SyncedFrame latest_;
    std::mutex latest_mutex_;
};