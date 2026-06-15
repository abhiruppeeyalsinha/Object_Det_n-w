

// processing_controller.cpp
#include "processing_controller.hpp"

ProcessingController::ProcessingController(FrameQueue<SyncedFrame> &q)
    : queue_(q)
{
}

void ProcessingController::start()
{
    running_ = true;
    worker_ = std::thread(&ProcessingController::run, this);
}

void ProcessingController::stop()
{
    running_ = false;

    if (worker_.joinable())
        worker_.join();
}

void ProcessingController::setMode(PipelineMode m)
{
    mode_ = m;
}

bool ProcessingController::getLatest(SyncedFrame &out)
{
    std::lock_guard<std::mutex> lock(latest_mutex_);

    if (!latest_.ir.valid || !latest_.rgb.valid)
        return false;

    out = latest_;
    out.ir.frame = latest_.ir.frame.clone();
    out.rgb.frame = latest_.rgb.frame.clone();
    out.sync_delta_us = latest_.sync_delta_us;

    return true;
}

void ProcessingController::run()
{
    SyncedFrame sf;

    while (running_)
    {
        if (!queue_.pop(sf))
            continue;

        std::lock_guard<std::mutex> lock(latest_mutex_);

        latest_ = sf;
        latest_.ir.frame = sf.ir.frame.clone();
        latest_.rgb.frame = sf.rgb.frame.clone();
        latest_.sync_delta_us = sf.sync_delta_us;

        latest_.ir.valid = true;
        latest_.rgb.valid = true;
    }
}
