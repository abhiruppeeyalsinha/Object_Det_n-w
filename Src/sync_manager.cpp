// sync_manager.cpp
#include "sync_manager.hpp"
#include <iostream>
#include <cmath>

using namespace std;

SyncManager::SyncManager(FrameQueue<FramePacket> &ir_q,
                         FrameQueue<FramePacket> &rgb_q,
                         FrameQueue<SyncedFrame> &out_q)
    : ir_queue_(ir_q), rgb_queue_(rgb_q), synced_queue_(out_q) {}

void SyncManager::start()
{
    running_ = true;
    worker_ = thread(&SyncManager::run, this);
}

void SyncManager::stop()
{

    running_ = false;
    if (worker_.joinable())
        worker_.join();
}

void SyncManager::run()
{
    FramePacket ir_pkt, rgb_pkt;

    while (running_)
    {
        if (!ir_queue_.pop(ir_pkt))
            continue;
        if (!rgb_queue_.pop(rgb_pkt))
            continue;

        int64_t diff = std::llabs(
            (long long)ir_pkt.timestamp_us -
            (long long)rgb_pkt.timestamp_us);



        // Create the sync frame
        SyncedFrame sf;
        sf.ir = ir_pkt;
        sf.rgb = rgb_pkt;
        sf.sync_delta_us = diff;

        synced_queue_.push(sf);
    }
}