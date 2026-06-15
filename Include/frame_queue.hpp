// frame_queue.hpp
#pragma once

#include <queue>
#include <mutex>
#include <condition_variable>
#include <algorithm>
#include <chrono>

template <typename T>
class FrameQueue
{
public:
    explicit FrameQueue(size_t max_size)
        : max_size_(std::max<size_t>(1, max_size))
    {
    }

    void push(const T &item)
    {
        std::unique_lock<std::mutex> lock(mutex_);

        if (queue_.size() >= max_size_)
            queue_.pop();

        queue_.push(item);
        cv_.notify_one();
    }

    bool pop(T &out)
    {
        std::unique_lock<std::mutex> lock(mutex_);

        if (!cv_.wait_for(lock, std::chrono::milliseconds(100),
                          [&]
                          { return !queue_.empty(); }))
        {
            return false;
        }

        out = queue_.front();
        queue_.pop();
        return true;
    }

private:
    std::queue<T> queue_;
    std::mutex mutex_;
    std::condition_variable cv_;
    size_t max_size_;
};
