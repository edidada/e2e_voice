#pragma once

#include <queue>
#include <vector>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <thread>
#include <string>

struct AudioData {
    std::vector<float> samples;
    int sample_rate;
    std::string text; // For debugging
    
    AudioData() : sample_rate(22050) {}
    AudioData(std::vector<float> s, int sr, const std::string& t = std::string()) 
        : samples(std::move(s)), sample_rate(sr), text(t) {}
};

class AudioQueue {
public:
    AudioQueue();
    ~AudioQueue();

    // Add audio to the queue for sequential playback
    void enqueue(const AudioData& audio);
    
    // Start the playback thread
    void start();
    
    // Stop the playback thread and clear queue
    void stop();
    
    // Check if queue is empty
    bool empty() const;

private:
    void playbackWorker();
    void playAudioBlocking(const std::vector<float>& samples, int sample_rate);
    
    std::queue<AudioData> audio_queue_;
    mutable std::mutex mutex_;
    std::condition_variable cv_;
    std::atomic<bool> stop_flag_;
    std::thread playback_thread_;
    bool started_;
};