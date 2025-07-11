#include "audio_queue.hpp"
#include <portaudio.h>
#include <iostream>
#include <algorithm>

AudioQueue::AudioQueue() : stop_flag_(false), started_(false) {
}

AudioQueue::~AudioQueue() {
    stop();
}

void AudioQueue::enqueue(const AudioData& audio) {
    if (stop_flag_) return;
    
    {
        std::lock_guard<std::mutex> lock(mutex_);
        audio_queue_.push(audio);
    }
    cv_.notify_one();
}

void AudioQueue::start() {
    if (started_) return;
    
    stop_flag_ = false;
    playback_thread_ = std::thread(&AudioQueue::playbackWorker, this);
    started_ = true;
}

void AudioQueue::stop() {
    if (!started_) return;
    
    stop_flag_ = true;
    cv_.notify_all();
    
    if (playback_thread_.joinable()) {
        playback_thread_.join();
    }
    
    // Clear remaining audio
    std::lock_guard<std::mutex> lock(mutex_);
    while (!audio_queue_.empty()) {
        audio_queue_.pop();
    }
    
    started_ = false;
}

bool AudioQueue::empty() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return audio_queue_.empty();
}

void AudioQueue::playbackWorker() {
    std::cout << "[AudioQueue] Playback thread started" << std::endl;
    
    while (!stop_flag_) {
        AudioData audio;
        
        // Wait for audio or stop signal
        {
            std::unique_lock<std::mutex> lock(mutex_);
            cv_.wait(lock, [this] { return !audio_queue_.empty() || stop_flag_; });
            
            if (stop_flag_) {
                std::cout << "[AudioQueue] Stop signal received" << std::endl;
                break;
            }
            
            if (!audio_queue_.empty()) {
                audio = std::move(audio_queue_.front());
                audio_queue_.pop();
                std::cout << "[AudioQueue] Dequeued audio: " << audio.text.substr(0, 20) << "..." << std::endl;
            } else {
                continue;
            }
        }
        
        // Play audio outside of lock
        if (!audio.samples.empty()) {
            std::cout << "[AudioQueue] Playing audio (" << audio.samples.size() << " samples)" << std::endl;
            playAudioBlocking(audio.samples, audio.sample_rate);
            std::cout << "[AudioQueue] Finished playing audio" << std::endl;
        }
    }
    
    std::cout << "[AudioQueue] Playback thread ended" << std::endl;
}

void AudioQueue::playAudioBlocking(const std::vector<float>& samples, int sample_rate) {
    if (samples.empty()) {
        std::cout << "[AudioQueue] Empty audio samples, skipping" << std::endl;
        return;
    }
    
    PaStream* stream = nullptr;
    
    // Open audio stream
    PaError err = Pa_OpenDefaultStream(&stream,
                                      0,          // no input channels
                                      1,          // mono output
                                      paFloat32,  // 32 bit floating point output
                                      sample_rate,
                                      256,        // frames per buffer
                                      nullptr,    // use blocking API
                                      nullptr);   // no user data
    
    if (err != paNoError) {
        std::cerr << "[AudioQueue] Failed to open audio stream: " << Pa_GetErrorText(err) << std::endl;
        return;
    }
    
    // Start stream
    err = Pa_StartStream(stream);
    if (err != paNoError) {
        std::cerr << "[AudioQueue] Failed to start audio stream: " << Pa_GetErrorText(err) << std::endl;
        Pa_CloseStream(stream);
        return;
    }
    
    // Write audio data in chunks (blocking writes)
    const size_t chunk_size = 1024;
    size_t total_written = 0;
    for (size_t i = 0; i < samples.size(); i += chunk_size) {
        if (stop_flag_) {
            std::cout << "[AudioQueue] Stop requested during playback" << std::endl;
            break;
        }
        
        size_t frames_to_write = std::min(chunk_size, samples.size() - i);
        err = Pa_WriteStream(stream, &samples[i], frames_to_write);
        total_written += frames_to_write;
        
        if (err != paNoError && err != paOutputUnderflowed) {
            std::cerr << "[AudioQueue] Error writing to audio stream: " << Pa_GetErrorText(err) << std::endl;
            break;
        }
    }
    
    std::cout << "[AudioQueue] Wrote " << total_written << "/" << samples.size() << " samples" << std::endl;
    
    // Clean up
    if (stream) {
        Pa_CloseStream(stream);
    }
    
    std::cout << "[AudioQueue] Audio playback completed" << std::endl;
}