#ifndef TTS_MODEL_HPP
#define TTS_MODEL_HPP

#include <string>
#include <vector>
#include <memory>
#include "tts_config.hpp"

namespace tts {

struct GeneratedAudio {
    std::vector<float> samples;
    int sample_rate;
    
    // Get duration in seconds
    float duration() const {
        return samples.empty() ? 0.0f : static_cast<float>(samples.size()) / sample_rate;
    }
};

class TTSModel {
public:
    explicit TTSModel(const TTSConfig& config);
    ~TTSModel();
    
    // Initialize the TTS model
    bool initialize();
    
    // Generate audio from text
    GeneratedAudio generate(const std::string& text);
    
    // Generate audio with specific speaker (for multi-speaker models)
    GeneratedAudio generate(const std::string& text, int speaker_id);
    
    // Generate audio with custom parameters
    GeneratedAudio generate(const std::string& text, int speaker_id, float speed);
    
    // Check if model is initialized
    bool isInitialized() const;
    
    // Get number of speakers in the model
    int getNumSpeakers() const;
    
    // Get sample rate
    int getSampleRate() const;

private:
    class Impl;
    std::unique_ptr<Impl> pImpl;
};

} // namespace tts

#endif // TTS_MODEL_HPP