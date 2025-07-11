#ifndef TTS_CONFIG_HPP
#define TTS_CONFIG_HPP

#include <string>

namespace tts {

struct TTSConfig {
    // Model paths
    std::string acoustic_model_path;  // Matcha acoustic model
    std::string vocoder_path;          // Vocoder model (HiFiGAN/Vocos)
    std::string lexicon_path;          // Lexicon file for pronunciation
    std::string tokens_path;           // Token vocabulary
    std::string dict_dir;              // Jieba dictionary directory (for Chinese)
    std::string data_dir;              // espeak-ng data directory (for English)
    
    // Model parameters
    float noise_scale = 1.0f;          // Controls variation (0.5-2.0)
    float length_scale = 1.0f;         // Speech speed (>1.0 = slower)
    int speaker_id = 0;                // For multi-speaker models
    
    // Runtime parameters
    int sample_rate = 22050;           // Output sample rate
    int max_num_sentences = 5;         // Max sentences per batch
    std::string language = "zh";       // Language: "zh" or "en"
    
    // Model type
    std::string model_type = "matcha"; // Currently only supporting matcha
    
    TTSConfig() = default;
};

} // namespace tts

#endif // TTS_CONFIG_HPP