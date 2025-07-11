#ifndef TTS_MODEL_DOWNLOADER_HPP
#define TTS_MODEL_DOWNLOADER_HPP

#include <string>
#include <vector>

namespace tts {

class TTSModelDownloader {
public:
    // Model file names
    static constexpr const char* MATCHA_ZH_MODEL = "matcha-icefall-zh-baker/model-steps-3.onnx";
    static constexpr const char* MATCHA_ZH_LEXICON = "matcha-icefall-zh-baker/lexicon.txt";
    static constexpr const char* MATCHA_ZH_TOKENS = "matcha-icefall-zh-baker/tokens.txt";
    static constexpr const char* MATCHA_ZH_DICT_DIR = "matcha-icefall-zh-baker/dict";
    static constexpr const char* VOCOS_VOCODER = "vocos-22khz-univ.onnx";
    
    TTSModelDownloader();
    ~TTSModelDownloader() = default;
    
    // Ensure all TTS models exist, download if necessary
    bool ensureModelsExist();
    
    // Get full path to a model file
    std::string getModelPath(const std::string& filename) const;
    
    // Check if a model file exists locally
    bool modelExists(const std::string& filename) const;
    
    // Download a specific model
    bool downloadModel(const std::string& filename);
    
    // Download and extract the complete TTS model tar.gz package
    bool downloadAndExtractTarGz();
    
    // Get cache directory
    std::string getCacheDir() const { return cache_dir_; }

private:
    std::string cache_dir_;
    
    // Get download URL for a model
    std::string getDownloadUrl(const std::string& filename) const;
    
    // Create cache directory if it doesn't exist
    bool ensureCacheDir();
    
    // Download file from URL
    bool downloadFile(const std::string& url, const std::string& dest_path);
    
    // Extract tar.gz archive
    bool extractTarGz(const std::string& archive_path, const std::string& dest_dir);
};

} // namespace tts

#endif // TTS_MODEL_DOWNLOADER_HPP