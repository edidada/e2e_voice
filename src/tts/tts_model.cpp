#include "tts/tts_model.hpp"
#include "audio_processor.hpp"
#include <onnxruntime_cxx_api.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <unordered_map>
#include <unordered_set>
#include <filesystem>
#include <cstdint>
#include <fftw3.h>
#include <regex>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Jieba for Chinese text segmentation
#include "jieba/Jieba.hpp"

namespace fs = std::filesystem;

namespace tts {

// Internal helper functions
namespace {

// Helper function to read custom metadata from ONNX model
std::string LookupCustomModelMetaData(const Ort::ModelMetadata& meta_data,
                                     const std::string& key,
                                     Ort::AllocatorWithDefaultOptions& allocator) {
    try {
        auto result = meta_data.LookupCustomMetadataMapAllocated(key.c_str(), allocator);
        if (result) {
            return std::string(result.get());
        }
    } catch (...) {
        // Ignore exceptions and return empty string
    }
    return "";
}

// Read tokens to ID mapping from file
std::unordered_map<std::string, int64_t> readTokenToIdMap(const std::string& path) {
    std::unordered_map<std::string, int64_t> token_to_id;
    std::ifstream file(path);
    if (!file) {
        throw std::runtime_error("Failed to open tokens file: " + path);
    }
    
    std::string line;
    int line_num = 0;
    while (std::getline(file, line)) {
        line_num++;
        if (!line.empty()) {
            std::istringstream iss(line);
            std::string token;
            int64_t id;
            
            if (iss >> token >> id) {
                // Format: "token_name token_id"
                token_to_id[token] = id;
            } else {
                // Fallback: use line number as ID (0-indexed)
                token_to_id[line] = line_num - 1;
            }
        }
    }
    
    return token_to_id;
}

// Read lexicon from file
std::unordered_map<std::string, std::string> readLexicon(const std::string& path) {
    std::unordered_map<std::string, std::string> lexicon;
    std::ifstream file(path);
    if (!file) {
        throw std::runtime_error("Failed to open lexicon file: " + path);
    }
    
    std::string line;
    while (std::getline(file, line)) {
        if (!line.empty()) {
            size_t space_pos = line.find(' ');
            if (space_pos != std::string::npos) {
                std::string word = line.substr(0, space_pos);
                std::string phones = line.substr(space_pos + 1);
                lexicon[word] = phones;
            }
        }
    }
    
    return lexicon;
}

// Convert mel spectrogram to audio using vocoder
std::vector<float> vocoderInference(Ort::Session& session, const std::vector<float>& mel, int mel_dim) {
    // Get input/output info
    Ort::AllocatorWithDefaultOptions allocator;
    
    
    // Prepare input tensor
    int64_t num_frames = mel.size() / mel_dim;
    std::vector<int64_t> input_shape = {1, mel_dim, num_frames};
    
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info, const_cast<float*>(mel.data()), mel.size(),
        input_shape.data(), input_shape.size()
    );
    
    // Use correct vocoder interface - get all outputs to see which one is audio
    const char* input_names[] = {"mels"};
    const char* output_names[] = {"mag", "x", "y"};
    
    auto output_tensors = session.Run(Ort::RunOptions{nullptr}, 
                                     input_names, &input_tensor, 1,
                                     output_names, 3);
    
    
    // Vocos outputs frequency domain data that needs ISTFT post-processing
    // Extract the three outputs: mag, x (real), y (imag)
    float* mag_data = output_tensors[0].GetTensorMutableData<float>();
    float* x_data = output_tensors[1].GetTensorMutableData<float>();
    float* y_data = output_tensors[2].GetTensorMutableData<float>();
    
    auto vocoder_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
    int32_t batch_size = vocoder_shape[0];
    int32_t n_fft_bins = vocoder_shape[1];  // 513 = (1024/2 + 1) for n_fft=1024
    int32_t vocoder_frames = vocoder_shape[2];
    
    
    // Reconstruct complex STFT from magnitude and phase components
    // Follow sherpa-onnx layout: (num_frames, n_fft/2+1)
    // real = mag * x, imag = mag * y
    std::vector<float> stft_real(vocoder_frames * n_fft_bins);
    std::vector<float> stft_imag(vocoder_frames * n_fft_bins);
    
    // Vocoder output is (batch, freq, time), we need (time, freq)
    for (int32_t frame = 0; frame < vocoder_frames; ++frame) {
        for (int32_t bin = 0; bin < n_fft_bins; ++bin) {
            int32_t vocoder_idx = bin * vocoder_frames + frame;  // (freq, time) layout
            int32_t stft_idx = frame * n_fft_bins + bin;         // (time, freq) layout
            
            stft_real[stft_idx] = mag_data[vocoder_idx] * x_data[vocoder_idx];
            stft_imag[stft_idx] = mag_data[vocoder_idx] * y_data[vocoder_idx];
        }
    }
    
    // Use proper ISTFT implementation following sherpa-onnx approach
    int32_t n_fft = 1024;
    int32_t hop_length = 256;
    int32_t win_length = 1024;
    
    // Calculate proper audio length (without center padding handling for now)
    int32_t audio_length = n_fft + (vocoder_frames - 1) * hop_length;
    std::vector<float> audio(audio_length, 0.0f);
    std::vector<float> denominator(audio_length, 0.0f);
    
    // Create Hann window
    std::vector<float> window(win_length);
    for (int32_t i = 0; i < win_length; ++i) {
        window[i] = 0.5f * (1.0f - std::cos(2.0f * M_PI * i / (win_length - 1)));
    }
    
    
    // Process each frame using correct ISTFT approach
    for (int32_t frame = 0; frame < vocoder_frames; ++frame) {
        // Prepare complex data for IFFT - follow kaldi-native-fbank format
        std::vector<float> ifft_data(n_fft);
        
        // Extract real and imag for this frame
        const float *p_real = stft_real.data() + frame * n_fft_bins;
        const float *p_imag = stft_imag.data() + frame * n_fft_bins;
        
        // Pack for real FFT (hermitian symmetry format)
        for (int32_t i = 0; i < n_fft / 2; ++i) {
            if (i == 0) {
                ifft_data[0] = p_real[0];  // DC component
                ifft_data[1] = p_real[n_fft / 2];  // Nyquist component  
            } else {
                ifft_data[2 * i] = p_real[i];      // Real part
                ifft_data[2 * i + 1] = p_imag[i];  // Imaginary part
            }
        }
        
        // Perform inverse FFT using FFTW
        fftwf_complex* in = fftwf_alloc_complex(n_fft / 2 + 1);
        float* out = fftwf_alloc_real(n_fft);
        fftwf_plan plan = fftwf_plan_dft_c2r_1d(n_fft, in, out, FFTW_ESTIMATE);
        
        // Copy to FFTW input format
        for (int32_t i = 0; i < n_fft_bins; ++i) {
            in[i][0] = p_real[i];
            in[i][1] = p_imag[i];
        }
        
        // Execute IFFT
        fftwf_execute(plan);
        
        // Apply IFFT normalization
        float scale = 1.0f / n_fft;
        for (int32_t i = 0; i < n_fft; ++i) {
            out[i] *= scale;
        }
        
        // Apply window
        for (int32_t i = 0; i < win_length; ++i) {
            out[i] *= window[i];
        }
        
        // Overlap-add
        int32_t start_pos = frame * hop_length;
        for (int32_t i = 0; i < n_fft; ++i) {
            if (start_pos + i < audio_length) {
                audio[start_pos + i] += out[i];
                denominator[start_pos + i] += window[i] * window[i];
            }
        }
        
        // Cleanup
        fftwf_destroy_plan(plan);
        fftwf_free(in);
        fftwf_free(out);
    }
    
    // Normalize by window overlap
    for (int32_t i = 0; i < audio_length; ++i) {
        if (denominator[i] > 1e-8f) {
            audio[i] /= denominator[i];
        }
    }
    
    // Apply volume scaling
    float max_amplitude = 0.0f;
    for (float sample : audio) {
        max_amplitude = std::max(max_amplitude, std::abs(sample));
    }
    
    if (max_amplitude > 0.0f) {
        float scale = 0.8f / max_amplitude;  // Scale to 80% for better volume
        for (float& sample : audio) {
            sample *= scale;
        }
    }
    
    return audio;
}

} // anonymous namespace

// TTSModel implementation
class TTSModel::Impl {
public:
    explicit Impl(const TTSConfig& config) : config_(config) {}
    
    bool initialize() {
        try {
            // Initialize ONNX Runtime
            env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "TTSModel");
            
            Ort::SessionOptions session_options;
            session_options.SetIntraOpNumThreads(4);
            session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
            
            // Load acoustic model
            acoustic_model_ = std::make_unique<Ort::Session>(*env_, config_.acoustic_model_path.c_str(), session_options);
            
            // Load vocoder model
            vocoder_model_ = std::make_unique<Ort::Session>(*env_, config_.vocoder_path.c_str(), session_options);
            
            // Load token to ID mapping
            token_to_id_ = readTokenToIdMap(config_.tokens_path);
            
            // Load lexicon
            lexicon_ = readLexicon(config_.lexicon_path);
            
            // Initialize Jieba for Chinese
            if (config_.language == "zh") {
                initializeJieba();
            }
            
            // Get model metadata
            extractModelMetadata();
            
            initialized_ = true;
            return true;
            
        } catch (const std::exception& e) {
            std::cerr << "Failed to initialize TTS model: " << e.what() << std::endl;
            return false;
        }
    }
    
    GeneratedAudio generate(const std::string& text, int speaker_id, float speed) {
        if (!initialized_) {
            throw std::runtime_error("TTS model not initialized");
        }
        
        // Preprocess text
        std::string processed_text = preprocessText(text);
        
        // Convert text to tokens
        std::vector<int64_t> token_ids = textToTokenIds(processed_text);
        
        if (token_ids.empty()) {
            GeneratedAudio audio;
            audio.sample_rate = config_.sample_rate;
            return audio;
        }
        
        // Add blank tokens between phonemes (Matcha requirement)
        std::vector<int64_t> tokens_with_blanks = addBlankTokens(token_ids);
        
        // Run acoustic model
        std::vector<float> mel = runAcousticModel(tokens_with_blanks, speaker_id, speed);
        
        if (mel.empty()) {
            GeneratedAudio audio;
            audio.sample_rate = config_.sample_rate;
            return audio;
        }
        
        // Run vocoder
        std::vector<float> audio_samples = vocoderInference(*vocoder_model_, mel, mel_dim_);
        
        // Create result
        GeneratedAudio audio;
        audio.samples = std::move(audio_samples);
        audio.sample_rate = config_.sample_rate;
        
        return audio;
    }
    
    bool isInitialized() const {
        return initialized_;
    }
    
    int getNumSpeakers() const {
        return num_speakers_;
    }
    
    int getSampleRate() const {
        return config_.sample_rate;
    }
    
private:
    void initializeJieba() {
        // Initialize Jieba with dictionary
        std::string dict_path = config_.dict_dir + "/jieba.dict.utf8";
        std::string hmm_path = config_.dict_dir + "/hmm_model.utf8";
        std::string user_dict = config_.dict_dir + "/user.dict.utf8";
        std::string idf_path = config_.dict_dir + "/idf.utf8";
        std::string stop_words = config_.dict_dir + "/stop_words.utf8";
        
        jieba_ = std::make_unique<cppjieba::Jieba>(
            dict_path, hmm_path, user_dict, idf_path, stop_words
        );
        
    }
    
    void extractModelMetadata() {
        // Get model inputs/outputs info
        Ort::AllocatorWithDefaultOptions allocator;
        
        // Check number of inputs
        size_t num_inputs = acoustic_model_->GetInputCount();
        
        // Read acoustic model metadata
        try {
            Ort::ModelMetadata acoustic_meta = acoustic_model_->GetModelMetadata();
            
            // Read pad_id from metadata
            auto pad_id_value = LookupCustomModelMetaData(acoustic_meta, "pad_id", allocator);
            if (!pad_id_value.empty()) {
                pad_id_ = std::stoi(pad_id_value);
            } else {
                pad_id_ = 0;  // Default value
            }
            
        } catch (const std::exception& e) {
            pad_id_ = 0;
        }
        
        // Extract ISTFT parameters from vocoder model metadata
        try {
            Ort::ModelMetadata vocoder_meta = vocoder_model_->GetModelMetadata();
            
            // Try to read ISTFT parameters from model metadata
            auto read_meta_int = [&](const char* key, int32_t& value, int32_t default_val) {
                try {
                    auto key_alloc = vocoder_meta.LookupCustomMetadataMapAllocated(key, allocator);
                    if (key_alloc) {
                        value = std::stoi(key_alloc.get());
                        // Loaded value from metadata
                    } else {
                        value = default_val;
                    }
                } catch (...) {
                    value = default_val;
                }
            };
            
            auto read_meta_bool = [&](const char* key, bool& value, bool default_val) {
                try {
                    auto key_alloc = vocoder_meta.LookupCustomMetadataMapAllocated(key, allocator);
                    if (key_alloc) {
                        std::string str_val = key_alloc.get();
                        value = (str_val == "true" || str_val == "True" || str_val == "1");
                        // Loaded value from metadata
                    } else {
                        value = default_val;
                    }
                } catch (...) {
                    value = default_val;
                }
            };
            
            auto read_meta_string = [&](const char* key, std::string& value, const std::string& default_val) {
                try {
                    auto key_alloc = vocoder_meta.LookupCustomMetadataMapAllocated(key, allocator);
                    if (key_alloc) {
                        value = key_alloc.get();
                        // Loaded value from metadata
                    } else {
                        value = default_val;
                    }
                } catch (...) {
                    value = default_val;
                }
            };
            
            // Read ISTFT parameters
            read_meta_int("n_fft", istft_n_fft_, 1024);
            read_meta_int("hop_length", istft_hop_length_, 256);
            read_meta_int("win_length", istft_win_length_, 1024);
            read_meta_bool("center", istft_center_, true);
            read_meta_bool("normalized", istft_normalized_, false);
            read_meta_string("window_type", istft_window_type_, "hann");
            read_meta_string("pad_mode", istft_pad_mode_, "reflect");
            
        } catch (const std::exception& e) {
            // Use sensible defaults for Vocos
            istft_n_fft_ = 1024;
            istft_hop_length_ = 256;
            istft_win_length_ = 1024;
            istft_center_ = true;
            istft_normalized_ = false;
            istft_window_type_ = "hann";
            istft_pad_mode_ = "reflect";
        }
        
        // For Matcha models, mel_dim is typically 80
        mel_dim_ = 80;
        
        // TODO: Extract number of speakers from model metadata
        num_speakers_ = 1;
    }
    
    std::string preprocessText(const std::string& text) {
        // Basic text normalization
        std::string processed = text;
        
        // Convert to lowercase for English
        if (config_.language == "en") {
            std::transform(processed.begin(), processed.end(), processed.begin(), ::tolower);
        }
        
        // TODO: Add more sophisticated text normalization
        // - Number to words conversion
        // - Abbreviation expansion
        // - Punctuation handling
        
        return processed;
    }
    
    std::string mapPhoneme(const std::string& phone) {
        // Handle common phoneme mismatches between lexicon and token vocabulary
        static std::unordered_map<std::string, std::string> phoneme_mapping = {
            // Common mappings for missing phonemes
            {"shei2", "she2"},  // who (谁) -> she sound
            {"cei2", "ce2"},    // missing variants
            {"den1", "de1"},    // missing variants
            {"den2", "de2"},
            {"den3", "de3"},
            {"den4", "de4"},
            {"kei2", "ke2"},    // missing variants
            {"kei3", "ke3"},
            {"nei1", "ne1"},    // missing variants
            {"pou1", "po1"},    // missing variants
            {"pou2", "po2"},
            {"pou3", "po3"},
            {"yo1", "yo"},      // missing tone variant
            {"m2", "m"},        // missing tone on nasal
            {"n2", "n"},        // missing tone on nasal
            {"ng2", "ng"},      // missing ng phoneme
            {"hm", "hm1"},      // add tone to hmm sound
        };
        
        auto it = phoneme_mapping.find(phone);
        if (it != phoneme_mapping.end()) {
            return it->second;
        }
        
        // If no direct mapping found, try removing or changing tone
        if (phone.length() > 1) {
            char last_char = phone.back();
            if (last_char >= '1' && last_char <= '4') {
                // Try without tone
                std::string base = phone.substr(0, phone.length() - 1);
                return base;
            } else {
                // Try adding tone 1 if no tone present
                return phone + "1";
            }
        }
        
        return phone;  // Return original if no mapping found
    }

    std::vector<int64_t> textToTokenIds(const std::string& text) {
        std::vector<int64_t> token_ids;
        
        if (config_.language == "zh" && jieba_) {
            // Follow sherpa-onnx's approach for Chinese text processing
            
            // Step 1: Replace punctuations (like sherpa-onnx)
            std::string processed_text = text;
            // Use regex to replace punctuations following sherpa-onnx pattern
            std::regex punct_re1("：|、|；");
            processed_text = std::regex_replace(processed_text, punct_re1, "，");
            std::regex punct_re2("[.]");
            processed_text = std::regex_replace(processed_text, punct_re2, "。");
            std::regex punct_re3("[?]");
            processed_text = std::regex_replace(processed_text, punct_re3, "？");
            std::regex punct_re4("[!]");
            processed_text = std::regex_replace(processed_text, punct_re4, "！");
            
            
            // Step 2: Jieba segmentation
            std::vector<std::string> words;
            jieba_->Cut(processed_text, words, true);  // Use HMM
            
            
            // Step 3: Remove redundant spaces and punctuations like sherpa-onnx
            std::vector<std::string> cleaned_words;
            for (size_t i = 0; i < words.size(); ++i) {
                if (i == 0) {
                    cleaned_words.push_back(words[i]);
                } else if (words[i] == " ") {
                    if (cleaned_words.back() == " " || isPunctuation(cleaned_words.back())) {
                        continue;  // Skip redundant spaces
                    } else {
                        cleaned_words.push_back(words[i]);
                    }
                } else if (isPunctuation(words[i])) {
                    if (cleaned_words.back() == " " || isPunctuation(cleaned_words.back())) {
                        continue;  // Skip redundant punctuations
                    } else {
                        cleaned_words.push_back(words[i]);
                    }
                } else {
                    cleaned_words.push_back(words[i]);
                }
            }
            
            
            // Step 4: Convert words to token IDs
            for (const auto& word : cleaned_words) {
                auto word_ids = convertWordToIds(word);
                if (!word_ids.empty()) {
                    token_ids.insert(token_ids.end(), word_ids.begin(), word_ids.end());
                }
            }
        } else {
            // For other languages, would need espeak-ng or similar
            // For now, just do character-based lookup
            for (char c : text) {
                if (c == ' ') {
                    auto it = token_to_id_.find(" ");
                    if (it != token_to_id_.end()) {
                        token_ids.push_back(it->second);
                    }
                } else {
                    std::string char_str(1, c);
                    auto it = token_to_id_.find(char_str);
                    if (it != token_to_id_.end()) {
                        token_ids.push_back(it->second);
                    }
                }
            }
        }
        
        
        return token_ids;
    }
    
    bool isPunctuation(const std::string& s) {
        static const std::unordered_set<std::string> puncts = {
            ",", ".", "!", "?", ":", "\"", "'", "，",
            "。", "！", "？", """, """, "'", "'", "；", "、"
        };
        return puncts.count(s);
    }
    
    std::vector<int64_t> convertWordToIds(const std::string& word) {
        // Convert word to lowercase for lookup (following sherpa-onnx)
        std::string lower_word = word;
        std::transform(lower_word.begin(), lower_word.end(), lower_word.begin(), ::tolower);
        
        // Try direct word lookup in lexicon first
        auto lex_it = lexicon_.find(lower_word);
        if (lex_it != lexicon_.end()) {
            return convertPhonemesToIds(lex_it->second);
        }
        
        // Try direct token lookup
        auto token_it = token_to_id_.find(word);
        if (token_it != token_to_id_.end()) {
            return {token_it->second};
        }
        
        // Handle punctuation mapping
        if (isPunctuation(word)) {
            // Map punctuations to their token equivalents
            std::string punct_token = mapPunctuation(word);
            if (!punct_token.empty()) {
                auto punct_it = token_to_id_.find(punct_token);
                if (punct_it != token_to_id_.end()) {
                    return {punct_it->second};
                }
            }
        }
        
        // Character-level fallback for OOV words
        std::vector<int64_t> result;
        std::vector<std::string> chars = splitUtf8(word);
        
        for (const auto& char_str : chars) {
            auto char_lex_it = lexicon_.find(char_str);
            if (char_lex_it != lexicon_.end()) {
                auto char_ids = convertPhonemesToIds(char_lex_it->second);
                result.insert(result.end(), char_ids.begin(), char_ids.end());
            } else {
            }
        }
        
        return result;
    }
    
    std::vector<int64_t> convertPhonemesToIds(const std::string& phonemes) {
        std::vector<int64_t> ids;
        std::istringstream iss(phonemes);
        std::string phone;
        
        while (iss >> phone) {
            auto token_it = token_to_id_.find(phone);
            if (token_it != token_to_id_.end()) {
                ids.push_back(token_it->second);
            } else {
                // Try fallback mappings
                std::string mapped_phone = mapPhoneme(phone);
                if (mapped_phone != phone) {
                    auto mapped_it = token_to_id_.find(mapped_phone);
                    if (mapped_it != token_to_id_.end()) {
                        ids.push_back(mapped_it->second);
                    }
                }
            }
        }
        
        return ids;
    }
    
    std::vector<std::string> splitUtf8(const std::string& str) {
        std::vector<std::string> result;
        for (size_t i = 0; i < str.length(); ) {
            int char_len = 1;
            unsigned char ch = str[i];
            if ((ch & 0x80) == 0) char_len = 1;
            else if ((ch & 0xE0) == 0xC0) char_len = 2;
            else if ((ch & 0xF0) == 0xE0) char_len = 3;
            else if ((ch & 0xF8) == 0xF0) char_len = 4;
            
            if (i + char_len <= str.length()) {
                result.push_back(str.substr(i, char_len));
            }
            i += char_len;
        }
        return result;
    }
    
    std::string mapPunctuation(const std::string& punct) {
        // Try to find the punctuation directly in tokens first
        auto direct_it = token_to_id_.find(punct);
        if (direct_it != token_to_id_.end()) {
            return punct;
        }
        
        // Simple ASCII punctuation mappings
        if (punct == "！") return "!";
        if (punct == "？") return "?";
        if (punct == "，") return ",";
        if (punct == "。") return ".";
        if (punct == "：") return ":";
        if (punct == "；") return ";";
        if (punct == "、") return ",";
        
        // Try to find common pause tokens for major punctuations
        if (punct == "。" || punct == "！" || punct == "？") {
            if (token_to_id_.count("sil")) return "sil";
            if (token_to_id_.count("sp")) return "sp";
            if (token_to_id_.count("<eps>")) return "<eps>";
        }
        
        return "";  // No mapping found
    }
    
    std::vector<int64_t> addBlankTokens(const std::vector<int64_t>& tokens) {
        // Matcha models expect blank tokens between phonemes
        // Use pad_id from model metadata (following sherpa-onnx approach)
        std::vector<int64_t> result(tokens.size() * 2 + 1, pad_id_);
        
        int32_t i = 1;
        for (auto token : tokens) {
            result[i] = token;
            i += 2;
        }
        
        return result;
    }
    
    std::vector<float> runAcousticModel(const std::vector<int64_t>& tokens, int speaker_id, float speed) {
        // Prepare inputs according to actual model signature
        std::vector<int64_t> token_shape = {1, static_cast<int64_t>(tokens.size())};
        std::vector<int64_t> length_data = {static_cast<int64_t>(tokens.size())};
        std::vector<int64_t> length_shape = {1};
        std::vector<float> noise_scale_data = {config_.noise_scale};
        std::vector<int64_t> noise_scale_shape = {1};
        std::vector<float> length_scale_data = {speed * config_.length_scale};
        std::vector<int64_t> length_scale_shape = {1};
        
        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        
        // Create input tensors according to model signature
        std::vector<Ort::Value> input_tensors;
        
        // Input 0: x (tokens)
        input_tensors.emplace_back(Ort::Value::CreateTensor<int64_t>(
            memory_info, const_cast<int64_t*>(tokens.data()), tokens.size(),
            token_shape.data(), token_shape.size()
        ));
        
        // Input 1: x_length
        input_tensors.emplace_back(Ort::Value::CreateTensor<int64_t>(
            memory_info, length_data.data(), 1,
            length_shape.data(), length_shape.size()
        ));
        
        // Input 2: noise_scale
        input_tensors.emplace_back(Ort::Value::CreateTensor<float>(
            memory_info, noise_scale_data.data(), 1,
            noise_scale_shape.data(), noise_scale_shape.size()
        ));
        
        // Input 3: length_scale
        input_tensors.emplace_back(Ort::Value::CreateTensor<float>(
            memory_info, length_scale_data.data(), 1,
            length_scale_shape.data(), length_scale_shape.size()
        ));
        
        // Get actual input names from the model
        Ort::AllocatorWithDefaultOptions allocator;
        std::vector<const char*> input_names;
        std::vector<const char*> output_names;
        
        
        // Use correct input names based on model signature
        const char* model_input_names[] = {"x", "x_length", "noise_scale", "length_scale"};
        const char* model_output_names[] = {"mel"};
        size_t num_inputs = 4;
        
        auto output_tensors = acoustic_model_->Run(
            Ort::RunOptions{nullptr},
            model_input_names, input_tensors.data(), num_inputs,
            model_output_names, 1
        );
        
        // Extract mel spectrogram
        float* mel_data = output_tensors[0].GetTensorMutableData<float>();
        auto mel_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
        size_t mel_size = std::accumulate(mel_shape.begin(), mel_shape.end(),
                                         1, std::multiplies<size_t>());
        
        return std::vector<float>(mel_data, mel_data + mel_size);
    }
    
private:
    TTSConfig config_;
    bool initialized_ = false;
    
    // ONNX Runtime
    std::unique_ptr<Ort::Env> env_;
    std::unique_ptr<Ort::Session> acoustic_model_;
    std::unique_ptr<Ort::Session> vocoder_model_;
    
    // Text processing
    std::unique_ptr<cppjieba::Jieba> jieba_;
    std::unordered_map<std::string, int64_t> token_to_id_;
    std::unordered_map<std::string, std::string> lexicon_;
    
    // Model info
    int mel_dim_ = 80;
    int num_speakers_ = 1;
    int64_t pad_id_ = 0;
    
    // ISTFT parameters from vocoder metadata
    int32_t istft_n_fft_ = 1024;
    int32_t istft_hop_length_ = 256;
    int32_t istft_win_length_ = 1024;
    bool istft_center_ = true;
    bool istft_normalized_ = false;
    std::string istft_window_type_ = "hann";
    std::string istft_pad_mode_ = "reflect";
};

// TTSModel public interface
TTSModel::TTSModel(const TTSConfig& config) 
    : pImpl(std::make_unique<Impl>(config)) {
}

TTSModel::~TTSModel() = default;

bool TTSModel::initialize() {
    return pImpl->initialize();
}

GeneratedAudio TTSModel::generate(const std::string& text) {
    return pImpl->generate(text, 0, 1.0f);  // Default speaker ID 0
}

GeneratedAudio TTSModel::generate(const std::string& text, int speaker_id) {
    return pImpl->generate(text, speaker_id, 1.0f);
}

GeneratedAudio TTSModel::generate(const std::string& text, int speaker_id, float speed) {
    return pImpl->generate(text, speaker_id, speed);
}

bool TTSModel::isInitialized() const {
    return pImpl->isInitialized();
}

int TTSModel::getNumSpeakers() const {
    return pImpl->getNumSpeakers();
}

int TTSModel::getSampleRate() const {
    return pImpl->getSampleRate();
}

} // namespace tts