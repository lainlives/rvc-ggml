// audio_utils.h — WAV loading/saving, resampling and mixing for rvc-ui.
#pragma once

#include <cmath>
#include <cstddef>
#include <string>
#include <utility>
#include <vector>

namespace audio {

struct Buffer {
    std::vector<float> data;  // interleaved
    int channels = 0;
    int sample_rate = 0;

    size_t frames() const {
        return channels > 0 ? data.size() / static_cast<size_t>(channels) : 0;
    }
};

// Load a WAV file (any dr_wav-supported PCM/float depth) as float32, native rate.
Buffer load_wav(const std::string& path);

// Save interleaved float32 WAV.
void save_wav(const std::string& path, const Buffer& buffer);

// Windowed-sinc resampling (same implementation as rvc-ggml/rvc-sep).
Buffer resample(const Buffer& in, int target_sample_rate);

// Convert any channel count to stereo (mono duplicated, >2 averaged).
Buffer to_stereo(const Buffer& in);

// a - b (sample-aligned, min length); sample rates/channels must match `a`.
Buffer subtract(const Buffer& a, const Buffer& b);

// Sum tracks (already at the same sample rate/channels) with linear gains;
// scales down evenly when the result would clip.
Buffer mix(const std::vector<std::pair<Buffer, float>>& tracks);

inline float db_to_linear(float db) {
    return std::pow(10.0f, db / 20.0f);
}

} // namespace audio
