// audio_utils.h — audio file I/O, resampling and mixing for rvc-ui.
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

// Load an audio file as float32 at its native sample rate and channel count.
// Decodes WAV / FLAC / MP3 / Ogg Vorbis through miniaudio.
Buffer load_audio(const std::string& path);

// Save interleaved float32 WAV.
void save_wav(const std::string& path, const Buffer& buffer);

// Windowed-sinc resampling (same implementation as rvc-ggml/rvc-sep).
Buffer resample(const Buffer& in, int target_sample_rate);

// Convert any channel count to stereo (mono duplicated, >2 averaged).
Buffer to_stereo(const Buffer& in);

// a - b (sample-aligned, min length); sample rates/channels must match `a`.
Buffer subtract(const Buffer& a, const Buffer& b);

// Sum tracks (already at the same sample rate/channels) with linear gains.
// No clipping guard: apply `apply_peak_normalization` on the result.
Buffer mix(const std::vector<std::pair<Buffer, float>>& tracks);

// Peak-normalisation policy: scale to `target_db` (-1 dBFS by default) when the
// peak would clip (>= 0 dBFS) or is unusually quiet (< quiet_db); otherwise the
// buffer is left untouched. Returns the applied gain in dB (0 when unchanged).
float apply_peak_normalization(Buffer& buffer, float quiet_db = -5.0f, float target_db = -1.0f);

inline float db_to_linear(float db) {
    return std::pow(10.0f, db / 20.0f);
}

} // namespace audio
