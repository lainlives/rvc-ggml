// audio_io.h
// WAV audio file read/write wrapper based on dr_wav.h
// Dependency: dr_wav.h (under the include directory)

#pragma once

#include <vector>
#include <string>
#include <cstdint>

struct AudioData {
    std::vector<float> samples;    // interleaved storage, length = num_channels * total_samples
    int sample_rate = 0;
    int num_channels = 0;
    uint64_t total_samples = 0;    // number of samples per channel

    size_t total_frames() const { return static_cast<size_t>(total_samples); }
    size_t size_bytes() const { return samples.size() * sizeof(float); }
};

// Load audio from a WAV file (supports int16/int24/int32/float32, converted to float automatically); throws std::runtime_error on failure
AudioData load_wav(const std::string& filepath);

// Save audio to a WAV file (16-bit PCM); returns success
bool save_wav(const std::string& filepath,
    const float* samples,
    int num_channels,
    int sample_rate,
    uint64_t total_samples);

// Save int16 PCM as WAV without modification (for pipeline output, avoiding double quantization)
bool save_wav_i16(const std::string& filepath,
    const int16_t* samples,
    int num_channels,
    int sample_rate,
    uint64_t total_samples);

// Linear-interpolation resampling (keeps the input channel count)
AudioData resample(const AudioData& audio, int target_sr);

// Mono 16k conversion: downmix + polyphase FIR resampling (quality on par with torchaudio kaiser_best)
std::vector<float> to_mono_16k(const AudioData& audio, int target_sr);

// Mono resampling (arbitrary sample rate, polyphase FIR)
std::vector<float> resample_mono(const std::vector<float>& samples, int src_sr, int dst_sr);
