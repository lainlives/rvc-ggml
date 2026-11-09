// load.h -- M4-1 audio loading and resampling (rvc-prep side)
//
// Semantic baseline (official 81eed5e train/preprocess.py:90 -> infer/audio.py:310-335, RVC_AUDIO_FORCE_CPU=1):
//   ffmpeg -f f32le -ac 1 -ar <target_sr> decode + downmix + resample.
// C++ side: dr_wav reads WAV (WAV only; users must convert non-WAV first) -> downmix (stereo 1/sqrt(2)) -> in-house Kaiser-window sinc
// resampling (not sample-exact vs ffmpeg swresample -> G4 statistical gate; see docs/handover.md §4.8 D/G).
//
// Precision chain: this file outputs double (matching the official lfilter's float64 arithmetic).

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace rvc_train::prep {

// Mono audio (double samples)
struct MonoAudio {
    std::vector<double> samples;
    int sample_rate = 0;
};

// WAV -> mono (channel average) -> resample to target_sr; returns false and fills err on failure
bool load_wav_mono(const std::string& path, int target_sr, MonoAudio* out, std::string* err);

// Raw F32 little-endian mono file (injection mode: golden post-load waveform)
bool load_raw_f32_mono(const std::string& path, int sample_rate, MonoAudio* out, std::string* err);

// Raw F32 dump (little-endian, contiguous)
bool write_raw_f32(const std::string& path, const float* data, std::size_t count, std::string* err);
bool write_raw_f32(const std::string& path, const std::vector<float>& data, std::string* err);

// Raw I32 dump (little-endian, contiguous; M4-2 f0bin)
bool write_raw_i32(const std::string& path, const int32_t* data, std::size_t count, std::string* err);
bool write_raw_i32(const std::string& path, const std::vector<int32_t>& data, std::string* err);

// double -> float (narrowed to F32, round to nearest)
std::vector<float> to_f32(const std::vector<double>& v);

// Resample (double wrapper around the in-house sinc resampler; called only when src != dst)
std::vector<double> resample_mono_double(const std::vector<double>& in, int src_sr, int dst_sr);

} // namespace rvc_train::prep
