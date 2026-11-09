// dsp.h
// RVC audio signal processing: high-pass filtering, RMS mixing, silent splitting, reflection pad, coarse F0 quantization.
// Parameters align with the Python reference implementation (scipy/librosa/numpy).

#pragma once

#include <vector>
#include <cstdint>

namespace rvc {
namespace dsp {

    // 5th-order Butterworth 48Hz high-pass (equivalent to scipy.signal.filtfilt at 16k sample rate)
    // Input and output are mono 16k float sequences; throws when the input is too short
    void high_pass_48hz_in_place(std::vector<float>& samples);

    // Reflection pad (numpy.pad mode='reflect'): left/right samples on each side
    std::vector<float> reflect_pad(const std::vector<float>& samples, int64_t left, int64_t right);

    // Frame RMS (aligned with librosa.feature.rms: frame_length = sr/2*2, hop = sr/2, centered, reflect padding)
    std::vector<float> frame_rms(const std::vector<float>& audio, int sample_rate);

    // Linearly interpolate the RMS curve to output_frames length (aligned with torch F.interpolate mode='linear')
    std::vector<float> interpolate_linear(const std::vector<float>& values, int64_t output_frames);

    // RVC RMS mixing: converted *= rms_src^(1-rate) * rms_out^(rate-1)
    // source_16k: original 16k mono; converted: target-sample-rate output (modified in place)
    void apply_rms_mix(const std::vector<float>& source_16k,
        std::vector<float>& converted,
        int target_sample_rate,
        float rms_mix_rate);

    // audio_sum curve (used internally by silent splitting; exported for validation)
    std::vector<float> audio_sum_curve(const std::vector<float>& audio);

    // Long-audio silent-point splitting (aligned with Python Pipeline.pipeline)
    // Returns split sample positions (16k domain); empty if there are no split points
    std::vector<int64_t> quiet_split_points(const std::vector<float>& audio,
        int split_query_sec,
        int split_center_sec,
        int split_threshold_sec);

    // F0(Hz) -> RVC coarse pitch quantization (1~255)
    int coarse_pitch_bin(float f0_hz);

    // Median filter (radius)
    void median_filter_f0(std::vector<float>& f0, int radius);

    // Linear interpolation over unvoiced segments (interpolate from neighboring non-zero values where f0==0)
    void interpolate_unvoiced(std::vector<float>& f0);

} // namespace dsp
} // namespace rvc
