// spec.h -- M4-4 linear spectrogram (official spectrogram_torch semantics, pocketfft host side)
//
// Semantic baseline = 81eed5e train/mel_processing.py::spectrogram_torch (see docs/handover.md §4.8 A):
//   reflect pad `int((n_fft-hop)/2)` on both sides -> periodic Hann -> torch.stft(center=False, onesided) ->
//   `sqrt(re²+im²+2e-7)`; input = 0_gt F32 waveform (data_utils.get_audio does no normalization, no wav round-trip).
//
// Frame count (M4-4 calibration) = `1 + (padded - n_fft) // hop`, `padded = gt_samples + 2*int((n_fft-hop)/2)`;
// v1-40k (2048/400) measured = `gt_samples // hop` (the old doc formula `1+(gt-n_fft)//hop` is corrected).
//
// Output layout: row-major [bins, frames] (matching torch [F,T]), F32.

#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace rvc_train::prep {

struct SpecConfig {
    int n_fft = 2048;
    int hop = 400;
    int win = 2048;
    int sample_rate = 40000;

    int bins() const { return n_fft / 2 + 1; }
    // Official frame count (reflect pad + center=False; 0 when padded < n_fft)
    int64_t frames_for(int64_t gt_samples) const;
    bool validate(std::string* err) const;
};

struct SpecResult {
    int64_t frames = 0;
    int64_t bins = 0;
    std::vector<float> values;  // row-major [bins, frames]
};

// gt: F32 waveform (0_gt); on false return, err contains the reason
bool compute_linear_spec(const float* gt, int64_t n, const SpecConfig& cfg, SpecResult* out,
                         std::string* err);

} // namespace rvc_train::prep
