// slicer.h -- bit-semantics replica of the official Slicer2 (M4-1)
//
// Semantic baseline: official 81eed5e train/dataset/slicer2.py (line-by-line replica of the whole file) and train/preprocess.py:40-47:
//   Slicer(sr=sr, threshold=-42, min_length=1500, min_interval=400, hop_size=15, max_sil_kept=500)
// Replication points (must not be changed):
//   - Python round() = banker's rounding (half-even) -> use std::nearbyint (FE_TONEAREST);
//   - Slicer.min_length/min_interval/max_sil_kept are frame counts internally;
//   - the odd `samples.shape[0] <= min_length` comparison (sample count vs frame count) is replicated as-is;
//   - get_rms: frame_length=win_size, hop=hop_size, frame_length//2 zero padding on each side, square root of the mean square;
//   - argmin takes the first occurrence of the minimum (np.argmin semantics).

#pragma once

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace rvc_train::prep {

// Millisecond parameters matching the official code (threshold dB supplied separately)
struct SlicerParams {
    double threshold_db = -42.0;
    double min_length_ms = 1500.0;
    double min_interval_ms = 400.0;
    double hop_size_ms = 15.0;
    double max_sil_kept_ms = 500.0;
};

// Frame-count parameters derived in the constructor (for cross-checks/docs calibration)
struct SlicerDerived {
    double threshold = 0.0;  // 10^(threshold_db/20)
    int hop_size = 0;        // hop size (samples)
    int win_size = 0;        // RMS window length (samples)
    int min_length = 0;      // frames
    int min_interval = 0;    // frames
    int max_sil_kept = 0;    // frames
};

// Slice range (sample domain [begin_sample, end_sample))
struct SliceRange {
    int64_t begin_sample = 0;
    int64_t end_sample = 0;
};

// Python round() half-even semantics
double round_half_even(double v);

// librosa-style frame RMS (frame_length/hop_length in samples; frame_length/2 zero padding on each side)
std::vector<double> get_rms(const std::vector<double>& samples, int frame_length, int hop_length);

class Slicer {
public:
    Slicer(int sample_rate, const SlicerParams& params);
    const SlicerDerived& derived() const { return derived_; }
    int sample_rate() const { return sample_rate_; }

    // Input: mono double waveform; output: slices (sample domain), ordered like the chunks returned by the official slice()
    std::vector<SliceRange> slice(const std::vector<double>& samples) const;

    // Debug: sil_tags from the last slice() (frame domain; for cross-checks/troubleshooting)
    const std::vector<std::pair<int64_t, int64_t>>& last_sil_tags() const { return last_sil_tags_; }
    const std::vector<double>& last_rms() const { return last_rms_; }
    const std::string& last_debug() const { return last_debug_; }

private:
    int sample_rate_ = 0;
    SlicerDerived derived_;
    mutable std::vector<std::pair<int64_t, int64_t>> last_sil_tags_;
    mutable std::vector<double> last_rms_;
    mutable std::string last_debug_;
};

} // namespace rvc_train::prep
