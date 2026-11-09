// slicer.cpp -- line-by-line replica of the official slicer2.py

#include "prep/slicer.h"

#include <algorithm>
#include <cfenv>
#include <cmath>
#include <cstddef>

namespace rvc_train::prep {

double round_half_even(double v) {
    // std::nearbyint uses the current rounding mode (default FE_TONEAREST = half-even, matching Python round)
    return std::nearbyint(v);
}

std::vector<double> get_rms(const std::vector<double>& samples, int frame_length, int hop_length) {
    std::vector<double> out;
    if (frame_length <= 0 || hop_length <= 0) {
        return out;
    }
    const int64_t n = static_cast<int64_t>(samples.size());
    const int64_t half = frame_length / 2;
    const int64_t pad = 2 * half;
    const int64_t n_win = n + pad - frame_length + 1;  // number of as_strided windows
    if (n_win <= 0) {
        return out;
    }
    const int64_t count = (n_win - 1) / hop_length + 1;  // xw[::hop] frame count
    out.resize(static_cast<std::size_t>(count));

    for (int64_t i = 0; i < count; ++i) {
        const int64_t start = i * hop_length;  // window start (coordinates after zero padding)
        double sum = 0.0;
        for (int64_t j = 0; j < frame_length; ++j) {
            const int64_t idx = start + j - half;  // map back to the original waveform
            if (idx >= 0 && idx < n) {
                const double v = samples[static_cast<std::size_t>(idx)];
                sum += v * v;
            }
        }
        out[static_cast<std::size_t>(i)] = std::sqrt(sum / static_cast<double>(frame_length));
    }
    return out;
}

namespace {

// np.argmin: return the index of the first occurrence of the minimum within [begin, end) (range must be non-empty)
int64_t argmin_range(const std::vector<double>& v, int64_t begin, int64_t end) {
    int64_t best = begin;
    for (int64_t i = begin + 1; i < end; ++i) {
        if (v[static_cast<std::size_t>(i)] < v[static_cast<std::size_t>(best)]) {
            best = i;
        }
    }
    return best;
}

} // namespace

Slicer::Slicer(int sample_rate, const SlicerParams& params) : sample_rate_(sample_rate) {
    // corresponds to slicer2.py Slicer.__init__ (operation order preserved)
    double min_interval = sample_rate * params.min_interval_ms / 1000.0;
    derived_.threshold = std::pow(10.0, params.threshold_db / 20.0);
    derived_.hop_size = static_cast<int>(round_half_even(sample_rate * params.hop_size_ms / 1000.0));
    const double min_interval_rounded = round_half_even(min_interval);
    derived_.win_size = std::min(static_cast<int>(min_interval_rounded), 4 * derived_.hop_size);
    derived_.min_length = static_cast<int>(
        round_half_even(sample_rate * params.min_length_ms / 1000.0 / derived_.hop_size));
    derived_.min_interval = static_cast<int>(round_half_even(min_interval / derived_.hop_size));
    derived_.max_sil_kept = static_cast<int>(
        round_half_even(sample_rate * params.max_sil_kept_ms / 1000.0 / derived_.hop_size));
}

std::vector<SliceRange> Slicer::slice(const std::vector<double>& samples) const {
    const int64_t n = static_cast<int64_t>(samples.size());

    // official quirk: sample count vs min_length (frame count)
    if (n <= derived_.min_length) {
        return {SliceRange{0, n}};
    }

    const std::vector<double> rms = get_rms(samples, derived_.win_size, derived_.hop_size);
    const int64_t total_frames = static_cast<int64_t>(rms.size());
    last_rms_ = rms;
    last_sil_tags_.clear();
    last_debug_.clear();

    std::vector<std::pair<int64_t, int64_t>> sil_tags;
    int64_t silence_start = -1;
    int64_t clip_start = 0;

    for (int64_t i = 0; i < total_frames; ++i) {
        const double r = rms[static_cast<std::size_t>(i)];
        if (r < derived_.threshold) {
            if (silence_start < 0) {
                silence_start = i;
            }
            continue;
        }
        if (silence_start < 0) {
            continue;
        }
        const bool is_leading_silence = (silence_start == 0) && (i > derived_.max_sil_kept);
        const bool need_slice_middle = (i - silence_start >= derived_.min_interval) &&
                                       (i - clip_start >= derived_.min_length);
        if (!is_leading_silence && !need_slice_middle) {
            silence_start = -1;
            continue;
        }
        last_debug_ += "event i=" + std::to_string(i) + " ss=" + std::to_string(silence_start) +
                       " lead=" + (is_leading_silence ? "1" : "0") +
                       " need=" + (need_slice_middle ? "1" : "0");
        if (i - silence_start <= derived_.max_sil_kept) {
            const int64_t pos = argmin_range(rms, silence_start, i + 1);  // absolute index
            if (silence_start == 0) {
                sil_tags.emplace_back(0, pos);
            } else {
                sil_tags.emplace_back(pos, pos);
            }
            clip_start = pos;
        } else if (i - silence_start <= derived_.max_sil_kept * 2) {
            const int64_t pos =
                argmin_range(rms, i - derived_.max_sil_kept, silence_start + derived_.max_sil_kept + 1);
            const int64_t pos_l =
                argmin_range(rms, silence_start, silence_start + derived_.max_sil_kept + 1);
            const int64_t pos_r = argmin_range(rms, i - derived_.max_sil_kept, i + 1);
            if (silence_start == 0) {
                sil_tags.emplace_back(0, pos_r);
                clip_start = pos_r;
            } else {
                sil_tags.emplace_back(std::min(pos_l, pos), std::max(pos_r, pos));
                clip_start = std::max(pos_r, pos);
            }
        } else {
            const int64_t pos_l =
                argmin_range(rms, silence_start, silence_start + derived_.max_sil_kept + 1);
            const int64_t pos_r = argmin_range(rms, i - derived_.max_sil_kept, i + 1);
            if (silence_start == 0) {
                sil_tags.emplace_back(0, pos_r);
            } else {
                sil_tags.emplace_back(pos_l, pos_r);
            }
            clip_start = pos_r;
        }
        if (!sil_tags.empty()) {
            last_debug_ += " tag=(" + std::to_string(sil_tags.back().first) + "," +
                           std::to_string(sil_tags.back().second) + ")\n";
        }
        silence_start = -1;
    }

    // trailing silence
    if (silence_start >= 0 && total_frames - silence_start >= derived_.min_interval) {
        const int64_t silence_end =
            std::min<int64_t>(total_frames, silence_start + derived_.max_sil_kept);
        const int64_t pos = argmin_range(rms, silence_start, silence_end + 1);  // absolute index
        sil_tags.emplace_back(pos, total_frames + 1);
    }

    if (sil_tags.empty()) {
        last_sil_tags_ = sil_tags;
        return {SliceRange{0, n}};
    }

    // _apply_slice: sample domain [begin*hop, min(n, end*hop))
    std::vector<SliceRange> chunks;
    auto apply = [&](int64_t begin_frame, int64_t end_frame) {
        const int64_t bs = begin_frame * derived_.hop_size;
        int64_t es = std::min<int64_t>(n, end_frame * derived_.hop_size);
        if (es < bs) {
            es = bs;  // official returns an empty array for an empty slice; equivalent to an empty range here
        }
        chunks.push_back(SliceRange{bs, es});
    };
    if (sil_tags.front().first > 0) {
        apply(0, sil_tags.front().first);
    }
    for (std::size_t i = 0; i + 1 < sil_tags.size(); ++i) {
        apply(sil_tags[i].second, sil_tags[i + 1].first);
    }
    if (sil_tags.back().second < total_frames) {
        apply(sil_tags.back().second, total_frames);
    }
    last_sil_tags_ = sil_tags;
    return chunks;
}

} // namespace rvc_train::prep
