// sinegen.cpp
// Host-side SineGen implementation (aligned with RVC models.py):
//   rad_values = (f0/sr) % 1 (per frame, nearest-neighbor upsampled to per sample)
//   tmp_over_one = cumsum(rad) * upp (linear interpolation to per sample with align_corners=True; wraparound detection via %1)
//   sine = sin(cumsum(rad + shift)) * 0.1
//   uv = f0 > 0 (nearest-neighbor upsampling); noise_amp = uv*0.003 + (1-uv)*0.1/3
//   output = sine*uv + noise_amp*noise

#include "rvc/sinegen.h"

#include <algorithm>
#include <cmath>
#include <random>

namespace rvc {
namespace sinegen {

namespace {
    constexpr float kSineAmp = 0.1f;
    constexpr float kNoiseStd = 0.003f;
    constexpr double kTwoPi = 6.283185307179586476925286766559;
} // namespace

std::vector<float> generate(const std::vector<float>& f0, int sample_rate, int upp,
    const std::vector<float>* noise, uint32_t seed) {
    const int64_t frames = static_cast<int64_t>(f0.size());
    const int64_t total = frames * static_cast<int64_t>(upp);
    std::vector<float> out(static_cast<size_t>(std::max<int64_t>(total, 0)), 0.0f);
    if (frames <= 0 || upp <= 0) {
        return out;
    }

    std::vector<float> noise_buf;
    const float* noise_data = nullptr;
    if (noise != nullptr && static_cast<int64_t>(noise->size()) >= total) {
        noise_data = noise->data();
    }
    else {
        noise_buf.resize(static_cast<size_t>(total));
        std::mt19937 rng(seed);
        std::normal_distribution<float> dist(0.0f, 1.0f);
        for (auto& value : noise_buf) {
            value = dist(rng);
        }
        noise_data = noise_buf.data();
    }

    // Per-frame phase increment and accumulation (cumsum * upp)
    std::vector<double> cumulative(static_cast<size_t>(frames), 0.0);
    double running = 0.0;
    for (int64_t t = 0; t < frames; ++t) {
        const float rad = std::fmod(std::max(0.0f, f0[static_cast<size_t>(t)]) /
            static_cast<float>(sample_rate), 1.0f);
        running += static_cast<double>(rad);
        cumulative[static_cast<size_t>(t)] = running * static_cast<double>(upp);
    }

    // align_corners=True linear interpolation: pos = i * (frames-1) / (total-1)
    auto interpolate = [&](int64_t i) -> double {
        if (total <= 1 || frames <= 1) {
            return cumulative[0];
        }
        const double pos = static_cast<double>(i) *
            static_cast<double>(frames - 1) / static_cast<double>(total - 1);
        const int64_t left = static_cast<int64_t>(std::floor(pos));
        const int64_t right = std::min<int64_t>(frames - 1, left + 1);
        const double frac = pos - static_cast<double>(left);
        return cumulative[static_cast<size_t>(left)] * (1.0 - frac) +
            cumulative[static_cast<size_t>(right)] * frac;
    };

    double phase = 0.0;      // cumsum(rad + cumsum_shift)
    double prev_mod = 0.0;
    for (int64_t i = 0; i < total; ++i) {
        const int64_t source_frame = std::min<int64_t>(frames - 1, i / upp);
        const float base_rad = std::fmod(
            std::max(0.0f, f0[static_cast<size_t>(source_frame)]) /
            static_cast<float>(sample_rate), 1.0f);
        const double interp = interpolate(i);
        const double mod = interp - std::floor(interp);
        if (i > 0 && (mod - prev_mod) < 0.0) {
            phase -= 1.0;    // Phase wraparound compensation (equivalent to cumsum_shift in Python)
        }
        prev_mod = mod;
        phase += static_cast<double>(base_rad);

        const bool voiced = f0[static_cast<size_t>(source_frame)] > 0.0f;
        const float noise_amp = voiced ? kNoiseStd : kSineAmp / 3.0f;
        const float sine = static_cast<float>(
            std::sin(phase * kTwoPi)) * kSineAmp;
        const float noise_value = noise_amp * noise_data[static_cast<size_t>(i)];
        out[static_cast<size_t>(i)] = voiced ? (sine + noise_value) : noise_value;
    }
    return out;
}

} // namespace sinegen
} // namespace rvc
