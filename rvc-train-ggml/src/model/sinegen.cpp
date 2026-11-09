// RVC-Train-GGML - SineGen implementation (src/model/sinegen.cpp)
//
// Algorithm (harmonic_num=0, formula-by-formula identical to RVC-GGML src/rvc/sinegen.cpp):
//   rad = (max(f0,0)/sr) % 1 (per frame)
//   cumulative = cumsum(rad) * upp, linearly interpolated per-sample with align_corners=True
//   wrap detection: where interp%1 decreases consecutively, phase -= 1 (equivalent to Python cumsum_shift)
//   sine = sin(phase)·0.1; uv = f0 > 0; noise_amp = uv·0.003 + (1-uv)·0.1/3
//   out = uv ? (sine + noise_amp·noise) : noise_amp·noise
#include "model/sinegen.h"

#include <algorithm>
#include <cmath>
#include <random>

namespace rvc_train {
namespace model {

namespace {

constexpr float kSineAmp = 0.1f;
constexpr float kNoiseStd = 0.003f;
constexpr double kTwoPi = 6.283185307179586476925286766559;

} // namespace

std::vector<float> sinegen_generate(const float* f0, int64_t frames, int sample_rate, int upp,
                                    const float* noise, int64_t noise_len, uint32_t seed) {
    if (f0 == nullptr || frames <= 0 || upp <= 0 || sample_rate <= 0) {
        return {};
    }
    const int64_t total = frames * static_cast<int64_t>(upp);
    std::vector<float> out(static_cast<size_t>(total), 0.0f);

    std::vector<float> noise_buf;
    const float* noise_data = nullptr;
    if (noise != nullptr && noise_len >= total) {
        noise_data = noise;
    } else {
        noise_buf.resize(static_cast<size_t>(total));
        std::mt19937 rng(seed);
        std::normal_distribution<float> dist(0.0f, 1.0f);
        for (float& value : noise_buf) {
            value = dist(rng);
        }
        noise_data = noise_buf.data();
    }

    // Per-frame phase increment and accumulation (cumsum * upp)
    std::vector<double> cumulative(static_cast<size_t>(frames), 0.0);
    double running = 0.0;
    for (int64_t t = 0; t < frames; ++t) {
        const float rad = std::fmod(std::max(0.0f, f0[t]) / static_cast<float>(sample_rate), 1.0f);
        running += static_cast<double>(rad);
        cumulative[static_cast<size_t>(t)] = running * static_cast<double>(upp);
    }

    // align_corners=True linear interpolation: pos = i * (frames-1) / (total-1)
    auto interpolate = [&](int64_t i) -> double {
        if (total <= 1 || frames <= 1) {
            return cumulative[0];
        }
        const double pos = static_cast<double>(i) * static_cast<double>(frames - 1) /
                           static_cast<double>(total - 1);
        const int64_t left = static_cast<int64_t>(std::floor(pos));
        const int64_t right = std::min<int64_t>(frames - 1, left + 1);
        const double frac = pos - static_cast<double>(left);
        return cumulative[static_cast<size_t>(left)] * (1.0 - frac) +
               cumulative[static_cast<size_t>(right)] * frac;
    };

    double phase = 0.0;
    double prev_mod = 0.0;
    for (int64_t i = 0; i < total; ++i) {
        const int64_t source_frame = std::min<int64_t>(frames - 1, i / upp);
        const float base_rad =
            std::fmod(std::max(0.0f, f0[source_frame]) / static_cast<float>(sample_rate), 1.0f);
        const double interp = interpolate(i);
        const double mod = interp - std::floor(interp);
        if (i > 0 && (mod - prev_mod) < 0.0) {
            phase -= 1.0;    // phase wrap compensation (equivalent to Python cumsum_shift)
        }
        prev_mod = mod;
        phase += static_cast<double>(base_rad);

        const bool voiced = f0[source_frame] > 0.0f;
        const float noise_amp = voiced ? kNoiseStd : kSineAmp / 3.0f;
        const float sine = static_cast<float>(std::sin(phase * kTwoPi)) * kSineAmp;
        const float noise_value = noise_amp * noise_data[static_cast<size_t>(i)];
        out[static_cast<size_t>(i)] = voiced ? (sine + noise_value) : noise_value;
    }
    return out;
}

} // namespace model
} // namespace rvc_train
