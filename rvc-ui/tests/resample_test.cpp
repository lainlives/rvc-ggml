// resample_test.cpp — regression test for the windowed-sinc resampler.
//
// Guards against the historical phase-table implementation, whose per-phase
// normalization degenerated for ratios such as 32000 -> 44100 (sign-flipped
// spikes: a 440 Hz sine came out with peak ~24 instead of ~1.0).
#include "audio_utils.h"

#include <cmath>
#include <cstdio>
#include <utility>

namespace {

bool check_case(int sr_in, int sr_out) {
    const double freq = 440.0;
    const size_t n = static_cast<size_t>(sr_in);  // 1 second
    audio::Buffer in;
    in.channels = 1;
    in.sample_rate = sr_in;
    in.data.resize(n);
    for (size_t i = 0; i < n; ++i) {
        in.data[i] = static_cast<float>(std::sin(2.0 * M_PI * freq * static_cast<double>(i) / sr_in));
    }
    const audio::Buffer out = audio::resample(in, sr_out);

    float peak = 0.0f;
    double sumsq = 0.0;
    for (float v : out.data) {
        peak = std::max(peak, std::fabs(v));
        sumsq += static_cast<double>(v) * v;
    }
    const double rms = std::sqrt(sumsq / static_cast<double>(out.data.size()));
    const bool ok = std::fabs(peak - 1.0) < 0.05 && std::fabs(rms - 0.7071) < 0.03;
    std::printf("%5d -> %5d: peak=%.4f rms=%.4f %s\n", sr_in, sr_out, peak, rms, ok ? "ok" : "FAIL");
    return ok;
}

} // namespace

int main() {
    const std::pair<int, int> cases[] = {
        {32000, 44100}, {32000, 16000}, {32000, 48000}, {22050, 44100}, {44100, 16000},
        {40000, 44100}, {48000, 44100}, {16000, 44100}, {44100, 44100},
    };
    bool ok = true;
    for (const auto& [sr_in, sr_out] : cases) {
        ok = check_case(sr_in, sr_out) && ok;  // keep checking all cases
    }
    return ok ? 0 : 1;
}
