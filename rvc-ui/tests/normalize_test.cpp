// normalize_test.cpp — regression test for the mix peak-normalisation policy.
//
// Policy: scale to -1 dBFS (0.891) when the peak would clip (>= 0 dBFS) or is
// unusually quiet (< -5 dBFS); otherwise leave the buffer untouched.
#include "audio_utils.h"

#include <cmath>
#include <cstdio>

namespace {

bool check(const char* label, float peak, float expect_gain_db, float expect_out) {
    audio::Buffer buffer;
    buffer.channels = 1;
    buffer.sample_rate = 44100;
    buffer.data.assign(1000, peak);
    const float gain_db = audio::apply_peak_normalization(buffer);
    const float out = buffer.data[0];
    const bool ok = std::fabs(gain_db - expect_gain_db) < 0.01f && std::fabs(out - expect_out) < 0.002f;
    std::printf("%-22s peak %.3f -> gain %+6.2f dB -> %.3f %s\n",
                label, peak, gain_db, out, ok ? "ok" : "FAIL");
    return ok;
}

} // namespace

int main() {
    bool ok = true;
    ok = check("clipping mix", 1.6f, -5.08f, 0.891f) && ok;
    ok = check("quiet mix (-12 dB)", 0.25f, +11.04f, 0.891f) && ok;
    ok = check("quiet mix (-6 dB)", 0.5f, +5.02f, 0.891f) && ok;
    ok = check("normal mix (-3 dB)", 0.707f, 0.0f, 0.707f) && ok;
    ok = check("valid mix (-0.5 dB)", 0.944f, 0.0f, 0.944f) && ok;
    ok = check("silent", 0.0f, 0.0f, 0.0f) && ok;
    return ok ? 0 : 1;
}
