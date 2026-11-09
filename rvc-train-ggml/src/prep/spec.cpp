// spec.cpp -- M4-4 linear spectrogram implementation (pocketfft, host side; semantics in the spec.h header comment)

#include "prep/spec.h"

#include <cmath>
#include <complex>

#define POCKETFFT_CACHE_SIZE 16
#if defined(_MSC_VER)
#pragma warning(push, 0)  // third-party header (pocketfft) emits /W4 warnings of its own, not this project's
#endif
#include "pocketfft_hdronly.h"
#if defined(_MSC_VER)
#pragma warning(pop)
#endif

namespace rvc_train::prep {
namespace {

constexpr double kMagEps = 2e-7;  // official sqrt(re²+im²+2e-7)
constexpr double kPi = 3.14159265358979323846;

} // namespace

int64_t SpecConfig::frames_for(int64_t gt_samples) const {
    // Python: int((n_fft-hop)/2) (truncation of a positive number); C++ positive-integer division matches
    const int64_t pad = (static_cast<int64_t>(n_fft) - hop) / 2;
    const int64_t padded = gt_samples + 2 * pad;
    if (padded < n_fft) {
        return 0;
    }
    return 1 + (padded - n_fft) / hop;
}

bool SpecConfig::validate(std::string* err) const {
    if (n_fft <= 0 || hop <= 0 || win <= 0) {
        if (err) {
            *err = "invalid spec config: n_fft/hop/win must be positive";
        }
        return false;
    }
    if (hop > n_fft) {
        if (err) {
            *err = "invalid spec config: hop > n_fft (official reflect pad semantics require hop<=n_fft)";
        }
        return false;
    }
    if (win > n_fft) {
        if (err) {
            *err = "invalid spec config: win > n_fft";
        }
        return false;
    }
    return true;
}

bool compute_linear_spec(const float* gt, int64_t n, const SpecConfig& cfg, SpecResult* out,
                         std::string* err) {
    out->frames = 0;
    out->bins = 0;
    out->values.clear();

    if (!cfg.validate(err)) {
        return false;
    }
    if (gt == nullptr || n <= 0) {
        if (err) {
            *err = "spec input is empty";
        }
        return false;
    }

    const int64_t pad = (static_cast<int64_t>(cfg.n_fft) - cfg.hop) / 2;
    if (pad >= n) {
        if (err) {
            *err = "spec input too short: reflect pad exceeds waveform length";
        }
        return false;
    }

    // ---------- reflect pad (numpy mode='reflect': left x[pad-k], right x[n-2-j]) ----------
    const int64_t padded_len = n + 2 * pad;
    std::vector<double> padded(static_cast<std::size_t>(padded_len));
    for (int64_t k = 0; k < pad; ++k) {
        padded[static_cast<std::size_t>(k)] = static_cast<double>(gt[pad - k]);
    }
    for (int64_t i = 0; i < n; ++i) {
        padded[static_cast<std::size_t>(pad + i)] = static_cast<double>(gt[i]);
    }
    for (int64_t j = 0; j < pad; ++j) {
        padded[static_cast<std::size_t>(pad + n + j)] = static_cast<double>(gt[n - 2 - j]);
    }

    // ---------- periodic Hann (centered zero padding when win < n_fft, torch semantics) ----------
    std::vector<double> window(static_cast<std::size_t>(cfg.n_fft), 0.0);
    const int woff = (cfg.n_fft - cfg.win) / 2;
    for (int i = 0; i < cfg.win; ++i) {
        window[static_cast<std::size_t>(woff + i)] =
            0.5 - 0.5 * std::cos(2.0 * kPi * static_cast<double>(i) / static_cast<double>(cfg.win));
    }

    const int64_t frames = cfg.frames_for(n);
    if (frames <= 0) {
        if (err) {
            *err = "spec frame count is 0 (waveform shorter than n_fft)";
        }
        return false;
    }
    const int64_t bins = cfg.bins();

    // ---------- per-frame r2c + magnitude ----------
    out->frames = frames;
    out->bins = bins;
    out->values.assign(static_cast<std::size_t>(bins * frames), 0.0f);

    const pocketfft::shape_t shape{static_cast<std::size_t>(cfg.n_fft)};
    const pocketfft::stride_t stride_in{static_cast<std::ptrdiff_t>(sizeof(double))};
    const pocketfft::stride_t stride_out{
        static_cast<std::ptrdiff_t>(sizeof(std::complex<double>))};
    const pocketfft::shape_t axes{0};

    std::vector<double> frame(static_cast<std::size_t>(cfg.n_fft));
    std::vector<std::complex<double>> spectrum(static_cast<std::size_t>(bins));

    for (int64_t t = 0; t < frames; ++t) {
        const int64_t base = t * cfg.hop;
        for (int i = 0; i < cfg.n_fft; ++i) {
            frame[static_cast<std::size_t>(i)] =
                padded[static_cast<std::size_t>(base + i)] * window[static_cast<std::size_t>(i)];
        }
        pocketfft::r2c(shape, stride_in, stride_out, axes, pocketfft::FORWARD, frame.data(),
                       spectrum.data(), 1.0);
        for (int64_t f = 0; f < bins; ++f) {
            const double re = spectrum[static_cast<std::size_t>(f)].real();
            const double im = spectrum[static_cast<std::size_t>(f)].imag();
            out->values[static_cast<std::size_t>(f * frames + t)] =
                static_cast<float>(std::sqrt(re * re + im * im + kMagEps));
        }
    }

    if (err) {
        err->clear();
    }
    return true;
}

} // namespace rvc_train::prep
