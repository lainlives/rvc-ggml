// mel.cpp
// RMVPE mel front-end implementation (pocketfft)

#include "mel.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <stdexcept>

#define POCKETFFT_CACHE_SIZE 16
#include "pocketfft_hdronly.h"

namespace rvc {
namespace audio {

namespace {

    constexpr double kPi = 3.141592653589793238462643383279502884;

    double hz_to_mel_htk(double hz) {
        return 2595.0 * std::log10(1.0 + hz / 700.0);
    }

    double mel_to_hz_htk(double mel) {
        return 700.0 * (std::pow(10.0, mel / 2595.0) - 1.0);
    }

} // namespace

std::vector<float> make_mel_filterbank(int sample_rate, int n_fft, int n_mels,
    float fmin, float fmax) {
    const int n_bins = n_fft / 2 + 1;
    std::vector<float> basis(static_cast<size_t>(n_mels) * n_bins, 0.0f);

    const double mel_min = hz_to_mel_htk(fmin);
    const double mel_max = hz_to_mel_htk(fmax);
    // librosa: mel_frequencies are evenly spaced in the mel domain
    std::vector<double> mels(static_cast<size_t>(n_mels) + 2);
    for (int i = 0; i <= n_mels + 1; ++i) {
        mels[static_cast<size_t>(i)] = mel_min + (mel_max - mel_min) * i / (n_mels + 1);
    }
    std::vector<double> freqs(static_cast<size_t>(n_mels) + 2);
    for (int i = 0; i <= n_mels + 1; ++i) {
        freqs[static_cast<size_t>(i)] = mel_to_hz_htk(mels[static_cast<size_t>(i)]);
    }

    // Triangular filters (librosa: slaney normalization is not used in htk mode)
    const double fft_freq_step = static_cast<double>(sample_rate) / n_fft;
    for (int m = 0; m < n_mels; ++m) {
        const double lower = freqs[static_cast<size_t>(m)];
        const double center = freqs[static_cast<size_t>(m + 1)];
        const double upper = freqs[static_cast<size_t>(m + 2)];
        if (center <= lower) continue;
        for (int k = 0; k < n_bins; ++k) {
            const double f = k * fft_freq_step;
            double weight = 0.0;
            if (f >= lower && f <= center) {
                weight = (f - lower) / (center - lower);
            }
            else if (f > center && f <= upper) {
                weight = (upper - f) / (upper - center);
            }
            if (weight > 0.0) {
                basis[static_cast<size_t>(m) * n_bins + k] = static_cast<float>(weight);
            }
        }
    }
    return basis;
}

std::vector<float> log_mel_spectrogram(const std::vector<float>& audio,
    const std::vector<float>& mel_basis,
    int sample_rate, int n_fft, int hop, int n_mels) {
    if (audio.empty()) return {};
    const int n_bins = n_fft / 2 + 1;
    if (static_cast<int>(mel_basis.size()) != n_mels * n_bins) {
        throw std::runtime_error("mel filterbank size mismatch");
    }

    // center=True: reflect pad n_fft/2
    const int64_t pad = n_fft / 2;
    const int64_t n = static_cast<int64_t>(audio.size());
    std::vector<double> padded(static_cast<size_t>(n + 2 * pad), 0.0);
    for (int64_t i = 0; i < pad; ++i) {
        int64_t idx = pad - i;
        if (idx >= n) idx = n - 1 - (idx - (n - 1));
        idx = std::clamp<int64_t>(idx, 0, n - 1);
        padded[static_cast<size_t>(i)] = audio[static_cast<size_t>(idx)];
    }
    for (int64_t i = 0; i < n; ++i) padded[static_cast<size_t>(pad + i)] = audio[static_cast<size_t>(i)];
    for (int64_t i = 0; i < pad; ++i) {
        int64_t idx = n - 2 - i;
        if (idx < 0) idx = -idx;
        idx = std::clamp<int64_t>(idx, 0, n - 1);
        padded[static_cast<size_t>(pad + n + i)] = audio[static_cast<size_t>(idx)];
    }

    const int64_t frames = (n + 2 * pad - n_fft) / hop + 1;

    // periodic Hann window (torch.hann_window defaults to periodic=True)
    std::vector<double> window(static_cast<size_t>(n_fft));
    for (int i = 0; i < n_fft; ++i) {
        window[static_cast<size_t>(i)] = 0.5 - 0.5 * std::cos(2.0 * kPi * i / n_fft);
    }

    pocketfft::shape_t shape{ static_cast<size_t>(n_fft) };
    pocketfft::stride_t stride_in{ static_cast<ptrdiff_t>(sizeof(double)) };
    pocketfft::stride_t stride_out{ static_cast<ptrdiff_t>(sizeof(std::complex<double>)) };
    pocketfft::shape_t axes{ 0 };

    std::vector<double> frame_in(static_cast<size_t>(n_fft));
    std::vector<std::complex<double>> frame_out(static_cast<size_t>(n_bins));
    std::vector<float> out(static_cast<size_t>(n_mels) * frames, 0.0f);

    for (int64_t t = 0; t < frames; ++t) {
        const int64_t offset = t * hop;
        for (int i = 0; i < n_fft; ++i) {
            frame_in[static_cast<size_t>(i)] =
                padded[static_cast<size_t>(offset + i)] * window[static_cast<size_t>(i)];
        }
        pocketfft::r2c(shape, stride_in, stride_out, axes, pocketfft::FORWARD,
            frame_in.data(), frame_out.data(), 1.0);

        for (int m = 0; m < n_mels; ++m) {
            double acc = 0.0;
            const float* basis = mel_basis.data() + static_cast<size_t>(m) * n_bins;
            for (int k = 0; k < n_bins; ++k) {
                const double re = frame_out[static_cast<size_t>(k)].real();
                const double im = frame_out[static_cast<size_t>(k)].imag();
                const double mag = std::sqrt(re * re + im * im);
                acc += basis[k] * mag;
            }
            const double clamped = std::max(acc, 1e-5);
            out[static_cast<size_t>(m) * frames + t] = static_cast<float>(std::log(clamped));
        }
    }
    return out;
}

} // namespace audio
} // namespace rvc
