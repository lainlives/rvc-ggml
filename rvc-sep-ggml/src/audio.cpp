// audio.cpp — WAV I/O and resampling for rvc-sep.
//
// dr_wav-based replacement for the upstream BSRoformer.cpp libav audio path.
// The resampler is a self-contained copy of the polyphase FIR implementation
// from rvc-ggml/src/audio/audio_io.cpp (this repository).

#define DR_WAV_IMPLEMENTATION
#include "dr_wav.h"

#include "bs_roformer/audio.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <stdexcept>

namespace {

    constexpr double kPi = 3.141592653589793238462643383279502884;

    // ----------------------------------------------------------
    // Kaiser window function (I0 series expansion), used for the resampling low-pass filter
    // ----------------------------------------------------------
    double bessel_i0(double x) {
        double sum = 1.0;
        double term = 1.0;
        for (int k = 1; k <= 32; ++k) {
            term *= (x / (2.0 * k)) * (x / (2.0 * k));
            sum += term;
            if (term < 1e-16 * sum) break;
        }
        return sum;
    }

    double kaiser(double n, double half_len, double beta) {
        if (half_len <= 0.0) return 1.0;
        const double r = n / half_len;
        const double arg = 1.0 - r * r;
        if (arg <= 0.0) return 0.0;
        return bessel_i0(beta * std::sqrt(arg)) / bessel_i0(beta);
    }

    int64_t gcd_i64(int64_t a, int64_t b) {
        while (b != 0) {
            const int64_t t = a % b;
            a = b;
            b = t;
        }
        return a < 0 ? -a : a;
    }

    // Rational-ratio polyphase FIR resampling:
    //   dst/src = D/U (U = upsampling, D = decimation)
    //   ideal low-pass cutoff fc = 0.95 * min(src,dst), Kaiser(beta=7.865) window, length L = 11 source samples
    //   coefficient table [U][2H+1], H is the half-width in the max(U,D) domain
    std::vector<float> windowed_sinc_resample(
        const std::vector<float>& in, int src_sr, int dst_sr) {
        if (src_sr <= 0 || dst_sr <= 0) {
            throw std::runtime_error("Resampling sample rates must be positive");
        }
        if (src_sr == dst_sr || in.empty()) {
            return in;
        }
        const int64_t g = gcd_i64(src_sr, dst_sr);
        const int64_t U = dst_sr / g;   // upsampling factor
        const int64_t D = src_sr / g;   // decimation factor
        if (U > 512) {
            // Sample-rate ratio is too extreme; fall back to plain linear interpolation
            const int64_t n = static_cast<int64_t>(in.size());
            const int64_t out_len = std::max<int64_t>(1, static_cast<int64_t>(std::llround(n * static_cast<double>(dst_sr) / src_sr)));
            std::vector<float> out(static_cast<size_t>(out_len));
            for (int64_t i = 0; i < out_len; ++i) {
                const double pos = static_cast<double>(i) * src_sr / dst_sr;
                const int64_t l = std::clamp<int64_t>(static_cast<int64_t>(pos), 0, n - 1);
                const int64_t r = std::min<int64_t>(n - 1, l + 1);
                const double f = pos - static_cast<double>(l);
                out[static_cast<size_t>(i)] = static_cast<float>(
                    in[static_cast<size_t>(l)] * (1.0 - f) + in[static_cast<size_t>(r)] * f);
            }
            return out;
        }

        const double fs_max = static_cast<double>(std::max(U, D));
        const double fc = 0.95 * static_cast<double>(std::min(src_sr, dst_sr)) / fs_max; // normalized cutoff in the intermediate domain (0..0.5)
        const double beta = 7.865;
        const int64_t L = 11;                       // one-sided zero-crossing count (source sample domain)
        const int64_t H = static_cast<int64_t>(std::ceil(L * U)); // intermediate-domain half-width
        const int64_t ntaps = 2 * H + 1;

        // Coefficient table [U][ntaps]: phase p corresponds to the remaining input-sample offset p/U
        std::vector<double> phase_coef(static_cast<size_t>(U) * ntaps, 0.0);
        for (int64_t p = 0; p < U; ++p) {
            double* row = phase_coef.data() + static_cast<size_t>(p) * ntaps;
            double sum = 0.0;
            for (int64_t k = 0; k < ntaps; ++k) {
                const double t = static_cast<double>(k - H) - static_cast<double>(p) / U;
                // sinc(2*fc*t) (normalized frequency)
                const double x = 2.0 * fc * t;
                const double sinc = (std::abs(x) < 1e-12) ? 1.0 : std::sin(kPi * x) / (kPi * x);
                const double w = kaiser(t, static_cast<double>(H) / U, beta);
                row[k] = 2.0 * fc * sinc * w;
                sum += row[k];
            }
            if (sum != 0.0) {
                for (int64_t k = 0; k < ntaps; ++k) row[k] /= sum;   // DC normalization
            }
        }

        // Reflection padding to avoid edge artifacts; center at H
        const int64_t n = static_cast<int64_t>(in.size());
        std::vector<float> padded(static_cast<size_t>(n + 2 * H));
        for (int64_t i = 0; i < H; ++i) {
            int64_t idx = H - i;
            if (idx >= n) idx = n - 1 - (idx - (n - 1));
            padded[static_cast<size_t>(i)] = in[static_cast<size_t>(std::clamp<int64_t>(idx, 0, n - 1))];
        }
        for (int64_t i = 0; i < n; ++i) padded[static_cast<size_t>(H + i)] = in[static_cast<size_t>(i)];
        for (int64_t i = 0; i < H; ++i) {
            int64_t idx = n - 2 - i;
            if (idx < 0) idx = -idx;
            padded[static_cast<size_t>(H + n + i)] = in[static_cast<size_t>(std::clamp<int64_t>(idx, 0, n - 1))];
        }

        const int64_t out_len = std::max<int64_t>(1, static_cast<int64_t>(std::llround(
            n * static_cast<double>(dst_sr) / src_sr)));
        std::vector<float> out(static_cast<size_t>(out_len));

        // Output sample i maps to intermediate-domain position m0 = i*D; nearest input sample j0 = floor(m0/U), phase p = m0 - j0*U
        for (int64_t i = 0; i < out_len; ++i) {
            const int64_t m0 = i * D;
            int64_t j0 = m0 / U;
            const int64_t p = m0 - j0 * U;
            const double* row = phase_coef.data() + static_cast<size_t>(p) * ntaps;
            double acc = 0.0;
            int64_t base = j0 - H;
            for (int64_t k = 0; k < ntaps; ++k) {
                acc += static_cast<double>(padded[static_cast<size_t>(base + k + H)]) * row[k];
            }
            out[static_cast<size_t>(i)] = static_cast<float>(acc);
        }
        return out;
    }

} // namespace

AudioBuffer AudioFile::Load(const std::string& path, int target_sample_rate) {
    drwav wav;
#ifdef _WIN32
    const std::filesystem::path p(path);
    const bool ok = drwav_init_file_w(&wav, p.wstring().c_str(), nullptr);
#else
    const bool ok = drwav_init_file(&wav, path.c_str(), nullptr);
#endif
    if (!ok) {
        throw std::runtime_error("Cannot open WAV file: " + path);
    }

    const drwav_uint64 frames = wav.totalPCMFrameCount;
    const unsigned int channels = wav.channels;
    const unsigned int sample_rate = wav.sampleRate;
    std::vector<float> interleaved(static_cast<size_t>(frames) * channels);
    const drwav_uint64 got = drwav_read_pcm_frames_f32(&wav, frames, interleaved.data());
    drwav_uninit(&wav);
    if (got != frames) {
        throw std::runtime_error("Failed to read WAV data: " + path);
    }

    // Convert to interleaved stereo float32
    std::vector<float> stereo;
    if (channels == 0) {
        throw std::runtime_error("WAV file has no channels: " + path);
    }
    if (channels == 1) {
        stereo.resize(static_cast<size_t>(frames) * 2);
        for (drwav_uint64 i = 0; i < frames; ++i) {
            const float v = interleaved[static_cast<size_t>(i)];
            stereo[static_cast<size_t>(i) * 2] = v;
            stereo[static_cast<size_t>(i) * 2 + 1] = v;
        }
    } else if (channels == 2) {
        stereo = std::move(interleaved);
    } else {
        // >2 channels: average to mono, then expand to stereo
        stereo.resize(static_cast<size_t>(frames) * 2);
        const float scale = 1.0f / static_cast<float>(channels);
        for (drwav_uint64 i = 0; i < frames; ++i) {
            float acc = 0.0f;
            for (unsigned int ch = 0; ch < channels; ++ch) {
                acc += interleaved[static_cast<size_t>(i) * channels + ch];
            }
            const float v = acc * scale;
            stereo[static_cast<size_t>(i) * 2] = v;
            stereo[static_cast<size_t>(i) * 2 + 1] = v;
        }
    }

    // Resample to the target rate when needed
    if (static_cast<int>(sample_rate) != target_sample_rate && frames > 0) {
        std::vector<float> left(static_cast<size_t>(frames));
        std::vector<float> right(static_cast<size_t>(frames));
        for (drwav_uint64 i = 0; i < frames; ++i) {
            left[static_cast<size_t>(i)] = stereo[static_cast<size_t>(i) * 2];
            right[static_cast<size_t>(i)] = stereo[static_cast<size_t>(i) * 2 + 1];
        }
        left = windowed_sinc_resample(left, static_cast<int>(sample_rate), target_sample_rate);
        right = windowed_sinc_resample(right, static_cast<int>(sample_rate), target_sample_rate);
        stereo.resize(left.size() * 2);
        for (size_t i = 0; i < left.size(); ++i) {
            stereo[i * 2] = left[i];
            stereo[i * 2 + 1] = right[i];
        }
        AudioBuffer out;
        out.data = std::move(stereo);
        out.channels = 2;
        out.sampleRate = static_cast<unsigned int>(target_sample_rate);
        out.samples = out.data.size();
        return out;
    }

    AudioBuffer out;
    out.data = std::move(stereo);
    out.channels = 2;
    out.sampleRate = sample_rate;
    out.samples = out.data.size();
    return out;
}

void AudioFile::Save(const std::string& path, const AudioBuffer& buffer) {
    if (buffer.channels == 0 || buffer.sampleRate == 0) {
        throw std::runtime_error("Invalid audio buffer (channels/sample rate)");
    }
    drwav_data_format fmt{};
    fmt.container = drwav_container_riff;
    fmt.format = DR_WAVE_FORMAT_IEEE_FLOAT;
    fmt.channels = buffer.channels;
    fmt.sampleRate = buffer.sampleRate;
    fmt.bitsPerSample = 32;

    drwav wav;
#ifdef _WIN32
    const std::filesystem::path p(path);
    const bool ok = drwav_init_file_write_w(&wav, p.wstring().c_str(), &fmt, nullptr);
#else
    const bool ok = drwav_init_file_write(&wav, path.c_str(), &fmt, nullptr);
#endif
    if (!ok) {
        throw std::runtime_error("Cannot open WAV file for writing: " + path);
    }
    const drwav_uint64 frames = buffer.samples / buffer.channels;
    const drwav_uint64 written = drwav_write_pcm_frames(&wav, frames, buffer.data.data());
    drwav_uninit(&wav);
    if (written != frames) {
        throw std::runtime_error("Failed to write WAV data: " + path);
    }
}
