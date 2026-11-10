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

    
    // Rational-ratio polyphase FIR resampling:
    //   dst/src = D/U (U = upsampling, D = decimation)
    //   ideal low-pass cutoff fc = 0.95 * min(src,dst), Kaiser(beta=7.865) window, length L = 11 source samples
    //   coefficient table [U][2H+1], H is the half-width in the max(U,D) domain
        // Windowed-sinc resampling, evaluated directly per output sample:
    //   pos = i / ratio (input-sample position), fc = 0.475 * min(1, ratio) cycles/sample,
    //   Kaiser window (beta 8.5), half-width 16 input samples, per-output normalization.
    //   The kernel is precomputed on a fine grid and linearly interpolated.
    // (Replaces an earlier phase-table implementation whose per-phase DC
    //  normalization degenerated for ratios such as 32000 -> 44100.)
    std::vector<float> windowed_sinc_resample(const std::vector<float>& in, int src_sr, int dst_sr) {
        if (src_sr <= 0 || dst_sr <= 0) {
            throw std::runtime_error("Resampling sample rates must be positive");
        }
        if (src_sr == dst_sr || in.empty()) {
            return in;
        }
        const int64_t n = static_cast<int64_t>(in.size());
        const int64_t out_len = std::max<int64_t>(1, static_cast<int64_t>(std::llround(
            static_cast<double>(n) * dst_sr / src_sr)));
        const double ratio = static_cast<double>(dst_sr) / src_sr;
        const double fc = 0.475 * std::min(1.0, ratio);  // cycles per input sample
        const double half = 16.0;                        // kernel half-width (input samples)
        const double beta = 8.5;

        // Kernel lookup table (128 grid points per input sample).
        const int64_t os = 128;
        const int64_t table_n = static_cast<int64_t>(std::ceil(half * os));
        std::vector<double> table(static_cast<size_t>(2 * table_n + 1));
        const double inv_i0_beta = 1.0 / bessel_i0(beta);
        for (int64_t k = -table_n; k <= table_n; ++k) {
            const double x = static_cast<double>(k) / os;
            const double r = x / half;
            const double arg = 1.0 - r * r;
            const double w = arg > 0.0 ? bessel_i0(beta * std::sqrt(arg)) * inv_i0_beta : 0.0;
            const double sx = 2.0 * fc * x;
            const double sinc = (std::abs(sx) < 1e-12) ? 1.0 : std::sin(kPi * sx) / (kPi * sx);
            table[static_cast<size_t>(k + table_n)] = 2.0 * fc * sinc * w;
        }
        auto kernel_at = [&](double x) -> double {
            const double idx = std::clamp(x * os, -static_cast<double>(table_n),
                                          static_cast<double>(table_n));
            const int64_t i0i = static_cast<int64_t>(std::floor(idx)) + table_n;
            const int64_t i1i = std::min<int64_t>(i0i + 1, 2 * table_n);
            const double f = idx - std::floor(idx);
            return table[static_cast<size_t>(i0i)] * (1.0 - f) +
                   table[static_cast<size_t>(i1i)] * f;
        };

        auto sample_at = [n, &in](int64_t j) -> double {
            if (j < 0) j = -j;
            if (j >= n) j = 2 * (n - 1) - j;
            if (j < 0) j = 0;
            if (j >= n) j = n - 1;
            return static_cast<double>(in[static_cast<size_t>(j)]);
        };

        std::vector<float> out(static_cast<size_t>(out_len));
        for (int64_t i = 0; i < out_len; ++i) {
            const double pos = static_cast<double>(i) / ratio;
            const int64_t j0 = static_cast<int64_t>(std::floor(pos - half));
            const int64_t j1 = static_cast<int64_t>(std::ceil(pos + half));
            double acc = 0.0;
            double wsum = 0.0;
            for (int64_t j = j0; j <= j1; ++j) {
                const double t = static_cast<double>(j) - pos;
                const double w = kernel_at(t);
                acc += sample_at(j) * w;
                wsum += w;
            }
            out[static_cast<size_t>(i)] = (wsum != 0.0) ? static_cast<float>(acc / wsum) : 0.0f;
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
