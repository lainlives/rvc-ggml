// audio_io.cpp
// WAV read/write and resampling implementation (dr_wav + polyphase FIR resampling)

#define DR_WAV_IMPLEMENTATION
#include "dr_wav.h"

#include "audio_io.h"
#include "utils.h"

#include <algorithm>
#include <cmath>
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

AudioData load_wav(const std::string& filepath) {
    drwav wav;
    if (!drwav_init_file(&wav, filepath.c_str(), nullptr)) {
        throw std::runtime_error("Cannot open WAV file: " + filepath);
    }

    AudioData audio;
    audio.sample_rate = static_cast<int>(wav.sampleRate);
    audio.num_channels = static_cast<int>(wav.channels);
    audio.total_samples = wav.totalPCMFrameCount;

    if (audio.num_channels <= 0 || audio.total_samples == 0) {
        drwav_uninit(&wav);
        throw std::runtime_error("WAV file is empty or has an invalid channel count: " + filepath);
    }

    audio.samples.resize(static_cast<size_t>(audio.total_samples) * audio.num_channels);
    const drwav_uint64 frames_read = drwav_read_pcm_frames_f32(
        &wav, audio.total_samples, audio.samples.data());
    drwav_uninit(&wav);

    if (frames_read == 0) {
        throw std::runtime_error("Failed to read WAV data: " + filepath);
    }
    audio.total_samples = frames_read;
    audio.samples.resize(static_cast<size_t>(frames_read) * audio.num_channels);
    return audio;
}

bool save_wav(const std::string& filepath,
    const float* samples,
    int num_channels,
    int sample_rate,
    uint64_t total_samples) {
    drwav_data_format format{};
    format.container = drwav_container_riff;
    format.format = DR_WAVE_FORMAT_PCM;
    format.channels = static_cast<drwav_uint32>(num_channels);
    format.sampleRate = static_cast<drwav_uint32>(sample_rate);
    format.bitsPerSample = 16;

    drwav wav;
    if (!drwav_init_file_write(&wav, filepath.c_str(), &format, nullptr)) {
        return false;
    }

    // float -> int16 (clamped)
    std::vector<drwav_int16> pcm(static_cast<size_t>(total_samples) * num_channels);
    for (size_t i = 0; i < pcm.size(); ++i) {
        float v = samples[i];
        if (v > 1.0f) v = 1.0f;
        if (v < -1.0f) v = -1.0f;
        pcm[i] = static_cast<drwav_int16>(std::lround(v * 32767.0f));
    }
    const drwav_uint64 written = drwav_write_pcm_frames(&wav, total_samples, pcm.data());
    drwav_uninit(&wav);
    return written == total_samples;
}

bool save_wav_i16(const std::string& filepath,
    const int16_t* samples,
    int num_channels,
    int sample_rate,
    uint64_t total_samples) {
    drwav_data_format format{};
    format.container = drwav_container_riff;
    format.format = DR_WAVE_FORMAT_PCM;
    format.channels = static_cast<drwav_uint32>(num_channels);
    format.sampleRate = static_cast<drwav_uint32>(sample_rate);
    format.bitsPerSample = 16;

    drwav wav;
    if (!drwav_init_file_write(&wav, filepath.c_str(), &format, nullptr)) {
        return false;
    }
    const drwav_uint64 written = drwav_write_pcm_frames(&wav, total_samples,
        reinterpret_cast<const drwav_int16*>(samples));
    drwav_uninit(&wav);
    return written == total_samples;
}

AudioData resample(const AudioData& audio, int target_sr) {
    if (audio.sample_rate == target_sr || audio.total_samples == 0) {
        return audio;
    }
    AudioData out;
    out.sample_rate = target_sr;
    out.num_channels = audio.num_channels;
    out.samples.resize(audio.samples.size());
    for (int ch = 0; ch < audio.num_channels; ++ch) {
        std::vector<float> channel(audio.total_samples);
        for (uint64_t n = 0; n < audio.total_samples; ++n) {
            channel[n] = audio.samples[n * audio.num_channels + ch];
        }
        std::vector<float> resampled = resample_mono(channel, audio.sample_rate, target_sr);
        if (ch == 0) {
            out.total_samples = resampled.size();
            out.samples.resize(static_cast<size_t>(out.total_samples) * audio.num_channels);
        }
        const uint64_t n_out = std::min<uint64_t>(out.total_samples, resampled.size());
        for (uint64_t n = 0; n < n_out; ++n) {
            out.samples[n * audio.num_channels + ch] = resampled[n];
        }
    }
    return out;
}

std::vector<float> resample_mono(const std::vector<float>& samples, int src_sr, int dst_sr) {
    return windowed_sinc_resample(samples, src_sr, dst_sr);
}

std::vector<float> to_mono_16k(const AudioData& audio, int target_sr) {
    std::vector<float> mono(audio.total_samples, 0.0f);
    const float scale = 1.0f / static_cast<float>(audio.num_channels);
    for (uint64_t n = 0; n < audio.total_samples; ++n) {
        float acc = 0.0f;
        for (int ch = 0; ch < audio.num_channels; ++ch) {
            acc += audio.samples[n * audio.num_channels + ch];
        }
        mono[n] = acc * scale;
    }
    if (audio.sample_rate == target_sr) {
        return mono;
    }
    return resample_mono(mono, audio.sample_rate, target_sr);
}
