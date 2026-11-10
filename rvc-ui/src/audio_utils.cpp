// audio_utils.cpp — WAV I/O, resampling and mixing for rvc-ui.
//
// dr_wav-based; the resampler is the same polyphase FIR implementation as
// rvc-ggml/src/audio/audio_io.cpp and rvc-sep-ggml/src/audio.cpp.

#define DR_WAV_IMPLEMENTATION
#include "dr_wav.h"

#include "audio_utils.h"

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <stdexcept>

namespace {

constexpr double kPi = 3.141592653589793238462643383279502884;

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

namespace audio {

Buffer load_wav(const std::string& path) {
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
    Buffer out;
    out.channels = static_cast<int>(wav.channels);
    out.sample_rate = static_cast<int>(wav.sampleRate);
    out.data.resize(static_cast<size_t>(frames) * out.channels);
    const drwav_uint64 got = drwav_read_pcm_frames_f32(&wav, frames, out.data.data());
    drwav_uninit(&wav);
    if (got != frames) {
        throw std::runtime_error("Failed to read WAV data: " + path);
    }
    return out;
}

void save_wav(const std::string& path, const Buffer& buffer) {
    if (buffer.channels == 0 || buffer.sample_rate == 0) {
        throw std::runtime_error("Invalid audio buffer (channels/sample rate)");
    }
    drwav_data_format fmt{};
    fmt.container = drwav_container_riff;
    fmt.format = DR_WAVE_FORMAT_IEEE_FLOAT;
    fmt.channels = static_cast<drwav_uint32>(buffer.channels);
    fmt.sampleRate = static_cast<drwav_uint32>(buffer.sample_rate);
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
    const drwav_uint64 frames = static_cast<drwav_uint64>(buffer.frames());
    const drwav_uint64 written = drwav_write_pcm_frames(&wav, frames, buffer.data.data());
    drwav_uninit(&wav);
    if (written != frames) {
        throw std::runtime_error("Failed to write WAV data: " + path);
    }
}

Buffer resample(const Buffer& in, int target_sample_rate) {
    if (in.sample_rate == target_sample_rate || in.sample_rate <= 0 || in.frames() == 0) {
        return in;
    }
    Buffer out;
    out.channels = in.channels;
    out.sample_rate = target_sample_rate;
    const size_t frames = in.frames();
    const size_t out_frames = static_cast<size_t>(std::llround(
        static_cast<double>(frames) * target_sample_rate / in.sample_rate));
    out.data.assign(out_frames * static_cast<size_t>(in.channels), 0.0f);

    std::vector<float> channel(frames);
    for (int ch = 0; ch < in.channels; ++ch) {
        for (size_t i = 0; i < frames; ++i) {
            channel[i] = in.data[i * static_cast<size_t>(in.channels) + static_cast<size_t>(ch)];
        }
        const std::vector<float> resampled = windowed_sinc_resample(channel, in.sample_rate, target_sample_rate);
        const size_t n = std::min(resampled.size(), out_frames);
        for (size_t i = 0; i < n; ++i) {
            out.data[i * static_cast<size_t>(in.channels) + static_cast<size_t>(ch)] = resampled[i];
        }
    }
    return out;
}

Buffer to_stereo(const Buffer& in) {
    if (in.channels == 2) {
        return in;
    }
    Buffer out;
    out.channels = 2;
    out.sample_rate = in.sample_rate;
    const size_t frames = in.frames();
    out.data.resize(frames * 2);
    if (in.channels == 1) {
        for (size_t i = 0; i < frames; ++i) {
            const float v = in.data[i];
            out.data[i * 2] = v;
            out.data[i * 2 + 1] = v;
        }
    } else if (in.channels > 2) {
        const float scale = 1.0f / static_cast<float>(in.channels);
        for (size_t i = 0; i < frames; ++i) {
            float acc = 0.0f;
            for (int ch = 0; ch < in.channels; ++ch) {
                acc += in.data[i * static_cast<size_t>(in.channels) + static_cast<size_t>(ch)];
            }
            const float v = acc * scale;
            out.data[i * 2] = v;
            out.data[i * 2 + 1] = v;
        }
    } else {
        throw std::runtime_error("Audio buffer has no channels");
    }
    return out;
}

Buffer subtract(const Buffer& a, const Buffer& b) {
    if (a.sample_rate != b.sample_rate || a.channels != b.channels) {
        throw std::runtime_error("subtract: sample rate/channel mismatch");
    }
    Buffer out = a;
    const size_t n = std::min(a.data.size(), b.data.size());
    for (size_t i = 0; i < n; ++i) {
        out.data[i] -= b.data[i];
    }
    return out;
}

Buffer mix(const std::vector<std::pair<Buffer, float>>& tracks) {
    if (tracks.empty()) {
        return {};
    }
    const int sr = tracks.front().first.sample_rate;
    const int ch = tracks.front().first.channels;
    size_t frames = 0;
    for (const auto& t : tracks) {
        if (t.first.sample_rate != sr || t.first.channels != ch) {
            throw std::runtime_error("mix: sample rate/channel mismatch");
        }
        frames = std::max(frames, t.first.frames());
    }
    Buffer out;
    out.channels = ch;
    out.sample_rate = sr;
    out.data.assign(frames * static_cast<size_t>(ch), 0.0f);
    for (const auto& [buffer, gain] : tracks) {
        const size_t n = std::min(buffer.data.size(), out.data.size());
        for (size_t i = 0; i < n; ++i) {
            out.data[i] += buffer.data[i] * gain;
        }
    }
    float peak = 0.0f;
    for (float v : out.data) {
        peak = std::max(peak, std::fabs(v));
    }
    if (peak > 0.999f) {
        const float scale = 0.999f / peak;
        for (float& v : out.data) {
            v *= scale;
        }
    }
    return out;
}

} // namespace audio
