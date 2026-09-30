// audio_io.cpp
// WAV 读写与重采样实现（dr_wav + 多相 FIR 重采样）

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
    // Kaiser 窗函数（I0 级数展开），用于重采样低通滤波器
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

    // 有理比多相 FIR 重采样：
    //   dst/src = D/U（U 上采样、D 抽取）
    //   理想低通截止 fc = 0.95 * min(src,dst)，Kaiser(beta=7.865) 窗，长度 L = 11 个源样本
    //   系数表 [U][2H+1]，H 为 max(U,D) 域半宽
    std::vector<float> windowed_sinc_resample(
        const std::vector<float>& in, int src_sr, int dst_sr) {
        if (src_sr <= 0 || dst_sr <= 0) {
            throw std::runtime_error("重采样采样率必须为正");
        }
        if (src_sr == dst_sr || in.empty()) {
            return in;
        }
        const int64_t g = gcd_i64(src_sr, dst_sr);
        const int64_t U = dst_sr / g;   // 上采样因子
        const int64_t D = src_sr / g;   // 抽取因子
        if (U > 512) {
            // 采样率比过离谱，直接线性插值兜底
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
        const double fc = 0.95 * static_cast<double>(std::min(src_sr, dst_sr)) / fs_max; // 中间域归一化截止 (0..0.5)
        const double beta = 7.865;
        const int64_t L = 11;                       // 单边零交叉数（源样本域）
        const int64_t H = static_cast<int64_t>(std::ceil(L * U)); // 中间域半宽
        const int64_t ntaps = 2 * H + 1;

        // 系数表 [U][ntaps]：phase p 对应输入样本剩余偏移 p/U
        std::vector<double> phase_coef(static_cast<size_t>(U) * ntaps, 0.0);
        for (int64_t p = 0; p < U; ++p) {
            double* row = phase_coef.data() + static_cast<size_t>(p) * ntaps;
            double sum = 0.0;
            for (int64_t k = 0; k < ntaps; ++k) {
                const double t = static_cast<double>(k - H) - static_cast<double>(p) / U;
                // sinc(2*fc*t)（归一化频率）
                const double x = 2.0 * fc * t;
                const double sinc = (std::abs(x) < 1e-12) ? 1.0 : std::sin(kPi * x) / (kPi * x);
                const double w = kaiser(t, static_cast<double>(H) / U, beta);
                row[k] = 2.0 * fc * sinc * w;
                sum += row[k];
            }
            if (sum != 0.0) {
                for (int64_t k = 0; k < ntaps; ++k) row[k] /= sum;   // DC 归一化
            }
        }

        // 反射填充，避免边界黑边；中心位于 H
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

        // 输出样本 i 对应中间域位置 m0 = i*D；最近输入样本 j0 = floor(m0/U)，相位 p = m0 - j0*U
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

AudioData load_wav(const std::string& filepath) {
    drwav wav;
    if (!drwav_init_file(&wav, filepath.c_str(), nullptr)) {
        throw std::runtime_error("无法打开 WAV 文件: " + filepath);
    }

    AudioData audio;
    audio.sample_rate = static_cast<int>(wav.sampleRate);
    audio.num_channels = static_cast<int>(wav.channels);
    audio.total_samples = wav.totalPCMFrameCount;

    if (audio.num_channels <= 0 || audio.total_samples == 0) {
        drwav_uninit(&wav);
        throw std::runtime_error("WAV 文件为空或声道数无效: " + filepath);
    }

    audio.samples.resize(static_cast<size_t>(audio.total_samples) * audio.num_channels);
    const drwav_uint64 frames_read = drwav_read_pcm_frames_f32(
        &wav, audio.total_samples, audio.samples.data());
    drwav_uninit(&wav);

    if (frames_read == 0) {
        throw std::runtime_error("读取 WAV 数据失败: " + filepath);
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

    // float -> int16（钳位）
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
