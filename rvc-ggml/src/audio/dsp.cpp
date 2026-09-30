// dsp.cpp
// RVC 音频信号处理实现。系数与边界处理对齐 scipy（filtfilt/lfilter_zi）与 librosa。

#include "dsp.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace rvc {
namespace dsp {

namespace {

    constexpr int kOrder = 5;

    // scipy.signal.butter(5, 48, btype='high', fs=16000)
    constexpr double kB[6] = {
        0.9699606451838447,
        -4.849803225919223,
        9.699606451838447,
        -9.699606451838447,
        4.849803225919223,
        -0.9699606451838447,
    };
    constexpr double kA[6] = {
        1.0,
        -4.939001819168364,
        9.757863526739543,
        -9.639544849413458,
        4.761506797356209,
        -0.9408236532054606,
    };
    // scipy.signal.lfilter_zi(b, a)
    constexpr double kZi[5] = {
        -0.9699605304939276,
        3.8798421289715876,
        -5.819763203738301,
        3.879842142541546,
        -0.9699605372808581,
    };

    // 转置直接 II 型 IIR（对齐 scipy.signal.lfilter 的 zi 语义）
    void lfilter_tdf2(const double* b, const double* a,
        const double* zi,
        std::vector<double>& x) {
        double state[kOrder];
        const double x0 = x.empty() ? 0.0 : x.front();
        for (int i = 0; i < kOrder; ++i) {
            state[i] = zi[i] * x0;
        }
        for (double& value : x) {
            const double y = b[0] * value + state[0];
            for (int i = 1; i < kOrder; ++i) {
                state[i - 1] = b[i] * value + state[i] - a[i] * y;
            }
            state[kOrder - 1] = b[kOrder] * value - a[kOrder] * y;
            value = y;
        }
    }

} // namespace

void high_pass_48hz_in_place(std::vector<float>& samples) {
    if (samples.empty()) return;
    // scipy.signal.filtfilt 默认 padlen = 3 * max(len(b), len(a)) = 3 * (order+1) = 18
    const int padlen = 3 * (kOrder + 1);

    // scipy filtfilt 的 odd 扩展：
    //   left  = 2*x[0] - x[padlen:0:-1]（远端先出现：2*x0-x[padlen] ... 2*x0-x[1]）
    //   right = 2*x[-1] - x[-2:-padlen-2:-1]（近端先出现：2*x-1-x[-2] ...）
    const size_t n = samples.size();
    if (static_cast<int64_t>(n) <= padlen) {
        // scipy 同样要求 len(x) > padlen；极短输入不做扩展（防御）
        return;
    }
    std::vector<double> ext(n + 2 * static_cast<size_t>(padlen), 0.0);
    for (int i = 0; i < padlen; ++i) {
        ext[static_cast<size_t>(i)] =
            2.0 * samples[0] - samples[static_cast<size_t>(padlen - i)];
    }
    for (size_t i = 0; i < n; ++i) {
        ext[static_cast<size_t>(padlen) + i] = samples[i];
    }
    for (int i = 0; i < padlen; ++i) {
        ext[static_cast<size_t>(padlen) + n + i] =
            2.0 * samples[n - 1] - samples[n - 2 - static_cast<size_t>(i)];
    }

    lfilter_tdf2(kB, kA, kZi, ext);
    std::reverse(ext.begin(), ext.end());
    lfilter_tdf2(kB, kA, kZi, ext);
    std::reverse(ext.begin(), ext.end());

    for (size_t i = 0; i < n; ++i) {
        samples[i] = static_cast<float>(ext[static_cast<size_t>(padlen) + i]);
    }
}

std::vector<float> reflect_pad(const std::vector<float>& samples, int64_t left, int64_t right) {
    if (samples.empty()) return {};
    const int64_t n = static_cast<int64_t>(samples.size());
    std::vector<float> out(static_cast<size_t>(n + left + right));
    for (int64_t i = 0; i < left; ++i) {
        // 反射（不含边缘）：index = -i (i=1 -> samples[1] ... 但 numpy reflect 为 samples[i])
        int64_t idx = left - i;
        if (idx >= n) idx = n - 1 - (idx - (n - 1));
        idx = std::clamp<int64_t>(idx, 0, n - 1);
        out[static_cast<size_t>(i)] = samples[static_cast<size_t>(idx)];
    }
    for (int64_t i = 0; i < n; ++i) {
        out[static_cast<size_t>(left + i)] = samples[static_cast<size_t>(i)];
    }
    for (int64_t i = 0; i < right; ++i) {
        int64_t idx = n - 2 - i;
        if (idx < 0) idx = -idx;
        idx = std::clamp<int64_t>(idx, 0, n - 1);
        out[static_cast<size_t>(left + n + i)] = samples[static_cast<size_t>(idx)];
    }
    return out;
}

std::vector<float> frame_rms(const std::vector<float>& audio, int sample_rate) {
    const int frame_length = (sample_rate / 2) * 2;
    const int hop = sample_rate / 2;
    if (frame_length <= 0 || hop <= 0 || audio.empty()) return {};

    // 对齐 librosa.feature.rms(center=True)：默认 pad_mode='constant'（零填充），
    //   padded = np.pad(x, frame_length//2, mode='constant')
    //   帧数 = 1 + (len(padded) - frame_length) // hop
    const int64_t n = static_cast<int64_t>(audio.size());
    const int64_t pad = frame_length / 2;
    const int64_t padded_len = n + 2 * pad;
    const int64_t frame_count = std::max<int64_t>(1, 1 + (padded_len - frame_length) / hop);

    std::vector<float> rms;
    rms.reserve(static_cast<size_t>(frame_count));
    for (int64_t f = 0; f < frame_count; ++f) {
        const int64_t begin = f * hop;
        double sum = 0.0;
        for (int i = 0; i < frame_length; ++i) {
            const int64_t src = begin + i - pad;   // padded 坐标 -> 原始样本坐标
            if (src < 0 || src >= n) continue;     // 零填充区域
            const float v = audio[static_cast<size_t>(src)];
            sum += static_cast<double>(v) * static_cast<double>(v);
        }
        rms.push_back(static_cast<float>(std::sqrt(sum / static_cast<double>(frame_length))));
    }
    return rms;
}

std::vector<float> interpolate_linear(const std::vector<float>& values, int64_t output_frames) {
    if (values.empty() || output_frames <= 0) return {};
    if (output_frames == 1 || values.size() == 1) {
        return std::vector<float>(static_cast<size_t>(output_frames), values.front());
    }
    const int64_t in_n = static_cast<int64_t>(values.size());
    std::vector<float> out(static_cast<size_t>(output_frames));
    for (int64_t i = 0; i < output_frames; ++i) {
        const double pos =
            static_cast<double>(i) * static_cast<double>(in_n - 1) /
            static_cast<double>(output_frames - 1);
        const int64_t left = static_cast<int64_t>(std::floor(pos));
        const int64_t right = std::min<int64_t>(in_n - 1, left + 1);
        const double frac = pos - static_cast<double>(left);
        out[static_cast<size_t>(i)] = static_cast<float>(
            values[static_cast<size_t>(left)] * (1.0 - frac) +
            values[static_cast<size_t>(right)] * frac);
    }
    return out;
}

void apply_rms_mix(const std::vector<float>& source_16k,
    std::vector<float>& converted,
    int target_sample_rate,
    float rms_mix_rate) {
    if (rms_mix_rate == 1.0f || converted.empty()) return;
    const auto rms1 = frame_rms(source_16k, 16000);
    const auto rms2 = frame_rms(converted, target_sample_rate);
    if (rms1.empty() || rms2.empty()) return;

    const int64_t out_frames = static_cast<int64_t>(converted.size());
    const auto rms1i = interpolate_linear(rms1, out_frames);
    auto rms2i = interpolate_linear(rms2, out_frames);
    for (int64_t i = 0; i < out_frames; ++i) {
        float r2 = std::max(rms2i[static_cast<size_t>(i)], 1e-6f);
        const float factor =
            std::pow(rms1i[static_cast<size_t>(i)], 1.0f - rms_mix_rate) *
            std::pow(r2, rms_mix_rate - 1.0f);
        converted[static_cast<size_t>(i)] *= factor;
    }
}

std::vector<float> audio_sum_curve(const std::vector<float>& audio) {
    // audio_sum[t] = sum_{i=0..159} |audio_pad[i + t]|，audio_pad 左右各反射 80
    constexpr int64_t window = 160;
    const int64_t n = static_cast<int64_t>(audio.size());
    const auto padded = reflect_pad(audio, window / 2, window / 2);
    std::vector<float> audio_sum(static_cast<size_t>(n), 0.0f);
    for (int64_t i = 0; i < window; ++i) {
        for (int64_t t = 0; t < n; ++t) {
            audio_sum[static_cast<size_t>(t)] += std::abs(padded[static_cast<size_t>(i + t)]);
        }
    }
    return audio_sum;
}

std::vector<int64_t> quiet_split_points(const std::vector<float>& audio,
    int split_query_sec,
    int split_center_sec,
    int split_threshold_sec) {
    const int64_t t_query = static_cast<int64_t>(split_query_sec) * 16000;
    const int64_t t_center = static_cast<int64_t>(split_center_sec) * 16000;
    const int64_t t_max = static_cast<int64_t>(split_threshold_sec) * 16000;
    const int64_t n = static_cast<int64_t>(audio.size());

    std::vector<int64_t> splits;
    if (n <= t_max) return splits;
    if (t_query <= 0 || t_center <= 0 || t_max <= 0) {
        throw std::runtime_error("静音切块参数必须为正");
    }

    const auto audio_sum = audio_sum_curve(audio);

    for (int64_t t = t_center; t < n; t += t_center) {
        const int64_t begin = std::max<int64_t>(0, t - t_query);
        const int64_t end = std::min<int64_t>(n, t + t_query);
        if (begin >= end) continue;
        int64_t best = begin;
        float best_value = audio_sum[static_cast<size_t>(begin)];
        for (int64_t i = begin + 1; i < end; ++i) {
            const float v = audio_sum[static_cast<size_t>(i)];
            if (v < best_value) {
                best_value = v;
                best = i;
            }
        }
        splits.push_back(t - t_query + (best - begin));
    }
    return splits;
}

int coarse_pitch_bin(float f0_hz) {
    if (f0_hz <= 0.0f) return 1;
    const float f0_mel = 1127.0f * std::log(1.0f + f0_hz / 700.0f);
    const float f0_min = 1127.0f * std::log(1.0f + 50.0f / 700.0f);
    const float f0_max = 1127.0f * std::log(1.0f + 1100.0f / 700.0f);
    const float scaled = (f0_mel - f0_min) * 254.0f / (f0_max - f0_min) + 1.0f;
    float clamped = scaled;
    if (clamped < 1.0f) clamped = 1.0f;
    if (clamped > 255.0f) clamped = 255.0f;
    return static_cast<int>(std::lrint(clamped));
}

void median_filter_f0(std::vector<float>& f0, int radius) {
    if (radius <= 0 || f0.empty()) return;
    std::vector<float> filtered(f0.size(), 0.0f);
    std::vector<float> window;
    window.reserve(static_cast<size_t>(radius * 2 + 1));
    for (size_t i = 0; i < f0.size(); ++i) {
        window.clear();
        const size_t begin = i > static_cast<size_t>(radius) ? i - static_cast<size_t>(radius) : 0;
        const size_t end = std::min(f0.size(), i + static_cast<size_t>(radius) + 1);
        for (size_t j = begin; j < end; ++j) window.push_back(f0[j]);
        const auto mid = window.begin() + static_cast<std::ptrdiff_t>(window.size() / 2);
        std::nth_element(window.begin(), mid, window.end());
        filtered[i] = *mid;
    }
    f0 = std::move(filtered);
}

void interpolate_unvoiced(std::vector<float>& f0) {
    const size_t n = f0.size();
    if (n == 0) return;
    size_t i = 0;
    while (i < n) {
        if (f0[i] != 0.0f) {
            ++i;
            continue;
        }
        size_t j = i;
        while (j < n && f0[j] == 0.0f) ++j;
        // 区间 [i, j) 为 0
        if (i == 0 && j >= n) {
            // 全 0
            break;
        }
        if (i == 0) {
            for (size_t k = 0; k < j; ++k) f0[k] = f0[j];
        }
        else if (j >= n) {
            for (size_t k = i; k < n; ++k) f0[k] = f0[i - 1];
        }
        else {
            const float left = f0[i - 1];
            const float right = f0[j];
            const size_t span = j - i + 1;
            for (size_t k = i; k < j; ++k) {
                const float t = static_cast<float>(k - i + 1) / static_cast<float>(span);
                f0[k] = left + (right - left) * t;
            }
        }
        i = j;
    }
}

} // namespace dsp
} // namespace rvc
