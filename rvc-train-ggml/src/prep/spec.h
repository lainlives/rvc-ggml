// spec.h —— M4-4 线性谱（官方 spectrogram_torch 语义，pocketfft 主机侧）
//
// 语义基准 = 81eed5e train/mel_processing.py::spectrogram_torch（见 docs/交接说明.md §4.8 A）：
//   reflect pad `int((n_fft-hop)/2)` 两侧 → 周期 Hann → torch.stft(center=False, onesided) →
//   `sqrt(re²+im²+2e-7)`；输入 = 0_gt F32 波形（data_utils.get_audio 不做归一化、不 wav 往返）。
//
// 帧数（M4-4 校准）= `1 + (padded - n_fft) // hop`，`padded = gt_samples + 2*int((n_fft-hop)/2)`；
// v1-40k（2048/400）实测 = `gt_samples // hop`（旧文档式 `1+(gt-n_fft)//hop` 已更正）。
//
// 输出布局：行主序 [bins, frames]（与 torch [F,T] 一致），F32。

#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace rvc_train::prep {

struct SpecConfig {
    int n_fft = 2048;
    int hop = 400;
    int win = 2048;
    int sample_rate = 40000;

    int bins() const { return n_fft / 2 + 1; }
    // 官方帧数（reflect pad + center=False；padded < n_fft 时为 0）
    int64_t frames_for(int64_t gt_samples) const;
    bool validate(std::string* err) const;
};

struct SpecResult {
    int64_t frames = 0;
    int64_t bins = 0;
    std::vector<float> values;  // 行主序 [bins, frames]
};

// gt：F32 波形（0_gt）；返回 false 时 err 含原因
bool compute_linear_spec(const float* gt, int64_t n, const SpecConfig& cfg, SpecResult* out,
                         std::string* err);

} // namespace rvc_train::prep
