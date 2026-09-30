// RVC-Train-GGML - 可微 STFT / mel 前端（src/train/stft_mel.h）
//
// 语义基准：官方 81eed5e `train/mel_processing.py`（训练损失专用；与运行时 infer_pack
// 的 mel 参数略有差异——官方训练代码为 sqrt(re²+im²+2e-7)、clamp min=2e-6，且
// spectrogram_torch 在 center=False 前显式 reflect pad (n_fft-hop)/2）。
//
// 实现方案（docs/架构设计.md §5.3）：reflect pad → Hann 窗折进 DFT 核 → conv_1d
// （核 [n_fft,1,2*bins]，常量图张量非 PARAM）→ 平方和开方 → mul_mat(mel_basis) → log/clamp。
// 布局：y=[T,B] → spec=[frames,bins,B] → mel=[frames,n_mels,B]（torch 形状反序的 ne 序）。
#pragma once

#include "traincpp/traincpp.h"

#include <string>

namespace rvc_train {
namespace train {

// STFT/mel 配置（对齐官方 configs/*.json 的 data 节）
struct StftMelConfig {
    int64_t n_fft  = 2048;
    int64_t hop    = 400;
    int64_t win    = 2048;
    int64_t n_mels = 125;
    double  sr     = 40000.0;
    double  fmin   = 0.0;
    double  fmax   = 0.0;       // <=0 表示 sr/2（官方 config 中 mel_fmax 为 null）
    float   mag_eps  = 2e-7f;   // |X| = sqrt(re²+im²+mag_eps)
    float   log_clip = 2e-6f;   // log(clamp(x, min=log_clip))
    bool    validate(std::string* err) const;
    int64_t bins() const { return n_fft / 2 + 1; }
    int64_t pad() const { return (n_fft - hop) / 2; }
    double  fmax_eff() const { return fmax > 0.0 ? fmax : sr * 0.5; }
    // 显式 reflect pad 后的帧数（= torch.stft(center=False) 的 frames）
    int64_t frames(int64_t t) const;
};

// 图内常量叶子（分配前创建；数据在 buffer 分配后由 stft_mel_fill_* 写入）
struct StftMelConsts {
    traincpp::Tensor* dft_kernel = nullptr;  // [n_fft,1,2*bins]，channels [0,bins)=cos、[bins,2bins)=-sin
    traincpp::Tensor* mel_basis  = nullptr;  // [bins,n_mels]（torch [n_mels,bins] 反序）
    traincpp::Tensor* mag_eps    = nullptr;  // [1] 标量叶子（sqrt 内的 2e-7）
};

void stft_mel_consts_init(traincpp::Context* ctx, const StftMelConfig& cfg, StftMelConsts* out);
void stft_mel_fill_dft_kernel(StftMelConsts* c, const StftMelConfig& cfg);
void stft_mel_fill_mel_basis(StftMelConsts* c, const StftMelConfig& cfg);
// 主机端生成 librosa（norm='slaney'、htk=False）口径的三角滤波器组；
// out 布局 [n_mels][bins] 行主序（与 golden mel_basis 文件布局一致）
void make_mel_filterbank_slaney(const StftMelConfig& cfg, float* out);

// 前向（均可微）：y=[T,B]（原始波形，未 padding）
// internals 非空时回填中间张量指针（调试用；生命周期同 Context）
struct StftInternals {
    traincpp::Tensor* kernel = nullptr;  // DFT 核 [n_fft,1,2*bins]
    traincpp::Tensor* padded = nullptr;  // reflect pad 后 [T+2p,1,B]
    traincpp::Tensor* conv   = nullptr;  // DFT conv 输出 [frames,2*bins,B]
    traincpp::Tensor* re     = nullptr;  // 实部 [frames,bins,B]
    traincpp::Tensor* im     = nullptr;  // 虚部 [frames,bins,B]
    traincpp::Tensor* spec   = nullptr;  // |X| [frames,bins,B]
};
traincpp::Tensor* stft_forward(traincpp::Context* ctx, const StftMelConsts& c,
                               const StftMelConfig& cfg, traincpp::Tensor* y,
                               StftInternals* internals = nullptr);
traincpp::Tensor* mel_spectrogram_forward(traincpp::Context* ctx, const StftMelConsts& c,
                                          const StftMelConfig& cfg, traincpp::Tensor* y);

} // namespace train
} // namespace rvc_train
