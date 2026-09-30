// rmvpe.h
// RMVPE 音高提取（DeepUnet + BiGRU + 360 类显著性头）。
// 结构对齐参考实现（0xShug0/audio.cpp，MIT）：4 张子图
//   feature（DeepUnet 窗口推理）/ gru_forward / gru_reverse / head。
// 数值目标：与 RVC Python（infer/lib/rmvpe.py）corr ≥ 0.999。

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <ggml.h>
#include <ggml-backend.h>

#include "gguf/gguf_model.h"

namespace rvc {
namespace rmvpe {

    struct Config {
        int sample_rate = 16000;
        int n_fft = 1024;
        int hop_length = 160;
        int win_length = 1024;
        int n_mels = 128;
        int n_classes = 360;
        int hidden_dim = 256;
        int feature_dim = 384;
        float mel_fmin = 30.0f;
        float mel_fmax = 8000.0f;
        float mel_clamp = 1e-5f;
        bool conv_flat = false;     // GGUF rvc.conv_flat：卷积权重为展平布局（量化友好）
    };

    Config read_config(const GgufModel& model);

    struct Result {
        int64_t frames = 0;                 // log-mel 帧数（未 pad）
        int64_t padded_frames = 0;          // 32 对齐后的帧数
        std::vector<float> mel;             // [n_mels, frames]（mel 行主序）
        std::vector<float> hidden;          // [frames, n_classes] 显著性
        std::vector<float> hidden_padded;   // [padded_frames, n_classes]
        std::vector<float> f0;              // [frames] Hz，0 表示无声
    };

    // 提取器：持有 4 张子图的上下文与缓存，可多次调用；
    // 输入长度变化时自动重建对应尺寸的图。
    class RmvpeExtractor {
    public:
        RmvpeExtractor(const GgufModel& model, ggml_backend_t backend);
        ~RmvpeExtractor();
        RmvpeExtractor(const RmvpeExtractor&) = delete;
        RmvpeExtractor& operator=(const RmvpeExtractor&) = delete;

        // 单声道 16k 浮点音频 -> 显著性/ f0（threshold 对应 Python thred=0.03）
        Result infer(const std::vector<float>& audio_16k, float threshold = 0.03f);

        // 仅 mel 前端（P3 自检用）
        std::vector<float> compute_mel(const std::vector<float>& audio_16k,
            int64_t* frames_out = nullptr) const;

    private:
        struct Impl;
        std::unique_ptr<Impl> impl_;
    };

} // namespace rmvpe
} // namespace rvc
