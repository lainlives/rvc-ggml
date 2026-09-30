// synthesizer.h
// RVC 合成器（SynthesizerTrnMs256/768 NSF）的 ggml 实现：
//   enc_p（TextEncoder 相对位置注意力 + pitch 嵌入）
//   -> flow（4×残差耦合逆变换 + Flip）
//   -> dec（HiFiGAN NSF 生成器，sine 激励 + 噪声卷积）
// 布局约定：图中张量统一 [C, T]（C 在 ne0，与 PyTorch [B,C,T] 内存转置对应），
//          生成器内部单独使用 [T, C]（卷积自然序）。

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <ggml.h>
#include <ggml-backend.h>

#include "gguf/gguf_model.h"

namespace rvc {
class GgmlGraphRunner;   // engine.h

namespace vits {

    struct Config {
        std::string version;          // "v1"/"v2"
        bool has_f0 = true;
        int sample_rate = 40000;
        int n_spk = 0;
        int inter_channels = 192;
        int hidden_channels = 192;
        int filter_channels = 768;
        int n_heads = 2;
        int n_layers = 6;             // enc_p 注意力层数
        int kernel_size = 3;          // enc_p FFN 卷积核
        int window_size = 10;         // 相对位置窗口
        std::vector<int> resblock_kernel_sizes;
        std::vector<int> resblock_dilation_sizes;
        std::vector<int> upsample_rates;
        std::vector<int> upsample_kernel_sizes;
        int upsample_initial_channel = 512;
        int spk_embed_dim = 0;
        int gin_channels = 256;
        int flow_n_layers = 3;        // flow WN 层数（从权重探测）
        int64_t hop_samples = 0;      // prod(upsample_rates)
        bool conv_flat = false;       // GGUF rvc.conv_flat：Conv1d 权重为展平布局（量化友好）
    };

    Config read_config(const GgufModel& model);

    // 构建期登记的常量输入（相对位置嵌入、翻转矩阵等，分配到后端后写入一次）
    struct GraphInput {
        ggml_tensor* tensor = nullptr;
        std::vector<float> values;
    };

    struct Input {
        int64_t frames = 0;           // 帧数（100fps）
        int feature_dim = 0;          // 256/768
        int speaker_id = 0;
        std::vector<float> features;  // [frames, feature_dim] 行主序
        std::vector<int32_t> pitch;   // [frames]（has_f0）
        std::vector<float> source;    // [frames*hop] raw sine（has_f0）
        std::vector<float> noise;     // [frames, inter_channels] 行主序
    };

    struct Output {
        int sample_rate = 0;
        int64_t samples = 0;          // 音频样本数
        std::vector<float> audio;
        std::vector<float> m_p;       // [frames, inter] 行主序（验证用）
        std::vector<float> logs_p;
        std::vector<float> z_p;
        std::vector<float> z;
    };

    class Synthesizer {
    public:
        Synthesizer(const GgufModel& model, ggml_backend_t backend);
        ~Synthesizer();
        Synthesizer(const Synthesizer&) = delete;
        Synthesizer& operator=(const Synthesizer&) = delete;

        const Config& config() const { return cfg_; }

        // 运行一次完整合成（相同 frames/feature_dim/sid 会复用静态图）
        Output infer(const Input& input);

    private:
        void build(int64_t frames, int feature_dim, int speaker_id);

        Config cfg_;
        const GgufModel& model_;
        ggml_backend_t backend_ = nullptr;

        std::unique_ptr<GgmlGraphRunner> runner_;
        int64_t built_frames_ = -1;
        int built_feature_dim_ = -1;
        int built_sid_ = -1;
        bool constants_uploaded_ = false;

        std::vector<GraphInput> constants_;
        ggml_tensor* in_features_ = nullptr;
        ggml_tensor* in_pitch_ = nullptr;
        ggml_tensor* in_noise_ = nullptr;
        ggml_tensor* in_source_ = nullptr;
        ggml_tensor* out_audio_ = nullptr;
        ggml_tensor* out_m_ = nullptr;
        ggml_tensor* out_logs_ = nullptr;
        ggml_tensor* out_zp_ = nullptr;
        ggml_tensor* out_z_ = nullptr;
    };

    // 子图构建（供 synthesizer 调用，也可单测）
    //   stats: enc_p 输出 [inter*2, T]（前 half=m，后 half=logs）
    ggml_tensor* build_text_encoder_stats(ggml_context* ctx, const GgufModel& model,
        const Config& cfg, ggml_tensor* features, ggml_tensor* pitch_i32,
        int64_t frames, std::vector<GraphInput>& constants);

    //   z_p [inter,T] -> z [inter,T]（reverse=True 的逆流）
    ggml_tensor* build_flow_reverse(ggml_context* ctx, const GgufModel& model,
        const Config& cfg, ggml_tensor* z_p, ggml_tensor* g, int64_t frames,
        ggml_tensor* flip_matrix);

    //   z [inter,T] -> audio [T_final]（生成器，输出张量 [T_final, 1]）
    ggml_tensor* build_generator(ggml_context* ctx, const GgufModel& model,
        const Config& cfg, ggml_tensor* z, ggml_tensor* g, ggml_tensor* source,
        int64_t frames);

} // namespace vits
} // namespace rvc
