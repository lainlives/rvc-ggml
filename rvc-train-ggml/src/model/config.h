// RVC-Train-GGML - RVC 模型配置（src/model/config.h）
//
// ModelConfig 由训练态 GGUF 的 rvc.train.* 元数据（TrainMeta）提升而来，
// 供 M2 起各层模块装配与形状推导使用；字段语义见 docs/架构设计.md §4.1/§5.6。
#pragma once

#include "model/weights.h"

#include <cstdint>
#include <string>
#include <vector>

namespace rvc_train {
namespace model {

// RVC 模型配置（M2-1 骨架）：
//   - 身份：kind（G/D）、version（v1/v2）、has_f0、form、resblock
//   - 音频/谱：sample_rate/hop/filter/win/mel/segment（small 形态可能缺省为 0）
//   - 结构：inter/hidden/filter/n_heads/n_layers/kernel/gin/spk/upsample_*
// 约定：spk_embed_dim 以 emb_g.weight 实际行数为准（转换器 M1 规则已处理）。
struct ModelConfig {
    std::string kind;      // "G"（生成器，训练态含 enc_q）| "D"（判别器）
    std::string version;   // "v1"（256 特征）| "v2"（768 特征）
    bool        has_f0 = false;
    std::string form;      // 源形态："train" | "small" | "raw"
    std::string resblock;  // 残差块类型（当前仅支持 "1" = ResBlock1）

    // ---------------------------------------------------------- 音频/谱
    int64_t sample_rate    = 0;
    int64_t spec_channels  = 0;   // filter_length/2 + 1（enc_q 输入通道）
    int64_t filter_length  = 0;
    int64_t hop_length     = 0;
    int64_t win_length     = 0;
    int64_t n_mel_channels = 0;
    float   mel_fmin = 0.0f;
    float   mel_fmax = 0.0f;
    int64_t segment_size   = 0;   // 训练段长（采样点）；small 形态可能为 0

    // ---------------------------------------------------------- 模型结构
    int64_t inter_channels  = 0;
    int64_t hidden_channels = 0;
    int64_t filter_channels = 0;
    int64_t n_heads         = 0;
    int64_t n_layers        = 0;
    int64_t kernel_size     = 0;
    float   p_dropout       = 0.0f;
    int64_t gin_channels    = 0;
    int64_t spk_embed_dim   = 0;
    int64_t upsample_initial_channel = 0;
    std::vector<int32_t> resblock_kernel_sizes;
    std::vector<int32_t> resblock_dilation_sizes;  // 展平（每组 kernel 3 个 dilation）
    std::vector<int32_t> upsample_rates;
    std::vector<int32_t> upsample_kernel_sizes;
    std::vector<int32_t> d_periods;                // 仅 D 模型：判别器周期

    static ModelConfig from_meta(const TrainMeta& meta);

    // 结构校验（模型装配所必需项）；失败时写入 err 并返回 false。
    // 音频字段在 small 形态可能缺省（0），不做强制；结构字段必须完整。
    bool validate(std::string* err) const;

    bool is_generator() const { return kind == "G"; }
    bool is_discriminator() const { return kind == "D"; }
    bool is_v2() const { return version == "v2"; }

    // ∏ upsample_rates：z 域 1 帧对应的采样点数（v1-40k = 400）
    int64_t hop_product() const;
    // segment_size（采样点）/ hop_length → 模型内 segment 帧数（v1-40k = 32）；未知返回 0
    int64_t z_frames() const;
};

// M7-2：v1-40k 默认模型结构（TrainerConfig 默认值与 config 缺省 model 段的回退）
ModelConfig default_model_config_v1_40k();

// V2-2：v2-48k 默认模型结构（config `model.version=="v2"` 时作为回退基底，见 trainer_config_from_json_text）
ModelConfig default_model_config_v2_48k();

} // namespace model
} // namespace rvc_train
