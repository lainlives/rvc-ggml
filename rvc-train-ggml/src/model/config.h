// RVC-Train-GGML - RVC model configuration (src/model/config.h)
//
// ModelConfig is promoted from the training-state GGUF rvc.train.* metadata (TrainMeta),
// used by layer module assembly and shape inference since M2; for field semantics see docs/architecture.md §4.1/§5.6.
#pragma once

#include "model/weights.h"

#include <cstdint>
#include <string>
#include <vector>

namespace rvc_train {
namespace model {

// RVC model configuration (M2-1 skeleton):
//   - identity: kind (G/D), version (v1/v2), has_f0, form, resblock
//   - audio/spectrogram: sample_rate/hop/filter/win/mel/segment (small form may default to 0)
//   - structure: inter/hidden/filter/n_heads/n_layers/kernel/gin/spk/upsample_*
// Convention: spk_embed_dim follows the actual row count of emb_g.weight (handled by converter M1 rules).
struct ModelConfig {
    std::string kind;      // "G" (generator; training form includes enc_q) | "D" (discriminator)
    std::string version;   // "v1" (256 features) | "v2" (768 features)
    bool        has_f0 = false;
    std::string form;      // source form: "train" | "small" | "raw"
    std::string resblock;  // residual block type (currently only "1" = ResBlock1)

    // ---------------------------------------------------------- audio/spectrogram
    int64_t sample_rate    = 0;
    int64_t spec_channels  = 0;   // filter_length/2 + 1 (enc_q input channels)
    int64_t filter_length  = 0;
    int64_t hop_length     = 0;
    int64_t win_length     = 0;
    int64_t n_mel_channels = 0;
    float   mel_fmin = 0.0f;
    float   mel_fmax = 0.0f;
    int64_t segment_size   = 0;   // training segment length (samples); small form may be 0

    // ---------------------------------------------------------- model structure
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
    std::vector<int32_t> resblock_dilation_sizes;  // flattened (3 dilations per kernel)
    std::vector<int32_t> upsample_rates;
    std::vector<int32_t> upsample_kernel_sizes;
    std::vector<int32_t> d_periods;                // D model only: discriminator periods

    static ModelConfig from_meta(const TrainMeta& meta);

    // Structure validation (required for model assembly); on failure writes err and returns false.
    // Audio fields may default to 0 in small form and are not enforced; structure fields must be complete.
    bool validate(std::string* err) const;

    bool is_generator() const { return kind == "G"; }
    bool is_discriminator() const { return kind == "D"; }
    bool is_v2() const { return version == "v2"; }

    // ∏ upsample_rates: number of samples per z-domain frame (v1-40k = 400)
    int64_t hop_product() const;
    // segment_size (samples) / hop_length -> segment frames inside the model (v1-40k = 32); 0 if unknown
    int64_t z_frames() const;
};

// M7-2: v1-40k default model structure (fallback for TrainerConfig defaults and config model section defaults)
ModelConfig default_model_config_v1_40k();

// V2-2: v2-48k default model structure (fallback base when config `model.version=="v2"`, see trainer_config_from_json_text)
ModelConfig default_model_config_v2_48k();

} // namespace model
} // namespace rvc_train
