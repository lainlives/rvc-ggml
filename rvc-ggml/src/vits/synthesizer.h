// synthesizer.h
// ggml implementation of the RVC synthesizer (SynthesizerTrnMs256/768 NSF):
//   enc_p (TextEncoder relative-position attention + pitch embedding)
//   -> flow (4× residual coupling inverse transform + Flip)
//   -> dec (HiFiGAN NSF generator, sine excitation + noise convolution)
// Layout convention: graph tensors are uniformly [C, T] (C in ne0, corresponding to PyTorch [B,C,T] memory transpose),
//          the generator internally uses [T, C] alone (natural convolution order).

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
        int n_layers = 6;             // number of enc_p attention layers
        int kernel_size = 3;          // enc_p FFN kernel size
        int window_size = 10;         // relative-position window
        std::vector<int> resblock_kernel_sizes;
        std::vector<int> resblock_dilation_sizes;
        std::vector<int> upsample_rates;
        std::vector<int> upsample_kernel_sizes;
        int upsample_initial_channel = 512;
        int spk_embed_dim = 0;
        int gin_channels = 256;
        int flow_n_layers = 3;        // number of flow WN layers (detected from weights)
        int64_t hop_samples = 0;      // prod(upsample_rates)
        bool conv_flat = false;       // GGUF rvc.conv_flat: Conv1d weights use the flattened layout (quantization-friendly)
    };

    Config read_config(const GgufModel& model);

    // Constant inputs registered at build time (relative-position embeddings, flip matrix, etc.; written once after backend allocation)
    struct GraphInput {
        ggml_tensor* tensor = nullptr;
        std::vector<float> values;
    };

    struct Input {
        int64_t frames = 0;           // frame count (100fps)
        int feature_dim = 0;          // 256/768
        int speaker_id = 0;
        std::vector<float> features;  // [frames, feature_dim] row-major
        std::vector<int32_t> pitch;   // [frames] (has_f0)
        std::vector<float> source;    // [frames*hop] raw sine (has_f0)
        std::vector<float> noise;     // [frames, inter_channels] row-major
    };

    struct Output {
        int sample_rate = 0;
        int64_t samples = 0;          // number of audio samples
        std::vector<float> audio;
        std::vector<float> m_p;       // [frames, inter] row-major (for validation)
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

        // Run one full synthesis (a static graph is reused for the same frames/feature_dim/sid)
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

    // Subgraph builders (called by synthesizer; also unit-testable)
    //   stats: enc_p output [inter*2, T] (first half=m, second half=logs)
    ggml_tensor* build_text_encoder_stats(ggml_context* ctx, const GgufModel& model,
        const Config& cfg, ggml_tensor* features, ggml_tensor* pitch_i32,
        int64_t frames, std::vector<GraphInput>& constants);

    //   z_p [inter,T] -> z [inter,T] (inverse flow with reverse=True)
    ggml_tensor* build_flow_reverse(ggml_context* ctx, const GgufModel& model,
        const Config& cfg, ggml_tensor* z_p, ggml_tensor* g, int64_t frames,
        ggml_tensor* flip_matrix);

    //   z [inter,T] -> audio [T_final] (generator; output tensor [T_final, 1])
    ggml_tensor* build_generator(ggml_context* ctx, const GgufModel& model,
        const Config& cfg, ggml_tensor* z, ggml_tensor* g, ggml_tensor* source,
        int64_t frames);

} // namespace vits
} // namespace rvc
