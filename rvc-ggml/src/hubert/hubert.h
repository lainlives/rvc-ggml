// hubert.h
// HuBERT/ContentVec feature extraction (ggml graph).
// Aligned with fairseq HuBERTModel.forward:
//   conv feature extractor -> layer_norm -> post_extract_proj -> pos_conv -> layer_norm
//   -> 12 x TransformerEncoderLayer -> output (v1: layer 10 + final_proj; v2: layer 12)

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <ggml.h>
#include <ggml-backend.h>

#include "gguf/gguf_model.h"

namespace rvc {
namespace hubert {

    struct Config {
        int n_layers = 12;
        int hidden = 768;
        int ffn = 3072;
        int heads = 12;
        int conv_pos = 128;
        int conv_pos_groups = 16;
        int final_dim = 256;
        bool conv_flat = false;     // GGUF rvc.conv_flat: front-end conv weights use a flattened layout
        std::vector<int> conv_dim{ 512, 512, 512, 512, 512, 512, 512 };
        std::vector<int> conv_kernel{ 10, 3, 3, 3, 3, 2, 2 };
        std::vector<int> conv_stride{ 5, 2, 2, 2, 2, 2, 2 };
    };

    Config read_config(const GgufModel& model);

    // Debug: when the RVC_HUBERT_DUMP=1 environment variable is set, build_graph records intermediate tensors
    // (the caller must set_output + expand them before they can be read and compared)
    struct DebugTensor {
        std::string name;
        ggml_tensor* tensor;
    };
    const std::vector<DebugTensor>& debug_tensors();

    // Build the graph. input: [T, 1] (T = sample count after padding to a multiple of 320)
    // Returns: pointer to the last node built; the caller takes tensors via build_graph's out params.
    // out_layer9  : [T_tokens, hidden] (for v1)
    // out_layer12 : [T_tokens, hidden] (for v2)
    // out_proj    : [T_tokens, final_dim] (v1 final features)
    void build_graph(ggml_context* ctx,
        const GgufModel& model,
        const Config& cfg,
        ggml_tensor* input,
        ggml_tensor** out_proj,
        ggml_tensor** out_layer9,
        ggml_tensor** out_layer12);

    // Convenience: load model + build graph + run + fetch results (for self-test)
    //   audio: mono 16k (automatically padded to a multiple of 320 internally)
    //   outputs: result tensors (original length, unpadded)
    struct Output {
        std::vector<float> v1_final_proj;   // [T_tokens, 256]
        std::vector<float> v1_layer9;       // [T_tokens, 768]
        std::vector<float> v2_layer12;      // [T_tokens, 768]
        int64_t frames = 0;
    };

    // Features [frames, dim] (row-major, matching PyTorch [T, C])
    struct Features {
        int64_t frames = 0;
        int64_t dim = 0;
        std::vector<float> values;
    };

    // Reusable encoder (caches graphs by input length) for the full pipeline
    class HubertEncoder {
    public:
        HubertEncoder(const GgufModel& model, ggml_backend_t backend);
        ~HubertEncoder();
        HubertEncoder(const HubertEncoder&) = delete;
        HubertEncoder& operator=(const HubertEncoder&) = delete;

        const Config& config() const { return cfg_; }

        // audio: mono 16k float; v2=true returns the last layer's 768 dims, otherwise v1 final_proj's 256 dims
        Features extract(const std::vector<float>& audio_16k, bool v2);

    private:
        void build(int64_t samples);

        struct Impl;
        std::unique_ptr<Impl> impl_;
        Config cfg_;
    };

} // namespace hubert
} // namespace rvc
