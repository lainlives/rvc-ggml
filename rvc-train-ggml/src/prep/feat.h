// feat.h -- M4-3 HuBERT features (official 81eed5e extract_hubert_feature / runtime fairseq semantics)
//
// Semantic baseline (golden = runtime `infer/modules/train/extract_feature_print.py`, equivalent to the official HF path):
//   audio16k (raw F32, no normalization, no padding) -> fairseq extract_features:
//     v1 = final_proj(layer9)  -> 256 dims
//     v2 = layer12             -> 768 dims
//   frames = floor((L-400)/320)+1 (conv frontend is 7 layers with no padding; L=16k sample count)
//
// Implementation notes (M4-3, Q22):
//   rvc_core's `hubert::build_graph` at pos_conv uses a self-built `pad1d` (two **uninitialized** zero-constant
//   tensors + concat) to express padding; a single graph run is correct, but after **repeated computes** on the
//   same graph the gallocr-reused buffers leave stale data in the pad tensors -> silent drift (measured: deviation
//   from the 2nd run on, even NaN after rebuilding across lengths). This file uses the rvc_core public API
//   (`hubert::build_graph` + `GgmlGraphRunner`) to build its own executor, explicitly zeroing the zero-constant
//   leaves named "pad_*" before every compute, making it consistent with fairseq and repeatable (RVC-GGML sources unmodified).
//   Evidence and reproduction: `docs/feedback-rvc-ggml-hubert-duplicate-compute-Q22.md`.
//
// Output is row-major [frames, dim], matching torch [T, C].

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "gguf/gguf_model.h"
#include "hubert/hubert.h"

namespace rvc_train::prep {

struct FeatResult {
    int64_t frames = 0;
    int64_t dim = 0;
    std::vector<float> values;  // [frames, dim] row-major (= torch [T, C])
};

// Version string -> v2 flag (true only for "v2"; callers should validate the value first)
inline bool version_is_v2(const std::string& version) { return version == "v2"; }

// HuBERT feature extractor (caches graphs by input length; safe for repeated computes)
class HubertFeatureExtractor {
public:
    HubertFeatureExtractor(const GgufModel& model, ggml_backend_t backend);
    ~HubertFeatureExtractor();
    HubertFeatureExtractor(const HubertFeatureExtractor&) = delete;
    HubertFeatureExtractor& operator=(const HubertFeatureExtractor&) = delete;

    const rvc::hubert::Config& config() const;

    // Input: normalized 16k chunk (matches official 1_16k_wavs content; original length, no padding, no normalization)
    // v2=false -> v1 256 dims; v2=true -> v2 768 dims
    FeatResult extract(const std::vector<float>& audio_16k, bool v2);

private:
    void build(int64_t samples);
    void zero_pads();

    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// Compatibility wrapper: same semantics as `HubertFeatureExtractor::extract`
FeatResult compute_hubert_feat(const std::vector<float>& audio16k,
                               HubertFeatureExtractor& hubert, bool v2);

} // namespace rvc_train::prep
