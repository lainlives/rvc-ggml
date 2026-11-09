// model_gguf.h -- M6: export training-state G GGUF as an RVC-GGML-readable inference GGUF
//
// Contract (mirrors the model export in RVC-GGML/scripts/export_gguf.py):
//   - Export only the inference subset: enc_p / flow / dec / emb_g (filters out enc_q and the discriminators)
//   - Fold weight_norm: w = g·v/‖v‖ (standard dim0 convention)
//   - Flatten Conv1d to conv_flat v2: [O,I,K] -> (O, I*K); dec.ups.* (ConvT1d) is not flattened
//   - Write the full rvc.* metadata (upsample_rates/kernel_sizes are required, conv_flat=true)
//
// Implementation uses train.cpp's built-in GGUF read/write (traincpp::GgufFile / GgufWriter),
// without depending on ggml; compiled into rvc-train (host-only, no device required).
#pragma once

#include <string>
#include <unordered_map>
#include <vector>

namespace rvc_train {
namespace exporter {

struct ModelExportOptions {
    std::string in_path;       // training-state G GGUF (general.architecture = "rvc-train")
    std::string out_path;      // output inference GGUF (general.architecture = "rvc")
    std::string general_name;  // optional general.name; empty uses the output file stem
    // When the input training-state GGUF loses the true rank because a trailing 1-dim is folded (e.g. Conv1d weights with O=1),
    // use the source file's true ndims==3 list to force Conv1d flattening (used by the M7-4 live-snapshot bridge;
    // when the list is empty, behavior is identical to M6).
    std::vector<std::string> force_flatten;
};

// Export an inference GGUF. Returns true on success; returns false on failure and writes err.
bool export_inference_gguf(const ModelExportOptions& opt, std::string* err);

// Scan the GGUF header and return each tensor's **stored n_dims** (low-level, read-only; lets the bridge repair folded dims).
bool gguf_read_tensor_ndims(const std::string& path, std::unordered_map<std::string, int>* out,
                            std::string* err);

} // namespace exporter
} // namespace rvc_train
