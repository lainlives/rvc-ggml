// RVC-Train-GGML - forward comparison harness (src/model/selftest.h)
//
// Mechanism (since M2-2):
//   tools/gen_golden_forward.py uses the runtime infer_pack to dump input/output/intermediate tensors per module
//   (.raw F32/I32 little-endian + manifest.<module>.json); this module reads the manifest,
//   assembles the module with ModelParams -> build graph -> allocate -> load params -> inject inputs -> compute -> compare tensor by tensor.
// Verdict: shapes match one by one; F32 relative error = max_abs / max(1, max|ref|) ≤ 1e-4.
#pragma once

#include "common/json.h"
#include "model/config.h"
#include "model/model.h"
#include "model/weights.h"

#include "traincpp/traincpp.h"

#include <array>
#include <cstdint>
#include <functional>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace rvc_train {
namespace model {

// golden tensor entry (shape = torch shape; ne = reversed)
struct GoldenTensor {
    std::string name;
    std::string file;      // relative to the golden dir
    std::string dtype;     // "f32" | "i32"
    std::string role;      // "input" | "output" | "intermediate"
    std::array<int64_t, 4> ne{1, 1, 1, 1};
    int nd = 1;            // number of shape dims (1..4)

    bool is_input() const { return role == "input"; }
};

struct GoldenManifest {
    std::string module;
    std::string dir;
    std::vector<GoldenTensor> tensors;
    common::JsonValue params;    // module-specific parameters (Null if none)

    const GoldenTensor* find_tensor(const std::string& name) const;
    int64_t param_int(const char* key, int64_t def = 0) const;
    std::string param_str(const char* key, const char* def = "") const;
    void param_int_array(const char* key, std::vector<int64_t>* out) const;
};

// Read {dir}/manifest.<module>.json (falls back to manifest.json when missing and validates the module field)
bool golden_manifest_load(const std::string& dir, const std::string& module, GoldenManifest* out,
                          std::string* err);

// Module build context: the runner pre-creates input tensors; builders register tensors to compare
struct SelftestBuildIO {
    traincpp::Context* ctx = nullptr;
    const ModelConfig* cfg = nullptr;
    const ModelParams* params = nullptr;
    const GoldenManifest* golden = nullptr;
    std::map<std::string, traincpp::Tensor*> inputs;
    std::vector<std::pair<std::string, traincpp::Tensor*>> compares;
    // Optional prepare callback: called after the runner injects input data and before graph_compute
    // (for host-side computations that depend on input data, e.g. dec's SineGen excitation source; must write into pre-built leaf tensors itself)
    std::function<bool(std::string*)> prepare;

    traincpp::Tensor* find_input(const std::string& name) const;
    void add_compare(const std::string& name, traincpp::Tensor* t);
};

using SelftestBuildFn = bool (*)(SelftestBuildIO& io, std::string* err);

// Per-module builders (registered in selftest.cpp)
bool build_selftest_emb_g(SelftestBuildIO& io, std::string* err);
bool build_selftest_wn(SelftestBuildIO& io, std::string* err);
bool build_selftest_resblock1(SelftestBuildIO& io, std::string* err);
bool build_selftest_dec(SelftestBuildIO& io, std::string* err);
bool build_selftest_flow(SelftestBuildIO& io, std::string* err);
bool build_selftest_enc_p(SelftestBuildIO& io, std::string* err);
bool build_selftest_enc_q(SelftestBuildIO& io, std::string* err);
// The discriminator uses a separate D weights file (f0D40k.train.gguf) and is not part of module="all"
bool build_selftest_disc(SelftestBuildIO& io, std::string* err);
// Whole-model training forward (SynthesizerTrn: two dec paths, no-slice + fixed ids slice)
bool build_selftest_synth(SelftestBuildIO& io, std::string* err);

// Forward comparison entry (module="all" runs emb_g/wn/resblock1/dec/flow/enc_p/enc_q/synth in order;
// disc must be run separately with the D weights specified)
// Returns: 0 = all passed; 1 = failed; 2 = parameter/environment error
int run_forward_selftest(const std::string& weights_path, const std::string& golden_dir,
                         const std::string& module, traincpp::Device* dev);

// Backward comparison entry (M3-1: fixed cotangent VJP, gate rel ≤1e-3; gradcheck gate ≤2e-2)
//   golden = manifest.backward.<module>.json (role: input/cotangent/grad/loss) +
//            <module>.cot_<output>.f32.raw / <module>.<name>.grad.f32.raw;
//   module="all" as in forward (disc must be run separately with the D weights specified);
//   gradcheck is a semicolon-separated list of "param_name[:element_index]" (empty = skip; "all" = first/middle element of all params);
//   when dump_grads is non-empty, write the C++ side grad_acc/loss to that directory (for diagnostics; file names follow the same convention as Python golden).
int run_backward_selftest(const std::string& weights_path, const std::string& golden_dir,
                          const std::string& module, const std::string& gradcheck,
                          const std::string& dump_grads, traincpp::Device* dev);

} // namespace model
} // namespace rvc_train
