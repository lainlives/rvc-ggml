// RVC-Train-GGML - RVC model parameter container (src/model/model.h)
//
// Tensor names = torch state_dict key names (consistent with training-state GGUF). M2-1 only creates parameter tensors
// (F32, shapes from the file; no graph building, no data loading) for layer modules to fetch by name;
// data loading and graph assembly are wired in by each module since M2-2 (this file is only a parameter container).
#pragma once

#include "model/weights.h"

#include "traincpp/traincpp.h"

#include <string>
#include <vector>

namespace rvc_train {
namespace model {

// Create a single parameter tensor by name: after checking existence, create an F32 tensor with the file's shape (data allocated by the backend buffer).
// Returns nullptr and writes err on failure.
traincpp::Tensor* build_param(traincpp::Context* ctx, const TrainWeights& weights,
                              const char* name, std::string* err);

// Model parameter container: flat parameter list (for the optimizer) + grouping by top-level module (for reports/assembly) + expected manifest.
class ModelParams {
public:
    // Create parameters for all tensors in the file (order = file order); returns false if any one fails.
    bool build_all(traincpp::Context* ctx, const TrainWeights& weights, std::string* err);

    int64_t count() const { return (int64_t) entries_.size(); }
    traincpp::Tensor* find(const char* name) const;

    // Top-level module names (e.g. enc_p/enc_q/flow/dec/emb_g/discriminators; in first-appearance order)
    const std::vector<std::string>& group_names() const { return group_names_; }
    int64_t group_count(const std::string& group) const;

    // Expected manifest (name + file ne) for TrainWeights::check_expected self-test
    const std::vector<TrainWeights::Expect>& expected() const { return expected_; }
    // Flat parameter list (order = file order; used by the M5 optimizer)
    const traincpp::ParamList& params() const { return params_; }

private:
    struct Entry {
        std::string name;
        traincpp::Tensor* tensor = nullptr;
        int64_t group_index = -1;
    };

    std::vector<Entry> entries_;
    std::vector<std::string> group_names_;
    std::vector<int64_t> group_counts_;
    std::vector<TrainWeights::Expect> expected_;
    traincpp::ParamList params_;
};

} // namespace model
} // namespace rvc_train
