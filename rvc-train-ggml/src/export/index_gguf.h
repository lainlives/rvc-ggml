// index_gguf.h -- M4-5a/M4-5b: building/writing/validating the RVC retrieval index
//
// The output format matches the RVC-GGML read contract (see RVC-GGML src/rvc/retrieval.cpp:28-63):
//   KV: general.architecture="rvc_index", index.dim/nlist/ntotal/metric_type/nprobe (UINT32)
//   tensors (F32): index.centroids[dim,nlist], index.vectors[dim,ntotal] (concatenated per list),
//                index.list_offsets[nlist], index.list_lengths[nlist]
// - M4-5a MVP (single list): nlist=1, offsets=[0], lengths=[ntotal], nprobe=1, centroids=column means;
//   vector order is not shuffled (one-to-one with source feature row ids).
// - M4-5b full IVF (aligned with official train_index.py; not bit-reproducible):
//   permutation (reproducible via seed) -> when N>200000, MiniBatchKMeans reduces to 10000 centers ->
//   nlist=max(1,min(int(16*sqrt(N)),N//39)) -> k-means centroids -> grouped and concatenated by nearest list.

#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <ggml-backend.h>

namespace rvc_train::exporter {

struct IndexSummary {
    int64_t dim = 0;
    int64_t nlist = 0;
    int64_t ntotal = 0;
    int64_t nprobe = 0;
};

// features: row-major [n, dim]; writes index.gguf
bool build_write_index_gguf(const std::string& out_path, const float* features, int64_t n,
                            int64_t dim, IndexSummary* out, std::string* err);

// M4-5b full IVF build options
struct IvfBuildOptions {
    int64_t nlist = 0;          // 0 = automatic via the official formula (max(1,min(int(16*sqrt(N)),N//39)))
    uint32_t nprobe = 1;        // written to KV index.nprobe (official trained/added index uses 1)
    uint64_t seed = 1234;       // permutation/k-means seed (official has none; reproducible here)
    int64_t kmeans_iters = 25;  // IVF k-means iteration cap
    int64_t mini_centroids = 10000;  // MiniBatchKMeans target center count when N>200000 (official 10000)
    int64_t mini_epochs = 10;        // MiniBatchKMeans epoch cap (official sklearn max_iter=100)
    bool no_permute = false;    // debug: skip permutation
    bool verbose = true;
};

struct IvfBuildStats {
    int64_t ntotal = 0;
    int64_t nlist = 0;
    int64_t list_min = 0;
    int64_t list_max = 0;
    int64_t list_empty = 0;
    int64_t reduced_centroids = 0;  // >0 means MiniBatchKMeans was used (value = center count)
    double kmeans_seconds = 0.0;
    double total_seconds = 0.0;
};

// Full IVF build: features may be permuted in place / replaced by cluster centers (contents no longer match input row order after the call).
bool build_write_index_gguf_ivf(const std::string& out_path, std::vector<float>* features,
                                int64_t n, int64_t dim, const IvfBuildOptions& opt,
                                IndexSummary* out, IvfBuildStats* stats, std::string* err);

// Read back and validate through the RVC-GGML public API (rvc::RetrievalIndex); when query is non-null, run one top-1 self-query
bool validate_index_gguf(const std::string& path, ggml_backend_t backend, const float* query,
                         int64_t query_dim, IndexSummary* out, std::string* err);

} // namespace rvc_train::exporter
