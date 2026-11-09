// retrieval.h
// RVC FAISS IVF retrieval index (GGUF sidecar): nearest-neighbor search + feature blending.
// Aligned with infer/modules/vc/pipeline.py:
//   score, ix = index.search(npy, k=8)          # IndexIVFFlat, nprobe=1, L2²
//   weight = (1/score)^2 normalized; npy = Σ big_npy[ix] * weight
//   feats = npy * index_rate + (1 - index_rate) * feats

#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <ggml-backend.h>

class GgufModel;

namespace rvc {

    class RetrievalIndex {
    public:
        struct Hit {
            float score = 0.0f;
            int64_t position = 0;   // Position in vectors_ (list grouping order)
        };

        // Load a GGUF index (weights are uploaded via backend then copied back to host; the temporary model can be freed after loading)
        void load(const std::string& path, ggml_backend_t backend);

        bool loaded() const { return dim_ > 0; }
        int64_t dim() const { return dim_; }
        int64_t ntotal() const { return ntotal_; }
        int64_t nlist() const { return nlist_; }

        // Single-frame top-k (ascending L2²); search the nprobe nearest lists (aligned with FAISS IndexIVFFlat)
        void search(const float* query, int64_t k, std::vector<Hit>& out) const;

        // Batch blending: query [frames, dim] row-major -> [frames, dim] (after index_rate blending)
        std::vector<float> blend(const std::vector<float>& query, int64_t frames,
            int64_t k, float index_rate) const;

    private:
        int64_t dim_ = 0;
        int64_t nlist_ = 0;
        int64_t ntotal_ = 0;
        int64_t nprobe_ = 1;
        std::vector<float> centroids_;      // [dim, nlist]
        std::vector<float> vectors_;        // [dim, ntotal] (concatenated grouped by list)
        std::vector<int64_t> offsets_;      // [nlist]
        std::vector<int64_t> lengths_;      // [nlist]
    };

} // namespace rvc
