// retrieval.cpp
// C++ implementation of FAISS IndexIVFFlat retrieval (nprobe=1).

#include "rvc/retrieval.h"

#include "gguf/gguf_model.h"

#include <algorithm>
#include <cmath>
#include <cfloat>
#include <stdexcept>

namespace rvc {

namespace {

    inline float l2_sqr(const float* a, const float* b, int64_t dim) {
        float sum = 0.0f;
        for (int64_t i = 0; i < dim; ++i) {
            const float d = a[i] - b[i];
            sum += d * d;
        }
        return sum;
    }

} // namespace

void RetrievalIndex::load(const std::string& path, ggml_backend_t backend) {
    GgufModel model;
    model.load(path, backend);

    dim_ = model.meta_u32("index.dim", 0);
    nlist_ = model.meta_u32("index.nlist", 0);
    ntotal_ = model.meta_u32("index.ntotal", 0);
    nprobe_ = model.meta_u32("index.nprobe", 1);
    if (dim_ <= 0 || nlist_ <= 0) {
        throw std::runtime_error("Retrieval index metadata is missing (dim/nlist)");
    }
    if (nprobe_ <= 0) {
        nprobe_ = 1;
    }
    if (nprobe_ > nlist_) {
        nprobe_ = nlist_;
    }

    centroids_ = model.read_f32("index.centroids");
    vectors_ = model.read_f32("index.vectors");
    const auto offsets = model.read_f32("index.list_offsets");
    const auto lengths = model.read_f32("index.list_lengths");
    if (static_cast<int64_t>(offsets.size()) != nlist_ ||
        static_cast<int64_t>(lengths.size()) != nlist_) {
        throw std::runtime_error("Retrieval index list_offsets/lengths size mismatch");
    }
    offsets_.resize(static_cast<size_t>(nlist_));
    lengths_.resize(static_cast<size_t>(nlist_));
    for (int64_t i = 0; i < nlist_; ++i) {
        offsets_[static_cast<size_t>(i)] = static_cast<int64_t>(offsets[static_cast<size_t>(i)]);
        lengths_[static_cast<size_t>(i)] = static_cast<int64_t>(lengths[static_cast<size_t>(i)]);
    }
    if (static_cast<int64_t>(centroids_.size()) != dim_ * nlist_ ||
        static_cast<int64_t>(vectors_.size()) != dim_ * ntotal_) {
        throw std::runtime_error("Retrieval index tensor element count mismatch");
    }
}

void RetrievalIndex::search(const float* query, int64_t k, std::vector<Hit>& out) const {
    out.clear();
    if (!loaded() || k <= 0) {
        return;
    }
    // 1) Centroid distances; take the nprobe nearest lists (ascending)
    std::vector<Hit> lists;
    lists.reserve(static_cast<size_t>(nlist_));
    for (int64_t i = 0; i < nlist_; ++i) {
        lists.push_back({ l2_sqr(query, centroids_.data() + i * dim_, dim_), i });
    }
    const int64_t probe = std::min(nprobe_, nlist_);
    std::partial_sort(lists.begin(), lists.begin() + probe, lists.end(),
        [](const Hit& a, const Hit& b) {
            return a.score != b.score ? a.score < b.score : a.position < b.position;
        });

    // 2) Scan these lists in turn, maintaining a top-k max-heap (aligned with FAISS scan_one_list semantics)
    std::vector<Hit>* heap = &out;
    auto cmp = [](const Hit& a, const Hit& b) { return a.score < b.score; };
    for (int64_t li = 0; li < probe; ++li) {
        const int64_t list_id = lists[static_cast<size_t>(li)].position;
        const int64_t begin = offsets_[static_cast<size_t>(list_id)];
        const int64_t count = lengths_[static_cast<size_t>(list_id)];
        for (int64_t j = 0; j < count; ++j) {
            const int64_t pos = begin + j;
            const float d = l2_sqr(query, vectors_.data() + pos * dim_, dim_);
            if (static_cast<int64_t>(heap->size()) < k) {
                heap->push_back({ d, pos });
                std::push_heap(heap->begin(), heap->end(), cmp);
            }
            else if (d < heap->front().score) {
                std::pop_heap(heap->begin(), heap->end(), cmp);
                heap->back() = { d, pos };
                std::push_heap(heap->begin(), heap->end(), cmp);
            }
        }
    }
    // 3) Output in ascending order (ties on score broken by position, ensuring determinism)
    std::sort(heap->begin(), heap->end(),
        [](const Hit& a, const Hit& b) {
            return a.score != b.score ? a.score < b.score : a.position < b.position;
        });
}

std::vector<float> RetrievalIndex::blend(const std::vector<float>& query, int64_t frames,
    int64_t k, float index_rate) const {
    std::vector<float> out(query.size(), 0.0f);
    if (!loaded() || frames <= 0) {
        return out;
    }
    std::vector<Hit> hits;
    std::vector<float> weights;
    for (int64_t f = 0; f < frames; ++f) {
        const float* q = query.data() + f * dim_;
        search(q, k, hits);
        if (hits.empty()) {
            continue;
        }
        weights.resize(hits.size());
        float wsum = 0.0f;
        for (size_t i = 0; i < hits.size(); ++i) {
            const float inv = hits[i].score > 0.0f ? 1.0f / hits[i].score : 1.0e30f;
            const float w = inv * inv;
            weights[i] = w;
            wsum += w;
        }
        if (!(wsum > 0.0f)) {
            continue;
        }
        float* dst = out.data() + f * dim_;
        for (size_t i = 0; i < hits.size(); ++i) {
            const float w = weights[i] / wsum;
            const float* vec = vectors_.data() + hits[i].position * dim_;
            for (int64_t c = 0; c < dim_; ++c) {
                dst[c] += w * vec[c];
            }
        }
        // RVC blending: npy * index_rate + (1 - index_rate) * feats
        const float* src = query.data() + f * dim_;
        for (int64_t c = 0; c < dim_; ++c) {
            dst[c] = dst[c] * index_rate + (1.0f - index_rate) * src[c];
        }
    }
    return out;
}

} // namespace rvc
