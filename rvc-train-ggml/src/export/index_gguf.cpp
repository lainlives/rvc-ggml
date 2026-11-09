// index_gguf.cpp -- M4-5a/M4-5b implementation (see the index_gguf.h header comment)

#include "export/index_gguf.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <random>
#include <sstream>
#include <vector>

#include "export/gguf_writer.h"
#include "export/kmeans.h"
#include "rvc/retrieval.h"

namespace rvc_train::exporter {
namespace {

// Official train_index.py: n_ivf = max(1, min(int(16*sqrt(N)), N // 39))
int64_t official_nlist(int64_t n) {
    const int64_t a = static_cast<int64_t>(16.0 * std::sqrt(static_cast<double>(n)));
    const int64_t b = n / 39;
    int64_t v = std::min(a, b);
    if (v < 1) {
        v = 1;
    }
    return v;
}

// Row-level random permutation (Fisher-Yates + in-place cycle permutation; uniform random order, reproducible via seed)
void shuffle_rows(std::vector<float>* data, int64_t rows, int64_t dim, uint64_t seed) {
    if (rows <= 1 || dim <= 0) {
        return;
    }
    std::vector<int64_t> order(static_cast<std::size_t>(rows));
    std::iota(order.begin(), order.end(), int64_t{0});
    std::mt19937 rng(static_cast<uint32_t>(seed));
    std::shuffle(order.begin(), order.end(), rng);
    for (int64_t i = 0; i < rows; ++i) {
        while (order[static_cast<std::size_t>(i)] != i) {
            const int64_t j = order[static_cast<std::size_t>(i)];
            float* ri = data->data() + i * dim;
            float* rj = data->data() + j * dim;
            for (int64_t d = 0; d < dim; ++d) {
                std::swap(ri[d], rj[d]);
            }
            std::swap(order[static_cast<std::size_t>(i)], order[static_cast<std::size_t>(j)]);
        }
    }
}

} // namespace

bool build_write_index_gguf(const std::string& out_path, const float* features, int64_t n,
                            int64_t dim, IndexSummary* out, std::string* err) {
    if (features == nullptr || n <= 0 || dim <= 0) {
        if (err) {
            *err = "index build input is empty/invalid (n=" + std::to_string(n) + " dim=" + std::to_string(dim) +
                   ")";
        }
        return false;
    }

    // MVP: single list; centroids = column means (only to satisfy the reader-side tensor contract; no effect when nlist=1)
    std::vector<float> centroids(static_cast<std::size_t>(dim), 0.0f);
    for (int64_t i = 0; i < n; ++i) {
        const float* row = features + i * dim;
        for (int64_t d = 0; d < dim; ++d) {
            centroids[static_cast<std::size_t>(d)] += row[d];
        }
    }
    for (int64_t d = 0; d < dim; ++d) {
        centroids[static_cast<std::size_t>(d)] /= static_cast<float>(n);
    }
    const std::vector<float> offsets{0.0f};
    const std::vector<float> lengths{static_cast<float>(n)};

    GgufWriter w;
    w.set_str("general.architecture", "rvc_index");
    w.set_str("general.name", "RVC-Train-GGML index (MVP single-list)");
    w.set_str("general.description", "FAISS IndexIVFFlat sidecar for RVC retrieval blending");
    w.set_u32("index.dim", static_cast<uint32_t>(dim));
    w.set_u32("index.nlist", 1);
    w.set_u32("index.ntotal", static_cast<uint32_t>(n));
    w.set_u32("index.metric_type", 1);  // METRIC_L2 (ignored by the RVC-GGML reader)
    w.set_u32("index.nprobe", 1);

    if (!w.add_f32_2d("index.centroids", centroids.data(), 1, dim, err) ||
        !w.add_f32_2d("index.vectors", features, n, dim, err) ||
        !w.add_f32_1d("index.list_offsets", offsets.data(), 1, err) ||
        !w.add_f32_1d("index.list_lengths", lengths.data(), 1, err)) {
        return false;
    }
    if (!w.write(out_path, err)) {
        return false;
    }

    if (out != nullptr) {
        out->dim = dim;
        out->nlist = 1;
        out->ntotal = n;
        out->nprobe = 1;
    }
    if (err) {
        err->clear();
    }
    return true;
}

bool validate_index_gguf(const std::string& path, ggml_backend_t backend, const float* query,
                         int64_t query_dim, IndexSummary* out, std::string* err) {
    try {
        rvc::RetrievalIndex idx;
        idx.load(path, backend);
        if (!idx.loaded()) {
            if (err) {
                *err = "index is empty after loading: " + path;
            }
            return false;
        }
        std::ostringstream ss;
        if (out != nullptr) {
            out->dim = idx.dim();
            out->nlist = idx.nlist();
            out->ntotal = idx.ntotal();
            out->nprobe = 0;  // RetrievalIndex does not expose nprobe
        }
        ss << "dim=" << idx.dim() << " nlist=" << idx.nlist() << " ntotal=" << idx.ntotal();
        if (query != nullptr && query_dim == idx.dim()) {
            std::vector<rvc::RetrievalIndex::Hit> hits;
            idx.search(query, 1, hits);
            if (hits.empty()) {
                if (err) {
                    *err = "self-query returned no hits: " + path;
                }
                return false;
            }
            ss << " self_top1_score=" << hits[0].score;
        }
        if (err) {
            *err = ss.str();
        }
        return true;
    } catch (const std::exception& e) {
        if (err) {
            *err = std::string("failed to load index: ") + e.what();
        }
        return false;
    }
}

bool build_write_index_gguf_ivf(const std::string& out_path, std::vector<float>* features,
                                int64_t n, int64_t dim, const IvfBuildOptions& opt,
                                IndexSummary* out, IvfBuildStats* stats, std::string* err) {
    using clock = std::chrono::steady_clock;
    const auto t_begin = clock::now();
    if (features == nullptr || n <= 0 || dim <= 0 ||
        features->size() != static_cast<std::size_t>(n) * static_cast<std::size_t>(dim)) {
        if (err) {
            *err = "invalid full IVF build input (n=" + std::to_string(n) + " dim=" + std::to_string(dim) +
                   ")";
        }
        return false;
    }
    if (opt.kmeans_iters <= 0 || opt.mini_centroids <= 0 || opt.mini_epochs < 0 || opt.nprobe < 1) {
        if (err) {
            *err = "invalid full IVF build parameters (iters/mini_centroids/mini_epochs/nprobe)";
        }
        return false;
    }

    // 1) Permutation (official np.random.permutation; reproducible via seed here)
    if (!opt.no_permute) {
        if (opt.verbose) {
            std::printf("[index] permuting feature rows (seed=%llu)\n",
                        static_cast<unsigned long long>(opt.seed));
        }
        shuffle_rows(features, n, dim, opt.seed);
    }

    // 2) N>200000: MiniBatchKMeans reduces to mini_centroids (official 10000)
    int64_t n_used = n;
    if (n > 200000) {
        if (opt.verbose) {
            std::printf("[index] N=%lld>200000 -> MiniBatchKMeans(%lld, epochs<=%lld)\n",
                        static_cast<long long>(n), static_cast<long long>(opt.mini_centroids),
                        static_cast<long long>(opt.mini_epochs));
        }
        MiniBatchKMeansOptions mo;
        mo.max_epochs = opt.mini_epochs;
        mo.seed = opt.seed + 1;
        mo.verbose = opt.verbose;
        std::vector<float> centers;
        if (!minibatch_kmeans_f32(features->data(), n, dim, opt.mini_centroids, mo, &centers, err)) {
            return false;
        }
        *features = std::move(centers);
        n_used = opt.mini_centroids;
    }

    // 3) nlist
    int64_t nlist = opt.nlist > 0 ? opt.nlist : official_nlist(n_used);
    nlist = std::clamp<int64_t>(nlist, 1, n_used);

    // 4) IVF k-means centroids
    const auto t_kmeans = clock::now();
    KMeansOptions ko;
    ko.iters = opt.kmeans_iters;
    ko.seed = opt.seed + 2;
    ko.verbose = opt.verbose;
    std::vector<float> centroids;
    if (!kmeans_f32(features->data(), n_used, dim, nlist, ko, &centroids, err)) {
        return false;
    }
    const double kmeans_seconds =
        std::chrono::duration<double>(clock::now() - t_kmeans).count();

    // 5) Assign to nearest centroids + group and concatenate by list
    std::vector<int32_t> assign;
    assign_to_centroids(features->data(), n_used, dim, centroids.data(), nlist, &assign, 0);
    std::vector<int64_t> lengths(static_cast<std::size_t>(nlist), 0);
    for (int64_t i = 0; i < n_used; ++i) {
        lengths[static_cast<std::size_t>(assign[static_cast<std::size_t>(i)])] += 1;
    }
    std::vector<int64_t> offsets(static_cast<std::size_t>(nlist), 0);
    for (int64_t j = 1; j < nlist; ++j) {
        offsets[static_cast<std::size_t>(j)] =
            offsets[static_cast<std::size_t>(j - 1)] + lengths[static_cast<std::size_t>(j - 1)];
    }
    std::vector<float> grouped(static_cast<std::size_t>(n_used) * static_cast<std::size_t>(dim),
                               0.0f);
    {
        std::vector<int64_t> cursor = offsets;
        for (int64_t i = 0; i < n_used; ++i) {
            const int32_t j = assign[static_cast<std::size_t>(i)];
            const int64_t row = cursor[static_cast<std::size_t>(j)]++;
            std::memcpy(grouped.data() + row * dim, features->data() + i * dim,
                        static_cast<std::size_t>(dim) * sizeof(float));
        }
    }

    // 6) Write GGUF (offsets/lengths as F32 -- the RVC-GGML reader contract; >2^24 loses integer precision)
    if (n_used > 16777216) {
        if (err) {
            *err = "ntotal exceeds the exact F32 integer range (>2^24); RVC-GGML list_offsets semantics do not allow this";
        }
        return false;
    }
    std::vector<float> off_f(static_cast<std::size_t>(nlist), 0.0f);
    std::vector<float> len_f(static_cast<std::size_t>(nlist), 0.0f);
    for (int64_t j = 0; j < nlist; ++j) {
        off_f[static_cast<std::size_t>(j)] = static_cast<float>(offsets[static_cast<std::size_t>(j)]);
        len_f[static_cast<std::size_t>(j)] = static_cast<float>(lengths[static_cast<std::size_t>(j)]);
    }

    std::ostringstream name;
    name << "RVC-Train-GGML index (IVF" << nlist << ", nprobe=" << opt.nprobe << ")";
    GgufWriter w;
    w.set_str("general.architecture", "rvc_index");
    w.set_str("general.name", name.str());
    w.set_str("general.description", "FAISS IndexIVFFlat sidecar for RVC retrieval blending");
    w.set_u32("index.dim", static_cast<uint32_t>(dim));
    w.set_u32("index.nlist", static_cast<uint32_t>(nlist));
    w.set_u32("index.ntotal", static_cast<uint32_t>(n_used));
    w.set_u32("index.metric_type", 1);  // METRIC_L2 (ignored by the RVC-GGML reader)
    w.set_u32("index.nprobe", opt.nprobe);
    if (!w.add_f32_2d("index.centroids", centroids.data(), nlist, dim, err) ||
        !w.add_f32_2d("index.vectors", grouped.data(), n_used, dim, err) ||
        !w.add_f32_1d("index.list_offsets", off_f.data(), nlist, err) ||
        !w.add_f32_1d("index.list_lengths", len_f.data(), nlist, err)) {
        return false;
    }
    if (!w.write(out_path, err)) {
        return false;
    }

    if (out != nullptr) {
        out->dim = dim;
        out->nlist = nlist;
        out->ntotal = n_used;
        out->nprobe = static_cast<int64_t>(opt.nprobe);
    }
    if (stats != nullptr) {
        int64_t lmin = n_used;
        int64_t lmax = 0;
        int64_t lempty = 0;
        for (int64_t j = 0; j < nlist; ++j) {
            const int64_t len = lengths[static_cast<std::size_t>(j)];
            lmin = std::min(lmin, len);
            lmax = std::max(lmax, len);
            if (len == 0) {
                ++lempty;
            }
        }
        stats->ntotal = n_used;
        stats->nlist = nlist;
        stats->list_min = lmin;
        stats->list_max = lmax;
        stats->list_empty = lempty;
        stats->reduced_centroids = n > 200000 ? n_used : 0;
        stats->kmeans_seconds = kmeans_seconds;
        stats->total_seconds = std::chrono::duration<double>(clock::now() - t_begin).count();
    }
    if (err) {
        err->clear();
    }
    return true;
}

} // namespace rvc_train::exporter
