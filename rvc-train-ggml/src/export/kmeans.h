// kmeans.h -- M4-5b: pure C++ k-means (Lloyd / MiniBatch) for RVC retrieval IVF index building
//
// Semantics reference (not bit-reproducible; the official code has no seed contract):
//   - Initialization: randomly sample k points (faiss 1.7.2 Clustering default; this project uses **distinct** sampling,
//     controlled by seed and reproducible);
//   - Lloyd: default iteration cap 25 (faiss ClusteringParameters.niter=25), F32 distance
//     (|x-c|² = |x|²+|c|²-2x·c, sequential accumulation), empty-cluster split (reseed with a point from the largest cluster),
//     early stop when assignment is unchanged;
//   - MiniBatch: official train_index.py uses sklearn MiniBatchKMeans(n_clusters=10000,
//     batch_size=256*n_cpu, init="random", compute_labels=False); this project uses batch=256×thread count,
//     incremental center update c += (x-c)/count, shuffle per epoch, early stop after 2 stagnant epochs (relative improvement <1e-3);
//     `max_epochs` defaults to 10 (official sklearn max_iter=100, usually terminated early by early stopping).
// Depends only on the C++ standard library (std::thread); threads only do parallel chunking, no third-party dependencies.

#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace rvc_train::exporter {

struct KMeansOptions {
    int64_t iters = 25;     // Lloyd iteration cap (<=0 means the default 25)
    uint64_t seed = 1234;   // matches faiss default seed=1234 (reproducible here)
    int threads = 0;        // 0 = automatic (hardware_concurrency, internal cap 8)
    bool verbose = true;    // print inertia each iteration
};

// Standard Lloyd k-means. x=[n,dim] row-major; k must satisfy 1<=k<=n; centroids output [k,dim].
bool kmeans_f32(const float* x, int64_t n, int64_t dim, int64_t k, const KMeansOptions& opt,
                std::vector<float>* centroids, std::string* err);

struct MiniBatchKMeansOptions {
    int64_t max_epochs = 10;  // cap on full-data passes (official sklearn uses 100; this project compromises and documents it)
    uint64_t seed = 1234;
    int threads = 0;
    bool verbose = true;
};

// MiniBatchKMeans style (random-sample init + mini-batch incremental updates + early stopping).
// Used when N>200000 to reduce features to mini_centroids centers (official=10000).
bool minibatch_kmeans_f32(const float* x, int64_t n, int64_t dim, int64_t k,
                          const MiniBatchKMeansOptions& opt, std::vector<float>* centroids,
                          std::string* err);

// Nearest centroid per row (L2²; ties go to the smaller centroid id); assign outputs [n]; returns inertia (Σ minimum L2²).
double assign_to_centroids(const float* x, int64_t n, int64_t dim, const float* centroids, int64_t k,
                           std::vector<int32_t>* assign, int threads);

} // namespace rvc_train::exporter
