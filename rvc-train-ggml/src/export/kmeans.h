// kmeans.h —— M4-5b：纯 C++ k-means（Lloyd / MiniBatch），用于 RVC 检索 IVF 索引构建
//
// 语义参照（不可逐位复现，官方无种子约定）：
//   - 初始化：随机采样 k 个点（faiss 1.7.2 Clustering 默认行为；本项目改为**不重复**采样，
//     由 seed 控制、可复现）；
//   - Lloyd：迭代上限默认 25（faiss ClusteringParameters.niter=25）、F32 距离
//     （|x-c|² = |x|²+|c|²-2x·c，顺序累加）、空簇拆分（取最大簇的点重播种）、
//     assignment 不变即早停；
//   - MiniBatch：官方 train_index.py 用 sklearn MiniBatchKMeans(n_clusters=10000,
//     batch_size=256*n_cpu, init="random", compute_labels=False)；本项目 batch=256×线程数、
//     增量中心更新 c += (x-c)/count、按 epoch 打乱、停滞 2 轮（相对改善 <1e-3）即早停；
//     `max_epochs` 默认 10（官方 sklearn max_iter=100，通常由 early stopping 提前结束）。
// 仅依赖 C++ 标准库（std::thread）；线程只做并行分块，不引入第三方依赖。

#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace rvc_train::exporter {

struct KMeansOptions {
    int64_t iters = 25;     // Lloyd 迭代上限（<=0 视为默认 25）
    uint64_t seed = 1234;   // 对齐 faiss 默认 seed=1234（本项目可复现）
    int threads = 0;        // 0 = 自动（hardware_concurrency，内部上限 8）
    bool verbose = true;    // 打印每轮 inertia
};

// 标准 Lloyd k-means。x=[n,dim] 行主序；k 要求 1<=k<=n；centroids 输出 [k,dim]。
bool kmeans_f32(const float* x, int64_t n, int64_t dim, int64_t k, const KMeansOptions& opt,
                std::vector<float>* centroids, std::string* err);

struct MiniBatchKMeansOptions {
    int64_t max_epochs = 10;  // 遍历全数据轮数上限（官方 sklearn 为 100；本项目折衷，文档记录）
    uint64_t seed = 1234;
    int threads = 0;
    bool verbose = true;
};

// MiniBatchKMeans 风格（随机采样 init + 小批量增量更新 + 早停）。
// 用于 N>200000 时把特征压到 mini_centroids 个中心（官方=10000）。
bool minibatch_kmeans_f32(const float* x, int64_t n, int64_t dim, int64_t k,
                          const MiniBatchKMeansOptions& opt, std::vector<float>* centroids,
                          std::string* err);

// 逐行最近质心（L2²；并列取更小质心 id）；assign 输出 [n]；返回 inertia（Σ 最小 L2²）。
double assign_to_centroids(const float* x, int64_t n, int64_t dim, const float* centroids, int64_t k,
                           std::vector<int32_t>* assign, int threads);

} // namespace rvc_train::exporter
