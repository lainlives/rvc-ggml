// index_gguf.h —— M4-5a/M4-5b：RVC 检索索引的构建/写出/自检
//
// 产物格式与 RVC-GGML 读取约定一致（见 RVC-GGML src/rvc/retrieval.cpp:28-63）：
//   KV：general.architecture="rvc_index"、index.dim/nlist/ntotal/metric_type/nprobe（UINT32）
//   张量（F32）：index.centroids[dim,nlist]、index.vectors[dim,ntotal]（按 list 拼接）、
//                index.list_offsets[nlist]、index.list_lengths[nlist]
// - M4-5a MVP（单 list）：nlist=1、offsets=[0]、lengths=[ntotal]、nprobe=1、centroids=列均值；
//   向量顺序不重排（与源特征行 id 一一对应）。
// - M4-5b 完整 IVF（对齐官方 train_index.py，不可逐位复现）：
//   置换（seed 可复现）→ N>200000 时 MiniBatchKMeans 压到 10000 中心 →
//   nlist=max(1,min(int(16√N),N//39)) → k-means 质心 → 按最近 list 分组拼接。

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

// features：行主序 [n, dim]；写出 index.gguf
bool build_write_index_gguf(const std::string& out_path, const float* features, int64_t n,
                            int64_t dim, IndexSummary* out, std::string* err);

// M4-5b 完整 IVF 构建参数
struct IvfBuildOptions {
    int64_t nlist = 0;          // 0 = 官方公式自动（max(1,min(int(16√N),N//39))）
    uint32_t nprobe = 1;        // 写入 KV index.nprobe（官方 trained/added 索引为 1）
    uint64_t seed = 1234;       // 置换/k-means 种子（官方无种子；本项目可复现）
    int64_t kmeans_iters = 25;  // IVF k-means 迭代上限
    int64_t mini_centroids = 10000;  // N>200000 时 MiniBatchKMeans 目标中心数（官方 10000）
    int64_t mini_epochs = 10;        // MiniBatchKMeans 轮数上限（官方 sklearn max_iter=100）
    bool no_permute = false;    // 调试：跳过置换
    bool verbose = true;
};

struct IvfBuildStats {
    int64_t ntotal = 0;
    int64_t nlist = 0;
    int64_t list_min = 0;
    int64_t list_max = 0;
    int64_t list_empty = 0;
    int64_t reduced_centroids = 0;  // >0 表示走了 MiniBatchKMeans（值为中心数）
    double kmeans_seconds = 0.0;
    double total_seconds = 0.0;
};

// 完整 IVF 构建：features 可被就地置换/替换为聚类中心（调用后内容不再等于输入行序）。
bool build_write_index_gguf_ivf(const std::string& out_path, std::vector<float>* features,
                                int64_t n, int64_t dim, const IvfBuildOptions& opt,
                                IndexSummary* out, IvfBuildStats* stats, std::string* err);

// 用 RVC-GGML 公共 API（rvc::RetrievalIndex）回读自检；query 非空时做一次 top-1 自查询
bool validate_index_gguf(const std::string& path, ggml_backend_t backend, const float* query,
                         int64_t query_dim, IndexSummary* out, std::string* err);

} // namespace rvc_train::exporter
