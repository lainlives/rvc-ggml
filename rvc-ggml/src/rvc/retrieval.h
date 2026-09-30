// retrieval.h
// RVC FAISS IVF 检索索引（GGUF sidecar）：最近邻搜索 + 特征混合。
// 对齐 infer/modules/vc/pipeline.py：
//   score, ix = index.search(npy, k=8)          # IndexIVFFlat, nprobe=1, L2²
//   weight = (1/score)^2 归一化；npy = Σ big_npy[ix] * weight
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
            int64_t position = 0;   // 在 vectors_ 中的位置（list 分组序）
        };

        // 加载 GGUF 索引（权重经 backend 上传后拷回主机，加载完即可释放临时模型）
        void load(const std::string& path, ggml_backend_t backend);

        bool loaded() const { return dim_ > 0; }
        int64_t dim() const { return dim_; }
        int64_t ntotal() const { return ntotal_; }
        int64_t nlist() const { return nlist_; }

        // 单帧 top-k（升序 L2²）；搜索 nprobe 个最近 list（对齐 FAISS IndexIVFFlat）
        void search(const float* query, int64_t k, std::vector<Hit>& out) const;

        // 批量混合：query [frames, dim] 行主序 -> [frames, dim]（index_rate 混合后）
        std::vector<float> blend(const std::vector<float>& query, int64_t frames,
            int64_t k, float index_rate) const;

    private:
        int64_t dim_ = 0;
        int64_t nlist_ = 0;
        int64_t ntotal_ = 0;
        int64_t nprobe_ = 1;
        std::vector<float> centroids_;      // [dim, nlist]
        std::vector<float> vectors_;        // [dim, ntotal]（按 list 分组拼接）
        std::vector<int64_t> offsets_;      // [nlist]
        std::vector<int64_t> lengths_;      // [nlist]
    };

} // namespace rvc
