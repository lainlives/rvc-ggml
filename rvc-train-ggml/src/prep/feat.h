// feat.h —— M4-3 HuBERT 特征（官方 81eed5e extract_hubert_feature / 运行时 fairseq 语义）
//
// 语义基准（golden = 运行时 `infer/modules/train/extract_feature_print.py`，与官方 HF 路径等价）：
//   audio16k（原始 F32、不归一化、不 pad）→ fairseq extract_features：
//     v1 = final_proj(layer9)  → 256 维
//     v2 = layer12             → 768 维
//   帧数 = floor((L-400)/320)+1（conv 前端 7 层无 padding；L=16k 样本数）
//
// 实现说明（M4-3，Q22）：
//   rvc_core 的 `hubert::build_graph` 在 pos_conv 处用自建 `pad1d`（两个**未初始化**的
//   零常量张量 + concat）表达 padding；单独执行一次图时结果正确，但同一图**重复 compute**
//   后，gallocr 复用的缓冲会把脏数据留在 pad 张量里 → 结果静默漂移（实测第 2 次起偏离、
//   跨长度重建后甚至 NaN）。本文件用 rvc_core 公共 API（`hubert::build_graph` +
//   `GgmlGraphRunner`）自建执行器，在每次 compute 前把名为 "pad_*" 的零常量叶子显式清零，
//   从而与 fairseq 一致且可重复（不修改 RVC-GGML 源码）。
//   证据与复现见 `docs/回传-RVC-GGML-HuBERT重复compute缺陷-Q22.md`。
//
// 输出行主序 [frames, dim]，与 torch [T, C] 一致。

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "gguf/gguf_model.h"
#include "hubert/hubert.h"

namespace rvc_train::prep {

struct FeatResult {
    int64_t frames = 0;
    int64_t dim = 0;
    std::vector<float> values;  // [frames, dim] 行主序（= torch [T, C]）
};

// 版本字符串 → v2 标志（仅 "v2" 为真；调用方应先校验取值合法）
inline bool version_is_v2(const std::string& version) { return version == "v2"; }

// HuBERT 特征提取器（按输入长度缓存图；重复 compute 安全）
class HubertFeatureExtractor {
public:
    HubertFeatureExtractor(const GgufModel& model, ggml_backend_t backend);
    ~HubertFeatureExtractor();
    HubertFeatureExtractor(const HubertFeatureExtractor&) = delete;
    HubertFeatureExtractor& operator=(const HubertFeatureExtractor&) = delete;

    const rvc::hubert::Config& config() const;

    // 输入：归一化后的 16k chunk（与官方 1_16k_wavs 内容一致；原长度、不 pad、不归一化）
    // v2=false → v1 256 维；v2=true → v2 768 维
    FeatResult extract(const std::vector<float>& audio_16k, bool v2);

private:
    void build(int64_t samples);
    void zero_pads();

    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// 兼容包装：语义同 `HubertFeatureExtractor::extract`
FeatResult compute_hubert_feat(const std::vector<float>& audio16k,
                               HubertFeatureExtractor& hubert, bool v2);

} // namespace rvc_train::prep
