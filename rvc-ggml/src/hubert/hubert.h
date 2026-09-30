// hubert.h
// HuBERT/ContentVec 特征提取（ggml 图）。
// 对齐 fairseq HuBERTModel.forward：
//   conv feature extractor -> layer_norm -> post_extract_proj -> pos_conv -> layer_norm
//   -> 12 x TransformerEncoderLayer -> 输出（v1: 第 10 层 + final_proj；v2: 第 12 层）

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <ggml.h>
#include <ggml-backend.h>

#include "gguf/gguf_model.h"

namespace rvc {
namespace hubert {

    struct Config {
        int n_layers = 12;
        int hidden = 768;
        int ffn = 3072;
        int heads = 12;
        int conv_pos = 128;
        int conv_pos_groups = 16;
        int final_dim = 256;
        bool conv_flat = false;     // GGUF rvc.conv_flat：前端卷积权重为展平布局
        std::vector<int> conv_dim{ 512, 512, 512, 512, 512, 512, 512 };
        std::vector<int> conv_kernel{ 10, 3, 3, 3, 3, 2, 2 };
        std::vector<int> conv_stride{ 5, 2, 2, 2, 2, 2, 2 };
    };

    Config read_config(const GgufModel& model);

    // 调试：设置环境变量 RVC_HUBERT_DUMP=1 时，build_graph 记录中间张量
    // （调用方需对它们 set_output + expand，然后可读取比对）
    struct DebugTensor {
        std::string name;
        ggml_tensor* tensor;
    };
    const std::vector<DebugTensor>& debug_tensors();

    // 构建图。input: [T, 1]（T = pad 到 320 倍数后的样本数）
    // 返回：当前构建的最后一个节点指针；调用方用 build_graph 的 out 参数取张量。
    // out_layer9  : [T_tokens, hidden]（v1 用）
    // out_layer12 : [T_tokens, hidden]（v2 用）
    // out_proj    : [T_tokens, final_dim]（v1 最终特征）
    void build_graph(ggml_context* ctx,
        const GgufModel& model,
        const Config& cfg,
        ggml_tensor* input,
        ggml_tensor** out_proj,
        ggml_tensor** out_layer9,
        ggml_tensor** out_layer12);

    // 便捷：加载模型 + 建立图 + 运行 + 取结果（用于自检）
    //   audio: 单声道 16k（内部自动 pad 到 320 倍数）
    //   outputs: 结果张量（未 pad 的原始长度）
    struct Output {
        std::vector<float> v1_final_proj;   // [T_tokens, 256]
        std::vector<float> v1_layer9;       // [T_tokens, 768]
        std::vector<float> v2_layer12;      // [T_tokens, 768]
        int64_t frames = 0;
    };

    // 特征 [frames, dim]（行主序，与 PyTorch [T, C] 一致）
    struct Features {
        int64_t frames = 0;
        int64_t dim = 0;
        std::vector<float> values;
    };

    // 可复用的编码器（按输入长度缓存图），供全链路管线使用
    class HubertEncoder {
    public:
        HubertEncoder(const GgufModel& model, ggml_backend_t backend);
        ~HubertEncoder();
        HubertEncoder(const HubertEncoder&) = delete;
        HubertEncoder& operator=(const HubertEncoder&) = delete;

        const Config& config() const { return cfg_; }

        // audio: 单声道 16k 浮点；v2=true 返回最后一层 768 维，否则 v1 final_proj 256 维
        Features extract(const std::vector<float>& audio_16k, bool v2);

    private:
        void build(int64_t samples);

        struct Impl;
        std::unique_ptr<Impl> impl_;
        Config cfg_;
    };

} // namespace hubert
} // namespace rvc
