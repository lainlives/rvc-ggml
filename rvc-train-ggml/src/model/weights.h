// RVC-Train-GGML - 训练态 GGUF 装载（src/model/weights.h）
//
// 训练态 GGUF（由 tools/pth_to_train_gguf.py 生成，元数据前缀 rvc.train.*）：
//   - 张量名 = torch state_dict 键名（含 weight_g/weight_v，不折叠）
//   - 形状为 ggml/traincpp 的 ne 序（torch 形状反序）
//
// 本模块只负责"打开 + 元数据 + 按名装载 + 形状校验"，
// 不含 RVC 模型结构知识（模型装配在 M2 的 src/model/ 各层模块里）。
#pragma once

#include "traincpp/traincpp.h"

#include <array>
#include <string>
#include <vector>

namespace rvc_train {
namespace model {

// 训练态 GGUF 元数据（rvc.train.*；缺失字段为 0/空）
struct TrainMeta {
    std::string kind;              // "G" | "D"
    std::string version;           // "v1" | "v2"
    bool        has_f0 = false;
    std::string form;              // "train" | "small" | "raw"
    std::string source_pth;
    std::string out_dtype;         // "F32" | "F16"
    int64_t     tensor_count = 0;

    int64_t sample_rate  = 0;
    int64_t spec_channels = 0;
    int64_t filter_length = 0;
    int64_t hop_length    = 0;
    int64_t win_length    = 0;
    int64_t n_mel_channels = 0;
    float   mel_fmin = 0.0f;
    float   mel_fmax = 0.0f;
    int64_t segment_size = 0;

    int64_t inter_channels  = 0;
    int64_t hidden_channels = 0;
    int64_t filter_channels = 0;
    int64_t n_heads         = 0;
    int64_t n_layers        = 0;
    int64_t kernel_size     = 0;
    float   p_dropout       = 0.0f;
    int64_t gin_channels    = 0;
    int64_t spk_embed_dim   = 0;
    int64_t upsample_initial_channel = 0;
    std::string resblock;
    std::vector<int32_t> resblock_kernel_sizes;
    std::vector<int32_t> resblock_dilation_sizes;   // 展平（3 个 kernel × 3 个 dilation）
    std::vector<int32_t> upsample_rates;
    std::vector<int32_t> upsample_kernel_sizes;

    std::vector<int32_t> d_periods;                 // D 模型：判别器周期
};

// 训练态 GGUF（只读）。文件名不存在/格式错误时 open 返回 nullptr 并给出错误。
class TrainWeights {
public:
    static TrainWeights* open(const std::string& path, std::string* err);
    ~TrainWeights();

    TrainWeights(const TrainWeights&) = delete;
    TrainWeights& operator=(const TrainWeights&) = delete;

    const std::string& path() const { return path_; }
    int64_t tensor_count() const { return count_; }
    const char* tensor_name(int64_t index) const;   // 越界返回 nullptr

    bool has(const char* name) const;
    // 文件内张量的 ne（4 维，补 1）；name 不存在返回 false
    bool tensor_ne(const char* name, std::array<int64_t, 4>* ne) const;
    // 文件内张量类型（未找到返回 TYPE_F32 并打印错误）
    traincpp::Type tensor_type(const char* name) const;

    bool read_meta(TrainMeta* meta) const;

    // 按文件形状在 ctx 中创建 F32 张量（名字不存在返回 nullptr）
    traincpp::Tensor* new_tensor(traincpp::Context* ctx, const char* name) const;

    // 按名装载到已创建张量：
    //   - 名字不存在、形状与文件不一致、dst 非 F32、数据未分配 → 返回 false 并给出明确错误
    bool load(traincpp::Tensor* dst, const char* name, std::string* err) const;

    // 期望清单校验（M2 模型装配用）：
    //   missing   = 期望但文件中不存在
    //   mismatched = 存在但 ne 不一致
    struct Expect {
        std::string name;
        std::array<int64_t, 4> ne{1, 1, 1, 1};
    };
    void check_expected(const std::vector<Expect>& expected, std::vector<std::string>* missing,
                        std::vector<std::string>* mismatched) const;

private:
    TrainWeights() = default;

    traincpp::GgufFile* file_ = nullptr;
    std::string path_;
    int64_t count_ = 0;
};

} // namespace model
} // namespace rvc_train
