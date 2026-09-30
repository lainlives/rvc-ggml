// RVC-Train-GGML - RVC 模型参数容器（src/model/model.h）
//
// 张量名 = torch state_dict 键名（与训练态 GGUF 一致）。M2-1 只创建参数张量
// （F32、形状来自文件；不建图、不装载数据），供各层模块按名取用；
// 数据装载与图装配自 M2-2 起由各模块接入（本文件仅参数容器）。
#pragma once

#include "model/weights.h"

#include "traincpp/traincpp.h"

#include <string>
#include <vector>

namespace rvc_train {
namespace model {

// 按名创建单个参数张量：校验存在性后以文件形状创建 F32 张量（数据由后端 buffer 分配）。
// 失败返回 nullptr 并写入 err。
traincpp::Tensor* build_param(traincpp::Context* ctx, const TrainWeights& weights,
                              const char* name, std::string* err);

// 模型参数容器：扁平参数表（优化器用）+ 按一级模块分组（报告/装配用）+ 期望清单。
class ModelParams {
public:
    // 遍历文件中全部张量创建参数（顺序 = 文件顺序）；任意一张失败即返回 false。
    bool build_all(traincpp::Context* ctx, const TrainWeights& weights, std::string* err);

    int64_t count() const { return (int64_t) entries_.size(); }
    traincpp::Tensor* find(const char* name) const;

    // 一级模块名（如 enc_p/enc_q/flow/dec/emb_g/discriminators；按首次出现顺序）
    const std::vector<std::string>& group_names() const { return group_names_; }
    int64_t group_count(const std::string& group) const;

    // 期望清单（name + 文件 ne），供 TrainWeights::check_expected 自检
    const std::vector<TrainWeights::Expect>& expected() const { return expected_; }
    // 扁平参数表（顺序 = 文件顺序；M5 优化器使用）
    const traincpp::ParamList& params() const { return params_; }

private:
    struct Entry {
        std::string name;
        traincpp::Tensor* tensor = nullptr;
        int64_t group_index = -1;
    };

    std::vector<Entry> entries_;
    std::vector<std::string> group_names_;
    std::vector<int64_t> group_counts_;
    std::vector<TrainWeights::Expect> expected_;
    traincpp::ParamList params_;
};

} // namespace model
} // namespace rvc_train
