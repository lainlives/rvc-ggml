// RVC-Train-GGML - RVC 模型参数容器实现（src/model/model.cpp）
#include "model/model.h"

#include <array>
#include <cstring>

namespace rvc_train {
namespace model {

namespace {

void set_err(std::string* err, const std::string& msg) {
    if (err != nullptr) {
        *err = msg;
    }
}

// 一级模块名：取名字中第一个 '.' 之前的部分（emb_g.weight → emb_g）
std::string top_group(const char* name) {
    const char* dot = std::strchr(name, '.');
    if (dot == nullptr) {
        return std::string(name);
    }
    return std::string(name, (size_t) (dot - name));
}

bool same_ne(const std::array<int64_t, 4>& a, const traincpp::Tensor* t) {
    for (int i = 0; i < 4; ++i) {
        if (a[(size_t) i] != t->ne[i]) {
            return false;
        }
    }
    return true;
}

} // namespace

traincpp::Tensor* build_param(traincpp::Context* ctx, const TrainWeights& weights,
                              const char* name, std::string* err) {
    if (name == nullptr || name[0] == '\0') {
        set_err(err, "空张量名");
        return nullptr;
    }
    if (!weights.has(name)) {
        set_err(err, std::string("缺失张量: ") + name);
        return nullptr;
    }
    std::array<int64_t, 4> ne{1, 1, 1, 1};
    if (!weights.tensor_ne(name, &ne)) {
        set_err(err, std::string("读取张量形状失败: ") + name);
        return nullptr;
    }
    traincpp::Tensor* t = weights.new_tensor(ctx, name);  // F32，形状来自文件
    if (t == nullptr) {
        set_err(err, std::string("创建参数张量失败: ") + name);
        return nullptr;
    }
    if (!same_ne(ne, t)) {
        set_err(err, std::string("创建后形状不符: ") + name);
        return nullptr;
    }
    return t;
}

bool ModelParams::build_all(traincpp::Context* ctx, const TrainWeights& weights, std::string* err) {
    entries_.clear();
    group_names_.clear();
    group_counts_.clear();
    expected_.clear();
    params_.items.clear();

    const int64_t n = weights.tensor_count();
    entries_.reserve((size_t) n);
    expected_.reserve((size_t) n);
    params_.items.reserve((size_t) n);

    for (int64_t i = 0; i < n; ++i) {
        const char* name = weights.tensor_name(i);
        if (name == nullptr) {
            set_err(err, "张量名读取失败: index " + std::to_string(i));
            return false;
        }
        traincpp::Tensor* t = build_param(ctx, weights, name, err);
        if (t == nullptr) {
            return false;
        }

        const std::string group = top_group(name);
        int64_t gi = -1;
        for (size_t k = 0; k < group_names_.size(); ++k) {
            if (group_names_[k] == group) {
                gi = (int64_t) k;
                break;
            }
        }
        if (gi < 0) {
            gi = (int64_t) group_names_.size();
            group_names_.push_back(group);
            group_counts_.push_back(0);
        }
        group_counts_[(size_t) gi] += 1;

        Entry e;
        e.name        = name;
        e.tensor      = t;
        e.group_index = gi;
        entries_.push_back(e);
        params_.items.push_back(t);

        TrainWeights::Expect exp;
        exp.name = name;
        std::array<int64_t, 4> ne{1, 1, 1, 1};
        if (!weights.tensor_ne(name, &ne)) {
            set_err(err, std::string("读取张量形状失败: ") + name);
            return false;
        }
        exp.ne = ne;
        expected_.push_back(exp);
    }
    return true;
}

traincpp::Tensor* ModelParams::find(const char* name) const {
    if (name == nullptr) {
        return nullptr;
    }
    for (const Entry& e : entries_) {
        if (e.name == name) {
            return e.tensor;
        }
    }
    return nullptr;
}

int64_t ModelParams::group_count(const std::string& group) const {
    for (size_t i = 0; i < group_names_.size(); ++i) {
        if (group_names_[i] == group) {
            return group_counts_[i];
        }
    }
    return 0;
}

} // namespace model
} // namespace rvc_train
