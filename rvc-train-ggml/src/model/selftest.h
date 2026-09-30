// RVC-Train-GGML - 前向对拍 harness（src/model/selftest.h）
//
// 机制（M2-2 起）：
//   tools/gen_golden_forward.py 用运行时 infer_pack 逐模块 dump 输入/输出/中间张量
//   （.raw F32/I32 小端 + manifest.<module>.json）；本模块读取 manifest，
//   用 ModelParams 装配模块 → 建图 → 分配 → 装载参数 → 注入输入 → 计算 → 逐张量对比。
// 判定：形状逐一对齐；F32 相对误差 = max_abs / max(1, max|ref|) ≤ 1e-4。
#pragma once

#include "common/json.h"
#include "model/config.h"
#include "model/model.h"
#include "model/weights.h"

#include "traincpp/traincpp.h"

#include <array>
#include <cstdint>
#include <functional>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace rvc_train {
namespace model {

// golden 张量条目（shape = torch 形状；ne = 反序）
struct GoldenTensor {
    std::string name;
    std::string file;      // 相对 golden 目录
    std::string dtype;     // "f32" | "i32"
    std::string role;      // "input" | "output" | "intermediate"
    std::array<int64_t, 4> ne{1, 1, 1, 1};
    int nd = 1;            // 形状维数（1..4）

    bool is_input() const { return role == "input"; }
};

struct GoldenManifest {
    std::string module;
    std::string dir;
    std::vector<GoldenTensor> tensors;
    common::JsonValue params;    // 模块专属参数（无则为 Null）

    const GoldenTensor* find_tensor(const std::string& name) const;
    int64_t param_int(const char* key, int64_t def = 0) const;
    std::string param_str(const char* key, const char* def = "") const;
    void param_int_array(const char* key, std::vector<int64_t>* out) const;
};

// 读取 {dir}/manifest.<module>.json（缺失时回退 manifest.json 并校验 module 字段）
bool golden_manifest_load(const std::string& dir, const std::string& module, GoldenManifest* out,
                          std::string* err);

// 模块构建上下文：runner 预创建输入张量；构建器登记待比较张量
struct SelftestBuildIO {
    traincpp::Context* ctx = nullptr;
    const ModelConfig* cfg = nullptr;
    const ModelParams* params = nullptr;
    const GoldenManifest* golden = nullptr;
    std::map<std::string, traincpp::Tensor*> inputs;
    std::vector<std::pair<std::string, traincpp::Tensor*>> compares;
    // 可选准备回调：在 runner 注入输入数据之后、graph_compute 之前调用
    // （用于依赖输入数据的主机侧计算，如 dec 的 SineGen 激励源；须自行写入预先建好的叶子张量）
    std::function<bool(std::string*)> prepare;

    traincpp::Tensor* find_input(const std::string& name) const;
    void add_compare(const std::string& name, traincpp::Tensor* t);
};

using SelftestBuildFn = bool (*)(SelftestBuildIO& io, std::string* err);

// 各模块构建器（selftest.cpp 注册）
bool build_selftest_emb_g(SelftestBuildIO& io, std::string* err);
bool build_selftest_wn(SelftestBuildIO& io, std::string* err);
bool build_selftest_resblock1(SelftestBuildIO& io, std::string* err);
bool build_selftest_dec(SelftestBuildIO& io, std::string* err);
bool build_selftest_flow(SelftestBuildIO& io, std::string* err);
bool build_selftest_enc_p(SelftestBuildIO& io, std::string* err);
bool build_selftest_enc_q(SelftestBuildIO& io, std::string* err);
// 判别器用独立的 D 权重文件（f0D40k.train.gguf），不并入 module="all"
bool build_selftest_disc(SelftestBuildIO& io, std::string* err);
// 整模型训练前向（SynthesizerTrn：不切片 + 固定 ids 切片两条 dec 路径）
bool build_selftest_synth(SelftestBuildIO& io, std::string* err);

// 前向对拍入口（module="all" 依次跑 emb_g/wn/resblock1/dec/flow/enc_p/enc_q/synth；
// disc 需单独运行并指定 D 权重）
// 返回：0 = 全部通过；1 = 失败；2 = 参数/环境错误
int run_forward_selftest(const std::string& weights_path, const std::string& golden_dir,
                         const std::string& module, traincpp::Device* dev);

// 反向对拍入口（M3-1：固定 cotangent VJP，门 rel ≤1e-3；gradcheck 门 ≤2e-2）
//   golden = manifest.backward.<module>.json（role: input/cotangent/grad/loss）+
//            <module>.cot_<输出>.f32.raw / <module>.<name>.grad.f32.raw；
//   module="all" 同前向（disc 需单独运行并指定 D 权重）；
//   gradcheck 为分号分隔的 "参数名[:元素下标]" 列表（空 = 跳过；"all" = 全部参数取首/中元素）；
//   dump_grads 非空时把 C++ 侧 grad_acc/loss 落盘到该目录（诊断用，文件名与 Python golden 同约定）。
int run_backward_selftest(const std::string& weights_path, const std::string& golden_dir,
                          const std::string& module, const std::string& gradcheck,
                          const std::string& dump_grads, traincpp::Device* dev);

} // namespace model
} // namespace rvc_train
