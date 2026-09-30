// RVC-Train-GGML - SynthesizerTrn 训练前向装配（src/model/synthesizer.h）
//
// 对应运行时 `infer_pack/models.py:739-762` 的 `SynthesizerTrnMs*.forward`（@torch.jit.ignore）：
//   g = emb_g(sid)；m_p/logs_p = enc_p(phone,pitch,phone_lengths)；
//   z/m_q/logs_q = enc_q(spec, y_lengths, g)；z_p = flow(z, y_mask, g)；
//   rand_slice_segments(z, y_lengths, segment_size) → z_slice/ids_slice；
//   pitchf_slice = slice_segments2(pitchf, ids_slice, segment_size)；
//   o = dec(z_slice, pitchf_slice, g)。
// 本模块仅负责**模型链条装配与前向**：切片选择矩阵、SineGen 激励源、attn_valid、Flip 置换矩阵
// 均由调用方（selftest builder / 训练循环）在 buffer 分配后准备并作为输入/叶子传入。
//
// 布局沿用各子模块：phone [Cin,T,B]、pitch [T,B]、x_mask/y_mask [T,1,B]、spec [spec,T,B]、
// q_noise [Cq,T,B]、g [1,gin,B]、z/z_p [C,T,B]；切片选择一律用 `slice_mat [T,seg,B]`（0/1）
// 经 `mul_mat` 表达（逐样本不同 ids 安全，且天然可反传；不切片模式直接喂整段 z/pitchf）。
// 手工装配 POD（Q13），张量名保持 GGUF 原名。
#pragma once

#include "model/dec.h"
#include "model/enc_p.h"
#include "model/enc_q.h"
#include "model/flow.h"
#include "model/model.h"

#include "traincpp/traincpp.h"

#include <cstdint>
#include <string>

namespace rvc_train {
namespace model {

struct SynthConfig {
    EncPConfig enc_p;
    EncQConfig enc_q;
    FlowConfig flow;
    DecConfig  dec;
    int64_t frames = 0;        // 谱帧数（T）
    int64_t batch = 0;         // 批大小（B）
    int64_t segment_size = 0;  // 训练切片帧数（z_frames，v1-40k = 32）
};

struct SynthModules {
    SynthConfig cfg;
    EncPModule enc_p;
    EncQModule enc_q;
    FlowModule flow;
    DecModule  dec;
};

// 按名装配四个子模块（内部各自手工 POD）；失败返回 false 并写 err。
bool synth_load(const ModelParams& params, const SynthConfig& cfg, SynthModules* out,
                std::string* err);

// 前向输入（均为建图期已存在的张量/叶子）：
//   sid [B] I32；phone/pitch/x_mask/attn_valid（enc_p）；spec/y_mask/q_noise（enc_q）；
//   flip_perm [C,C]（flow）；slice_mat [T,seg,B]（0/1）；sine_source_full [1,T*upp,B]；
//   sine_source_slice [1,seg*upp,B]（主机 SineGen 在 prepare 中填充）
struct SynthInputs {
    traincpp::Tensor* sid = nullptr;
    traincpp::Tensor* phone = nullptr;
    traincpp::Tensor* pitch = nullptr;
    traincpp::Tensor* x_mask = nullptr;
    traincpp::Tensor* attn_valid = nullptr;
    traincpp::Tensor* spec = nullptr;
    traincpp::Tensor* y_mask = nullptr;
    traincpp::Tensor* q_noise = nullptr;
    traincpp::Tensor* flip_perm = nullptr;
    traincpp::Tensor* slice_mat = nullptr;
    traincpp::Tensor* sine_source_full = nullptr;
    traincpp::Tensor* sine_source_slice = nullptr;
    // true（默认）= 同时构建整段 dec 路径（M2-8 前向对拍用）；
    // false = 仅切片路径（训练循环/M3-3：官方训练只用切片 y_hat，省一半显存）
    bool include_full = true;
};

struct SynthOutputs {
    traincpp::Tensor* g = nullptr;         // [1,gin,B]
    traincpp::Tensor* m_p = nullptr;       // [C,T,B]
    traincpp::Tensor* logs_p = nullptr;
    traincpp::Tensor* z = nullptr;
    traincpp::Tensor* m_q = nullptr;
    traincpp::Tensor* logs_q = nullptr;
    traincpp::Tensor* z_p = nullptr;
    traincpp::Tensor* z_slice = nullptr;   // [C,seg,B]（mul_mat(slice_mat,z)）
    traincpp::Tensor* o_full = nullptr;    // 不切片：dec(z, pitchf) [T*upp,1,B]
    traincpp::Tensor* o_slice = nullptr;   // 切片：dec(z_slice, pitchf_slice) [seg*upp,1,B]

    EncPCaptures enc_p_caps;
    EncQCaptures enc_q_caps;
    FlowCaptures flow_caps;
    DecCaptures  dec_full_caps;
    DecCaptures  dec_slice_caps;
};

// 整模型训练前向（含不切片与固定切片两条 dec 路径）。
// 失败返回空（全部字段为 nullptr）。
SynthOutputs synth_forward(traincpp::Context* ctx, const ModelParams& params,
                           const SynthModules& m, const SynthInputs& in);

} // namespace model
} // namespace rvc_train
