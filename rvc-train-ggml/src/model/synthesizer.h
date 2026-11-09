// RVC-Train-GGML - SynthesizerTrn training forward assembly (src/model/synthesizer.h)
//
// Corresponds to runtime `infer_pack/models.py:739-762` `SynthesizerTrnMs*.forward` (@torch.jit.ignore):
//   g = emb_g(sid); m_p/logs_p = enc_p(phone,pitch,phone_lengths);
//   z/m_q/logs_q = enc_q(spec, y_lengths, g); z_p = flow(z, y_mask, g);
//   rand_slice_segments(z, y_lengths, segment_size) -> z_slice/ids_slice;
//   pitchf_slice = slice_segments2(pitchf, ids_slice, segment_size);
//   o = dec(z_slice, pitchf_slice, g).
// This module is only responsible for **model chain assembly and forward**: the slice selection matrix, SineGen excitation source, attn_valid, and Flip permutation matrix
// are all prepared by the caller (selftest builder / training loop) after buffer allocation and passed in as inputs/leaves.
//
// Layout follows each submodule: phone [Cin,T,B], pitch [T,B], x_mask/y_mask [T,1,B], spec [spec,T,B],
// q_noise [Cq,T,B], g [1,gin,B], z/z_p [C,T,B]; slice selection always uses `slice_mat [T,seg,B]` (0/1)
// expressed via `mul_mat` (safe for per-sample different ids, and naturally differentiable; no-slice mode feeds the whole segment z/pitchf directly).
// Manually assembled POD (Q13); tensor names keep their original GGUF names.
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
    int64_t frames = 0;        // number of spectral frames (T)
    int64_t batch = 0;         // batch size (B)
    int64_t segment_size = 0;  // training slice frames (z_frames, v1-40k = 32)
};

struct SynthModules {
    SynthConfig cfg;
    EncPModule enc_p;
    EncQModule enc_q;
    FlowModule flow;
    DecModule  dec;
};

// Assemble the four submodules by name (each manually POD inside); returns false and writes err on failure.
bool synth_load(const ModelParams& params, const SynthConfig& cfg, SynthModules* out,
                std::string* err);

// Forward inputs (all tensors/leaves that already exist at graph build time):
//   sid [B] I32; phone/pitch/x_mask/attn_valid (enc_p); spec/y_mask/q_noise (enc_q);
//   flip_perm [C,C] (flow); slice_mat [T,seg,B] (0/1); sine_source_full [1,T*upp,B];
//   sine_source_slice [1,seg*upp,B] (filled by host SineGen in prepare)
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
    // true (default) = also build the whole-segment dec path (for M2-8 forward comparison);
    // false = slice path only (training loop/M3-3: official training only uses the slice y_hat, saving half the VRAM)
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
    traincpp::Tensor* z_slice = nullptr;   // [C,seg,B] (mul_mat(slice_mat,z))
    traincpp::Tensor* o_full = nullptr;    // no-slice: dec(z, pitchf) [T*upp,1,B]
    traincpp::Tensor* o_slice = nullptr;   // slice: dec(z_slice, pitchf_slice) [seg*upp,1,B]

    EncPCaptures enc_p_caps;
    EncQCaptures enc_q_caps;
    FlowCaptures flow_caps;
    DecCaptures  dec_full_caps;
    DecCaptures  dec_slice_caps;
};

// Whole-model training forward (includes both the no-slice and fixed-slice dec paths).
// Returns empty on failure (all fields nullptr).
SynthOutputs synth_forward(traincpp::Context* ctx, const ModelParams& params,
                           const SynthModules& m, const SynthInputs& in);

} // namespace model
} // namespace rvc_train
