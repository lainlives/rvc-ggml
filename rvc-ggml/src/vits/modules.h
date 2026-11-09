// modules.h
// ggml neural network operator wrappers: Conv1d/ConvTranspose1d/Linear/LayerNorm/GroupNorm/attention, etc.
// Shared by the three graph builders HuBERT / RMVPE / VITS.

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <ggml.h>
#include <ggml-backend.h>

#include "gguf/gguf_model.h"

namespace rvc {
namespace vits {

    // Graph build context: owns the ggml_context/cgraph lifetime to avoid memory leaks
    class GraphContext {
    public:
        GraphContext(size_t mem_size, size_t max_nodes);
        ~GraphContext();
        GraphContext(const GraphContext&) = delete;
        GraphContext& operator=(const GraphContext&) = delete;

        ggml_context* ctx() const { return ctx_; }
        ggml_cgraph* graph() const { return graph_; }

        // Create a writable input tensor (ne given in ggml order)
        ggml_tensor* make_input(ggml_type type, int n0, int n1 = 1, int n2 = 1, int n3 = 1,
            const char* name = nullptr);

    private:
        ggml_context* ctx_ = nullptr;
        ggml_cgraph* graph_ = nullptr;
    };

    // ------------------------------------------------------------
    // Common operators (all output F32, compatible with quantized weights)
    // ------------------------------------------------------------

    // 1D sliding window (equiv. torch.nn.functional.pad(x,(n-1,0)).unfold(0,n,1))
    ggml_tensor* sliding_window(ggml_context* ctx, ggml_tensor* x, int64_t n);

    // LayerNorm (normalized along dim0), gamma/beta are [H]
    ggml_tensor* layer_norm(ggml_context* ctx, ggml_tensor* x,
        ggml_tensor* gamma, ggml_tensor* beta, float eps);

    // Linear layer: x [K, T] (any leading batch dims) times W[K,N] + b[N]
    ggml_tensor* linear(ggml_context* ctx, ggml_tensor* x, ggml_tensor* w, ggml_tensor* b);

    // Conv1d: input [T, C, B], weight [k, C_in, C_out] (already exported in ggml order)
    // flat=true: weight uses the flattened layout [C_in*K, C_out] (GGUF rvc.conv_flat, quantization-friendly):
    //   implemented internally as im2col + mul_mat(weight as src0); supports q8_0/q5_1/q4_1/q4_0 and other quantized weights
    ggml_tensor* conv1d(ggml_context* ctx, ggml_tensor* input, ggml_tensor* weight,
        ggml_tensor* bias, int stride, int padding, int dilation, bool flat = false);

    // Grouped Conv1d (ggml_conv_1d has no groups support; degenerate conv_2d with height=1)
    //   input [T, C, B], weight [k, C/groups, C_out], output [T_out, C_out, B]
    ggml_tensor* conv1d_grouped(ggml_context* ctx, ggml_tensor* input, ggml_tensor* weight,
        ggml_tensor* bias, int groups, int stride, int padding, int dilation);

    // GroupNorm (input [T, C, B]), w/b [C]
    ggml_tensor* group_norm(ggml_context* ctx, ggml_tensor* x,
        ggml_tensor* w, ggml_tensor* b, int n_groups, float eps);

    // Row lookup: embedding table lookup (F32 table), idx is a 1D I32 tensor
    ggml_tensor* get_rows(ggml_context* ctx, ggml_tensor* table, ggml_tensor* idx);

    // Relative-position attention helper: convert relative logits [frames, 2*frames-1, heads] to absolute [frames, frames, heads]
    ggml_tensor* relative_to_absolute(ggml_context* ctx, ggml_tensor* x, int64_t frames, int64_t heads, int64_t head_dim);
    ggml_tensor* absolute_to_relative(ggml_context* ctx, ggml_tensor* x, int64_t frames, int64_t heads, int64_t head_dim);

    // ------------------------------------------------------------
    // VITS-specific helpers ([C,T] layout: C in ne0, consistent with linear/attention)
    // ------------------------------------------------------------

    // Conv1d: [C,T] layout, internally transposed to [T,C] then routed to conv1d (flat passed through)
    ggml_tensor* conv1d_ct(ggml_context* ctx, ggml_tensor* x_ct, ggml_tensor* weight,
        ggml_tensor* bias, int stride, int padding, int dilation, bool flat = false);

    // 1x1 convolution: equivalent to a linear layer; weight is Conv1d's [1, K, N] (ggml order)
    ggml_tensor* conv1x1_ct(ggml_context* ctx, ggml_tensor* x_ct, ggml_tensor* weight,
        ggml_tensor* bias);

    // Channel slice ([C,T] -> [len,T]), returns a view (no cont)
    ggml_tensor* slice_channels(ggml_context* ctx, ggml_tensor* x_ct, int64_t begin, int64_t len);

    // Constant input tensor (registered at build time; written once by the caller after backend allocation)
    ggml_tensor* make_const_f32(ggml_context* ctx, int64_t n0, int64_t n1,
        const char* name = nullptr);

    // Ensure the tensor is F32 (F16 computation from quantized weights takes the integer/fast path, so promote uniformly)
    ggml_tensor* to_f32(ggml_context* ctx, ggml_tensor* x);

    // Multiply/add with broadcast checks (throws on incompatible shapes and prints both shapes)
    ggml_tensor* mul_checked(ggml_context* ctx, ggml_tensor* a, ggml_tensor* b);
    ggml_tensor* add_checked(ggml_context* ctx, ggml_tensor* a, ggml_tensor* b);

    // Dimension assertions (to locate dimension-order bugs during graph construction)
    void check_dim(const ggml_tensor* t, int64_t n0, int64_t n1, const char* where);
    void check_dim3(const ggml_tensor* t, int64_t n0, int64_t n1, int64_t n2, const char* where);

    // Shorthand for permute + cont (axis = source dimension order for target dim0..dim3)
    ggml_tensor* permute_cont(ggml_context* ctx, ggml_tensor* x,
        int a0, int a1, int a2, int a3);

} // namespace vits
} // namespace rvc
