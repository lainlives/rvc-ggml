// modules.h
// ggml 神经网络算子封装：Conv1d/ConvTranspose1d/Linear/LayerNorm/GroupNorm/注意力等。
// 供 HuBERT / RMVPE / VITS 三个图构建器共用。

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

    // 图构建上下文：持有 ggml_context/cgraph 生命周期，避免内存泄漏
    class GraphContext {
    public:
        GraphContext(size_t mem_size, size_t max_nodes);
        ~GraphContext();
        GraphContext(const GraphContext&) = delete;
        GraphContext& operator=(const GraphContext&) = delete;

        ggml_context* ctx() const { return ctx_; }
        ggml_cgraph* graph() const { return graph_; }

        // 创建一个可写输入张量（ne 按 ggml 顺序给）
        ggml_tensor* make_input(ggml_type type, int n0, int n1 = 1, int n2 = 1, int n3 = 1,
            const char* name = nullptr);

    private:
        ggml_context* ctx_ = nullptr;
        ggml_cgraph* graph_ = nullptr;
    };

    // ------------------------------------------------------------
    // 常用算子（全部输出 F32，兼容量化权重）
    // ------------------------------------------------------------

    // 一维滑动窗口（equiv. torch.nn.functional.pad(x,(n-1,0)).unfold(0,n,1)）
    ggml_tensor* sliding_window(ggml_context* ctx, ggml_tensor* x, int64_t n);

    // LayerNorm（dim0 归一化），gamma/beta 为 [H]
    ggml_tensor* layer_norm(ggml_context* ctx, ggml_tensor* x,
        ggml_tensor* gamma, ggml_tensor* beta, float eps);

    // 线性层：x [K, T]（任意前置 batch 维）乘 W[K,N] + b[N]
    ggml_tensor* linear(ggml_context* ctx, ggml_tensor* x, ggml_tensor* w, ggml_tensor* b);

    // Conv1d：input [T, C, B]，weight [k, C_in, C_out]（已按 ggml 序导出）
    // flat=true 时 weight 为展平布局 [C_in*K, C_out]（GGUF rvc.conv_flat，量化友好）：
    //   内部用 im2col + mul_mat(权重作 src0) 实现，支持 q8_0/q5_1/q4_1/q4_0 等量化权重
    ggml_tensor* conv1d(ggml_context* ctx, ggml_tensor* input, ggml_tensor* weight,
        ggml_tensor* bias, int stride, int padding, int dilation, bool flat = false);

    // 分组 Conv1d（ggml_conv_1d 不支持 groups；用 conv_2d 高度=1 退化实现）
    //   input [T, C, B]，weight [k, C/groups, C_out]，输出 [T_out, C_out, B]
    ggml_tensor* conv1d_grouped(ggml_context* ctx, ggml_tensor* input, ggml_tensor* weight,
        ggml_tensor* bias, int groups, int stride, int padding, int dilation);

    // GroupNorm（input [T, C, B]），w/b [C]
    ggml_tensor* group_norm(ggml_context* ctx, ggml_tensor* x,
        ggml_tensor* w, ggml_tensor* b, int n_groups, float eps);

    // 取张量的行：embedding 查表（F32 表），idx 为 I32 一维
    ggml_tensor* get_rows(ggml_context* ctx, ggml_tensor* table, ggml_tensor* idx);

    // 相对位置注意力的辅助：把相对 logits [frames, 2*frames-1, heads] 转为绝对 [frames, frames, heads]
    ggml_tensor* relative_to_absolute(ggml_context* ctx, ggml_tensor* x, int64_t frames, int64_t heads, int64_t head_dim);
    ggml_tensor* absolute_to_relative(ggml_context* ctx, ggml_tensor* x, int64_t frames, int64_t heads, int64_t head_dim);

    // ------------------------------------------------------------
    // VITS 专用辅助（[C,T] 布局：C 在 ne0，与线性/注意力一致）
    // ------------------------------------------------------------

    // Conv1d：[C,T] 布局，内部转 [T,C] 后走 conv1d（flat 透传）
    ggml_tensor* conv1d_ct(ggml_context* ctx, ggml_tensor* x_ct, ggml_tensor* weight,
        ggml_tensor* bias, int stride, int padding, int dilation, bool flat = false);

    // 1x1 卷积：等价于线性层，权重为 Conv1d 的 [1, K, N]（ggml 序）
    ggml_tensor* conv1x1_ct(ggml_context* ctx, ggml_tensor* x_ct, ggml_tensor* weight,
        ggml_tensor* bias);

    // 通道切片（[C,T] -> [len,T]），返回视图（无 cont）
    ggml_tensor* slice_channels(ggml_context* ctx, ggml_tensor* x_ct, int64_t begin, int64_t len);

    // 常量输入张量（构建期登记，分配到后端后由调用方写入一次）
    ggml_tensor* make_const_f32(ggml_context* ctx, int64_t n0, int64_t n1,
        const char* name = nullptr);

    // 确保张量为 F32（量化权重产生的 F16 计算走整数/快速路径，统一提升）
    ggml_tensor* to_f32(ggml_context* ctx, ggml_tensor* x);

    // 带广播检查的乘/加（形状不兼容时抛出异常并打印双方维度）
    ggml_tensor* mul_checked(ggml_context* ctx, ggml_tensor* a, ggml_tensor* b);
    ggml_tensor* add_checked(ggml_context* ctx, ggml_tensor* a, ggml_tensor* b);

    // 维度断言（用于图构建阶段定位维序错误）
    void check_dim(const ggml_tensor* t, int64_t n0, int64_t n1, const char* where);
    void check_dim3(const ggml_tensor* t, int64_t n0, int64_t n1, int64_t n2, const char* where);

    // permute + cont 的简写（axis 为目标 dim0..dim3 的源维序）
    ggml_tensor* permute_cont(ggml_context* ctx, ggml_tensor* x,
        int a0, int a1, int a2, int a3);

} // namespace vits
} // namespace rvc
