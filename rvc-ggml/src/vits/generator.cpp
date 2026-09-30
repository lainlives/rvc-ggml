// generator.cpp
// RVC HiFiGAN NSF 生成器（dec）：
//   conv_pre -> (+cond(g)) -> 4×[LeakyReLU + ConvTranspose1d + (+NoiseConv(source)) + ResBlock1×3]
//   -> LeakyReLU -> conv_post -> tanh
// 布局：x 为 [T, C]（卷积自然序），与 flow/enc_p 的 [C, T] 在入口/出口转换。

#include "vits/synthesizer.h"
#include "vits/modules.h"

#include <stdexcept>

namespace rvc {
namespace vits {

namespace {

    constexpr float kLReluSlope = 0.1f;
    constexpr float kLReluSlopeFinal = 0.01f;   // Generator 末尾 F.leaky_relu(x) 默认 0.01

    ggml_tensor* require(const GgufModel& model, const std::string& name) {
        ggml_tensor* t = model.weight(name);
        if (!t) {
            throw std::runtime_error("VITS 缺失权重: " + name);
        }
        return t;
    }

    // ConvTranspose1d（PyTorch 语义）：ggml 无 padding，先全量卷积再裁掉两侧 pad
    ggml_tensor* conv_transpose1d_pytorch(ggml_context* ctx, ggml_tensor* x,
        ggml_tensor* weight, ggml_tensor* bias, int64_t stride, int64_t kernel) {
        const int64_t pad = (kernel - stride) / 2;
        ggml_tensor* raw = ggml_conv_transpose_1d(ctx, weight, x,
            static_cast<int>(stride), 0, 1);                      // [L_raw, Cout, 1]
        const int64_t trimmed = raw->ne[0] - 2 * pad;
        if (trimmed <= 0) {
            throw std::runtime_error("VITS 生成器 ConvTranspose 裁剪后长度为 0");
        }
        ggml_tensor* out = ggml_cont(ctx, ggml_view_2d(ctx, raw, trimmed, raw->ne[1],
            raw->nb[1], static_cast<size_t>(pad) * raw->nb[0]));
        if (bias) {
            out = add_checked(ctx, out, ggml_reshape_2d(ctx, bias, 1, bias->ne[0]));
        }
        return out;
    }

    // ResBlock1（convs1 带膨胀，convs2 无膨胀），x [T,C]
    ggml_tensor* resblock1(ggml_context* ctx, const GgufModel& model, ggml_tensor* x,
        int64_t index, int64_t kernel, bool conv_flat) {
        const std::string prefix = "dec.resblocks." + std::to_string(index);
        const int64_t dilations[3] = { 1, 3, 5 };
        for (int64_t l = 0; l < 3; ++l) {
            const std::string c1 = prefix + ".convs1." + std::to_string(l);
            const std::string c2 = prefix + ".convs2." + std::to_string(l);
            const int64_t d = dilations[l];
            const int64_t pad1 = (kernel * d - d) / 2;
            const int64_t pad2 = (kernel - 1) / 2;
            ggml_tensor* xt = ggml_leaky_relu(ctx, x, kLReluSlope, false);
            xt = conv1d(ctx, xt, require(model, c1 + ".weight"),
                require(model, c1 + ".bias"), 1, static_cast<int>(pad1),
                static_cast<int>(d), conv_flat);
            xt = ggml_leaky_relu(ctx, xt, kLReluSlope, false);
            xt = conv1d(ctx, xt, require(model, c2 + ".weight"),
                require(model, c2 + ".bias"), 1, static_cast<int>(pad2), 1, conv_flat);
            x = add_checked(ctx, x, xt);
        }
        return x;
    }

} // namespace

ggml_tensor* build_generator(ggml_context* ctx, const GgufModel& model,
    const Config& cfg, ggml_tensor* z, ggml_tensor* g, ggml_tensor* source,
    int64_t frames) {
    (void)frames;
    const int64_t n_ups = static_cast<int64_t>(cfg.upsample_rates.size());
    const int64_t n_kernels = static_cast<int64_t>(cfg.resblock_kernel_sizes.size());
    if (n_ups <= 0 || n_kernels != 3) {
        throw std::runtime_error("VITS 生成器配置异常（upsamples/resblock kernels）");
    }
    const int64_t init_ch = cfg.upsample_initial_channel;

    // z [C,T] -> [T,C]
    ggml_tensor* x = permute_cont(ctx, z, 1, 0, 2, 3);
    x = conv1d(ctx, x, require(model, "dec.conv_pre.weight"),
        require(model, "dec.conv_pre.bias"), 1, 3, 1, cfg.conv_flat);   // [T,512]

    // 条件项：cond(g)，g [gin,1] -> [1,gin,1]
    ggml_tensor* g3 = ggml_reshape_3d(ctx, g, 1, cfg.gin_channels, 1);
    ggml_tensor* cond = conv1d(ctx, g3, require(model, "dec.cond.weight"),
        require(model, "dec.cond.bias"), 1, 0, 1, cfg.conv_flat);       // [1,512,1]
    cond = ggml_repeat(ctx, cond, x);                             // [T,512,1]
    x = add_checked(ctx, x, cond);

    for (int64_t i = 0; i < n_ups; ++i) {
        const int64_t rate = cfg.upsample_rates[static_cast<size_t>(i)];
        const int64_t kernel = cfg.upsample_kernel_sizes[static_cast<size_t>(i)];
        const int64_t out_ch = init_ch >> (i + 1);
        const std::string up = "dec.ups." + std::to_string(i);

        x = ggml_leaky_relu(ctx, x, kLReluSlope, false);
        x = conv_transpose1d_pytorch(ctx, x, require(model, up + ".weight"),
            require(model, up + ".bias"), rate, kernel);          // [T_i, out_ch]
        if (x->ne[1] != out_ch) {
            throw std::runtime_error("VITS 生成器上采样通道数异常");
        }

        if (cfg.has_f0) {
            if (source == nullptr) {
                throw std::runtime_error("VITS 生成器缺少 sine 激励输入");
            }
            // 噪声卷积：kernel/stride 由后续上采样率决定（最后一个为 1x1）
            int64_t stride_f0 = 1;
            for (int64_t nxt = i + 1; nxt < n_ups; ++nxt) {
                stride_f0 *= cfg.upsample_rates[static_cast<size_t>(nxt)];
            }
            const int64_t nk = (stride_f0 == 1) ? 1 : stride_f0 * 2;
            const int64_t np = stride_f0 / 2;
            const std::string nc = "dec.noise_convs." + std::to_string(i);
            ggml_tensor* x_source = conv1d(ctx, source, require(model, nc + ".weight"),
                require(model, nc + ".bias"), static_cast<int>(stride_f0),
                static_cast<int>(np), 1, cfg.conv_flat);
            x = add_checked(ctx, x, x_source);
        }

        ggml_tensor* sum = nullptr;
        for (int64_t j = 0; j < n_kernels; ++j) {
            ggml_tensor* rb = resblock1(ctx, model, x,
                i * n_kernels + j, cfg.resblock_kernel_sizes[static_cast<size_t>(j)],
                cfg.conv_flat);
            sum = sum ? add_checked(ctx, sum, rb) : rb;
        }
        x = ggml_scale(ctx, sum, 1.0f / static_cast<float>(n_kernels));
    }

    x = ggml_leaky_relu(ctx, x, kLReluSlopeFinal, false);
    x = conv1d(ctx, x, require(model, "dec.conv_post.weight"), nullptr, 1, 3, 1,
        cfg.conv_flat);
    return ggml_tanh(ctx, x);                                     // [T_final, 1]
}

} // namespace vits
} // namespace rvc
