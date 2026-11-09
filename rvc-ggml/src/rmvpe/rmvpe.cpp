// rmvpe.cpp
// RMVPE graph implementation: mel front end (buffer_mel_basis in GGUF) + DeepUnet features +
// bidirectional GRU (512-frame chunking, state handoff) + 360-class salience head + cents decoding.

#include "rmvpe/rmvpe.h"

#include "audio/mel.h"
#include "engine.h"
#include "vits/modules.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>

namespace rvc {
namespace rmvpe {

namespace {

using rvc::vits::add_checked;
using rvc::vits::linear;
using rvc::vits::mul_checked;
using rvc::vits::to_f32;

constexpr int64_t kMelBins = 128;
constexpr int64_t kFeatureDim = 384;
constexpr int64_t kHiddenDim = 256;
constexpr int64_t kClasses = 360;

// Feature network window (aligned with reference implementation 0xShug0/audio.cpp)
constexpr int64_t kFeatureActiveFrames = 4096;
constexpr int64_t kFeatureOverlapFrames = 2048;
constexpr int64_t kFeatureGraphFrames = kFeatureActiveFrames + 2 * kFeatureOverlapFrames;
// Single-window limit: at or below this frame count, run whole-segment inference in one shot (strictly matching Python's full-mel inference);
// above it, fall back to windowed inference with context (boundary frames differ slightly from Python; see technical_02_ggml_vulkan_details.md)
// Limit derivation (Vulkan): flat conv first layer OC=16 ≤ 32 -> small tile, wg_denoms={32,32,1},
// matmul N = 128 × frames, and ggml-vulkan does not split the N dimension, so it must satisfy
//   128 × T / 32 ≤ maxComputeWorkGroupCount[1] = 65535
// Frames aligned to 32: 16352×4=65408 is safe, 16384×4=65536 exceeds the limit -> use 16352 (about 163.5s of audio)
constexpr int64_t kFeatureSingleShotLimit = 16352;

// GRU chunk length (graphs statically unrolled over 512 frames, with state handoff between chunks)
constexpr int64_t kGruChunkFrames = 512;

constexpr float kBatchNormEps = 1e-5f;

// Decoding constants (20 * arange(360) + 1997.3794084376191)
constexpr float kCentsStep = 20.0f;
constexpr float kCentsBase = 1997.3794084376191f;

const bool kDebug = getenv("RVC_RMVPE_DEBUG") != nullptr;

ggml_tensor* require(const GgufModel& model, const std::string& name) {
    ggml_tensor* t = model.weight(name);
    if (!t) {
        throw std::runtime_error("RMVPE is missing tensor: " + name);
    }
    return t;
}

// [C] -> [1,1,C,1]: conv2d output channel-dim broadcast (ne[2])
ggml_tensor* channel4(ggml_context* ctx, ggml_tensor* t) {
    return ggml_reshape_4d(ctx, to_f32(ctx, t), 1, 1, t->ne[0], 1);
}

ggml_tensor* sub_checked(ggml_context* ctx, ggml_tensor* a, ggml_tensor* b) {
    if (getenv("RVC_DEBUG_OP")) {
        fprintf(stderr, "[rmvpe sub] a=[%lld,%lld,%lld,%lld] b=[%lld,%lld,%lld,%lld]\n",
            (long long)a->ne[0], (long long)a->ne[1], (long long)a->ne[2], (long long)a->ne[3],
            (long long)b->ne[0], (long long)b->ne[1], (long long)b->ne[2], (long long)b->ne[3]);
    }
    return ggml_sub(ctx, a, b);
}

ggml_tensor* div_checked(ggml_context* ctx, ggml_tensor* a, ggml_tensor* b) {
    if (getenv("RVC_DEBUG_OP")) {
        fprintf(stderr, "[rmvpe div] a=[%lld,%lld,%lld,%lld] b=[%lld,%lld,%lld,%lld]\n",
            (long long)a->ne[0], (long long)a->ne[1], (long long)a->ne[2], (long long)a->ne[3],
            (long long)b->ne[0], (long long)b->ne[1], (long long)b->ne[2], (long long)b->ne[3]);
    }
    return ggml_div(ctx, a, b);
}

// BatchNorm2d (eval): y = (x - mean) / sqrt(var + eps) * w + b
// x: [W, H, C, N]; parameters [C] broadcast to the channel dim ne[2]
ggml_tensor* batch_norm2d(ggml_context* ctx, ggml_tensor* x,
    const GgufModel& model, const std::string& prefix) {
    ggml_tensor* w = channel4(ctx, require(model, prefix + ".weight"));
    ggml_tensor* b = channel4(ctx, require(model, prefix + ".bias"));
    ggml_tensor* mean = channel4(ctx, require(model, prefix + ".running_mean"));
    ggml_tensor* var = channel4(ctx, require(model, prefix + ".running_var"));
    ggml_tensor* denom = ggml_sqrt(ctx, ggml_scale_bias(ctx, var, 1.0f, kBatchNormEps));
    ggml_tensor* norm = div_checked(ctx, sub_checked(ctx, x, mean), denom);
    return add_checked(ctx, mul_checked(ctx, norm, w), b);
}

// Conv2d: x [W,H,IC,N], w [KW,KH,IC,OC], pad is padding on all four sides
// when flat=true, w uses a flattened layout [IC*KH*KW, OC] (GGUF rvc.conv_flat):
//   im2col(F32/F16) + mul_mat(quantized weights as src0), supporting quantized conv weights
ggml_tensor* conv2d_flat_impl(ggml_context* ctx, ggml_tensor* x, ggml_tensor* w,
    ggml_tensor* b, int pad, int stride, int64_t kh, int64_t kw) {
    const int64_t IC = x->ne[2];
    if (w->ne[2] != 1 || w->ne[3] != 1 || w->ne[0] != IC * kh * kw) {
        char buf[256];
        snprintf(buf, sizeof(buf),
            "conv2d(flat): weight/input mismatch (w=[%lld,%lld,%lld,%lld] in=[%lld,%lld,%lld,%lld] k=%lldx%lld)",
            static_cast<long long>(w->ne[0]), static_cast<long long>(w->ne[1]),
            static_cast<long long>(w->ne[2]), static_cast<long long>(w->ne[3]),
            static_cast<long long>(x->ne[0]), static_cast<long long>(x->ne[1]),
            static_cast<long long>(x->ne[2]), static_cast<long long>(x->ne[3]),
            static_cast<long long>(kh), static_cast<long long>(kw));
        throw std::runtime_error(buf);
    }
    const int64_t OC = w->ne[1];

    // im2col only reads the kernel shape (not its data)
    ggml_tensor* kernel = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, kw, kh, IC, 1);
    ggml_set_name(kernel, "conv2d_flat_kernel");
    ggml_set_input(kernel);

    ggml_tensor* cols = ggml_im2col(ctx, kernel, to_f32(ctx, x),
        stride, stride, pad, pad, 1, 1, true,
        w->type == GGML_TYPE_F16 ? GGML_TYPE_F16 : GGML_TYPE_F32);   // [KW*KH*IC, OW, OH, N]
    const int64_t OW = cols->ne[1];
    const int64_t OH = cols->ne[2];
    const int64_t N = cols->ne[3];
    ggml_tensor* cols2 = ggml_reshape_2d(ctx, cols, cols->ne[0], OW * OH * N);

    ggml_tensor* y = ggml_mul_mat(ctx, w, cols2);        // [OC, OW*OH*N]
    y = ggml_reshape_4d(ctx, y, OC, OW, OH, N);
    // ggml_permute args = source dim i mapped to target axis axis_i: src0(OC)->2, src1(OW)->0, src2(OH)->1
    y = rvc::vits::permute_cont(ctx, y, 2, 0, 1, 3);     // [OW, OH, OC, N]
    if (b) {
        y = add_checked(ctx, y, channel4(ctx, b));
    }
    return y;
}

ggml_tensor* conv2d(ggml_context* ctx, ggml_tensor* x, ggml_tensor* w, ggml_tensor* b,
    int pad, int stride = 1, int64_t kh = 3, int64_t kw = 3, bool flat = false) {
    if (getenv("RVC_DEBUG_CONV")) {
        fprintf(stderr, "[rmvpe conv2d] w=%s w.ne=[%lld,%lld,%lld,%lld] x.ne=[%lld,%lld,%lld,%lld] p=%d s=%d flat=%d\n",
            w->name,
            (long long)w->ne[0], (long long)w->ne[1], (long long)w->ne[2], (long long)w->ne[3],
            (long long)x->ne[0], (long long)x->ne[1], (long long)x->ne[2], (long long)x->ne[3],
            pad, stride, flat ? 1 : 0);
    }
    if (flat) {
        return conv2d_flat_impl(ctx, x, w, b, pad, stride, kh, kw);
    }
    ggml_tensor* y = ggml_conv_2d(ctx, w, to_f32(ctx, x), stride, stride, pad, pad, 1, 1);
    y = to_f32(ctx, y);
    if (b) {
        y = add_checked(ctx, y, channel4(ctx, b));
    }
    return y;
}

ggml_tensor* conv_bn_relu(ggml_context* ctx, ggml_tensor* x,
    const GgufModel& model, const std::string& conv_prefix, const std::string& bn_prefix,
    bool flat) {
    ggml_tensor* y = conv2d(ctx, x, require(model, conv_prefix + ".weight"), nullptr, 1, 1,
        3, 3, flat);
    y = batch_norm2d(ctx, y, model, bn_prefix);
    return ggml_relu(ctx, y);
}

// ConvBlockRes: conv.0/1 + conv.3/4 (3x3, pad1) + shortcut (1x1, only when channels change)
ggml_tensor* conv_block_res(ggml_context* ctx, ggml_tensor* x,
    const GgufModel& model, const std::string& prefix, bool flat) {
    const int64_t in_c = x->ne[2];
    ggml_tensor* y = conv_bn_relu(ctx, x, model, prefix + ".conv.0", prefix + ".conv.1", flat);
    const int64_t out_c = y->ne[2];
    y = conv_bn_relu(ctx, y, model, prefix + ".conv.3", prefix + ".conv.4", flat);
    if (in_c != out_c) {
        ggml_tensor* res = conv2d(ctx, x, require(model, prefix + ".shortcut.weight"),
            require(model, prefix + ".shortcut.bias"), 0, 1, 1, 1, flat);
        y = add_checked(ctx, y, res);
    }
    else {
        y = add_checked(ctx, y, x);
    }
    return y;
}

// AvgPool2d(2): non-overlapping 2x2 average
ggml_tensor* avg_pool2d_2x2(ggml_context* ctx, ggml_tensor* x) {
    return ggml_pool_2d(ctx, x, GGML_OP_POOL_AVG, 2, 2, 2, 2, 0.0f, 0.0f);
}

// ConvTranspose2d(k=3, s=2, p=1, output_padding=1): equivalent to 2x upsampling.
// ggml's no-padding version outputs (2*in+1); PyTorch outputs full[1 : 1+2*in] (pad=1 + op=1).
ggml_tensor* conv_transpose2x(ggml_context* ctx, ggml_tensor* x, ggml_tensor* w) {
    const int64_t W = x->ne[0];
    const int64_t H = x->ne[1];
    const int64_t OC = w->ne[2];
    const int64_t N = x->ne[3];
    ggml_tensor* full = ggml_conv_transpose_2d_p0(ctx, w, to_f32(ctx, x), 2);
    ggml_tensor* win = ggml_view_4d(ctx, full,
        2 * W, 2 * H, OC, N,
        full->nb[1], full->nb[2], full->nb[3],
        full->nb[0] + full->nb[1]);
    return ggml_cont(ctx, win);
}

// ------------------------------------------------------------------
// Subgraph construction
// ------------------------------------------------------------------

// DeepUnet + cnn: input [frames, 128] (mel row-major), output [384, frames]
ggml_tensor* build_feature_graph(ggml_context* ctx, ggml_tensor* input,
    const GgufModel& model, ggml_tensor** out, bool flat) {
    // [frames,128] -> [128,frames] -> [1,1,frames,128] (row-major = PyTorch [1,1,T,128])
    ggml_tensor* x = ggml_cont(ctx, ggml_transpose(ctx, to_f32(ctx, input)));
    const int64_t frames = x->ne[1];
    x = ggml_reshape_4d(ctx, x, kMelBins, frames, 1, 1);

    x = batch_norm2d(ctx, x, model, "unet.encoder.bn");

    std::vector<ggml_tensor*> skips;
    skips.reserve(5);
    for (int layer = 0; layer < 5; ++layer) {
        const std::string base = "unet.encoder.layers." + std::to_string(layer);
        for (int block = 0; block < 4; ++block) {
            x = conv_block_res(ctx, x, model, base + ".conv." + std::to_string(block), flat);
        }
        skips.push_back(x);
        x = avg_pool2d_2x2(ctx, x);
    }

    for (int layer = 0; layer < 4; ++layer) {
        const std::string base = "unet.intermediate.layers." + std::to_string(layer);
        for (int block = 0; block < 4; ++block) {
            x = conv_block_res(ctx, x, model, base + ".conv." + std::to_string(block), flat);
        }
    }

    for (int layer = 0; layer < 5; ++layer) {
        const std::string base = "unet.decoder.layers." + std::to_string(layer);
        const int64_t dec_out = x->ne[2] / 2;
        ggml_tensor* up = conv_transpose2x(ctx, x, require(model, base + ".conv1.0.weight"));
        up = batch_norm2d(ctx, up, model, base + ".conv1.1");
        up = ggml_relu(ctx, up);
        ggml_tensor* skip = skips.back();
        skips.pop_back();
        x = ggml_concat(ctx, up, skip, 2);   // PyTorch cat((x, skip), dim=1)
        for (int block = 0; block < 4; ++block) {
            x = conv_block_res(ctx, x, model, base + ".conv2." + std::to_string(block), flat);
        }
    }

    x = conv2d(ctx, x, require(model, "cnn.weight"), require(model, "cnn.bias"), 1, 1, 3, 3, flat);  // [128,frames,3,1]
    // [128,3,frames] -> [384,frames]: flatten (c,m) with m fastest (aligned with PyTorch flatten(-2))
    x = ggml_cont(ctx, ggml_permute(ctx, x, 0, 2, 1, 3));
    *out = ggml_reshape_2d(ctx, x, kFeatureDim, frames);
    return *out;
}

// BiGRU single direction (512-frame unroll): input [384,512], keep [512], hidden [256,1]
// Output sequence [256,512] and final_hidden [256,1]
void build_gru_graph(ggml_context* ctx, ggml_tensor* input, ggml_tensor* keep,
    ggml_tensor* hidden, const GgufModel& model, bool reverse,
    ggml_tensor** sequence_out, ggml_tensor** final_out) {
    const int64_t frames = input->ne[1];
    const std::string suffix = reverse ? "_reverse" : "";
    const std::string base = "fc.0.gru.";
    ggml_tensor* w_ih = require(model, base + "weight_ih_l0" + suffix);
    ggml_tensor* b_ih = require(model, base + "bias_ih_l0" + suffix);
    ggml_tensor* w_hh = require(model, base + "weight_hh_l0" + suffix);
    ggml_tensor* b_hh = require(model, base + "bias_hh_l0" + suffix);

    ggml_tensor* x_proj = linear(ctx, input, w_ih, b_ih);   // [768, frames]

    // Slice per gate: take [off, off+256) from [768,1]
    auto gate = [&](ggml_tensor* src, int64_t off) {
        return ggml_cont(ctx, ggml_view_2d(ctx, src, kHiddenDim, 1, src->nb[1],
            static_cast<size_t>(off) * src->nb[0]));
    };

    ggml_tensor* h = hidden;
    std::vector<ggml_tensor*> steps(static_cast<size_t>(frames));
    for (int64_t index = 0; index < frames; ++index) {
        const int64_t t = reverse ? (frames - 1 - index) : index;
        ggml_tensor* xi = ggml_cont(ctx, ggml_view_2d(ctx, x_proj, 3 * kHiddenDim, 1,
            x_proj->nb[1], static_cast<size_t>(t) * x_proj->nb[1]));
        ggml_tensor* hi = linear(ctx, h, w_hh, b_hh);       // [768,1]

        ggml_tensor* r = ggml_sigmoid(ctx, add_checked(ctx, gate(xi, 0), gate(hi, 0)));
        ggml_tensor* z = ggml_sigmoid(ctx, add_checked(ctx, gate(xi, kHiddenDim), gate(hi, kHiddenDim)));
        ggml_tensor* n = ggml_tanh(ctx, add_checked(ctx,
            gate(xi, 2 * kHiddenDim),
            mul_checked(ctx, r, gate(hi, 2 * kHiddenDim))));
        // h' = (1 - z) * n + z * h
        ggml_tensor* updated = add_checked(ctx, n, mul_checked(ctx, z, sub_checked(ctx, h, n)));

        ggml_tensor* keep_t = ggml_cont(ctx, ggml_view_1d(ctx, keep, 1,
            static_cast<size_t>(t) * keep->nb[0]));
        h = add_checked(ctx, h, mul_checked(ctx, sub_checked(ctx, updated, h), keep_t));
        steps[static_cast<size_t>(t)] = h;
    }

    ggml_tensor* sequence = steps[0];
    for (int64_t t = 1; t < frames; ++t) {
        sequence = ggml_concat(ctx, sequence, steps[static_cast<size_t>(t)], 1);
    }
    *sequence_out = sequence;
    *final_out = h;
}

// head: forward/reverse [256,512] -> concat [512,512] -> Linear(360) -> Sigmoid
ggml_tensor* build_head_graph(ggml_context* ctx, ggml_tensor* forward, ggml_tensor* reverse,
    const GgufModel& model) {
    ggml_tensor* cat = ggml_concat(ctx, forward, reverse, 0);   // [512,512]
    ggml_tensor* logits = linear(ctx, cat, require(model, "fc.1.weight"), require(model, "fc.1.bias"));
    return ggml_sigmoid(ctx, logits);   // [360,512]
}

// ------------------------------------------------------------------
// Host-side data movement
// ------------------------------------------------------------------

// Copy padded mel frames [start, start+frames) into the window (mel row-major; window size graph_frames)
void copy_mel_window(const std::vector<float>& mel, int64_t mel_frames, int64_t graph_frames,
    int64_t start, int64_t frames, std::vector<float>& out) {
    out.assign(static_cast<size_t>(kMelBins * graph_frames), 0.0f);
    for (int64_t m = 0; m < kMelBins; ++m) {
        std::copy_n(mel.begin() + static_cast<std::ptrdiff_t>(m * mel_frames + start),
            frames, out.begin() + static_cast<std::ptrdiff_t>(m * graph_frames));
    }
}

// Feature row copy: source [rows,384] row source_start -> destination row dst_start
void copy_feature_rows(const std::vector<float>& source, int64_t source_start, int64_t rows,
    std::vector<float>& destination, int64_t destination_start) {
    for (int64_t row = 0; row < rows; ++row) {
        std::copy_n(source.begin() + static_cast<std::ptrdiff_t>((source_start + row) * kFeatureDim),
            kFeatureDim,
            destination.begin() + static_cast<std::ptrdiff_t>((destination_start + row) * kFeatureDim));
    }
}

void fill_gru_chunk(const std::vector<float>& features, int64_t start, int64_t frames,
    std::vector<float>& chunk, std::vector<float>& keep) {
    chunk.assign(static_cast<size_t>(kGruChunkFrames * kFeatureDim), 0.0f);
    keep.assign(static_cast<size_t>(kGruChunkFrames), 0.0f);
    for (int64_t row = 0; row < frames; ++row) {
        std::copy_n(features.begin() + static_cast<std::ptrdiff_t>((start + row) * kFeatureDim),
            kFeatureDim,
            chunk.begin() + static_cast<std::ptrdiff_t>(row * kFeatureDim));
        keep[static_cast<size_t>(row)] = 1.0f;
    }
}

// Salience [frames,360] -> f0 (aligned with Python to_local_average_cents + decode)
std::vector<float> decode_salience(const std::vector<float>& salience, int64_t frames,
    float threshold) {
    std::vector<float> f0(static_cast<size_t>(frames), 0.0f);
    for (int64_t t = 0; t < frames; ++t) {
        const float* row = salience.data() + static_cast<std::ptrdiff_t>(t * kClasses);
        int64_t center = 0;
        float best = row[0];
        for (int64_t c = 1; c < kClasses; ++c) {
            if (row[c] > best) {
                best = row[c];
                center = c;
            }
        }
        if (best <= threshold) {
            continue;
        }
        float weighted = 0.0f;
        float weight_sum = 0.0f;
        for (int offset = -4; offset <= 4; ++offset) {
            const int64_t c = center + offset;
            if (c < 0 || c >= kClasses) {
                continue;
            }
            const float value = row[c];
            weighted += value * (kCentsStep * static_cast<float>(c) + kCentsBase);
            weight_sum += value;
        }
        if (weight_sum > 0.0f) {
            const float cents = weighted / weight_sum;
            f0[static_cast<size_t>(t)] = 10.0f * std::pow(2.0f, cents / 1200.0f);
        }
    }
    return f0;
}

} // namespace

// ------------------------------------------------------------------
// Impl
// ------------------------------------------------------------------

struct RmvpeExtractor::Impl {
    const GgufModel* model = nullptr;
    ggml_backend_t backend = nullptr;
    Config cfg;
    std::vector<float> mel_basis;

    // feature graph (size varies with input; rebuilt lazily)
    std::unique_ptr<GgmlGraphRunner> feature;
    ggml_tensor* feature_in = nullptr;
    ggml_tensor* feature_out = nullptr;
    int64_t feature_frames = 0;

    // GRU graphs (fixed 512 frames)
    std::unique_ptr<GgmlGraphRunner> fwd;
    ggml_tensor* fwd_in = nullptr;
    ggml_tensor* fwd_keep = nullptr;
    ggml_tensor* fwd_hidden = nullptr;
    ggml_tensor* fwd_sequence = nullptr;
    ggml_tensor* fwd_final = nullptr;

    std::unique_ptr<GgmlGraphRunner> rev;
    ggml_tensor* rev_in = nullptr;
    ggml_tensor* rev_keep = nullptr;
    ggml_tensor* rev_hidden = nullptr;
    ggml_tensor* rev_sequence = nullptr;
    ggml_tensor* rev_final = nullptr;

    // head graph (fixed 512 frames)
    std::unique_ptr<GgmlGraphRunner> head;
    ggml_tensor* head_fwd = nullptr;
    ggml_tensor* head_rev = nullptr;
    ggml_tensor* head_out = nullptr;

    void release_feature() {
        feature.reset();
        feature_in = nullptr;
        feature_out = nullptr;
        feature_frames = 0;
    }

    void ensure_feature_graph(int64_t frames) {
        if (frames <= 0 || frames % 32 != 0) {
            throw std::runtime_error("RMVPE feature graph requires a positive frame count aligned to 32");
        }
        if (feature && feature_frames == frames) {
            return;
        }
        release_feature();

        feature = std::make_unique<GgmlGraphRunner>(backend, 256ull * 1024 * 1024, 16384);
        feature_in = ggml_new_tensor_2d(feature->ctx(), GGML_TYPE_F32, frames, kMelBins);
        ggml_set_name(feature_in, "rmvpe.mel");
        ggml_set_input(feature_in);
        build_feature_graph(feature->ctx(), feature_in, *model, &feature_out, cfg.conv_flat);
        ggml_set_name(feature_out, "rmvpe.feature");
        ggml_set_output(feature_out);
        feature->expand(feature_out);
        if (!feature->allocate()) {
            release_feature();
            throw std::runtime_error("RMVPE feature graph buffer allocation failed");
        }
        feature_frames = frames;
        if (kDebug) {
            fprintf(stderr, "[rmvpe] feature graph: frames=%lld nodes=%d buffer=%zu MiB\n",
                (long long)frames, feature->node_count(),
                feature->buffer_size() / (1024 * 1024));
        }
    }

    void build_gru_runners() {
        fwd = std::make_unique<GgmlGraphRunner>(backend, 128ull * 1024 * 1024, 32768);
        fwd_in = ggml_new_tensor_2d(fwd->ctx(), GGML_TYPE_F32, kFeatureDim, kGruChunkFrames);
        fwd_keep = ggml_new_tensor_1d(fwd->ctx(), GGML_TYPE_F32, kGruChunkFrames);
        fwd_hidden = ggml_new_tensor_2d(fwd->ctx(), GGML_TYPE_F32, kHiddenDim, 1);
        ggml_set_name(fwd_in, "rmvpe.gru_in");
        ggml_set_name(fwd_keep, "rmvpe.gru_keep");
        ggml_set_name(fwd_hidden, "rmvpe.gru_h");
        ggml_set_input(fwd_in);
        ggml_set_input(fwd_keep);
        ggml_set_input(fwd_hidden);
        build_gru_graph(fwd->ctx(), fwd_in, fwd_keep, fwd_hidden, *model, false,
            &fwd_sequence, &fwd_final);
        ggml_set_output(fwd_sequence);
        ggml_set_output(fwd_final);
        fwd->expand(fwd_sequence);
        fwd->expand(fwd_final);
        if (!fwd->allocate()) {
            throw std::runtime_error("RMVPE forward GRU graph buffer allocation failed");
        }

        rev = std::make_unique<GgmlGraphRunner>(backend, 128ull * 1024 * 1024, 32768);
        rev_in = ggml_new_tensor_2d(rev->ctx(), GGML_TYPE_F32, kFeatureDim, kGruChunkFrames);
        rev_keep = ggml_new_tensor_1d(rev->ctx(), GGML_TYPE_F32, kGruChunkFrames);
        rev_hidden = ggml_new_tensor_2d(rev->ctx(), GGML_TYPE_F32, kHiddenDim, 1);
        ggml_set_name(rev_in, "rmvpe.gru_rev_in");
        ggml_set_name(rev_keep, "rmvpe.gru_rev_keep");
        ggml_set_name(rev_hidden, "rmvpe.gru_rev_h");
        ggml_set_input(rev_in);
        ggml_set_input(rev_keep);
        ggml_set_input(rev_hidden);
        build_gru_graph(rev->ctx(), rev_in, rev_keep, rev_hidden, *model, true,
            &rev_sequence, &rev_final);
        ggml_set_output(rev_sequence);
        ggml_set_output(rev_final);
        rev->expand(rev_sequence);
        rev->expand(rev_final);
        if (!rev->allocate()) {
            throw std::runtime_error("RMVPE reverse GRU graph buffer allocation failed");
        }

        head = std::make_unique<GgmlGraphRunner>(backend, 64ull * 1024 * 1024, 8192);
        head_fwd = ggml_new_tensor_2d(head->ctx(), GGML_TYPE_F32, kHiddenDim, kGruChunkFrames);
        head_rev = ggml_new_tensor_2d(head->ctx(), GGML_TYPE_F32, kHiddenDim, kGruChunkFrames);
        ggml_set_name(head_fwd, "rmvpe.head_fwd");
        ggml_set_name(head_rev, "rmvpe.head_rev");
        ggml_set_input(head_fwd);
        ggml_set_input(head_rev);
        head_out = build_head_graph(head->ctx(), head_fwd, head_rev, *model);
        ggml_set_name(head_out, "rmvpe.head_out");
        ggml_set_output(head_out);
        head->expand(head_out);
        if (!head->allocate()) {
            throw std::runtime_error("RMVPE head graph buffer allocation failed");
        }

        if (kDebug) {
            fprintf(stderr, "[rmvpe] gru/head graph: fwd=%d rev=%d head=%d nodes\n",
                fwd->node_count(), rev->node_count(), head->node_count());
        }
    }
};

RmvpeExtractor::RmvpeExtractor(const GgufModel& model, ggml_backend_t backend)
    : impl_(std::make_unique<Impl>()) {
    impl_->model = &model;
    impl_->backend = backend;
    impl_->cfg = read_config(model);

    impl_->mel_basis = model.read_f32("buffer_mel_basis");
    const size_t want = static_cast<size_t>(impl_->cfg.n_mels) * (impl_->cfg.n_fft / 2 + 1);
    if (impl_->mel_basis.size() != want) {
        throw std::runtime_error("RMVPE mel filter bank size mismatch: " +
            std::to_string(impl_->mel_basis.size()) + " != " + std::to_string(want));
    }

    impl_->build_gru_runners();
}

RmvpeExtractor::~RmvpeExtractor() = default;

Config read_config(const GgufModel& model) {
    Config cfg;
    const std::string kp = "rmvpe.";
    cfg.sample_rate = model.meta_u32(kp + "sample_rate", cfg.sample_rate);
    cfg.n_fft = model.meta_u32(kp + "n_fft", cfg.n_fft);
    cfg.hop_length = model.meta_u32(kp + "hop_length", cfg.hop_length);
    cfg.win_length = model.meta_u32(kp + "win_length", cfg.win_length);
    cfg.n_mels = model.meta_u32(kp + "n_mels", cfg.n_mels);
    cfg.n_classes = model.meta_u32(kp + "n_classes", cfg.n_classes);
    cfg.hidden_dim = model.meta_u32(kp + "hidden_dim", cfg.hidden_dim);
    cfg.feature_dim = model.meta_u32(kp + "feature_dim", cfg.feature_dim);
    cfg.mel_fmin = model.meta_f32(kp + "mel_fmin", cfg.mel_fmin);
    cfg.mel_fmax = model.meta_f32(kp + "mel_fmax", cfg.mel_fmax);
    cfg.mel_clamp = model.meta_f32(kp + "mel_clamp", cfg.mel_clamp);
    cfg.conv_flat = model.meta_bool("rvc.conv_flat", false);
    return cfg;
}

std::vector<float> RmvpeExtractor::compute_mel(const std::vector<float>& audio_16k,
    int64_t* frames_out) const {
    const Config& cfg = impl_->cfg;
    std::vector<float> mel = audio::log_mel_spectrogram(audio_16k, impl_->mel_basis,
        cfg.sample_rate, cfg.n_fft, cfg.hop_length, cfg.n_mels);
    const int64_t frames = static_cast<int64_t>(mel.size()) / cfg.n_mels;
    if (frames_out) {
        *frames_out = frames;
    }
    return mel;
}

Result RmvpeExtractor::infer(const std::vector<float>& audio_16k, float threshold) {
    if (audio_16k.empty()) {
        throw std::runtime_error("RMVPE input audio is empty");
    }
    const Config& cfg = impl_->cfg;

    Result result;
    result.mel = compute_mel(audio_16k, &result.frames);
    const int64_t frames = result.frames;
    const int64_t padded = 32 * ((frames - 1) / 32 + 1);
    result.padded_frames = padded;

    // Pad the mel time axis to a multiple of 32 (zero padding, aligned with Python F.pad)
    std::vector<float> mel_padded(static_cast<size_t>(kMelBins * padded), 0.0f);
    for (int64_t m = 0; m < kMelBins; ++m) {
        std::copy_n(result.mel.begin() + static_cast<std::ptrdiff_t>(m * frames),
            frames, mel_padded.begin() + static_cast<std::ptrdiff_t>(m * padded));
    }

    // ---- feature (DeepUnet): prefer a single whole-segment window; fall back to windowed inference when too long ----
    const bool single_shot = padded <= kFeatureSingleShotLimit;
    const int64_t step = single_shot ? padded : kFeatureActiveFrames;
    const int64_t graph_frames = single_shot ? padded : kFeatureGraphFrames;
    impl_->ensure_feature_graph(graph_frames);

    std::vector<float> features(static_cast<size_t>(padded * kFeatureDim), 0.0f);
    std::vector<float> window;
    for (int64_t active_start = 0; active_start < padded; active_start += step) {
        const int64_t active_frames = std::min<int64_t>(step, padded - active_start);
        const int64_t context_start = std::max<int64_t>(0, active_start - kFeatureOverlapFrames);
        const int64_t context_end = std::min<int64_t>(padded,
            active_start + active_frames + kFeatureOverlapFrames);
        const int64_t context_frames = context_end - context_start;
        const int64_t active_offset = active_start - context_start;

        copy_mel_window(mel_padded, padded, graph_frames, context_start, context_frames, window);
        impl_->feature->set_input_vec(impl_->feature_in, window);
        impl_->feature->compute();

        std::vector<float> feature_out;
        impl_->feature->get_output_vec(impl_->feature_out, feature_out);
        copy_feature_rows(feature_out, active_offset, active_frames, features, active_start);
    }

    // ---- forward GRU (512-frame chunking, state handoff) ----
    std::vector<float> forward_states(static_cast<size_t>(padded * kHiddenDim), 0.0f);
    std::vector<float> hidden(static_cast<size_t>(kHiddenDim), 0.0f);
    std::vector<float> chunk;
    std::vector<float> keep;
    std::vector<float> seq;
    for (int64_t start = 0; start < padded; start += kGruChunkFrames) {
        const int64_t chunk_frames = std::min<int64_t>(kGruChunkFrames, padded - start);
        fill_gru_chunk(features, start, chunk_frames, chunk, keep);
        impl_->fwd->set_input_vec(impl_->fwd_in, chunk);
        impl_->fwd->set_input_vec(impl_->fwd_keep, keep);
        impl_->fwd->set_input_vec(impl_->fwd_hidden, hidden);
        impl_->fwd->compute();
        impl_->fwd->get_output_vec(impl_->fwd_sequence, seq);
        for (int64_t row = 0; row < chunk_frames; ++row) {
            std::copy_n(seq.begin() + static_cast<std::ptrdiff_t>(row * kHiddenDim), kHiddenDim,
                forward_states.begin() + static_cast<std::ptrdiff_t>((start + row) * kHiddenDim));
        }
        impl_->fwd->get_output_vec(impl_->fwd_final, hidden);
    }

    // ---- reverse GRU + head ----
    std::vector<float> salience(static_cast<size_t>(padded * kClasses), 0.0f);
    std::vector<float> reverse_hidden(static_cast<size_t>(kHiddenDim), 0.0f);
    std::vector<float> head_fwd(static_cast<size_t>(kGruChunkFrames * kHiddenDim), 0.0f);
    std::vector<float> head_out;
    for (int64_t end = padded; end > 0;) {
        const int64_t start = std::max<int64_t>(0, end - kGruChunkFrames);
        const int64_t chunk_frames = end - start;

        fill_gru_chunk(features, start, chunk_frames, chunk, keep);
        impl_->rev->set_input_vec(impl_->rev_in, chunk);
        impl_->rev->set_input_vec(impl_->rev_keep, keep);
        impl_->rev->set_input_vec(impl_->rev_hidden, reverse_hidden);
        impl_->rev->compute();
        impl_->rev->get_output_vec(impl_->rev_sequence, seq);
        impl_->rev->get_output_vec(impl_->rev_final, reverse_hidden);

        std::fill(head_fwd.begin(), head_fwd.end(), 0.0f);
        for (int64_t row = 0; row < chunk_frames; ++row) {
            std::copy_n(
                forward_states.begin() + static_cast<std::ptrdiff_t>((start + row) * kHiddenDim),
                kHiddenDim,
                head_fwd.begin() + static_cast<std::ptrdiff_t>(row * kHiddenDim));
        }
        impl_->head->set_input_vec(impl_->head_fwd, head_fwd);
        impl_->head->set_input_vec(impl_->head_rev, seq);
        impl_->head->compute();
        impl_->head->get_output_vec(impl_->head_out, head_out);
        for (int64_t row = 0; row < chunk_frames; ++row) {
            std::copy_n(head_out.begin() + static_cast<std::ptrdiff_t>(row * kClasses), kClasses,
                salience.begin() + static_cast<std::ptrdiff_t>((start + row) * kClasses));
        }
        end = start;
    }

    result.hidden_padded = salience;
    result.hidden.assign(salience.begin(),
        salience.begin() + static_cast<std::ptrdiff_t>(frames * kClasses));
    result.f0 = decode_salience(result.hidden, frames, threshold);
    return result;
}

} // namespace rmvpe
} // namespace rvc
