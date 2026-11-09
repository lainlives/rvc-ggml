// synthesizer.cpp
// VITS synthesizer orchestration: read GGUF metadata/weights, build a static graph (enc_p -> flow -> dec),
// allocate buffers and execute; the same (frames, feature_dim, sid) reuses the graph.

#include "vits/synthesizer.h"

#include "vits/modules.h"
#include "engine.h"
#include "utils.h"

#include <algorithm>
#include <stdexcept>

namespace rvc {
namespace vits {

namespace {

    std::vector<int> to_int_vec(const std::vector<int64_t>& values) {
        std::vector<int> out;
        out.reserve(values.size());
        for (int64_t v : values) {
            out.push_back(static_cast<int>(v));
        }
        return out;
    }

    // Per-element: input features [T,C_in] row-major
    void check_input(const Config& cfg, const Input& in) {
        if (in.frames <= 0 || in.feature_dim <= 0) {
            throw std::runtime_error("VITS input frame count/dimension is invalid");
        }
        const size_t feat = static_cast<size_t>(in.frames) * in.feature_dim;
        if (in.features.size() != feat) {
            throw std::runtime_error("VITS features size mismatch");
        }
        const size_t noise = static_cast<size_t>(in.frames) * cfg.inter_channels;
        if (in.noise.size() != noise) {
            throw std::runtime_error("VITS noise size mismatch");
        }
        if (cfg.has_f0) {
            if (static_cast<int64_t>(in.pitch.size()) != in.frames) {
                throw std::runtime_error("VITS pitch size mismatch");
            }
            if (static_cast<int64_t>(in.source.size()) != in.frames * cfg.hop_samples) {
                throw std::runtime_error("VITS source(sine) size mismatch");
            }
        }
    }

} // namespace

Config read_config(const GgufModel& model) {
    Config cfg;
    cfg.version = model.meta_str("rvc.version", "v1");
    cfg.has_f0 = model.meta_bool("rvc.has_f0", true);
    cfg.sample_rate = model.meta_u32("rvc.sample_rate", 40000);
    cfg.n_spk = model.meta_u32("rvc.n_spk", 0);
    cfg.inter_channels = model.meta_u32("rvc.inter_channels", 192);
    cfg.hidden_channels = model.meta_u32("rvc.hidden_channels", 192);
    cfg.filter_channels = model.meta_u32("rvc.filter_channels", 768);
    cfg.n_heads = model.meta_u32("rvc.n_heads", 2);
    cfg.n_layers = model.meta_u32("rvc.n_layers", 6);
    cfg.kernel_size = model.meta_u32("rvc.kernel_size", 3);
    cfg.resblock_kernel_sizes = to_int_vec(model.meta_array_i32("rvc.resblock_kernel_sizes"));
    cfg.resblock_dilation_sizes = to_int_vec(model.meta_array_i32("rvc.resblock_dilation_sizes"));
    cfg.upsample_rates = to_int_vec(model.meta_array_i32("rvc.upsample_rates"));
    cfg.upsample_kernel_sizes = to_int_vec(model.meta_array_i32("rvc.upsample_kernel_sizes"));
    cfg.upsample_initial_channel = model.meta_u32("rvc.upsample_initial_channel", 512);
    cfg.spk_embed_dim = model.meta_u32("rvc.spk_embed_dim", 0);
    cfg.gin_channels = model.meta_u32("rvc.gin_channels", 256);
    cfg.conv_flat = model.meta_bool("rvc.conv_flat", false);

    if (cfg.upsample_rates.empty() || cfg.upsample_kernel_sizes.size() != cfg.upsample_rates.size()) {
        throw std::runtime_error("VITS metadata is missing upsample_rates/kernel_sizes (re-export with the current export_gguf.py)");
    }
    cfg.hop_samples = 1;
    for (int r : cfg.upsample_rates) {
        cfg.hop_samples *= r;
    }

    // flow WN layer count detected from weights (may differ between training versions)
    int flow_layers = 0;
    while (model.has_weight("flow.flows.0.enc.in_layers." +
        std::to_string(flow_layers) + ".weight")) {
        ++flow_layers;
    }
    if (flow_layers == 0) {
        throw std::runtime_error("VITS is missing flow weights (flow.flows.0.enc.in_layers.*)");
    }
    cfg.flow_n_layers = flow_layers;
    return cfg;
}

Synthesizer::Synthesizer(const GgufModel& model, ggml_backend_t backend)
    : model_(model), backend_(backend) {
    cfg_ = read_config(model);
}

Synthesizer::~Synthesizer() = default;

void Synthesizer::build(int64_t frames, int feature_dim, int speaker_id) {
    const int64_t inter = cfg_.inter_channels;
    runner_ = std::make_unique<GgmlGraphRunner>(backend_, 1024ull * 1024 * 1024, 65536);
    ggml_context* ctx = runner_->ctx();
    constants_.clear();
    constants_uploaded_ = false;

    in_features_ = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, feature_dim, frames);
    ggml_set_name(in_features_, "vits.features");
    ggml_set_input(in_features_);

    in_noise_ = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, inter, frames);
    ggml_set_name(in_noise_, "vits.noise");
    ggml_set_input(in_noise_);

    if (cfg_.has_f0) {
        in_pitch_ = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, frames);
        ggml_set_name(in_pitch_, "vits.pitch");
        ggml_set_input(in_pitch_);

        in_source_ = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, frames * cfg_.hop_samples, 1, 1);
        ggml_set_name(in_source_, "vits.sine");
        ggml_set_input(in_source_);
    }
    else {
        in_pitch_ = nullptr;
        in_source_ = nullptr;
    }

    // Speaker embedding g = emb_g[sid]; sid is a constant tensor
    ggml_tensor* in_sid = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
    ggml_set_name(in_sid, "vits.sid");
    ggml_set_input(in_sid);
    ggml_tensor* g = get_rows(ctx, model_.weight("emb_g.weight"), in_sid);   // [gin,1]

    // Channel flip matrix (used by flow)
    ggml_tensor* flip = make_const_f32(ctx, inter, inter, "vits.flip");
    {
        std::vector<float> p(static_cast<size_t>(inter * inter), 0.0f);
        for (int64_t i = 0; i < inter; ++i) {
            p[static_cast<size_t>(i * inter + (inter - 1 - i))] = 1.0f;
        }
        constants_.push_back({ flip, std::move(p) });
    }

    ggml_tensor* stats = build_text_encoder_stats(ctx, model_, cfg_, in_features_,
        in_pitch_, frames, constants_);
    out_m_ = ggml_cont(ctx, slice_channels(ctx, stats, 0, inter));
    out_logs_ = ggml_cont(ctx, slice_channels(ctx, stats, inter, inter));
    out_zp_ = add_checked(ctx, out_m_,
        ggml_scale(ctx, mul_checked(ctx, ggml_exp(ctx, out_logs_), in_noise_), 0.66666f));
    out_z_ = build_flow_reverse(ctx, model_, cfg_, out_zp_, g, frames, flip);
    out_audio_ = build_generator(ctx, model_, cfg_, out_z_, g, in_source_, frames);

    for (ggml_tensor* t : { out_m_, out_logs_, out_zp_, out_z_, out_audio_ }) {
        ggml_set_output(t);
        runner_->expand(t);
    }

    if (!runner_->allocate()) {
        throw std::runtime_error("VITS compute graph buffer allocation failed");
    }
    Logger::info("VITS graph: frames=" + std::to_string(frames) +
        " nodes=" + std::to_string(runner_->node_count()) +
        " buffer=" + std::to_string(runner_->buffer_size() / (1024 * 1024)) + " MiB");

    // Write constants and sid only once
    for (auto& c : constants_) {
        runner_->set_input(c.tensor, c.values.data(), c.values.size() * sizeof(float));
    }
    const int32_t sid32 = static_cast<int32_t>(speaker_id);
    runner_->set_input(in_sid, &sid32, sizeof(sid32));
    constants_uploaded_ = true;

    built_frames_ = frames;
    built_feature_dim_ = feature_dim;
    built_sid_ = speaker_id;
}

Output Synthesizer::infer(const Input& input) {
    check_input(cfg_, input);
    if (built_frames_ != input.frames || built_feature_dim_ != input.feature_dim ||
        built_sid_ != input.speaker_id) {
        build(input.frames, input.feature_dim, input.speaker_id);
    }

    runner_->set_input(in_features_, input.features.data(),
        input.features.size() * sizeof(float));
    runner_->set_input(in_noise_, input.noise.data(), input.noise.size() * sizeof(float));
    if (cfg_.has_f0) {
        runner_->set_input(in_pitch_, input.pitch.data(), input.pitch.size() * sizeof(int32_t));
        runner_->set_input(in_source_, input.source.data(), input.source.size() * sizeof(float));
    }
    runner_->compute();

    const int64_t frames = input.frames;
    const int64_t inter = cfg_.inter_channels;

    Output out;
    out.sample_rate = cfg_.sample_rate;
    out.samples = static_cast<int64_t>(ggml_nelements(out_audio_));
    runner_->get_output_vec(out_audio_, out.audio);

    // [C,T] memory -> [T,C] row-major
    auto read_transposed = [&](ggml_tensor* t, std::vector<float>& dst) {
        std::vector<float> raw;
        runner_->get_output_vec(t, raw);
        dst.resize(raw.size());
        for (int64_t c = 0; c < inter; ++c) {
            for (int64_t tt = 0; tt < frames; ++tt) {
                dst[static_cast<size_t>(tt * inter + c)] =
                    raw[static_cast<size_t>(c + inter * tt)];
            }
        }
    };
    read_transposed(out_m_, out.m_p);
    read_transposed(out_logs_, out.logs_p);
    read_transposed(out_zp_, out.z_p);
    read_transposed(out_z_, out.z);
    return out;
}

} // namespace vits
} // namespace rvc
