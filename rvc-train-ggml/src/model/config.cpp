// RVC-Train-GGML - RVC model configuration implementation (src/model/config.cpp)
#include "model/config.h"

namespace rvc_train {
namespace model {

namespace {

void set_err(std::string* err, const std::string& msg) {
    if (err != nullptr) {
        *err = msg;
    }
}

} // namespace

ModelConfig ModelConfig::from_meta(const TrainMeta& m) {
    ModelConfig c;
    c.kind     = m.kind;
    c.version  = m.version;
    c.has_f0   = m.has_f0;
    c.form     = m.form;
    c.resblock = m.resblock;

    c.sample_rate    = m.sample_rate;
    c.spec_channels  = m.spec_channels;
    c.filter_length  = m.filter_length;
    c.hop_length     = m.hop_length;
    c.win_length     = m.win_length;
    c.n_mel_channels = m.n_mel_channels;
    c.mel_fmin       = m.mel_fmin;
    c.mel_fmax       = m.mel_fmax;
    c.segment_size   = m.segment_size;

    c.inter_channels  = m.inter_channels;
    c.hidden_channels = m.hidden_channels;
    c.filter_channels = m.filter_channels;
    c.n_heads         = m.n_heads;
    c.n_layers        = m.n_layers;
    c.kernel_size     = m.kernel_size;
    c.p_dropout       = m.p_dropout;
    c.gin_channels    = m.gin_channels;
    c.spk_embed_dim   = m.spk_embed_dim;
    c.upsample_initial_channel = m.upsample_initial_channel;
    c.resblock_kernel_sizes    = m.resblock_kernel_sizes;
    c.resblock_dilation_sizes  = m.resblock_dilation_sizes;
    c.upsample_rates           = m.upsample_rates;
    c.upsample_kernel_sizes    = m.upsample_kernel_sizes;
    c.d_periods                = m.d_periods;
    return c;
}

bool ModelConfig::validate(std::string* err) const {
    if (kind != "G" && kind != "D") {
        set_err(err, "invalid kind (expected G/D): " + kind);
        return false;
    }
    if (version != "v1" && version != "v2") {
        set_err(err, "invalid version (expected v1/v2): " + version);
        return false;
    }

    // Discriminator: only validate the period list (structure parameters are derived from the fully-connected convolution chain, implemented in M2-7)
    if (is_discriminator()) {
        if (d_periods.empty()) {
            set_err(err, "D model is missing rvc.train.d_periods");
            return false;
        }
        return true;
    }

    // ---------------------------------------------------------- G (generator)
    if (inter_channels <= 0 || hidden_channels <= 0 || filter_channels <= 0) {
        set_err(err, "invalid channel parameters (inter/hidden/filter must be > 0)");
        return false;
    }
    if (n_heads <= 0 || n_layers <= 0 || kernel_size <= 0) {
        set_err(err, "invalid Transformer parameters (n_heads/n_layers/kernel_size must be > 0)");
        return false;
    }
    if (inter_channels % n_heads != 0) {
        set_err(err, "inter_channels is not divisible by n_heads");
        return false;
    }
    if (gin_channels <= 0 || upsample_initial_channel <= 0 || spk_embed_dim <= 0) {
        set_err(err, "gin_channels/upsample_initial_channel/spk_embed_dim must be > 0");
        return false;
    }
    if (resblock != "1") {
        set_err(err, "resblock only supports \"1\" (ResBlock1): " + resblock);
        return false;
    }
    if (resblock_kernel_sizes.empty()) {
        set_err(err, "resblock_kernel_sizes is empty");
        return false;
    }
    if (resblock_dilation_sizes.size() != resblock_kernel_sizes.size() * 3) {
        set_err(err, "flattened resblock_dilation_sizes length does not match the kernel count");
        return false;
    }
    if (upsample_rates.empty() || upsample_rates.size() != upsample_kernel_sizes.size()) {
        set_err(err, "invalid upsample_rates/upsample_kernel_sizes (empty or length mismatch)");
        return false;
    }
    for (size_t i = 0; i < upsample_rates.size(); ++i) {
        if (upsample_rates[i] <= 0 || upsample_kernel_sizes[i] <= 0) {
            set_err(err, "upsample_rates/kernel_sizes contains a non-positive value");
            return false;
        }
    }
    if (hop_length > 0 && segment_size > 0 && segment_size % hop_length != 0) {
        set_err(err, "segment_size is not divisible by hop_length");
        return false;
    }
    if (filter_length > 0 && spec_channels > 0 && spec_channels != filter_length / 2 + 1) {
        set_err(err, "spec_channels is inconsistent with filter_length (expected filter_length/2+1)");
        return false;
    }
    return true;
}

int64_t ModelConfig::hop_product() const {
    int64_t p = 1;
    for (int32_t r : upsample_rates) {
        p *= (int64_t) r;
    }
    return p;
}

int64_t ModelConfig::z_frames() const {
    if (hop_length <= 0 || segment_size <= 0) {
        return 0;
    }
    return segment_size / hop_length;
}

ModelConfig default_model_config_v1_40k() {
    ModelConfig c;
    c.kind = "G";
    c.version = "v1";
    c.has_f0 = true;
    c.form = "train";
    c.resblock = "1";

    c.sample_rate = 40000;
    c.filter_length = 2048;
    c.hop_length = 400;
    c.win_length = 2048;
    c.n_mel_channels = 125;
    c.mel_fmin = 0.0f;
    c.mel_fmax = 0.0f;
    c.spec_channels = c.filter_length / 2 + 1;
    c.segment_size = 12800;

    c.inter_channels = 192;
    c.hidden_channels = 192;
    c.filter_channels = 768;
    c.n_heads = 2;
    c.n_layers = 6;
    c.kernel_size = 3;
    c.p_dropout = 0.0f;
    c.gin_channels = 256;
    c.spk_embed_dim = 109;
    c.upsample_initial_channel = 512;
    c.resblock_kernel_sizes = {3, 7, 11};
    c.resblock_dilation_sizes = {1, 3, 5, 1, 3, 5, 1, 3, 5};
    c.upsample_rates = {10, 10, 2, 2};
    c.upsample_kernel_sizes = {16, 16, 4, 4};
    c.d_periods = {2, 3, 5, 7, 11, 17};
    return c;
}

ModelConfig default_model_config_v2_48k() {
    ModelConfig c;
    c.kind = "G";
    c.version = "v2";
    c.has_f0 = true;
    c.form = "train";
    c.resblock = "1";

    // v2-48k: hop = 480 (= sample_rate/100), feature width 768
    c.sample_rate = 48000;
    c.filter_length = 2048;
    c.hop_length = 480;
    c.win_length = 2048;
    c.n_mel_channels = 128;
    c.mel_fmin = 0.0f;
    c.mel_fmax = 0.0f;
    c.spec_channels = c.filter_length / 2 + 1;
    c.segment_size = 17280;   // official v2-48k segment_size (= 36 z frames)

    c.inter_channels = 192;
    c.hidden_channels = 192;
    c.filter_channels = 768;
    c.n_heads = 2;
    c.n_layers = 6;
    c.kernel_size = 3;
    c.p_dropout = 0.0f;
    c.gin_channels = 256;
    c.spk_embed_dim = 109;
    c.upsample_initial_channel = 512;
    c.resblock_kernel_sizes = {3, 7, 11};
    c.resblock_dilation_sizes = {1, 3, 5, 1, 3, 5, 1, 3, 5};
    c.upsample_rates = {12, 10, 2, 2};           // ∏ = 480
    c.upsample_kernel_sizes = {24, 20, 4, 4};
    c.d_periods = {2, 3, 5, 7, 11, 17, 23, 37};  // v2 MPD: 9 discriminators
    return c;
}

} // namespace model
} // namespace rvc_train
