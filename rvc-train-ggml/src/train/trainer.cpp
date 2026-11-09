// RVC-Train-GGML - M5-3 training loop implementation (src/train/trainer.cpp)
//
// Semantic baseline: official 81eed5e train/train.py + data_utils.py (see `docs/handover.md` §4.9 A/D).
// v1 simplification (Q25): single-bucket graph (TrainStep fixed shape; samples exceeding graph T are skipped and logged).
// M7-3: gradient accumulation (grad_accum); M7-4: epoch snapshots/summary + export every N epochs; M7-5: resume metadata.
#include "train/trainer.h"

#include "common/json.h"
#include "common/platform.h"

#include "export/mem_to_train_gguf.h"
#include "export/model_gguf.h"
#include "model/config.h"
#include "model/weights.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <system_error>
#include <unordered_map>
#include <utility>

namespace rvc_train {
namespace train {

namespace {

std::string join_path(const std::string& dir, const std::string& name) {
    if (dir.empty()) {
        return name;
    }
    const char last = dir.back();
    if (last == '/' || last == '\\') {
        return dir + name;
    }
    return dir + platform::path_sep() + name;
}

double json_num(const common::JsonValue* obj, const char* key, double default_value) {
    const common::JsonValue* v = obj != nullptr ? obj->find(key) : nullptr;
    return (v != nullptr && v->is_number()) ? v->as_number(default_value) : default_value;
}

std::string json_escape(const std::string& s) {
    std::string out;
    out.reserve(s.size());
    for (char c : s) {
        switch (c) {
            case '\\': out += "\\\\"; break;
            case '"':  out += "\\\""; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:   out += c; break;
        }
    }
    return out;
}

int64_t json_int(const common::JsonValue* obj, const char* key, int64_t default_value) {
    const common::JsonValue* v = obj != nullptr ? obj->find(key) : nullptr;
    return (v != nullptr && v->is_number()) ? v->as_int(default_value) : default_value;
}

std::string json_str(const common::JsonValue* obj, const char* key, const std::string& default_value) {
    const common::JsonValue* v = obj != nullptr ? obj->find(key) : nullptr;
    return (v != nullptr && v->is_string()) ? v->as_string() : default_value;
}

bool json_bool(const common::JsonValue* obj, const char* key, bool default_value) {
    const common::JsonValue* v = obj != nullptr ? obj->find(key) : nullptr;
    return (v != nullptr && v->is_bool()) ? v->as_bool(default_value) : default_value;
}

void json_arr_i32(const common::JsonValue* obj, const char* key, std::vector<int32_t>* out) {
    const common::JsonValue* v = obj != nullptr ? obj->find(key) : nullptr;
    if (v == nullptr || !v->is_array()) {
        return;
    }
    out->clear();
    for (size_t i = 0; i < v->size(); ++i) {
        const common::JsonValue& e = v->at(i);
        if (e.is_number()) {
            out->push_back((int32_t) e.as_int(0));
        }
    }
}

} // namespace

bool trainer_config_from_json_text(const std::string& text, TrainerConfig* out, std::string* err) {
    if (out == nullptr) {
        if (err != nullptr) {
            *err = "trainer_config_from_json_text: out is null";
        }
        return false;
    }
    common::JsonValue root;
    if (!common::JsonValue::parse(text, &root, err)) {
        return false;
    }
    if (!root.is_object()) {
        if (err != nullptr) {
            *err = "config root is not a JSON object";
        }
        return false;
    }
    const common::JsonValue* ru = root.find("run");
    const common::JsonValue* tr = root.find("train");
    const common::JsonValue* da = root.find("data");
    const common::JsonValue* mo = root.find("model");

    if (ru != nullptr && ru->is_object()) {
        out->exp_dir         = json_str(ru, "exp", out->exp_dir);
        out->out_dir         = json_str(ru, "out", out->out_dir);
        out->weights_g       = json_str(ru, "pretrain_g", out->weights_g);
        out->weights_d       = json_str(ru, "pretrain_d", out->weights_d);
        out->resume          = json_str(ru, "resume", out->resume);
        out->export_run_json = json_str(ru, "export_run_json", out->export_run_json);
        out->device          = json_int(ru, "device", out->device);
        out->export_every    = json_int(ru, "export_every", out->export_every);
        out->snapshot_keep   = json_int(ru, "snapshot_keep", out->snapshot_keep);
        out->scratch         = json_bool(ru, "scratch", out->scratch);
    }
    if (tr != nullptr && tr->is_object()) {
        out->log_interval = json_int(tr, "log_interval", out->log_interval);
        out->seed         = json_int(tr, "seed", out->seed);
        out->epochs       = json_int(tr, "epochs", out->epochs);
        out->max_steps    = json_int(tr, "max_steps", out->max_steps);
        out->batch_size   = json_int(tr, "batch_size", out->batch_size);
        out->lr           = (float) json_num(tr, "learning_rate", out->lr);
        out->eps          = (float) json_num(tr, "eps", out->eps);
        out->weight_decay = (float) json_num(tr, "weight_decay", out->weight_decay);
        out->lr_decay     = (float) json_num(tr, "lr_decay", out->lr_decay);
        out->c_mel        = (float) json_num(tr, "c_mel", out->c_mel);
        out->c_kl         = (float) json_num(tr, "c_kl", out->c_kl);
        out->save_interval = json_int(tr, "save_interval", out->save_interval);
        out->disc_len      = json_int(tr, "disc_len", out->disc_len);
        out->bucket        = json_int(tr, "bucket", out->bucket);
        out->grad_accum    = json_int(tr, "grad_accum", out->grad_accum);
        const common::JsonValue* betas = tr->find("betas");
        if (betas != nullptr && betas->is_array() && betas->size() >= 2) {
            out->beta1 = (float) betas->at(0).as_number(out->beta1);
            out->beta2 = (float) betas->at(1).as_number(out->beta2);
        }
        const common::JsonValue* buckets = tr->find("buckets");
        if (buckets != nullptr && buckets->is_array()) {
            out->buckets.clear();
            for (size_t i = 0; i < buckets->size(); ++i) {
                const common::JsonValue& v = buckets->at(i);
                if (v.is_number()) {
                    out->buckets.push_back(v.as_int(0));
                }
            }
        }
        const double seg = json_num(tr, "segment_size", (double) out->segment_samples);
        if (seg > 0.0) {
            out->segment_samples = (int64_t) seg;
        }
    }
    if (da != nullptr && da->is_object()) {
        out->sr     = json_num(da, "sampling_rate", out->sr);
        out->n_fft  = json_int(da, "filter_length", out->n_fft);
        out->win    = json_int(da, "win_length", out->win);
        out->n_mels = json_int(da, "n_mel_channels", out->n_mels);
        out->fmin   = json_num(da, "mel_fmin", out->fmin);
        const common::JsonValue* fmax = da->find("mel_fmax");
        if (fmax != nullptr) {
            out->fmax = fmax->is_number() ? fmax->as_number(-1.0) : -1.0;
        }
        const int64_t hop = json_int(da, "hop_length", 0);
        if (hop > 0 && out->segment_samples % hop == 0) {
            out->segment_frames = out->segment_samples / hop;
        }
    }
    if (mo != nullptr && mo->is_object()) {
        out->q_channels = json_int(mo, "inter_channels", out->q_channels);
    }

    // ---------------- model structure (config data/model sections; used by M7-2 scratch) ----------------
    // Defaults to v1-40k (the project's initial target); the config `model` section overrides when present.
    model::ModelConfig& M = out->model_cfg;
    M = model::default_model_config_v1_40k();
    if (mo != nullptr && mo->is_object()) {
        // V2-2: when version=="v2" use v2-48k as the fallback base (so omitted fields do not fall back to v1 defaults), then override field by field.
        const std::string ver = json_str(mo, "version", M.version);
        if (ver == "v2") {
            M = model::default_model_config_v2_48k();
        }
        M.version = ver;
        M.has_f0 = json_bool(mo, "has_f0", M.has_f0);
        M.inter_channels = json_int(mo, "inter_channels", M.inter_channels);
        M.hidden_channels = json_int(mo, "hidden_channels", M.hidden_channels);
        M.filter_channels = json_int(mo, "filter_channels", M.filter_channels);
        M.n_heads = json_int(mo, "n_heads", M.n_heads);
        M.n_layers = json_int(mo, "n_layers", M.n_layers);
        M.kernel_size = json_int(mo, "kernel_size", M.kernel_size);
        M.p_dropout = (float) json_num(mo, "p_dropout", M.p_dropout);
        M.gin_channels = json_int(mo, "gin_channels", M.gin_channels);
        M.spk_embed_dim = json_int(mo, "spk_embed_dim", M.spk_embed_dim);
        M.upsample_initial_channel =
            json_int(mo, "upsample_initial_channel", M.upsample_initial_channel);
        M.resblock = json_str(mo, "resblock", M.resblock);
        json_arr_i32(mo, "resblock_kernel_sizes", &M.resblock_kernel_sizes);
        // resblock_dilation_sizes supports nested [[..],[..]] or flat forms
        const common::JsonValue* dil = mo->find("resblock_dilation_sizes");
        if (dil != nullptr && dil->is_array()) {
            M.resblock_dilation_sizes.clear();
            for (size_t i = 0; i < dil->size(); ++i) {
                const common::JsonValue& e = dil->at(i);
                if (e.is_array()) {
                    for (size_t j = 0; j < e.size(); ++j) {
                        if (e.at(j).is_number()) {
                            M.resblock_dilation_sizes.push_back((int32_t) e.at(j).as_int(0));
                        }
                    }
                } else if (e.is_number()) {
                    M.resblock_dilation_sizes.push_back((int32_t) e.as_int(0));
                }
            }
        }
        json_arr_i32(mo, "upsample_rates", &M.upsample_rates);
        json_arr_i32(mo, "upsample_kernel_sizes", &M.upsample_kernel_sizes);
        json_arr_i32(mo, "d_periods", &M.d_periods);
    }
    // data section -> audio/spectrogram
    M.sample_rate = (int64_t) out->sr;
    M.filter_length = out->n_fft;
    M.win_length = out->win;
    M.hop_length = out->segment_frames > 0 ? out->segment_samples / out->segment_frames : 0;
    M.n_mel_channels = out->n_mels;
    M.mel_fmin = (float) out->fmin;
    M.mel_fmax = (float) (out->fmax > 0.0 ? out->fmax : 0.0);
    M.spec_channels = out->n_fft / 2 + 1;
    M.segment_size = out->segment_samples;
    // When d_periods is not given explicitly, derive it from version
    if (M.d_periods.empty()) {
        M.d_periods = M.is_v2() ? std::vector<int32_t>{2, 3, 5, 7, 11, 17, 23, 37}
                                : std::vector<int32_t>{2, 3, 5, 7, 11, 17};
    }
    return true;
}

bool trainer_config_from_json(const std::string& path, TrainerConfig* out, std::string* err) {
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) {
        if (err != nullptr) {
            *err = "cannot open training config: " + path;
        }
        return false;
    }
    const std::string text((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    return trainer_config_from_json_text(text, out, err);
}

namespace {

void json_write_f(std::string* out, float v) {
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%.*g", std::numeric_limits<float>::max_digits10, (double) v);
    *out += buf;
}

void json_write_d(std::string* out, double v) {
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%.*g", std::numeric_limits<double>::max_digits10, v);
    *out += buf;
}

} // namespace

bool trainer_config_to_json(const TrainerConfig& cfg, std::string* out, std::string* err) {
    if (out == nullptr) {
        if (err != nullptr) {
            *err = "trainer_config_to_json: out is null";
        }
        return false;
    }
    std::string s;
    s.reserve(2048);
    s += "{\n";
    s += "  \"run\": {\n";
    s += "    \"exp\": \"" + json_escape(cfg.exp_dir) + "\",\n";
    s += "    \"out\": \"" + json_escape(cfg.out_dir) + "\",\n";
    s += "    \"pretrain_g\": \"" + json_escape(cfg.weights_g) + "\",\n";
    s += "    \"pretrain_d\": \"" + json_escape(cfg.weights_d) + "\",\n";
    s += "    \"device\": " + std::to_string((long long) cfg.device) + ",\n";
    s += "    \"resume\": \"" + json_escape(cfg.resume) + "\",\n";
    s += "    \"export_every\": " + std::to_string((long long) cfg.export_every) + ",\n";
    s += "    \"snapshot_keep\": " + std::to_string((long long) cfg.snapshot_keep) + ",\n";
    s += "    \"scratch\": ";
    s += cfg.scratch ? "true" : "false";
    s += ",\n";
    s += "    \"export_run_json\": \"" + json_escape(cfg.export_run_json) + "\"\n";
    s += "  },\n";
    s += "  \"train\": {\n";
    s += "    \"batch_size\": " + std::to_string((long long) cfg.batch_size) + ",\n";
    s += "    \"learning_rate\": ";  json_write_f(&s, cfg.lr); s += ",\n";
    s += "    \"betas\": [";  json_write_f(&s, cfg.beta1); s += ", "; json_write_f(&s, cfg.beta2); s += "],\n";
    s += "    \"eps\": ";          json_write_f(&s, cfg.eps); s += ",\n";
    s += "    \"weight_decay\": "; json_write_f(&s, cfg.weight_decay); s += ",\n";
    s += "    \"lr_decay\": ";     json_write_f(&s, cfg.lr_decay); s += ",\n";
    s += "    \"c_mel\": ";        json_write_f(&s, cfg.c_mel); s += ",\n";
    s += "    \"c_kl\": ";         json_write_f(&s, cfg.c_kl); s += ",\n";
    s += "    \"epochs\": " + std::to_string((long long) cfg.epochs) + ",\n";
    s += "    \"max_steps\": " + std::to_string((long long) cfg.max_steps) + ",\n";
    s += "    \"segment_size\": " + std::to_string((long long) cfg.segment_samples) + ",\n";
    s += "    \"bucket\": " + std::to_string((long long) cfg.bucket) + ",\n";
    s += "    \"disc_len\": " + std::to_string((long long) cfg.disc_len) + ",\n";
    s += "    \"log_interval\": " + std::to_string((long long) cfg.log_interval) + ",\n";
    s += "    \"save_interval\": " + std::to_string((long long) cfg.save_interval) + ",\n";
    s += "    \"seed\": " + std::to_string((long long) cfg.seed) + ",\n";
    s += "    \"grad_accum\": " + std::to_string((long long) cfg.grad_accum) + ",\n";
    s += "    \"buckets\": [";
    for (size_t i = 0; i < cfg.buckets.size(); ++i) {
        if (i > 0) {
            s += ", ";
        }
        s += std::to_string((long long) cfg.buckets[i]);
    }
    s += "]\n";
    s += "  },\n";
    s += "  \"data\": {\n";
    s += "    \"sampling_rate\": "; json_write_d(&s, cfg.sr); s += ",\n";
    s += "    \"filter_length\": " + std::to_string((long long) cfg.n_fft) + ",\n";
    s += "    \"hop_length\": " + std::to_string((long long) cfg.segment_frames > 0 ? cfg.segment_samples / cfg.segment_frames : 0) + ",\n";
    s += "    \"win_length\": " + std::to_string((long long) cfg.win) + ",\n";
    s += "    \"n_mel_channels\": " + std::to_string((long long) cfg.n_mels) + ",\n";
    s += "    \"mel_fmin\": "; json_write_d(&s, cfg.fmin); s += ",\n";
    s += "    \"mel_fmax\": ";
    if (cfg.fmax > 0.0) {
        json_write_d(&s, cfg.fmax);
    } else {
        s += "null";
    }
    s += "\n";
    s += "  },\n";
    s += "  \"model\": {\n";
    s += "    \"version\": \"" + json_escape(cfg.model_cfg.version) + "\",\n";
    s += "    \"has_f0\": ";
    s += cfg.model_cfg.has_f0 ? "true" : "false";
    s += ",\n";
    s += "    \"inter_channels\": " + std::to_string((long long) cfg.model_cfg.inter_channels) + ",\n";
    s += "    \"hidden_channels\": " + std::to_string((long long) cfg.model_cfg.hidden_channels) + ",\n";
    s += "    \"filter_channels\": " + std::to_string((long long) cfg.model_cfg.filter_channels) + ",\n";
    s += "    \"n_heads\": " + std::to_string((long long) cfg.model_cfg.n_heads) + ",\n";
    s += "    \"n_layers\": " + std::to_string((long long) cfg.model_cfg.n_layers) + ",\n";
    s += "    \"kernel_size\": " + std::to_string((long long) cfg.model_cfg.kernel_size) + ",\n";
    s += "    \"p_dropout\": "; json_write_f(&s, cfg.model_cfg.p_dropout); s += ",\n";
    s += "    \"resblock\": \"" + json_escape(cfg.model_cfg.resblock) + "\",\n";
    s += "    \"gin_channels\": " + std::to_string((long long) cfg.model_cfg.gin_channels) + ",\n";
    s += "    \"spk_embed_dim\": " + std::to_string((long long) cfg.model_cfg.spk_embed_dim) + ",\n";
    s += "    \"upsample_initial_channel\": " +
         std::to_string((long long) cfg.model_cfg.upsample_initial_channel) + ",\n";
    auto write_i32_arr = [&](const char* key, const std::vector<int32_t>& a, bool last) {
        s += "    \"";
        s += key;
        s += "\": [";
        for (size_t i = 0; i < a.size(); ++i) {
            if (i > 0) {
                s += ", ";
            }
            s += std::to_string((long long) a[i]);
        }
        s += "]";
        s += last ? "\n" : ",\n";
    };
    write_i32_arr("resblock_kernel_sizes", cfg.model_cfg.resblock_kernel_sizes, false);
    write_i32_arr("resblock_dilation_sizes", cfg.model_cfg.resblock_dilation_sizes, false);
    write_i32_arr("upsample_rates", cfg.model_cfg.upsample_rates, false);
    write_i32_arr("upsample_kernel_sizes", cfg.model_cfg.upsample_kernel_sizes, false);
    write_i32_arr("d_periods", cfg.model_cfg.d_periods, true);
    s += "  }\n";
    s += "}\n";
    *out = std::move(s);
    return true;
}

bool trainer_config_write_json(const TrainerConfig& cfg, const std::string& path, std::string* err) {
    std::error_code ec;
    const std::filesystem::path p(path);
    if (p.has_parent_path()) {
        std::filesystem::create_directories(p.parent_path(), ec);
        if (ec) {
            if (err != nullptr) {
                *err = "failed to create config directory: " + p.parent_path().string() + " (" + ec.message() + ")";
            }
            return false;
        }
    }
    std::string text;
    if (!trainer_config_to_json(cfg, &text, err)) {
        return false;
    }
    std::ofstream f(path, std::ios::trunc | std::ios::binary);
    if (!f.is_open()) {
        if (err != nullptr) {
            *err = "cannot write training config: " + path;
        }
        return false;
    }
    f << text;
    if (!f.good()) {
        if (err != nullptr) {
            *err = "failed to write training config: " + path;
        }
        return false;
    }
    return true;
}

bool trainer_config_equivalent(const TrainerConfig& a, const TrainerConfig& b, std::string* diff) {
    std::string ja;
    std::string jb;
    std::string err;
    if (!trainer_config_to_json(a, &ja, &err) || !trainer_config_to_json(b, &jb, &err)) {
        if (diff != nullptr) {
            *diff = "serialization failed: " + err;
        }
        return false;
    }
    if (ja == jb) {
        return true;
    }
    if (diff != nullptr) {
        std::string la;
        std::string lb;
        size_t ia = 0;
        size_t ib = 0;
        while (ia <= ja.size() || ib <= jb.size()) {
            const size_t ea = ja.find('\n', ia);
            const size_t eb = jb.find('\n', ib);
            la = ja.substr(ia, ea == std::string::npos ? std::string::npos : ea - ia);
            lb = jb.substr(ib, eb == std::string::npos ? std::string::npos : eb - ib);
            if (la != lb) {
                *diff = "first difference: expected \"" + lb + "\" got \"" + la + "\"";
                break;
            }
            if (ea == std::string::npos || eb == std::string::npos) {
                *diff = "config length/fields differ";
                break;
            }
            ia = ea + 1;
            ib = eb + 1;
        }
    }
    return false;
}

// ---------------------------------------------------------------- lifecycle

Trainer::~Trainer() {
    if (log_file_.is_open()) {
        log_file_.flush();
    }
    if (csv_file_.is_open()) {
        csv_file_.flush();
    }
    if (sched_g_ != nullptr) {
        traincpp::lr_scheduler_free(sched_g_);
        sched_g_ = nullptr;
    }
    if (sched_d_ != nullptr) {
        traincpp::lr_scheduler_free(sched_d_);
        sched_d_ = nullptr;
    }
    ts_.reset();
}

bool Trainer::fail(std::string* err, const std::string& msg) const {
    if (err != nullptr) {
        *err = msg;
    }
    return false;
}

Trainer* Trainer::create(const TrainerConfig& cfg, traincpp::Device* dev, std::string* err) {
    if (dev == nullptr) {
        if (err != nullptr) {
            *err = "Trainer::create: device is null";
        }
        return nullptr;
    }
    if (cfg.exp_dir.empty() || cfg.weights_g.empty() || cfg.weights_d.empty() ||
        cfg.out_dir.empty()) {
        if (err != nullptr) {
            *err = "Trainer::create: exp_dir/weights_g/weights_d/out_dir required";
        }
        return nullptr;
    }
    if (cfg.batch_size <= 0) {
        if (err != nullptr) {
            *err = "Trainer::create: invalid batch_size (must be >= 1)";
        }
        return nullptr;
    }
    std::unique_ptr<Trainer> t(new Trainer());
    t->cfg_ = cfg;
    t->dev_ = dev;

    data::TrainDataConfig dc;
    dc.exp_dir         = cfg.exp_dir;
    dc.batch_size      = cfg.batch_size;
    dc.segment_frames  = cfg.segment_frames;
    dc.segment_samples = cfg.segment_samples;
    dc.q_channels      = cfg.q_channels;
    dc.speaker_id      = 0;
    dc.buckets         = cfg.buckets;
    dc.n_fft           = cfg.n_fft;
    dc.win             = cfg.win;
    dc.n_mels          = cfg.n_mels;
    dc.sr              = cfg.sr;
    dc.fmin            = cfg.fmin;
    dc.fmax            = cfg.fmax;
    dc.log_clip        = cfg.log_clip;
    if (!data::Dataset::open(cfg.exp_dir, dc, &t->ds_, err)) {
        return nullptr;
    }

    // Version-consistency precheck (V2-8): the G weight feature dim must match the cache feature dim; otherwise
    // train_step fails with the cryptic "batch field phone element count mismatch (got != want)"; the cause is reported up front here (v1=256 / v2=768).
    {
        std::string werr;
        model::TrainWeights* gw = model::TrainWeights::open(cfg.weights_g, &werr);
        if (gw == nullptr) {
            t->fail(err, "Trainer::create: failed to open G weights: " + werr);
            return nullptr;
        }
        model::TrainMeta gmeta;
        gw->read_meta(&gmeta);
        delete gw;
        const model::ModelConfig gm = model::ModelConfig::from_meta(gmeta);

        std::string derr;
        model::TrainWeights* dw = model::TrainWeights::open(cfg.weights_d, &derr);
        if (dw == nullptr) {
            t->fail(err, "Trainer::create: failed to open D weights: " + derr);
            return nullptr;
        }
        model::TrainMeta dmeta;
        dw->read_meta(&dmeta);
        delete dw;
        const model::ModelConfig dm = model::ModelConfig::from_meta(dmeta);

        if (gm.version != dm.version) {
            t->fail(err, "G weights version (" + gm.version + ") and D weights version (" + dm.version +
                              ") differ; use G/D from the same version (v1 or v2).");
            return nullptr;
        }
        const int64_t want_feat = gm.is_v2() ? 768 : 256;
        int64_t cache_feat = 0;
        if (!t->ds_.cache().samples.empty()) {
            cache_feat = t->ds_.cache().samples[0].feat_dim;
        }
        if (cache_feat != 0 && cache_feat != want_feat) {
            t->fail(err, "model and training cache do not match: G weights are " + gm.version + " (feature dim " +
                              std::to_string(want_feat) + "), but the cache feature dim is " +
                              std::to_string(cache_feat) + " (cache version " +
                              t->ds_.cache().meta.version +
                              "). Use weights and config consistent with the cache version (v1=256 / v2=768).");
            return nullptr;
        }
    }

    int64_t bucket = cfg.bucket;
    if (bucket <= 0) {
        for (int64_t b : cfg.buckets) {
            bucket = std::max(bucket, b);
        }
    }
    if (bucket < cfg.segment_frames) {
        if (err != nullptr) {
            *err = "Trainer::create: invalid bucket T (must be >= segment_frames; specify it in config/CLI)";
        }
        return nullptr;
    }
    t->bucket_ = bucket;
    t->skipped_samples_ = 0;
    for (int64_t i = 0; i < t->ds_.size(); ++i) {
        const int64_t frames = t->ds_.aligned_frames(i);
        if (frames < cfg.segment_frames || frames > bucket) {
            ++t->skipped_samples_;
            continue;
        }
        t->sample_indices_.push_back(i);
    }
    if (t->sample_indices_.empty()) {
        if (err != nullptr) {
            *err = "Trainer::create: no usable samples within the bucket (try a larger --bucket)";
        }
        return nullptr;
    }
    // M7-3: one optimizer step = K micro-batches; steps_per_epoch counts optimizer steps (a trailing incomplete group is dropped)
    const int64_t accum = std::max<int64_t>(1, cfg.grad_accum);
    t->steps_per_epoch_ = ((int64_t) t->sample_indices_.size() / cfg.batch_size) / accum;
    if (t->steps_per_epoch_ <= 0) {
        if (err != nullptr) {
            *err = "Trainer::create: not enough trainable samples for one optimizer step (batch×grad_accum)";
        }
        return nullptr;
    }

    TrainStepConfig tc;
    tc.frames       = bucket;
    tc.batch        = cfg.batch_size;
    tc.grad_accum   = accum;
    tc.disc_len     = cfg.disc_len;
    tc.lr           = cfg.lr;
    tc.beta1        = cfg.beta1;
    tc.beta2        = cfg.beta2;
    tc.eps          = cfg.eps;
    tc.weight_decay = cfg.weight_decay;
    tc.c_mel        = cfg.c_mel;
    tc.c_kl         = cfg.c_kl;
    t->ts_.reset(TrainStep::create(cfg.weights_g, cfg.weights_d, tc, dev, err));
    if (t->ts_ == nullptr) {
        return nullptr;
    }

    t->sched_g_ = traincpp::lr_scheduler_exponential_new(t->ts_->optim_g(), cfg.lr_decay);
    t->sched_d_ = traincpp::lr_scheduler_exponential_new(t->ts_->optim_d(), cfg.lr_decay);
    if (t->sched_g_ == nullptr || t->sched_d_ == nullptr) {
        if (err != nullptr) {
            *err = "Trainer::create: failed to create the ExponentialLR scheduler";
        }
        return nullptr;
    }

    traincpp::rng_seed(&t->rng_, (uint64_t) cfg.seed);

    t->ckpt_path_ = join_path(cfg.out_dir, "train.ckpt.gguf");
    t->log_path_  = join_path(cfg.out_dir, "train.log");
    t->csv_path_  = join_path(cfg.out_dir, "loss.csv");
    t->epochs_csv_path_ = join_path(cfg.out_dir, "epochs.csv");

    if (!cfg.resume.empty()) {
        if (!t->load_checkpoint(cfg.resume, err)) {
            return nullptr;
        }
    } else {
        t->epoch_ = 0;
        t->batch_pos_ = 0;
        if (cfg.epochs <= 0) {
            t->finished_ = true;
        }
    }
    if (!t->open_logs(err)) {
        return nullptr;
    }
    return t.release();
}

// ---------------------------------------------------------------- loop

bool Trainer::rebuild_epoch_batches(int64_t epoch, std::string* err) {
    std::vector<int64_t> order = sample_indices_;
    traincpp::Rng shuffle_rng;
    traincpp::rng_seed(&shuffle_rng, (uint64_t) cfg_.seed + (uint64_t) epoch);
    for (size_t i = order.size(); i > 1; --i) {
        const uint32_t j = traincpp::rng_next_u32(&shuffle_rng) % (uint32_t) i;
        std::swap(order[i - 1], order[(size_t) j]);
    }
    const int64_t n_batches = (int64_t) order.size() / cfg_.batch_size;
    epoch_batches_.assign((size_t) n_batches, {});
    for (int64_t k = 0; k < n_batches; ++k) {
        const auto begin = order.begin() + (size_t) (k * cfg_.batch_size);
        epoch_batches_[(size_t) k].assign(begin, begin + cfg_.batch_size);
    }
    const int64_t dropped = (int64_t) order.size() - n_batches * cfg_.batch_size;
    if (dropped > 0) {
        std::printf("[trainer] epoch %lld: trailing %lld samples do not fill a batch, dropped per the M5 simplification\n",
                    (long long) epoch, (long long) dropped);
    }
    epoch_batches_epoch_ = epoch;
    epoch_batches_valid_ = true;
    (void) err;
    return true;
}

bool Trainer::build_batch_into(const std::vector<int64_t>& indices, data::BatchBuffers* out,
                               std::string* err) {
    if (!ds_.build_batch(indices, bucket_, &rng_, nullptr, out, err)) {
        return false;
    }
    // Test scale: truncate the D input to disc_len (the first disc_len samples per sample; ne [L,1,B] is contiguous per sample).
    if (cfg_.disc_len > 0 && cfg_.disc_len < cfg_.segment_samples) {
        const int64_t d = cfg_.disc_len;
        const int64_t l = cfg_.segment_samples;
        const int64_t b = out->batch;
        std::vector<float> wave((size_t) (d * b));
        for (int64_t i = 0; i < b; ++i) {
            std::memcpy(&wave[(size_t) (i * d)], &out->wave[(size_t) (i * l)],
                        (size_t) d * sizeof(float));
        }
        out->wave.swap(wave);
    }
    return true;
}

bool Trainer::run_one_step(std::string* err) {
    if (finished_) {
        return true;
    }
    const int64_t done = traincpp::optim_step_count(ts_->optim_g());
    if (cfg_.max_steps > 0 && done >= cfg_.max_steps) {
        finished_ = true;
        return true;
    }
    if (epoch_ >= cfg_.epochs) {
        finished_ = true;
        return true;
    }
    if (!epoch_batches_valid_ || epoch_batches_epoch_ != epoch_) {
        if (!rebuild_epoch_batches(epoch_, err)) {
            return false;
        }
    }
    // M7-3: one optimizer step consumes K=grad_accum consecutive micro-batches; batch_pos_ counts optimizer steps
    const int64_t accum = std::max<int64_t>(1, cfg_.grad_accum);
    const int64_t n_batches = (int64_t) epoch_batches_.size();
    const int64_t n_groups = n_batches / accum;  // a trailing incomplete group is dropped
    if (n_groups <= 0) {
        finished_ = true;
        return true;
    }
    if (batch_pos_ < 0 || batch_pos_ >= n_groups) {
        return fail(err, "Trainer: optimizer step position out of range (internal state error)");
    }
    const int64_t start = batch_pos_ * accum;

    std::vector<data::BatchBuffers> batches((size_t) accum);
    for (int64_t k = 0; k < accum; ++k) {
        if (!build_batch_into(epoch_batches_[(size_t) (start + k)], &batches[(size_t) k], err)) {
            return false;
        }
    }
    const auto t0 = std::chrono::steady_clock::now();
    if (!ts_->run_step_accum(batches.data(), accum, &last_, err)) {
        return false;
    }
    const auto t1 = std::chrono::steady_clock::now();
    const double dt_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    if (timed_steps_ == 0) {
        first_step_ms_ = dt_ms;
    }
    total_step_ms_ += dt_ms;
    ++timed_steps_;
    if ((int64_t) first_losses_.size() < kFirstLossCount) {
        first_losses_.push_back(last_);
    }
    ++batch_pos_;
    const int64_t step = traincpp::optim_step_count(ts_->optim_g());
    const bool to_console =
        (cfg_.log_interval > 0) && (step == 1 || (step % cfg_.log_interval) == 0);
    log_step(last_, to_console);

    if (batch_pos_ >= n_groups) {
        // Official: ExponentialLR steps once per epoch (train.py:283-288)
        traincpp::lr_scheduler_step(sched_g_);
        traincpp::lr_scheduler_step(sched_d_);
        ++epoch_;
        batch_pos_ = 0;
        epoch_batches_valid_ = false;
        if (!on_epoch_end(err)) {
            return false;
        }
    }
    if (cfg_.max_steps > 0 && step >= cfg_.max_steps) {
        finished_ = true;
    }
    if (epoch_ >= cfg_.epochs) {
        finished_ = true;
    }
    return true;
}

bool Trainer::run(std::string* err) {
    while (!finished_) {
        if (!run_one_step(err)) {
            return false;
        }
    }
    if (step_count() > 0) {
        if (!save_checkpoint(ckpt_path_, err)) {
            return false;
        }
    }
    if (!cfg_.export_run_json.empty()) {
        write_run_json();
    }
    const TrainStepMemory m = ts_->memory();
    std::printf("[trainer] done: steps=%lld epochs=%lld lr_g=%.3e lr_d=%.3e "
                "loss_disc=%.6f loss_gen_all=%.6f loss_gen=%.6f loss_fm=%.6f loss_mel=%.6f "
                "loss_kl=%.6f\n",
                (long long) step_count(), (long long) epoch_, (double) lr_g(), (double) lr_d(),
                (double) last_.loss_disc, (double) last_.loss_gen_all,
                (double) last_.loss_gen, (double) last_.loss_fm, (double) last_.loss_mel,
                (double) last_.loss_kl);
    std::printf("[trainer] buf_g=%.1f MiB buf_d=%.1f MiB; log %s; checkpoint %s\n",
                (double) m.buf_g_bytes / (1024.0 * 1024.0),
                (double) m.buf_d_bytes / (1024.0 * 1024.0), log_path_.c_str(),
                ckpt_path_.c_str());
    std::printf("[trainer] step time: first=%.1f ms avg=%.1f ms (%lld steps sampled)\n", first_step_ms_,
                mean_step_ms(), (long long) timed_steps_);
    return true;
}

// ---------------------------------------------------------------- logging

bool Trainer::open_logs(std::string* err) {
    std::error_code ec;
    std::filesystem::create_directories(cfg_.out_dir, ec);
    if (ec) {
        return fail(err, "failed to create output directory: " + cfg_.out_dir + " (" + ec.message() + ")");
    }
    log_file_.open(log_path_, std::ios::app);
    if (!log_file_.is_open()) {
        return fail(err, "cannot write train.log: " + log_path_);
    }
    csv_file_.open(csv_path_, std::ios::app);
    if (!csv_file_.is_open()) {
        return fail(err, "cannot write loss.csv: " + csv_path_);
    }
    std::error_code se;
    const auto size = std::filesystem::file_size(csv_path_, se);
    if (se || size == 0) {
        csv_file_ << "step,epoch,lr_g,lr_d,loss_disc,loss_gen_all,loss_gen,loss_fm,loss_mel,loss_kl\n";
        csv_file_.flush();
    }
    epochs_file_.open(epochs_csv_path_, std::ios::app);
    if (!epochs_file_.is_open()) {
        return fail(err, "cannot write epochs.csv: " + epochs_csv_path_);
    }
    std::error_code se2;
    const auto esize = std::filesystem::file_size(epochs_csv_path_, se2);
    if (se2 || esize == 0) {
        epochs_file_ << "epoch,step,lr_g,lr_d,loss_disc,loss_gen_all,loss_gen,loss_fm,loss_mel,loss_kl\n";
        epochs_file_.flush();
    }
    log_file_ << "[trainer] startup: bucket=" << bucket_ << " batch=" << cfg_.batch_size
              << " samples=" << sample_indices_.size() << " skipped=" << skipped_samples_
              << " steps/epoch=" << steps_per_epoch_ << " epochs=" << cfg_.epochs
              << " max_steps=" << cfg_.max_steps << " seed=" << cfg_.seed << "\n";
    log_file_.flush();
    // Also print the startup summary to the console (including steps/epoch and the total-step estimate, to directly gauge the training scale)
    const long long total_steps =
        cfg_.max_steps > 0 ? (long long) cfg_.max_steps
                           : (long long) cfg_.epochs * (long long) steps_per_epoch_;
    std::printf("[trainer] startup: bucket=%lld batch=%lld samples=%lld skipped=%lld "
                "steps/epoch=%lld epochs=%lld max_steps=%lld seed=%lld (est. total optimization steps ~=%lld)\n",
                (long long) bucket_, (long long) cfg_.batch_size,
                (long long) sample_indices_.size(), (long long) skipped_samples_,
                (long long) steps_per_epoch_, (long long) cfg_.epochs, (long long) cfg_.max_steps,
                (long long) cfg_.seed, total_steps);
    return true;
}

void Trainer::log_step(const TrainStepLosses& L, bool to_console) {
    const int64_t step = step_count();
    if (csv_file_.is_open()) {
        csv_file_ << step << ',' << epoch_ << ',' << lr_g() << ',' << lr_d() << ',' << L.loss_disc
                  << ',' << L.loss_gen_all << ',' << L.loss_gen << ',' << L.loss_fm << ','
                  << L.loss_mel << ',' << L.loss_kl << '\n';
        csv_file_.flush();
    }
    if (!to_console || step == last_log_step_) {
        return;
    }
    last_log_step_ = step;
    if (log_file_.is_open()) {
        log_file_ << "[trainer] step " << step << " epoch " << epoch_ << " lr_g " << lr_g()
                  << " lr_d " << lr_d() << " loss_disc " << L.loss_disc << " loss_gen_all "
                  << L.loss_gen_all << " loss_gen " << L.loss_gen << " loss_fm " << L.loss_fm
                  << " loss_mel " << L.loss_mel << " loss_kl " << L.loss_kl << "\n";
        log_file_.flush();
    }
    std::printf("[trainer] step %lld epoch %lld lr_g=%.3e lr_d=%.3e loss_disc=%.6f "
                "loss_gen_all=%.6f loss_gen=%.6f loss_fm=%.6f loss_mel=%.6f loss_kl=%.6f\n",
                (long long) step, (long long) epoch_, (double) lr_g(), (double) lr_d(),
                (double) L.loss_disc, (double) L.loss_gen_all, (double) L.loss_gen,
                (double) L.loss_fm, (double) L.loss_mel, (double) L.loss_kl);
}

void Trainer::write_run_json() const {
    std::ofstream f(cfg_.export_run_json, std::ios::trunc);
    if (!f.is_open()) {
        std::printf("[trainer] warning: cannot write run JSON: %s\n", cfg_.export_run_json.c_str());
        return;
    }
    const TrainStepMemory m = ts_->memory();
    const traincpp::ContextMemoryStats cm = traincpp::context_memory_stats(ts_->context());
    const traincpp::AllocStats as = traincpp::backend_alloc_stats();
    f << "{\n";
    f << "  \"device\": \"" << json_escape(dev_->props().name) << "\",\n";
    f << "  \"bucket\": " << bucket_ << ",\n";
    f << "  \"batch\": " << cfg_.batch_size << ",\n";
    f << "  \"disc_len\": " << (cfg_.disc_len > 0 ? cfg_.disc_len : cfg_.segment_samples) << ",\n";
    f << "  \"steps\": " << step_count() << ",\n";
    f << "  \"epochs\": " << epoch_ << ",\n";
    f << "  \"samples\": " << sample_indices_.size() << ",\n";
    f << "  \"skipped_samples\": " << skipped_samples_ << ",\n";
    f << "  \"lr_g\": " << lr_g() << ",\n";
    f << "  \"lr_d\": " << lr_d() << ",\n";
    f << "  \"step_time_first_ms\": " << first_step_ms_ << ",\n";
    f << "  \"step_time_mean_ms\": " << mean_step_ms() << ",\n";
    f << "  \"timed_steps\": " << timed_steps_ << ",\n";
    f << "  \"buf_g_mib\": " << (double) m.buf_g_bytes / (1024.0 * 1024.0) << ",\n";
    f << "  \"buf_d_mib\": " << (double) m.buf_d_bytes / (1024.0 * 1024.0) << ",\n";
    f << "  \"ctx_tensor_mib\": " << (double) cm.tensor_bytes / (1024.0 * 1024.0) << ",\n";
    f << "  \"ctx_persistent_mib\": " << (double) cm.persistent_bytes / (1024.0 * 1024.0) << ",\n";
    f << "  \"backend_peak_mib\": " << (double) as.peak_bytes / (1024.0 * 1024.0) << ",\n";
    f << "  \"loss\": {\"loss_disc\": " << last_.loss_disc << ", \"loss_gen_all\": "
      << last_.loss_gen_all << ", \"loss_gen\": " << last_.loss_gen << ", \"loss_fm\": "
      << last_.loss_fm << ", \"loss_mel\": " << last_.loss_mel << ", \"loss_kl\": "
      << last_.loss_kl << "},\n";
    f << "  \"first_losses\": [";
    for (size_t i = 0; i < first_losses_.size(); ++i) {
        const TrainStepLosses& L = first_losses_[i];
        if (i > 0) {
            f << ", ";
        }
        f << "{\"loss_disc\": " << L.loss_disc << ", \"loss_gen_all\": " << L.loss_gen_all
          << ", \"loss_gen\": " << L.loss_gen << ", \"loss_fm\": " << L.loss_fm
          << ", \"loss_mel\": " << L.loss_mel << ", \"loss_kl\": " << L.loss_kl << "}";
    }
    f << "],\n";
    f << "  \"checkpoint\": \"" << json_escape(ckpt_path_) << "\"\n";
    f << "}\n";
    std::printf("[trainer] run summary written: %s\n", cfg_.export_run_json.c_str());
}

// ---------------------------------------------------------------- M7-4 epoch boundary

bool Trainer::on_epoch_end(std::string* err) {
    write_epoch_summary();
    const int64_t ep = epoch_;
    if (cfg_.save_interval > 0 && (ep % cfg_.save_interval) == 0) {
        if (!save_epoch_snapshot(ep, err)) {
            return false;
        }
        if (!prune_snapshots(err)) {
            return false;
        }
    }
    if (cfg_.export_every > 0 && (ep % cfg_.export_every) == 0) {
        if (!export_epoch_inference(ep, err)) {
            return false;
        }
    }
    return true;
}

void Trainer::write_epoch_summary() {
    const TrainStepLosses& L = last_;
    if (epochs_file_.is_open()) {
        epochs_file_ << epoch_ << ',' << step_count() << ',' << lr_g() << ',' << lr_d() << ','
                     << L.loss_disc << ',' << L.loss_gen_all << ',' << L.loss_gen << ','
                     << L.loss_fm << ',' << L.loss_mel << ',' << L.loss_kl << '\n';
        epochs_file_.flush();
    }
    if (log_file_.is_open()) {
        log_file_ << "[trainer] epoch " << epoch_ << " summary step " << step_count() << " lr_g "
                  << lr_g() << " lr_d " << lr_d() << " loss_disc " << L.loss_disc
                  << " loss_gen_all " << L.loss_gen_all << " loss_gen " << L.loss_gen
                  << " loss_fm " << L.loss_fm << " loss_mel " << L.loss_mel << " loss_kl "
                  << L.loss_kl << "\n";
        log_file_.flush();
    }
    std::printf("[trainer] epoch %lld summary step=%lld lr_g=%.3e loss_gen_all=%.6f\n",
                (long long) epoch_, (long long) step_count(), (double) lr_g(),
                (double) L.loss_gen_all);
}

bool Trainer::save_epoch_snapshot(int64_t epoch, std::string* err) {
    const std::string path =
        join_path(cfg_.out_dir, "train.ep" + std::to_string(epoch) + ".ckpt.gguf");
    return save_checkpoint(path, err);
}

// Parse train.ep{N}.ckpt.gguf under out_dir and keep only the newest snapshot_keep of them (removing the
// same-named snapshot/export files together); snapshot_keep<=0 means keep all.
bool Trainer::prune_snapshots(std::string* err) {
    if (cfg_.snapshot_keep <= 0) {
        return true;
    }
    namespace fs = std::filesystem;
    const std::string pre = "train.ep";
    const std::string suf = ".ckpt.gguf";
    std::vector<std::pair<int64_t, std::string>> snaps;  // (epoch, N string)
    std::error_code ec;
    for (const auto& entry : fs::directory_iterator(cfg_.out_dir, ec)) {
        if (ec || !entry.is_regular_file()) {
            continue;
        }
        const std::string fn = entry.path().filename().string();
        if (fn.size() <= pre.size() + suf.size() || fn.rfind(pre, 0) != 0 ||
            fn.compare(fn.size() - suf.size(), suf.size(), suf) != 0) {
            continue;
        }
        const std::string mid = fn.substr(pre.size(), fn.size() - pre.size() - suf.size());
        try {
            snaps.emplace_back(std::stoll(mid), mid);
        } catch (...) {
        }
    }
    if ((int64_t) snaps.size() <= cfg_.snapshot_keep) {
        return true;
    }
    std::sort(snaps.begin(), snaps.end());
    const int64_t remove = (int64_t) snaps.size() - cfg_.snapshot_keep;
    for (int64_t i = 0; i < remove; ++i) {
        const std::string& n = snaps[(size_t) i].second;
        std::error_code re;
        fs::remove(join_path(cfg_.out_dir, "train.ep" + n + ".ckpt.gguf"), re);
        fs::remove(join_path(cfg_.out_dir, "train.ep" + n + ".snapshot.gguf"), re);
        fs::remove(join_path(cfg_.out_dir, "export.ep" + n + ".gguf"), re);
    }
    (void) err;
    return true;
}

// Live G parameters -> training-state GGUF snapshot -> inference GGUF (reuses the M6 exporter)
bool Trainer::export_epoch_inference(int64_t epoch, std::string* err) {
    std::string e;
    model::TrainWeights* wg = model::TrainWeights::open(cfg_.weights_g, &e);
    if (wg == nullptr) {
        return fail(err, "export inference GGUF: failed to open G weights: " + e);
    }
    model::TrainMeta mg;
    const bool meta_ok = wg->read_meta(&mg);
    delete wg;
    if (!meta_ok) {
        return fail(err, "export inference GGUF: failed to read G metadata");
    }
    const model::ModelConfig gcfg = model::ModelConfig::from_meta(mg);
    const std::string n = std::to_string(epoch);
    const std::string snap = join_path(cfg_.out_dir, "train.ep" + n + ".snapshot.gguf");
    const std::string out = join_path(cfg_.out_dir, "export.ep" + n + ".gguf");
    if (!exporter::export_train_gguf_live(gcfg, ts_->g_params(), snap,
                                          "rvc-train@epoch" + n, cfg_.weights_g, &e)) {
        return fail(err, "export inference GGUF: failed to write the live training-state GGUF: " + e);
    }
    exporter::ModelExportOptions opt;
    opt.in_path = snap;
    opt.out_path = out;
    // When the live snapshot is written the trailing 1-dim is collapsed (the real rank is lost); use the
    // ndims==3 list of the source training-state GGUF to force flattening (fixes O=1 Conv1d weights, e.g. dec.conv_post.weight).
    {
        std::unordered_map<std::string, int> ndims;
        if (exporter::gguf_read_tensor_ndims(cfg_.weights_g, &ndims, nullptr)) {
            for (const auto& kv : ndims) {
                if (kv.second == 3) {
                    opt.force_flatten.push_back(kv.first);
                }
            }
        }
    }
    if (!exporter::export_inference_gguf(opt, &e)) {
        return fail(err, "export inference GGUF: export failed: " + e);
    }
    std::printf("[trainer] epoch %lld exported inference GGUF: %s (training-state snapshot %s)\n", (long long) epoch,
                out.c_str(), snap.c_str());
    return true;
}

// ---------------------------------------------------------------- M7-5 resume metadata

void Trainer::compute_sample_fingerprint(int64_t* count, int64_t* total_frames,
                                         uint64_t* id_hash) const {
    const int64_t n = ds_.size();
    int64_t tf = 0;
    std::string ids;
    ids.reserve((size_t) (n > 0 ? n : 0) * 16);
    for (int64_t i = 0; i < n; ++i) {
        tf += ds_.aligned_frames(i);
        ids += ds_.sample_info(i).id;
        ids += '\n';
    }
    if (count != nullptr) {
        *count = n;
    }
    if (total_frames != nullptr) {
        *total_frames = tf;
    }
    if (id_hash != nullptr) {
        *id_hash = resume_fnv1a64(ids);
    }
}

ResumeMeta Trainer::build_resume_meta() const {
    ResumeMeta m;
    m.version         = 1;
    m.epoch           = epoch_;
    m.batch_pos       = batch_pos_;
    m.step            = step_count();
    m.grad_accum      = std::max<int64_t>(1, cfg_.grad_accum);
    m.config_signature = trainer_config_signature(cfg_);
    std::string joined;
    for (const std::string& line : m.config_signature) {
        joined += line;
        joined += '\n';
    }
    m.config_hash = resume_fnv1a64(joined);
    compute_sample_fingerprint(&m.sample_count, &m.total_frames, &m.id_hash);
    return m;
}

bool Trainer::validate_resume_meta(const ResumeMeta& m, std::string* err) const {
    // 1) training-related config signature
    const std::vector<std::string> cur = trainer_config_signature(cfg_);
    if (cur != m.config_signature) {
        std::string diff;
        const size_t n = std::min(cur.size(), m.config_signature.size());
        for (size_t i = 0; i < n; ++i) {
            if (cur[i] != m.config_signature[i]) {
                diff = "archive \"" + m.config_signature[i] + "\" / current \"" + cur[i] + "\"";
                break;
            }
        }
        if (diff.empty()) {
            diff = "field count mismatch (archive " + std::to_string(m.config_signature.size()) +
                   " / current " + std::to_string(cur.size()) + ")";
        }
        return fail(err, "resume validation failed: training config does not match the checkpoint (" + diff +
                             "). To change the config, start training anew; do not resume from this checkpoint.");
    }
    // 2) sample-set fingerprint (sample count + total frames + id list hash)
    int64_t sc = 0;
    int64_t tf = 0;
    uint64_t ih = 0;
    compute_sample_fingerprint(&sc, &tf, &ih);
    if (sc != m.sample_count || tf != m.total_frames || ih != m.id_hash) {
        return fail(err, "resume validation failed: sample set does not match the checkpoint (archive count/frames/id_hash=" +
                             std::to_string((long long) m.sample_count) + "/" +
                             std::to_string((long long) m.total_frames) + "/" +
                             std::to_string((unsigned long long) m.id_hash) + ", current " +
                             std::to_string((long long) sc) + "/" + std::to_string((long long) tf) +
                             "/" + std::to_string((unsigned long long) ih) + ")");
    }
    // 3) cross-check the optimizer step against the checkpoint
    const int64_t step = step_count();
    if (m.step != step) {
        return fail(err, "resume validation failed: checkpoint step=" + std::to_string((long long) step) +
                             " vs metadata step=" + std::to_string((long long) m.step) + " mismatch");
    }
    // 4) epoch/batch_pos consistent with step (steps_per_epoch_ has been computed from the current config)
    if (m.epoch < 0 || m.batch_pos < 0 || steps_per_epoch_ <= 0 || m.batch_pos >= steps_per_epoch_ ||
        m.epoch * steps_per_epoch_ + m.batch_pos != m.step) {
        return fail(err, "resume validation failed: metadata epoch/batch_pos inconsistent with step (epoch=" +
                             std::to_string((long long) m.epoch) + " batch_pos=" +
                             std::to_string((long long) m.batch_pos) + " step=" +
                             std::to_string((long long) m.step) + " steps_per_epoch=" +
                             std::to_string((long long) steps_per_epoch_) + ")");
    }
    return true;
}

bool Trainer::write_resume_meta(const std::string& ckpt_path, std::string* err) {
    const ResumeMeta m = build_resume_meta();
    const std::string mp = resume_meta_path_for(ckpt_path);
    if (!resume_meta_write(m, mp, err)) {
        return false;
    }
    std::printf("[trainer] resume metadata written: %s (epoch=%lld batch_pos=%lld step=%lld "
                "config_hash=%016llx)\n",
                mp.c_str(), (long long) m.epoch, (long long) m.batch_pos, (long long) m.step,
                (unsigned long long) m.config_hash);
    return true;
}

// ---------------------------------------------------------------- checkpoint

bool Trainer::save_checkpoint(const std::string& path, std::string* err) {
    traincpp::CheckpointOptimizer parts[2] = {
        {"G", ts_->optim_g(), sched_g_},
        {"D", ts_->optim_d(), sched_d_},
    };
    traincpp::CheckpointItems items{};
    items.rng              = &rng_;
    items.optimizers       = parts;
    items.optimizer_count  = 2;
    if (!traincpp::checkpoint_save(path.c_str(), items, "rvc-train")) {
        return fail(err, "checkpoint_save failed: " + path);
    }
    if (!write_resume_meta(path, err)) {
        return false;
    }
    std::printf("[trainer] checkpoint saved: %s (step=%lld epoch=%lld lr_g=%.3e)\n",
                path.c_str(), (long long) step_count(), (long long) epoch_,
                (double) lr_g());
    return true;
}

bool Trainer::load_checkpoint(const std::string& path, std::string* err) {
    traincpp::optim_alloc_state(ts_->optim_g());
    traincpp::optim_alloc_state(ts_->optim_d());
    traincpp::CheckpointOptimizer parts[2] = {
        {"G", ts_->optim_g(), sched_g_},
        {"D", ts_->optim_d(), sched_d_},
    };
    traincpp::CheckpointItems items{};
    items.rng             = &rng_;
    items.optimizers      = parts;
    items.optimizer_count = 2;
    if (!traincpp::checkpoint_load(path.c_str(), items)) {
        return fail(err, "checkpoint_load failed: " + path);
    }
    // M7-5: read and validate the sidecar resume metadata (epoch/batch_pos/config fingerprint/sample fingerprint)
    ResumeMeta meta;
    const std::string meta_path = resume_meta_path_for(path);
    std::string rerr;
    if (!resume_meta_read(meta_path, &meta, &rerr)) {
        return fail(err, "resume validation failed: cannot read resume metadata " + meta_path + " (" + rerr +
                             "). The checkpoint must be accompanied by the same-named .meta.json; do not resume if it is missing/corrupt.");
    }
    if (!validate_resume_meta(meta, err)) {
        return false;
    }
    epoch_     = meta.epoch;
    batch_pos_ = meta.batch_pos;
    epoch_batches_valid_ = false;
    finished_ = false;
    const int64_t step = step_count();
    if (cfg_.epochs > 0 && epoch_ >= cfg_.epochs) {
        finished_ = true;
    }
    if (cfg_.max_steps > 0 && step >= cfg_.max_steps) {
        finished_ = true;
    }
    std::printf("[trainer] checkpoint loaded: %s (step=%lld epoch=%lld lr_g=%.3e rng=0x%016llx)\n",
                path.c_str(), (long long) step, (long long) epoch_, (double) lr_g(),
                (unsigned long long) rng_.state);
    return true;
}

bool Trainer::verify_checkpoint_memory(const std::string& path, std::string* err) {
    traincpp::CheckpointOptimizer parts[2] = {
        {"G", ts_->optim_g(), sched_g_},
        {"D", ts_->optim_d(), sched_d_},
    };
    traincpp::CheckpointItems items{};
    items.rng             = &rng_;
    items.optimizers      = parts;
    items.optimizer_count = 2;
    if (!traincpp::checkpoint_verify(path.c_str(), items)) {
        return fail(err, "checkpoint_verify failed: " + path);
    }
    return true;
}

// ---------------------------------------------------------------- accessors

int64_t Trainer::step_count() const {
    return ts_ ? traincpp::optim_step_count(ts_->optim_g()) : 0;
}

float Trainer::lr_g() const {
    return ts_ ? traincpp::optim_get_lr(ts_->optim_g()) : 0.0f;
}

float Trainer::lr_d() const {
    return ts_ ? traincpp::optim_get_lr(ts_->optim_d()) : 0.0f;
}

} // namespace train
} // namespace rvc_train
