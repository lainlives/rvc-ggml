// RVC-Train-GGML - M7-5 续训元数据实现（src/train/resume_meta.cpp）
#include "train/resume_meta.h"

#include "common/json.h"
#include "train/trainer.h"

#include <cstdio>
#include <fstream>
#include <limits>

namespace rvc_train {
namespace train {

namespace {

void push_f(std::vector<std::string>* v, const char* key, float x) {
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%.*g", std::numeric_limits<float>::max_digits10, (double) x);
    v->push_back(std::string(key) + "=" + buf);
}

void push_d(std::vector<std::string>* v, const char* key, double x) {
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%.*g", std::numeric_limits<double>::max_digits10, x);
    v->push_back(std::string(key) + "=" + buf);
}

void push_i(std::vector<std::string>* v, const char* key, int64_t x) {
    v->push_back(std::string(key) + "=" + std::to_string((long long) x));
}

template <typename T>
void push_i_arr(std::vector<std::string>* v, const char* key, const std::vector<T>& a) {
    std::string s = std::string(key) + "=";
    for (size_t i = 0; i < a.size(); ++i) {
        if (i > 0) {
            s += ",";
        }
        s += std::to_string((long long) a[i]);
    }
    v->push_back(std::move(s));
}

std::string json_escape(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 8);
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

std::string hex64(uint64_t x) {
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%016llx", (unsigned long long) x);
    return buf;
}

bool parse_hex64(const std::string& s, uint64_t* out) {
    if (s.empty()) {
        return false;
    }
    uint64_t v = 0;
    for (char c : s) {
        int d;
        if (c >= '0' && c <= '9') {
            d = c - '0';
        } else if (c >= 'a' && c <= 'f') {
            d = c - 'a' + 10;
        } else if (c >= 'A' && c <= 'F') {
            d = c - 'A' + 10;
        } else {
            return false;
        }
        v = (v << 4) | (uint64_t) d;
    }
    *out = v;
    return true;
}

int64_t json_i(const common::JsonValue* obj, const char* key, int64_t def) {
    const common::JsonValue* v = obj != nullptr ? obj->find(key) : nullptr;
    return (v != nullptr && v->is_number()) ? v->as_int(def) : def;
}

std::string json_s(const common::JsonValue* obj, const char* key, const std::string& def) {
    const common::JsonValue* v = obj != nullptr ? obj->find(key) : nullptr;
    return (v != nullptr && v->is_string()) ? v->as_string() : def;
}

} // namespace

uint64_t resume_fnv1a64(const std::string& s) {
    uint64_t h = 1469598103934665603ull;  // offset basis
    for (unsigned char c : s) {
        h ^= (uint64_t) c;
        h *= 1099511628211ull;            // FNV prime
    }
    return h;
}

std::string resume_meta_path_for(const std::string& ckpt_path) {
    const std::string suf = ".gguf";
    if (ckpt_path.size() >= suf.size() &&
        ckpt_path.compare(ckpt_path.size() - suf.size(), suf.size(), suf) == 0) {
        return ckpt_path.substr(0, ckpt_path.size() - suf.size()) + ".meta.json";
    }
    return ckpt_path + ".meta.json";
}

std::vector<std::string> trainer_config_signature(const TrainerConfig& cfg) {
    std::vector<std::string> v;
    const model::ModelConfig& M = cfg.model_cfg;
    v.push_back("model.version=" + M.version);
    push_i(&v, "model.has_f0", M.has_f0 ? 1 : 0);
    push_i(&v, "model.inter_channels", M.inter_channels);
    push_i(&v, "model.hidden_channels", M.hidden_channels);
    push_i(&v, "model.filter_channels", M.filter_channels);
    push_i(&v, "model.n_heads", M.n_heads);
    push_i(&v, "model.n_layers", M.n_layers);
    push_i(&v, "model.kernel_size", M.kernel_size);
    push_f(&v, "model.p_dropout", M.p_dropout);
    push_i(&v, "model.gin_channels", M.gin_channels);
    push_i(&v, "model.spk_embed_dim", M.spk_embed_dim);
    push_i(&v, "model.upsample_initial_channel", M.upsample_initial_channel);
    v.push_back("model.resblock=" + M.resblock);
    push_i_arr(&v, "model.resblock_kernel_sizes", M.resblock_kernel_sizes);
    push_i_arr(&v, "model.resblock_dilation_sizes", M.resblock_dilation_sizes);
    push_i_arr(&v, "model.upsample_rates", M.upsample_rates);
    push_i_arr(&v, "model.upsample_kernel_sizes", M.upsample_kernel_sizes);
    push_i_arr(&v, "model.d_periods", M.d_periods);
    push_i(&v, "model.sample_rate", M.sample_rate);
    push_i(&v, "model.filter_length", M.filter_length);
    push_i(&v, "model.win_length", M.win_length);
    push_i(&v, "model.hop_length", M.hop_length);
    push_i(&v, "model.n_mel_channels", M.n_mel_channels);
    push_f(&v, "model.mel_fmin", M.mel_fmin);
    push_f(&v, "model.mel_fmax", M.mel_fmax);
    push_i(&v, "model.spec_channels", M.spec_channels);
    push_i(&v, "model.segment_size", M.segment_size);

    // 影响训练轨迹的数据/优化超参（run 段/out/device/resume/export/log_interval 不参与）
    push_i(&v, "train.batch_size", cfg.batch_size);
    push_i(&v, "train.bucket", cfg.bucket);
    push_i(&v, "train.disc_len", cfg.disc_len);
    push_i(&v, "train.segment_frames", cfg.segment_frames);
    push_i(&v, "train.segment_samples", cfg.segment_samples);
    push_i(&v, "train.q_channels", cfg.q_channels);
    push_i(&v, "train.grad_accum", cfg.grad_accum);
    push_i(&v, "train.seed", cfg.seed);
    push_f(&v, "train.learning_rate", cfg.lr);
    push_f(&v, "train.beta1", cfg.beta1);
    push_f(&v, "train.beta2", cfg.beta2);
    push_f(&v, "train.eps", cfg.eps);
    push_f(&v, "train.weight_decay", cfg.weight_decay);
    push_f(&v, "train.lr_decay", cfg.lr_decay);
    push_f(&v, "train.c_mel", cfg.c_mel);
    push_f(&v, "train.c_kl", cfg.c_kl);
    push_f(&v, "train.log_clip", cfg.log_clip);
    push_d(&v, "data.sampling_rate", cfg.sr);
    push_i(&v, "data.n_fft", cfg.n_fft);
    push_i(&v, "data.win", cfg.win);
    push_i(&v, "data.n_mels", cfg.n_mels);
    push_d(&v, "data.fmin", cfg.fmin);
    push_d(&v, "data.fmax", cfg.fmax);
    push_i_arr(&v, "data.buckets", cfg.buckets);
    return v;
}

bool resume_meta_write(const ResumeMeta& m, const std::string& path, std::string* err) {
    std::string joined;
    for (const std::string& line : m.config_signature) {
        joined += line;
        joined += "\n";
    }
    const uint64_t hash = resume_fnv1a64(joined);
    std::string s;
    s.reserve(1024);
    s += "{\n";
    s += "  \"version\": " + std::to_string(m.version) + ",\n";
    s += "  \"epoch\": " + std::to_string((long long) m.epoch) + ",\n";
    s += "  \"batch_pos\": " + std::to_string((long long) m.batch_pos) + ",\n";
    s += "  \"step\": " + std::to_string((long long) m.step) + ",\n";
    s += "  \"grad_accum\": " + std::to_string((long long) m.grad_accum) + ",\n";
    s += "  \"config_hash\": \"" + hex64(hash) + "\",\n";
    s += "  \"sample_count\": " + std::to_string((long long) m.sample_count) + ",\n";
    s += "  \"total_frames\": " + std::to_string((long long) m.total_frames) + ",\n";
    s += "  \"id_hash\": \"" + hex64(m.id_hash) + "\",\n";
    s += "  \"config_signature\": [";
    for (size_t i = 0; i < m.config_signature.size(); ++i) {
        if (i > 0) {
            s += ", ";
        }
        s += "\"" + json_escape(m.config_signature[i]) + "\"";
    }
    s += "]\n";
    s += "}\n";

    std::ofstream f(path, std::ios::trunc | std::ios::binary);
    if (!f.is_open()) {
        if (err != nullptr) {
            *err = "无法写入续训元数据: " + path;
        }
        return false;
    }
    f << s;
    if (!f.good()) {
        if (err != nullptr) {
            *err = "写入续训元数据失败: " + path;
        }
        return false;
    }
    return true;
}

bool resume_meta_read(const std::string& path, ResumeMeta* out, std::string* err) {
    if (out == nullptr) {
        if (err != nullptr) {
            *err = "resume_meta_read: out 为空";
        }
        return false;
    }
    common::JsonValue root;
    if (!common::json_read_file(path, &root, err)) {
        return false;
    }
    if (!root.is_object()) {
        if (err != nullptr) {
            *err = "续训元数据根不是 JSON 对象: " + path;
        }
        return false;
    }
    ResumeMeta m;
    m.version      = (int) json_i(&root, "version", 1);
    m.epoch        = json_i(&root, "epoch", 0);
    m.batch_pos    = json_i(&root, "batch_pos", 0);
    m.step         = json_i(&root, "step", 0);
    m.grad_accum   = json_i(&root, "grad_accum", 1);
    m.sample_count = json_i(&root, "sample_count", 0);
    m.total_frames = json_i(&root, "total_frames", 0);
    if (!parse_hex64(json_s(&root, "config_hash", ""), &m.config_hash)) {
        m.config_hash = 0;
    }
    if (!parse_hex64(json_s(&root, "id_hash", ""), &m.id_hash)) {
        m.id_hash = 0;
    }
    const common::JsonValue* sig = root.find("config_signature");
    if (sig == nullptr || !sig->is_array()) {
        if (err != nullptr) {
            *err = "续训元数据缺少 config_signature 数组: " + path;
        }
        return false;
    }
    m.config_signature.clear();
    for (size_t i = 0; i < sig->size(); ++i) {
        const common::JsonValue& e = sig->at(i);
        if (e.is_string()) {
            m.config_signature.push_back(e.as_string());
        }
    }
    *out = std::move(m);
    if (err != nullptr) {
        err->clear();
    }
    return true;
}

} // namespace train
} // namespace rvc_train
