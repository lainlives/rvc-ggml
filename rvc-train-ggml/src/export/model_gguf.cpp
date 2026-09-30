// model_gguf.cpp —— M6 推理 GGUF 导出实现（见 model_gguf.h）
#include "export/model_gguf.h"

#include "traincpp/traincpp.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

using namespace traincpp;

namespace rvc_train {
namespace exporter {

namespace {

bool starts_with(const std::string& s, const char* p) {
    const size_t n = std::strlen(p);
    return s.size() >= n && std::memcmp(s.data(), p, n) == 0;
}

bool ends_with(const std::string& s, const char* p) {
    const size_t n = std::strlen(p);
    return s.size() >= n && std::memcmp(s.data() + (s.size() - n), p, n) == 0;
}

// 推理子集：只保留 enc_p. / flow. / dec. / emb_g.
bool keep_tensor(const std::string& name) {
    return starts_with(name, "enc_p.") || starts_with(name, "flow.") ||
           starts_with(name, "dec.") || starts_with(name, "emb_g.");
}

int infer_ndim(const int64_t ne[4]) {
    int nd = 1;
    for (int i = 0; i < 4; ++i) {
        if (ne[i] > 1) {
            nd = i + 1;
        }
    }
    return nd;
}

std::string stem_of(const std::string& path) {
    size_t slash = path.find_last_of("/\\");
    std::string base = (slash == std::string::npos) ? path : path.substr(slash + 1);
    size_t dot = base.find_last_of('.');
    if (dot != std::string::npos && dot > 0) {
        base = base.substr(0, dot);
    }
    return base;
}

// ---- 最小 GGUF 头解析：只取每个张量的真实 n_dims ----
// train.cpp 的 GgufTensorInfo 不带 n_dims（其 Tensor 无此字段），而"是否展平"
// 必须用真实维数判定（如 dec.conv_post.weight [1,I,K] 的输出通道=1，用 ne>1 推断会
// 误判成 2D）。此处直接按 GGUF 规范扫描头部，跳过 KV 区后读 tensor info 的 n_dims。
bool skip_gguf_value(std::ifstream& f, uint32_t type) {
    auto skip = [&](size_t n) -> bool {
        f.seekg((std::streamoff) n, std::ios::cur);
        return (bool) f;
    };
    switch (type) {
        case 0:  // U8
        case 1:  // I8
        case 7:  // BOOL
            return skip(1);
        case 2:  // U16
        case 3:  // I16
            return skip(2);
        case 4:  // U32
        case 5:  // I32
        case 6:  // F32
            return skip(4);
        case 10:  // U64
        case 11:  // I64
        case 12:  // F64
            return skip(8);
        case 8: {  // STRING
            uint64_t n = 0;
            f.read((char*) &n, 8);
            return (bool) f && skip((size_t) n);
        }
        case 9: {  // ARRAY
            uint32_t et = 0;
            uint64_t n  = 0;
            f.read((char*) &et, 4);
            f.read((char*) &n, 8);
            if (!f) {
                return false;
            }
            for (uint64_t i = 0; i < n; ++i) {
                if (!skip_gguf_value(f, et)) {
                    return false;
                }
            }
            return true;
        }
        default:
            return false;
    }
}

bool load_tensor_ndims(const std::string& path, std::unordered_map<std::string, int>* out,
                       std::string* err) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        *err = "无法打开源 GGUF: " + path;
        return false;
    }
    char magic[4] = {0, 0, 0, 0};
    f.read(magic, 4);
    if (!f || std::memcmp(magic, "GGUF", 4) != 0) {
        *err = "源文件不是 GGUF: " + path;
        return false;
    }
    uint32_t version = 0;
    uint64_t tensor_count = 0;
    uint64_t kv_count     = 0;
    f.read((char*) &version, 4);
    f.read((char*) &tensor_count, 8);
    f.read((char*) &kv_count, 8);
    if (!f) {
        *err = "GGUF 头部损坏: " + path;
        return false;
    }
    if (version < 2 || version > 3) {
        *err = "不支持的 GGUF 版本: " + std::to_string(version);
        return false;
    }
    for (uint64_t i = 0; i < kv_count; ++i) {
        uint64_t klen = 0;
        f.read((char*) &klen, 8);
        f.seekg((std::streamoff) klen, std::ios::cur);
        uint32_t vt = 0;
        f.read((char*) &vt, 4);
        if (!f || !skip_gguf_value(f, vt)) {
            *err = "GGUF KV 区解析失败: " + path;
            return false;
        }
    }
    for (uint64_t i = 0; i < tensor_count; ++i) {
        uint64_t nlen = 0;
        f.read((char*) &nlen, 8);
        std::string name;
        name.resize((size_t) nlen);
        if (nlen > 0) {
            f.read(name.data(), (std::streamsize) nlen);
        }
        uint32_t nd = 0;
        f.read((char*) &nd, 4);
        f.seekg((std::streamoff) ((uint64_t) nd * 8), std::ios::cur);
        uint32_t ty  = 0;
        uint64_t off = 0;
        f.read((char*) &ty, 4);
        f.read((char*) &off, 8);
        if (!f) {
            *err = "GGUF tensor info 解析失败: " + path;
            return false;
        }
        (void) ty;
        (void) off;
        (*out)[name] = (int) nd;
    }
    return true;
}

// 读取源张量为 F32 主机数据（自动反量化）；返回 ggml ne、推断 ndim。
bool get_tensor(const GgufFile* f, const std::string& name, std::vector<float>* data,
                int64_t ne[4], int* ndim, std::string* err) {
    const int64_t idx = gguf_find_tensor(f, name.c_str());
    if (idx < 0) {
        *err = "源 GGUF 缺少张量: " + name;
        return false;
    }
    const GgufTensorInfo* info = gguf_tensor_info(f, idx);
    if (info == nullptr) {
        *err = "张量信息损坏: " + name;
        return false;
    }
    for (int i = 0; i < 4; ++i) {
        ne[i] = info->ne[i];
    }
    *ndim = infer_ndim(ne);
    const int64_t n = gguf_tensor_nelements(f, idx);
    data->resize((size_t) n);
    if (!gguf_tensor_to_f32(f, name.c_str(), data->data(), n)) {
        *err = "读取张量数据失败: " + name;
        return false;
    }
    return true;
}

// 折叠 weight_norm（标准 PyTorch weight_norm(dim=0)）：
//   v torch 形状 [O, ...]，g.numel()==O → w = v / clamp(‖v‖_rest, 1e-12) * g
// 数据为 torch 行主序；输出与 v 同形（ggml ne 不变）。
bool fold_weight_norm(const std::vector<float>& v, const int64_t v_ne[4], int v_ndim,
                      const std::vector<float>& g, int64_t g_numel, std::vector<float>* out,
                      std::string* err) {
    int64_t vshape[4] = {1, 1, 1, 1};
    for (int d = 0; d < v_ndim; ++d) {
        vshape[d] = v_ne[v_ndim - 1 - d];
    }
    int64_t total = 1;
    for (int d = 0; d < v_ndim; ++d) {
        total *= vshape[d];
    }
    if ((int64_t) v.size() != total) {
        *err = "weight_norm 折叠：源张量元素数异常";
        return false;
    }
    const int64_t O = vshape[0];
    if (g_numel != O) {
        char buf[192];
        std::snprintf(buf, sizeof(buf),
                      "weight_norm 折叠：暂不支持 fairseq dim=(0,1) 约定（g.numel=%lld, dim0=%lld）",
                      (long long) g_numel, (long long) O);
        *err = buf;
        return false;
    }
    const int64_t inner = (O > 0) ? total / O : 0;
    out->resize((size_t) total);
    for (int64_t o = 0; o < O; ++o) {
        const float* vp = v.data() + (size_t) (o * inner);
        double ss = 0.0;
        for (int64_t j = 0; j < inner; ++j) {
            const double x = (double) vp[j];
            ss += x * x;
        }
        double norm = std::sqrt(ss);
        if (norm < 1e-12) {
            norm = 1e-12;
        }
        const double gg = (double) g[(size_t) o];
        float* op = out->data() + (size_t) (o * inner);
        for (int64_t j = 0; j < inner; ++j) {
            op[j] = (float) ((double) vp[j] / norm * gg);
        }
    }
    return true;
}

// Conv1d 权重展平：[O,I,K] → (O, I*K)（数据不变，仅 ne 变化）。
// 规则与 export_gguf.py 一致：ndim==3 且名以 .weight 结尾且非 dec.ups.*。
void maybe_flatten(const std::string& name, int ndim, const int64_t ne_in[4], int* out_ndim,
                   int64_t out_ne[4]) {
    for (int i = 0; i < 4; ++i) {
        out_ne[i] = ne_in[i];
    }
    *out_ndim = ndim;
    if (ndim == 3 && ends_with(name, ".weight") && !starts_with(name, "dec.ups.")) {
        out_ne[0] = ne_in[0] * ne_in[1];
        out_ne[1] = ne_in[2];
        out_ne[2] = 1;
        out_ne[3] = 1;
        *out_ndim  = 2;
    }
}

bool copy_str_meta(const GgufFile* src, GgufWriter* w, const char* dst, const char* src_key,
                   bool required, std::string* err) {
    if (!gguf_has_key(src, src_key)) {
        if (required) {
            *err = std::string("源 GGUF 缺少元数据: ") + src_key;
            return false;
        }
        return true;
    }
    const char* s = gguf_get_str(src, src_key);
    gguf_writer_set_str(w, dst, s != nullptr ? s : "");
    return true;
}

bool copy_u32_meta(const GgufFile* src, GgufWriter* w, const char* dst, const char* src_key,
                   bool required, std::string* err) {
    if (!gguf_has_key(src, src_key)) {
        if (required) {
            *err = std::string("源 GGUF 缺少元数据: ") + src_key;
            return false;
        }
        return true;
    }
    gguf_writer_set_u32(w, dst, gguf_get_u32(src, src_key));
    return true;
}

bool copy_arr_i32_meta(const GgufFile* src, GgufWriter* w, const char* dst, const char* src_key,
                       bool required, std::string* err) {
    if (!gguf_has_key(src, src_key)) {
        if (required) {
            *err = std::string("源 GGUF 缺少元数据: ") + src_key;
            return false;
        }
        return true;
    }
    int64_t n = 0;
    const int32_t* v = gguf_get_arr_i32(src, src_key, &n);
    if (v == nullptr) {
        *err = std::string("元数据数组读取失败: ") + src_key;
        return false;
    }
    gguf_writer_set_arr_i32(w, dst, v, n);
    return true;
}

} // namespace

bool gguf_read_tensor_ndims(const std::string& path, std::unordered_map<std::string, int>* out,
                            std::string* err) {
    return load_tensor_ndims(path, out, err);
}

bool export_inference_gguf(const ModelExportOptions& opt, std::string* err) {
    if (opt.in_path.empty() || opt.out_path.empty()) {
        if (err != nullptr) {
            *err = "输入/输出路径不能为空";
        }
        return false;
    }

    GgufFile* src = gguf_open(opt.in_path.c_str(), /*load_data=*/true);
    if (src == nullptr) {
        if (err != nullptr) {
            *err = "无法打开/解析源 GGUF: " + opt.in_path;
        }
        return false;
    }

    std::unordered_map<std::string, int> ndims;
    {
        std::string perr;
        if (!load_tensor_ndims(opt.in_path, &ndims, &perr)) {
            if (err != nullptr) {
                *err = perr;
            }
            gguf_close(src);
            return false;
        }
    }
    auto true_ndim = [&](const std::string& n, int fallback) {
        auto it = ndims.find(n);
        return it != ndims.end() ? it->second : fallback;
    };
    // M7-4：实时快照/scratch 训练态 GGUF 会因尾部 1 维折叠丢失真实维数 → 用来源文件
    // ndims==3 名单强制展平；并固定加入 RVC 架构中输出通道=1 的 Conv1d 权重
    //（dec.conv_post；已核对 v1-40k 训练态 GGUF 仅此一处，v2 结构相同），
    // 使独立 `--export-gguf` 对 scratch GGUF 也能正确展平。
    std::unordered_set<std::string> force(opt.force_flatten.begin(), opt.force_flatten.end());
    force.insert("dec.conv_post.weight");
    force.insert("dec.conv_post.weight_v");

    // 输出数据容器：no_alloc=false 让张量数据分配在 ctx arena 内（写端只记指针）。
    ContextParams cp;
    cp.mem_size = 64 * 1024 * 1024;  // 不足时自动追加 arena
    cp.no_alloc = false;
    Context* ctx = context_new(cp);

    GgufWriter* w = gguf_writer_new();
    gguf_writer_set_arch(w, "rvc");
    gguf_writer_set_str(w, "general.name",
                        opt.general_name.empty() ? stem_of(opt.out_path).c_str()
                                                 : opt.general_name.c_str());
    gguf_writer_set_str(w, "general.description", "RVC inference model exported by RVC-Train-GGML");
    gguf_writer_set_u32(w, "general.file_type", 0);  // 全 F32

    std::string e;
    bool ok = true;

    // ---- 元数据：rvc.train.* → rvc.* ----
    ok = ok && copy_str_meta(src, w, "rvc.version", "rvc.train.version", true, &e);
    if (ok && gguf_has_key(src, "rvc.train.has_f0")) {
        gguf_writer_set_bool(w, "rvc.has_f0", gguf_get_bool(src, "rvc.train.has_f0"));
    }
    ok = ok && copy_u32_meta(src, w, "rvc.sample_rate", "rvc.train.sample_rate", true, &e);
    ok = ok && copy_u32_meta(src, w, "rvc.spec_channels", "rvc.train.spec_channels", true, &e);
    ok = ok && copy_u32_meta(src, w, "rvc.segment_size", "rvc.train.segment_size", false, &e);
    ok = ok && copy_u32_meta(src, w, "rvc.inter_channels", "rvc.train.inter_channels", true, &e);
    ok = ok && copy_u32_meta(src, w, "rvc.hidden_channels", "rvc.train.hidden_channels", true, &e);
    ok = ok && copy_u32_meta(src, w, "rvc.filter_channels", "rvc.train.filter_channels", true, &e);
    ok = ok && copy_u32_meta(src, w, "rvc.n_heads", "rvc.train.n_heads", true, &e);
    ok = ok && copy_u32_meta(src, w, "rvc.n_layers", "rvc.train.n_layers", true, &e);
    ok = ok && copy_u32_meta(src, w, "rvc.kernel_size", "rvc.train.kernel_size", true, &e);
    ok = ok && copy_str_meta(src, w, "rvc.resblock", "rvc.train.resblock", true, &e);
    ok = ok && copy_arr_i32_meta(src, w, "rvc.resblock_kernel_sizes",
                                 "rvc.train.resblock_kernel_sizes", true, &e);
    ok = ok && copy_arr_i32_meta(src, w, "rvc.resblock_dilation_sizes",
                                 "rvc.train.resblock_dilation_sizes", true, &e);
    ok = ok && copy_arr_i32_meta(src, w, "rvc.upsample_rates", "rvc.train.upsample_rates", true, &e);
    ok = ok && copy_u32_meta(src, w, "rvc.upsample_initial_channel",
                             "rvc.train.upsample_initial_channel", true, &e);
    ok = ok && copy_arr_i32_meta(src, w, "rvc.upsample_kernel_sizes",
                                 "rvc.train.upsample_kernel_sizes", true, &e);
    ok = ok && copy_u32_meta(src, w, "rvc.gin_channels", "rvc.train.gin_channels", true, &e);
    gguf_writer_set_bool(w, "rvc.conv_flat", true);

    // n_spk / spk_embed_dim 以 emb_g.weight 实际形状为准（torch [dim, spk] → ne [spk, dim]）
    int64_t emb_ne[4] = {1, 1, 1, 1};
    int emb_ndim = 0;
    {
        std::vector<float> tmp;
        if (!get_tensor(src, "emb_g.weight", &tmp, emb_ne, &emb_ndim, &e)) {
            ok = false;
        }
    }
    if (ok) {
        const uint32_t n_spk = (uint32_t) emb_ne[1];
        uint32_t spk_dim = n_spk;
        if (gguf_has_key(src, "rvc.train.spk_embed_dim")) {
            const uint32_t v = gguf_get_u32(src, "rvc.train.spk_embed_dim");
            if (v > 0) {
                spk_dim = v;
            }
        }
        gguf_writer_set_u32(w, "rvc.n_spk", n_spk);
        gguf_writer_set_u32(w, "rvc.spk_embed_dim", spk_dim);
    }

    // ---- 张量：按源顺序，跳过 weight_g（在 weight_v 处折叠）；enc_q/disc 已过滤 ----
    int64_t n_out = 0;
    const int64_t count = gguf_tensor_count(src);
    for (int64_t i = 0; ok && i < count; ++i) {
        const GgufTensorInfo* info = gguf_tensor_info(src, i);
        if (info == nullptr || info->name == nullptr) {
            continue;
        }
        const std::string name = info->name;
        if (!keep_tensor(name)) {
            continue;
        }
        if (ends_with(name, ".weight_g")) {
            continue;  // 由 .weight_v 处折叠产出
        }

        std::vector<float> out;
        int64_t out_ne[4] = {1, 1, 1, 1};
        int out_ndim = 1;
        const std::string out_name =
            ends_with(name, ".weight_v")
                ? (name.substr(0, name.size() - std::strlen(".weight_v")) + ".weight")
                : name;

        if (ends_with(name, ".weight_v")) {
            const std::string prefix = name.substr(0, name.size() - std::strlen(".weight_v"));
            std::vector<float> v, g;
            int64_t v_ne[4] = {1, 1, 1, 1};
            int64_t g_ne[4] = {1, 1, 1, 1};
            int v_ndim = 0, g_ndim = 0;
            if (!get_tensor(src, name, &v, v_ne, &v_ndim, &e) ||
                !get_tensor(src, prefix + ".weight_g", &g, g_ne, &g_ndim, &e)) {
                ok = false;
                break;
            }
            int64_t g_numel = 1;
            for (int d = 0; d < g_ndim; ++d) {
                g_numel *= g_ne[d];
            }
            if (!fold_weight_norm(v, v_ne, v_ndim, g, g_numel, &out, &e)) {
                ok = false;
                break;
            }
            int nd = true_ndim(name, infer_ndim(v_ne));
            if (force.count(name) != 0) {
                nd = 3;
            }
            maybe_flatten(out_name, nd, v_ne, &out_ndim, out_ne);
        } else {
            int64_t ne[4] = {1, 1, 1, 1};
            int ndim = 0;
            if (!get_tensor(src, name, &out, ne, &ndim, &e)) {
                ok = false;
                break;
            }
            int nd = true_ndim(name, ndim);
            if (force.count(name) != 0) {
                nd = 3;
            }
            maybe_flatten(name, nd, ne, &out_ndim, out_ne);
        }

        // 写出张量（数据已在 ctx arena；保持存活到 write 返回）
        Tensor* t = new_tensor_nd(ctx, TYPE_F32, out_ndim, out_ne);
        std::memcpy(t->data, out.data(), out.size() * sizeof(float));
        tensor_set_name(t, "%s", out_name.c_str());
        gguf_writer_add_tensor(w, t->name, t, TYPE_F32);
        ++n_out;
    }

    if (ok && !gguf_writer_write(w, opt.out_path.c_str())) {
        e = "写出 GGUF 失败: " + opt.out_path;
        ok = false;
    }

    if (ok) {
        std::printf("[export] 完成: %s -> %s（%lld 张量）\n", opt.in_path.c_str(),
                    opt.out_path.c_str(), (long long) n_out);
    } else if (err != nullptr) {
        *err = e;
    }

    gguf_writer_free(w);
    context_free(ctx);
    gguf_close(src);
    return ok;
}

} // namespace exporter
} // namespace rvc_train
