// rvc-prep - RVC 训练数据预处理程序
//
// 职责（M4 已实现）：
//   原始音频 → 切片/重采样/高通 → 16k / f0(RMVPE) / HuBERT 特征 / 线性谱缓存
//   复用 RVC-GGML 的 rvc_core（ggml Vulkan）；训练运行期零 Python 依赖。
//
// 当前阶段（M5-0 后）：M4 已全部完成（5a MVP + 5b 完整 IVF，见 `交接说明` §4.8 K/L）；
//   本 exe 的预处理/缓存/索引命令保持稳定，M5-1 数据加载器只新增缓存**读侧** API（不改写出格式）。
//   索引：`--index` 默认完整 IVF（置换 + 可选 MiniBatchKMeans + 官方 nlist 公式 + IVF k-means + list 分组；
//   `--index-mvp` 回退 M4-5a 单 list），另提供 `--search-index` 搜索 dump 供 G10 对拍。
//   `--debug-dump` 落 load/hp/gt/16k/f0/feat/spec/slices/chunks 供官方 golden 对拍（G1–G9；
//   索引另由 tools/verify_index_gguf.py 校验，G10）。
//
// 用法：
//   rvc-prep（无参数 / --interactive）                 交互式预处理向导（M8；双击启动即向导）
//   rvc-prep --list-devices
//   rvc-prep --selftest [--device N]                   设备自检（M0 冒烟；M8-3 起改为显式进入）
//   rvc-prep --input <dir|wav> --sr 40000 --version v1 [--debug-dump DIR] [--exp DIR] [选项]
//   rvc-prep --load-raw <case.load.f32.raw> --load-raw-sr 40000 [--debug-dump DIR]
//   rvc-prep --validate-cache <exp_dir>                缓存自校验（G9）
//   rvc-prep --index --features <dir|file.npy> --out <index.gguf>   完整 IVF 索引（M4-5b）
//   rvc-prep --validate-index <index.gguf>             索引自校验（M4-5a/5b）
//   rvc-prep --search-index <index.gguf> --queries <q.npy> --search-out <hits.raw>
//                                                      搜索 dump（G10 对拍用）
//
// 退出码：0 = 成功；1 = 失败；2 = 参数错误。

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include <windows.h>

#include <ggml.h>

#include "common/json.h"
#include "cli/prep_wizard.h"
#include "engine.h"
#include "export/index_gguf.h"
#include "export/npy.h"
#include "gguf/gguf_model.h"
#include "hubert/hubert.h"
#include "prep/cache.h"
#include "prep/feat.h"
#include "prep/pipeline.h"
#include "rmvpe/rmvpe.h"
#include "rvc/retrieval.h"

namespace fs = std::filesystem;

namespace {

// 环境变量是否存在（MSVC /W4 下用 _dupenv_s 替代 getenv，避免 C4996）
bool env_present(const char* name) {
    char*  value = nullptr;
    size_t size  = 0;
    if (_dupenv_s(&value, &size, name) == 0 && value != nullptr) {
        std::free(value);
        return true;
    }
    return false;
}

void print_devices() {
    const std::vector<rvc::GpuDeviceInfo> devs = rvc::list_gpu_devices();
    std::printf("[rvc-prep] ggml GPU 设备数: %zu\n", devs.size());
    for (const rvc::GpuDeviceInfo& d : devs) {
        std::printf("  [%d] %s | %s\n", d.index, d.name.c_str(), d.description.c_str());
    }
}

// 在选定设备上跑一次最小 ggml 图：c = a + b，校验结果
bool run_selftest(int device_id) {
    rvc::GgmlDevice dev(device_id);
    std::printf("[rvc-prep] 后端: %s（device_id=%d）\n", dev.name().c_str(), device_id);

    // 图/中间量由 GgmlGraphRunner 管理（ctx 元数据 4 MiB，最多 64 节点）
    rvc::GgmlGraphRunner runner(dev.backend(), static_cast<size_t>(4) << 20, 64);
    ggml_context* ctx = runner.ctx();

    ggml_tensor* a = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4);
    ggml_tensor* b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4);
    ggml_tensor* c = ggml_add(ctx, a, b);
    ggml_set_name(a, "a");
    ggml_set_name(b, "b");
    ggml_set_name(c, "c");

    runner.expand(c);  // 展开计算图
    if (!runner.allocate()) {
        std::printf("[rvc-prep] 计算缓冲分配失败\n");
        return false;
    }

    const float av[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    const float bv[4] = {10.0f, 20.0f, 30.0f, 40.0f};
    runner.set_input(a, av, sizeof(av));
    runner.set_input(b, bv, sizeof(bv));
    runner.compute();

    float cv[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    runner.get_output(c, cv, sizeof(cv));

    const float expect[4] = {11.0f, 22.0f, 33.0f, 44.0f};
    bool ok = true;
    for (int i = 0; i < 4; ++i) {
        if (std::fabs(cv[i] - expect[i]) > 1e-6f) {
            ok = false;
        }
    }
    std::printf("[rvc-prep] a+b = [%.1f, %.1f, %.1f, %.1f]（期望 [11, 22, 33, 44]）\n",
                cv[0], cv[1], cv[2], cv[3]);
    return ok;
}

void print_usage() {
    std::printf(
        "用法:\n"
        "  rvc-prep（无参数 / --interactive）  交互式预处理向导（M8）\n"
        "  rvc-prep --list-devices\n"
        "  rvc-prep --selftest                 M0 设备/图计算自检\n"
        "  rvc-prep [--device N]\n"
        "  rvc-prep --interactive [--answers FILE]      # M8-2 交互式预处理向导\n"
        "  rvc-prep --input <dir|wav> --sr 40000 --version v1 [--debug-dump DIR] [--exp DIR] [--limit N]\n"
        "  rvc-prep --load-raw <file.f32.raw> --load-raw-sr 40000 [--debug-dump DIR]\n"
        "  rvc-prep --validate-cache <exp_dir>\n"
        "  rvc-prep --index --features <dir|file.npy> --out <index.gguf>   # M4-5b 完整 IVF\n"
        "      [--exp <cache>] 也可作为特征源\n"
        "      [--index-mvp]            回退 M4-5a 单 list（全扫）\n"
        "      [--nprobe N]             写入 index.nprobe（默认 1，官方默认）\n"
        "      [--seed N]               置换/k-means 种子（默认 1234，可复现）\n"
        "      [--kmeans-iters N]       IVF k-means 迭代上限（默认 25）\n"
        "      [--mini-centroids N]     N>200000 时 MiniBatchKMeans 中心数（默认 10000）\n"
        "      [--mini-epochs N]        MiniBatchKMeans 轮数上限（默认 10）\n"
        "      [--no-permute]           调试：跳过特征行置换\n"
        "  rvc-prep --validate-index <index.gguf>\n"
        "  rvc-prep --search-index <index.gguf> --queries <file.npy> --search-out <hits.raw> [--k 8]\n"
        "选项: --exp DIR（训练缓存输出/索引特征输入） --has-f0 0|1 --per 3.7 --overlap 0.3\n"
        "      --no-16k --no-f0 --no-feat --no-spec\n"
        "      --rmvpe FILE（M4-2：RMVPE 权重，启用 f0 提取）\n"
        "      --hubert FILE（M4-3：HuBERT 权重，启用特征提取；--version v1|v2 决定 256/768 维）\n");
}

struct Args {
    bool list_only = false;
    bool device_set = false;
    int device_id = 0;

    // M8-2：交互式预处理向导
    bool interactive = false;            // --interactive（无参数双击亦进入）
    bool selftest_m0 = false;            // --selftest（M0 冒烟；M8-3 起改为显式进入）
    std::string answers_path;            // --answers <file>（非交互注入）
    bool selftest_prep_wizard = false;   // --selftest-prep-wizard

    std::string input;         // 目录或单个 WAV
    std::string load_raw;      // 注入模式
    int load_raw_sr = 0;       // 0 = 用 --sr
    std::string exp;             // M4-4 缓存输出目录
    std::string validate_cache;  // M4-4 --validate-cache
    std::string debug_dump;
    // M4-5a：索引模式
    bool index_mode = false;         // --index
    std::string features;            // --features <dir|file.npy>（也支持 --exp 缓存）
    std::string index_out;           // --out <index.gguf>
    std::string validate_index;      // --validate-index <index.gguf>
    // M4-5b：完整 IVF / 搜索 dump
    bool index_mvp = false;              // --index-mvp（回退 M4-5a 单 list）
    uint32_t nprobe = 1;                 // --nprobe
    uint64_t seed = 1234;                // --seed
    int64_t kmeans_iters = 25;           // --kmeans-iters
    int64_t mini_centroids = 10000;      // --mini-centroids
    int64_t mini_epochs = 10;            // --mini-epochs
    bool no_permute = false;             // --no-permute
    std::string search_index;            // --search-index <index.gguf>
    std::string search_queries;          // --queries <file.npy>
    std::string search_out;              // --search-out <hits.raw>
    int search_k = 8;                    // --k（默认 8，对齐 RVC-GGML pipeline.cpp:103）
    std::string hubert;
    std::string rmvpe;
    int limit = -1;

    rvc_train::prep::PrepOptions opt;
    bool no_f0 = false;
    bool no_feat = false;
    bool no_spec = false;

    bool has_work() const { return !input.empty() || !load_raw.empty(); }
};

bool parse_int(const char* s, int* out) {
    char* end = nullptr;
    const long v = std::strtol(s, &end, 10);
    if (end == s || *end != '\0') {
        return false;
    }
    *out = static_cast<int>(v);
    return true;
}

bool parse_double(const char* s, double* out) {
    char* end = nullptr;
    const double v = std::strtod(s, &end);
    if (end == s || *end != '\0') {
        return false;
    }
    *out = v;
    return true;
}

bool parse_i64(const char* s, int64_t* out) {
    char* end = nullptr;
    const long long v = std::strtoll(s, &end, 10);
    if (end == s || *end != '\0') {
        return false;
    }
    *out = static_cast<int64_t>(v);
    return true;
}

bool parse_u64(const char* s, uint64_t* out) {
    char* end = nullptr;
    const unsigned long long v = std::strtoull(s, &end, 10);
    if (end == s || *end != '\0') {
        return false;
    }
    *out = static_cast<uint64_t>(v);
    return true;
}

int parse_args(int argc, char** argv, Args* args) {
    auto need = [&](int i, const char* flag) -> const char* {
        if (i + 1 >= argc) {
            std::printf("[rvc-prep] 参数 %s 缺少取值\n", flag);
            return nullptr;
        }
        return argv[i + 1];
    };

    for (int i = 1; i < argc; ++i) {
        const char* a = argv[i];
        if (std::strcmp(a, "--list-devices") == 0) {
            args->list_only = true;
        } else if (std::strcmp(a, "--interactive") == 0) {
            args->interactive = true;
        } else if (std::strcmp(a, "--selftest") == 0) {
            args->selftest_m0 = true;
        } else if (std::strcmp(a, "--answers") == 0) {
            const char* v = need(i, a);
            if (v == nullptr) {
                return 2;
            }
            args->answers_path = v;
            ++i;
        } else if (std::strcmp(a, "--selftest-prep-wizard") == 0) {
            args->selftest_prep_wizard = true;
        } else if (std::strcmp(a, "--device") == 0) {
            const char* v = need(i, a);
            if (v == nullptr || !parse_int(v, &args->device_id)) {
                return 2;
            }
            args->device_set = true;
            ++i;
        } else if (std::strcmp(a, "--input") == 0) {
            const char* v = need(i, a);
            if (v == nullptr) {
                return 2;
            }
            args->input = v;
            ++i;
        } else if (std::strcmp(a, "--load-raw") == 0) {
            const char* v = need(i, a);
            if (v == nullptr) {
                return 2;
            }
            args->load_raw = v;
            ++i;
        } else if (std::strcmp(a, "--load-raw-sr") == 0) {
            const char* v = need(i, a);
            if (v == nullptr || !parse_int(v, &args->load_raw_sr)) {
                return 2;
            }
            ++i;
        } else if (std::strcmp(a, "--exp") == 0) {
            const char* v = need(i, a);
            if (v == nullptr) {
                return 2;
            }
            args->exp = v;
            ++i;
        } else if (std::strcmp(a, "--validate-cache") == 0) {
            const char* v = need(i, a);
            if (v == nullptr) {
                return 2;
            }
            args->validate_cache = v;
            ++i;
        } else if (std::strcmp(a, "--index") == 0) {
            args->index_mode = true;
        } else if (std::strcmp(a, "--features") == 0) {
            const char* v = need(i, a);
            if (v == nullptr) {
                return 2;
            }
            args->features = v;
            ++i;
        } else if (std::strcmp(a, "--out") == 0) {
            const char* v = need(i, a);
            if (v == nullptr) {
                return 2;
            }
            args->index_out = v;
            ++i;
        } else if (std::strcmp(a, "--validate-index") == 0) {
            const char* v = need(i, a);
            if (v == nullptr) {
                return 2;
            }
            args->validate_index = v;
            ++i;
        } else if (std::strcmp(a, "--index-mvp") == 0) {
            args->index_mvp = true;
        } else if (std::strcmp(a, "--nprobe") == 0) {
            const char* v = need(i, a);
            int64_t iv = 0;
            if (v == nullptr || !parse_i64(v, &iv) || iv < 1 || iv > 0x7fffffffLL) {
                return 2;
            }
            args->nprobe = static_cast<uint32_t>(iv);
            ++i;
        } else if (std::strcmp(a, "--seed") == 0) {
            const char* v = need(i, a);
            if (v == nullptr || !parse_u64(v, &args->seed)) {
                return 2;
            }
            ++i;
        } else if (std::strcmp(a, "--kmeans-iters") == 0) {
            const char* v = need(i, a);
            if (v == nullptr || !parse_i64(v, &args->kmeans_iters)) {
                return 2;
            }
            ++i;
        } else if (std::strcmp(a, "--mini-centroids") == 0) {
            const char* v = need(i, a);
            if (v == nullptr || !parse_i64(v, &args->mini_centroids)) {
                return 2;
            }
            ++i;
        } else if (std::strcmp(a, "--mini-epochs") == 0) {
            const char* v = need(i, a);
            if (v == nullptr || !parse_i64(v, &args->mini_epochs)) {
                return 2;
            }
            ++i;
        } else if (std::strcmp(a, "--no-permute") == 0) {
            args->no_permute = true;
        } else if (std::strcmp(a, "--search-index") == 0) {
            const char* v = need(i, a);
            if (v == nullptr) {
                return 2;
            }
            args->search_index = v;
            ++i;
        } else if (std::strcmp(a, "--queries") == 0) {
            const char* v = need(i, a);
            if (v == nullptr) {
                return 2;
            }
            args->search_queries = v;
            ++i;
        } else if (std::strcmp(a, "--search-out") == 0) {
            const char* v = need(i, a);
            if (v == nullptr) {
                return 2;
            }
            args->search_out = v;
            ++i;
        } else if (std::strcmp(a, "--k") == 0) {
            const char* v = need(i, a);
            if (v == nullptr || !parse_int(v, &args->search_k) || args->search_k < 1) {
                return 2;
            }
            ++i;
        } else if (std::strcmp(a, "--sr") == 0) {
            const char* v = need(i, a);
            if (v == nullptr || !parse_int(v, &args->opt.target_sr)) {
                return 2;
            }
            ++i;
        } else if (std::strcmp(a, "--version") == 0) {
            const char* v = need(i, a);
            if (v == nullptr) {
                return 2;
            }
            args->opt.version = v;
            ++i;
        } else if (std::strcmp(a, "--has-f0") == 0) {
            const char* v = need(i, a);
            int iv = 0;
            if (v == nullptr || !parse_int(v, &iv)) {
                return 2;
            }
            args->opt.has_f0 = iv != 0;
            ++i;
        } else if (std::strcmp(a, "--per") == 0) {
            const char* v = need(i, a);
            if (v == nullptr || !parse_double(v, &args->opt.per)) {
                return 2;
            }
            ++i;
        } else if (std::strcmp(a, "--overlap") == 0) {
            const char* v = need(i, a);
            if (v == nullptr || !parse_double(v, &args->opt.overlap)) {
                return 2;
            }
            ++i;
        } else if (std::strcmp(a, "--debug-dump") == 0) {
            const char* v = need(i, a);
            if (v == nullptr) {
                return 2;
            }
            args->debug_dump = v;
            ++i;
        } else if (std::strcmp(a, "--limit") == 0) {
            const char* v = need(i, a);
            if (v == nullptr || !parse_int(v, &args->limit)) {
                return 2;
            }
            ++i;
        } else if (std::strcmp(a, "--hubert") == 0) {
            const char* v = need(i, a);
            if (v == nullptr) {
                return 2;
            }
            args->hubert = v;
            ++i;
        } else if (std::strcmp(a, "--rmvpe") == 0) {
            const char* v = need(i, a);
            if (v == nullptr) {
                return 2;
            }
            args->rmvpe = v;
            ++i;
        } else if (std::strcmp(a, "--no-16k") == 0) {
            args->opt.no_16k = true;
        } else if (std::strcmp(a, "--no-f0") == 0) {
            args->no_f0 = true;
        } else if (std::strcmp(a, "--no-feat") == 0) {
            args->no_feat = true;
        } else if (std::strcmp(a, "--no-spec") == 0) {
            args->no_spec = true;
        } else {
            std::printf("[rvc-prep] 未知参数: %s\n", a);
            print_usage();
            return 2;
        }
    }
    return 0;
}

// 注入模式的 case 名：`<case>.load.f32.raw` → `<case>`
std::string case_from_load_raw(const std::string& path) {
    std::string name = fs::path(path).filename().string();
    const std::string raw_suffix = ".f32.raw";
    if (name.size() > raw_suffix.size() &&
        name.compare(name.size() - raw_suffix.size(), raw_suffix.size(), raw_suffix) == 0) {
        name = name.substr(0, name.size() - raw_suffix.size());
    }
    const std::string load_suffix = ".load";
    if (name.size() > load_suffix.size() &&
        name.compare(name.size() - load_suffix.size(), load_suffix.size(), load_suffix) == 0) {
        name = name.substr(0, name.size() - load_suffix.size());
    }
    return name.empty() ? std::string("inject") : name;
}

int run_prep(const Args& args) {
    rvc_train::prep::PrepOptions opt = args.opt;
    opt.debug_dump_dir = args.debug_dump;

    if (opt.version != "v1" && opt.version != "v2") {
        std::printf("[rvc-prep] 失败：--version 仅支持 v1|v2（当前 %s）\n", opt.version.c_str());
        return 2;
    }
    // M4-4：训练缓存输出目录（samples/*.bin 由 pipeline 写；索引在全部文件成功后写）
    opt.cache_dir = args.exp;
    if (!args.exp.empty()) {
        std::printf("[rvc-prep] 训练缓存输出: %s\n", args.exp.c_str());
    }

    // M4-4：线性谱配置（官方 config：n_fft=win=2048、hop=sr/100；v1-40k → 400）
    opt.spec.n_fft = 2048;
    opt.spec.win = 2048;
    opt.spec.hop = opt.target_sr / 100;
    opt.spec.sample_rate = opt.target_sr;
    if (args.no_spec) {
        opt.with_spec = false;
        std::printf("[rvc-prep] 线性谱：--no-spec，跳过\n");
    } else {
        std::printf("[rvc-prep] 线性谱: n_fft=%d hop=%d win=%d bins=%d sr=%d\n",
                    opt.spec.n_fft, opt.spec.hop, opt.spec.win, opt.spec.bins(),
                    opt.spec.sample_rate);
    }

    // M4-2/M4-3：f0（RMVPE）与 HuBERT 特征共用同一 GgmlDevice；
    // device/model/extractor/encoder 生命周期覆盖整个 run_prep。
    std::unique_ptr<rvc::GgmlDevice> device;
    std::unique_ptr<GgufModel> f0_model;
    std::unique_ptr<rvc::rmvpe::RmvpeExtractor> f0_extractor;
    std::unique_ptr<GgufModel> hubert_model;
    std::unique_ptr<rvc_train::prep::HubertFeatureExtractor> hubert_extractor;
    int device_id = -1;
    auto ensure_device = [&]() -> bool {
        if (device) {
            return true;
        }
        device_id = args.device_set ? args.device_id : (rvc::list_gpu_devices().empty() ? 99 : 0);
        try {
            device = std::make_unique<rvc::GgmlDevice>(device_id);
        } catch (const std::exception& e) {
            std::printf("[rvc-prep] 失败：创建设备（device_id=%d）：%s\n", device_id, e.what());
            return false;
        }
        return true;
    };

    if (args.no_f0) {
        opt.with_f0 = false;
        std::printf("[rvc-prep] f0：--no-f0，跳过\n");
    } else if (args.rmvpe.empty()) {
        opt.with_f0 = false;
        std::printf("[rvc-prep] 警告：未提供 --rmvpe，跳过 f0 提取（G5/G6 将无输出）\n");
    } else {
        if (!ensure_device()) {
            return 1;
        }
        try {
            std::printf("[rvc-prep] f0 后端: %s（device_id=%d）\n", device->name().c_str(),
                        device_id);
            f0_model = std::make_unique<GgufModel>();
            f0_model->load(args.rmvpe, device->backend());
            f0_extractor =
                std::make_unique<rvc::rmvpe::RmvpeExtractor>(*f0_model, device->backend());
            opt.rmvpe = f0_extractor.get();
            opt.with_f0 = true;
        } catch (const std::exception& e) {
            std::printf("[rvc-prep] 失败：加载 RMVPE 模型 %s：%s\n", args.rmvpe.c_str(), e.what());
            return 1;
        }
    }

    // M4-3：HuBERT 特征（与 f0 共用设备；维度由 --version 决定）
    if (args.no_feat) {
        opt.with_feat = false;
        std::printf("[rvc-prep] 特征：--no-feat，跳过\n");
    } else if (args.hubert.empty()) {
        opt.with_feat = false;
        std::printf("[rvc-prep] 警告：未提供 --hubert，跳过特征提取（G7 将无输出）\n");
    } else {
        if (!ensure_device()) {
            return 1;
        }
        try {
            hubert_model = std::make_unique<GgufModel>();
            hubert_model->load(args.hubert, device->backend());
            // Q22：不用 rvc::hubert::HubertEncoder（其 pad1d 未初始化零张量在重复 compute 下
            // 被 gallocr 复用缓冲污染 → 静默漂移）；用本项目自建执行器，每次 compute 前清零 pad。
            hubert_extractor = std::make_unique<rvc_train::prep::HubertFeatureExtractor>(
                *hubert_model, device->backend());
            opt.hubert = hubert_extractor.get();
            opt.with_feat = true;
            const rvc::hubert::Config& hcfg = hubert_extractor->config();
            std::printf(
                "[rvc-prep] HuBERT 后端: %s（device_id=%d；%s；v1=%d 维 / v2=%d 维）\n",
                device->name().c_str(), device_id, opt.version.c_str(), hcfg.final_dim,
                hcfg.hidden);
        } catch (const std::exception& e) {
            std::printf("[rvc-prep] 失败：加载 HuBERT 模型 %s：%s\n", args.hubert.c_str(),
                        e.what());
            return 1;
        }
    }

    rvc_train::prep::PrepFileResult result;
    std::string err;

    if (!args.load_raw.empty()) {
        // 注入模式：跳过解码/重采样
        const int raw_sr = args.load_raw_sr > 0 ? args.load_raw_sr : opt.target_sr;
        rvc_train::prep::MonoAudio load;
        if (!rvc_train::prep::load_raw_f32_mono(args.load_raw, raw_sr, &load, &err)) {
            std::printf("[rvc-prep] 失败: %s\n", err.c_str());
            return 1;
        }
        if (raw_sr != opt.target_sr) {
            load.samples =
                rvc_train::prep::resample_mono_double(load.samples, raw_sr, opt.target_sr);
            load.sample_rate = opt.target_sr;
            std::printf("[rvc-prep] 注入模式：raw sr=%d 重采样到 %d\n", raw_sr, opt.target_sr);
        }
        if (!rvc_train::prep::run_prep_injected(load, case_from_load_raw(args.load_raw), opt,
                                                &result, &err)) {
            std::printf("[rvc-prep] 失败: %s\n", err.c_str());
            return 1;
        }
        std::printf("[rvc-prep] 注入完成: case=%s load=%lld samples 切片=%zu 分块=%zu "
                    "f0=%.3fs feat=%lld帧 spec=%lld帧\n",
                    result.case_name.c_str(), static_cast<long long>(result.load_samples),
                    result.slices.size(), result.chunks.size(),
                    static_cast<double>(result.total_f0_frames) / 100.0,
                    static_cast<long long>(result.total_feat_frames),
                    static_cast<long long>(result.total_spec_frames));
        if (!args.exp.empty()) {
            rvc_train::prep::CacheMeta meta;
            meta.target_sr = opt.target_sr;
            meta.version = opt.version;
            meta.has_f0 = opt.has_f0;
            meta.per = opt.per;
            meta.overlap = opt.overlap;
            meta.slicer = opt.slicer;
            meta.spec = opt.spec;
            std::string cerr;
            if (result.samples.empty() ||
                !rvc_train::prep::write_cache_index(args.exp, meta, result.samples, &cerr)) {
                std::printf("[rvc-prep] 失败：写缓存索引：%s\n",
                            result.samples.empty() ? "无有效样本" : cerr.c_str());
                return 1;
            }
            std::printf("[rvc-prep] 缓存索引已写：%s（样本 %zu）\n", args.exp.c_str(),
                        result.samples.size());
        }
        return 0;
    }

    // 输入文件列表（目录：文件名排序，仅 .wav/.WAV）
    std::vector<std::string> files;
    const fs::path input_path(args.input);
    std::error_code ec;
    if (fs::is_directory(input_path, ec)) {
        std::vector<std::string> names;
        for (const auto& entry : fs::directory_iterator(input_path, ec)) {
            if (entry.is_regular_file(ec)) {
                names.push_back(entry.path().string());
            }
        }
        std::sort(names.begin(), names.end());
        for (const std::string& p : names) {
            std::string ext = fs::path(p).extension().string();
            std::transform(ext.begin(), ext.end(), ext.begin(),
                           [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
            if (ext == ".wav") {
                files.push_back(p);
            } else {
                std::printf("[rvc-prep] 跳过非 WAV: %s\n", p.c_str());
            }
        }
    } else if (fs::is_regular_file(input_path, ec)) {
        files.push_back(input_path.string());
    } else {
        std::printf("[rvc-prep] 输入路径不存在: %s\n", args.input.c_str());
        return 1;
    }

    if (files.empty()) {
        std::printf("[rvc-prep] 未找到可处理的 WAV\n");
        return 1;
    }
    if (args.limit > 0 && static_cast<int>(files.size()) > args.limit) {
        files.resize(static_cast<std::size_t>(args.limit));
    }

    bool all_ok = true;
    std::vector<rvc_train::prep::CacheSampleInfo> all_samples;
    for (const std::string& f : files) {
        const std::string case_name = fs::path(f).stem().string();
        rvc_train::prep::PrepFileResult r;
        std::string e;
        if (!rvc_train::prep::run_prep_wav(f, case_name, opt, &r, &e)) {
            std::printf("[rvc-prep] 失败: %s | %s\n", f.c_str(), e.c_str());
            all_ok = false;
            continue;
        }
        all_samples.insert(all_samples.end(), r.samples.begin(), r.samples.end());
        std::printf(
            "[rvc-prep] %s: load=%lld 切片=%zu 分块=%zu（有效 %zu）gt=%.3fs 16k=%.3fs "
            "f0=%.3fs feat=%lld帧 spec=%lld帧\n",
            case_name.c_str(), static_cast<long long>(r.load_samples), r.slices.size(),
            r.chunks.size(),
            static_cast<std::size_t>(std::count_if(r.chunks.begin(), r.chunks.end(),
                                                  [](const rvc_train::prep::ChunkInfo& c) {
                                                      return c.valid;
                                                  })),
            static_cast<double>(r.total_gt_samples) / opt.target_sr,
            static_cast<double>(r.total_16k_samples) / 16000.0,
            static_cast<double>(r.total_f0_frames) / 100.0,
            static_cast<long long>(r.total_feat_frames),
            static_cast<long long>(r.total_spec_frames));
    }

    if (!args.exp.empty()) {
        if (!all_ok) {
            std::printf("[rvc-prep] 警告：存在失败文件，跳过缓存索引写入（--exp %s）\n",
                        args.exp.c_str());
        } else if (all_samples.empty()) {
            std::printf("[rvc-prep] 失败：无有效样本，缓存索引未写\n");
            return 1;
        } else {
            rvc_train::prep::CacheMeta meta;
            meta.target_sr = opt.target_sr;
            meta.version = opt.version;
            meta.has_f0 = opt.has_f0;
            meta.per = opt.per;
            meta.overlap = opt.overlap;
            meta.slicer = opt.slicer;
            meta.spec = opt.spec;
            std::string cerr;
            if (!rvc_train::prep::write_cache_index(args.exp, meta, all_samples, &cerr)) {
                std::printf("[rvc-prep] 失败：写缓存索引：%s\n", cerr.c_str());
                return 1;
            }
            std::printf("[rvc-prep] 缓存索引已写：%s（样本 %zu）\n", args.exp.c_str(),
                        all_samples.size());
        }
    }
    return all_ok ? 0 : 1;
}

// 读取原始 F32 文件（缓存 samples/*.bin 用）
bool read_raw_f32_file(const std::string& path, std::vector<float>* out, std::string* err) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) {
        if (err) {
            *err = "无法打开: " + path;
        }
        return false;
    }
    const std::streamoff size = f.tellg();
    if (size < 0 || size % 4 != 0) {
        if (err) {
            *err = "F32 raw 大小非法: " + path;
        }
        return false;
    }
    out->resize(static_cast<std::size_t>(size / 4));
    f.seekg(0, std::ios::beg);
    if (size > 0 && !f.read(reinterpret_cast<char*>(out->data()), size)) {
        if (err) {
            *err = "读取失败: " + path;
        }
        return false;
    }
    return true;
}

// 索引构建（默认完整 IVF；`--index-mvp` 回退 MVP 单 list）——特征来源：--features <dir|file.npy> 或 --exp 缓存
int run_index(const Args& args) {
    std::vector<float> feats;
    int64_t rows = 0;
    int64_t dim = 0;
    std::string err;

    if (!args.features.empty()) {
        std::error_code ec;
        const bool ok = fs::is_directory(args.features, ec)
                            ? rvc_train::exporter::load_npy_dir_f32(args.features, &feats, &rows, &dim, &err)
                            : rvc_train::exporter::load_npy_f32(args.features, &feats, &rows, &dim, &err);
        if (!ok) {
            std::printf("[rvc-prep] 失败：加载特征 %s：%s\n", args.features.c_str(), err.c_str());
            return 1;
        }
        std::printf("[rvc-prep] 特征: %lld 行 × %lld 维（%s）\n", static_cast<long long>(rows),
                    static_cast<long long>(dim), args.features.c_str());
    } else if (!args.exp.empty()) {
        rvc_train::common::JsonValue root;
        const std::string ds = (fs::path(args.exp) / "dataset.json").string();
        if (!rvc_train::common::json_read_file(ds, &root, &err)) {
            std::printf("[rvc-prep] 失败：读缓存索引 %s：%s\n", ds.c_str(), err.c_str());
            return 1;
        }
        const rvc_train::common::JsonValue* samples = root.find("samples");
        if (samples == nullptr || !samples->is_array() || samples->size() == 0) {
            std::printf("[rvc-prep] 失败：%s 无 samples\n", ds.c_str());
            return 1;
        }
        int64_t n_used = 0;
        for (std::size_t i = 0; i < samples->size(); ++i) {
            const rvc_train::common::JsonValue& s = samples->at(i);
            const rvc_train::common::JsonValue* files = s.find("files");
            const rvc_train::common::JsonValue* feat = files != nullptr ? files->find("feat") : nullptr;
            if (feat == nullptr || !feat->is_string()) {
                continue;
            }
            const int64_t fdim = s.find("feat_dim") != nullptr ? s.find("feat_dim")->as_int() : 0;
            if (fdim <= 0) {
                continue;
            }
            std::vector<float> part;
            const std::string fp = (fs::path(args.exp) / feat->as_string()).string();
            if (!read_raw_f32_file(fp, &part, &err)) {
                std::printf("[rvc-prep] 失败：%s\n", err.c_str());
                return 1;
            }
            if (part.size() % static_cast<std::size_t>(fdim) != 0) {
                std::printf("[rvc-prep] 失败：%s 元素数与 feat_dim=%lld 不整除\n", fp.c_str(),
                            static_cast<long long>(fdim));
                return 1;
            }
            if (dim == 0) {
                dim = fdim;
            } else if (dim != fdim) {
                std::printf("[rvc-prep] 失败：feat_dim 不一致（%lld vs %lld）\n",
                            static_cast<long long>(dim), static_cast<long long>(fdim));
                return 1;
            }
            rows += static_cast<int64_t>(part.size()) / fdim;
            feats.insert(feats.end(), part.begin(), part.end());
            ++n_used;
        }
        if (n_used == 0 || feats.empty()) {
            std::printf("[rvc-prep] 失败：缓存中没有 feat 文件（需 --with-feat 产出的缓存）\n");
            return 1;
        }
        std::printf("[rvc-prep] 缓存特征: %lld 样本 / %lld 行 × %lld 维（%s）\n",
                    static_cast<long long>(n_used), static_cast<long long>(rows),
                    static_cast<long long>(dim), args.exp.c_str());
    } else {
        std::printf("[rvc-prep] --index 需要 --features <dir|file.npy> 或 --exp <cache_dir>\n");
        return 2;
    }

    if (args.opt.version == "v1" && dim != 256) {
        std::printf("[rvc-prep] 失败：--version v1 需要 256 维特征（当前 %lld）\n",
                    static_cast<long long>(dim));
        return 1;
    }
    if (args.opt.version == "v2" && dim != 768) {
        std::printf("[rvc-prep] 失败：--version v2 需要 768 维特征（当前 %lld）\n",
                    static_cast<long long>(dim));
        return 1;
    }
    if (args.index_out.empty()) {
        std::printf("[rvc-prep] --index 需要 --out <index.gguf>\n");
        return 2;
    }

    rvc_train::exporter::IndexSummary sum;
    if (args.index_mvp) {
        if (!rvc_train::exporter::build_write_index_gguf(args.index_out, feats.data(), rows, dim,
                                                         &sum, &err)) {
            std::printf("[rvc-prep] 失败：写索引 %s：%s\n", args.index_out.c_str(), err.c_str());
            return 1;
        }
        std::printf("[rvc-prep] MVP 索引已写出: %s（dim=%lld nlist=%lld ntotal=%lld nprobe=%lld）\n",
                    args.index_out.c_str(), static_cast<long long>(sum.dim),
                    static_cast<long long>(sum.nlist), static_cast<long long>(sum.ntotal),
                    static_cast<long long>(sum.nprobe));
    } else {
        rvc_train::exporter::IvfBuildOptions io;
        io.nlist = 0;
        io.nprobe = args.nprobe;
        io.seed = args.seed;
        io.kmeans_iters = args.kmeans_iters;
        io.mini_centroids = args.mini_centroids;
        io.mini_epochs = args.mini_epochs;
        io.no_permute = args.no_permute;
        io.verbose = true;
        rvc_train::exporter::IvfBuildStats st;
        if (!rvc_train::exporter::build_write_index_gguf_ivf(args.index_out, &feats, rows, dim, io,
                                                             &sum, &st, &err)) {
            std::printf("[rvc-prep] 失败：写完整 IVF 索引 %s：%s\n", args.index_out.c_str(),
                        err.c_str());
            return 1;
        }
        std::printf(
            "[rvc-prep] IVF 索引已写出: %s（dim=%lld nlist=%lld ntotal=%lld nprobe=%lld"
            " list[min/mean/max]=%lld/%.1f/%lld empty=%lld）\n",
            args.index_out.c_str(), static_cast<long long>(sum.dim),
            static_cast<long long>(sum.nlist), static_cast<long long>(sum.ntotal),
            static_cast<long long>(sum.nprobe), static_cast<long long>(st.list_min),
            static_cast<double>(st.ntotal) / static_cast<double>(st.nlist > 0 ? st.nlist : 1),
            static_cast<long long>(st.list_max), static_cast<long long>(st.list_empty));
        if (st.reduced_centroids > 0) {
            std::printf("[rvc-prep] N>200000：MiniBatchKMeans 已压到 %lld 个中心\n",
                        static_cast<long long>(st.reduced_centroids));
        }
        std::printf("[rvc-prep] 耗时：k-means %.1fs / 总计 %.1fs（seed=%llu）\n",
                    st.kmeans_seconds, st.total_seconds,
                    static_cast<unsigned long long>(args.seed));
    }

    // C++ 自检：CPU 后端 + RVC-GGML 公共读取 API（rvc::RetrievalIndex），并用首行做 top-1 自查询
    try {
        rvc::GgmlDevice dev(99);
        rvc_train::exporter::IndexSummary vs;
        std::string msg;
        if (!rvc_train::exporter::validate_index_gguf(args.index_out, dev.backend(), feats.data(),
                                                    dim, &vs, &msg)) {
            std::printf("[rvc-prep] 失败：索引自检：%s\n", msg.c_str());
            return 1;
        }
        std::printf("[rvc-prep] 索引自检通过: %s\n", msg.c_str());
    } catch (const std::exception& e) {
        std::printf("[rvc-prep] 失败：索引自检异常：%s\n", e.what());
        return 1;
    }
    return 0;
}

// M4-5b：搜索 dump（G10 对拍）——读 index.gguf + queries.npy，逐 query 输出 top-k
// 文件格式（小端）：header = int64 qrows / int64 k / int64 dim；
//   每个 query：int64 count + count × {float score, int64 position}
int run_search_index(const Args& args) {
    if (args.search_queries.empty() || args.search_out.empty()) {
        std::printf("[rvc-prep] --search-index 需要 --queries <file.npy> 与 --search-out <hits.raw>\n");
        return 2;
    }
    std::vector<float> q;
    int64_t qrows = 0;
    int64_t qdim = 0;
    std::string err;
    if (!rvc_train::exporter::load_npy_f32(args.search_queries, &q, &qrows, &qdim, &err)) {
        std::printf("[rvc-prep] 失败：加载 queries %s：%s\n", args.search_queries.c_str(),
                    err.c_str());
        return 1;
    }
    try {
        rvc::GgmlDevice dev(99);  // CPU 足够（RetrievalIndex 加载后纯主机搜索）
        rvc::RetrievalIndex idx;
        idx.load(args.search_index, dev.backend());
        if (!idx.loaded()) {
            std::printf("[rvc-prep] 失败：索引为空 %s\n", args.search_index.c_str());
            return 1;
        }
        if (qdim != idx.dim()) {
            std::printf("[rvc-prep] 失败：queries dim=%lld 与索引 dim=%lld 不一致\n",
                        static_cast<long long>(qdim), static_cast<long long>(idx.dim()));
            return 1;
        }
        std::ofstream f(args.search_out, std::ios::binary);
        if (!f) {
            std::printf("[rvc-prep] 失败：无法写 %s\n", args.search_out.c_str());
            return 1;
        }
        const int64_t hdr[3] = {qrows, static_cast<int64_t>(args.search_k), qdim};
        f.write(reinterpret_cast<const char*>(hdr), sizeof(hdr));
        std::vector<rvc::RetrievalIndex::Hit> hits;
        for (int64_t i = 0; i < qrows; ++i) {
            idx.search(q.data() + i * qdim, args.search_k, hits);
            const int64_t n_hits = static_cast<int64_t>(hits.size());
            f.write(reinterpret_cast<const char*>(&n_hits), sizeof(n_hits));
            for (const rvc::RetrievalIndex::Hit& h : hits) {
                const float s = h.score;
                const int64_t p = h.position;
                f.write(reinterpret_cast<const char*>(&s), sizeof(s));
                f.write(reinterpret_cast<const char*>(&p), sizeof(p));
            }
        }
        if (!f) {
            std::printf("[rvc-prep] 失败：写 %s 时出错\n", args.search_out.c_str());
            return 1;
        }
        std::printf(
            "[rvc-prep] 搜索 dump 完成: %s（queries=%lld k=%d dim=%lld；索引 nlist=%lld ntotal=%lld）\n",
            args.search_out.c_str(), static_cast<long long>(qrows), args.search_k,
            static_cast<long long>(qdim), static_cast<long long>(idx.nlist()),
            static_cast<long long>(idx.ntotal()));
        return 0;
    } catch (const std::exception& e) {
        std::printf("[rvc-prep] 失败：搜索 dump 异常：%s\n", e.what());
        return 1;
    }
}

} // namespace

// M8-2：交互式预处理向导入口——收集参数后执行与 --input 等价的管线，完成后自校验并报告。
int run_interactive_prep(Args& args) {
    rvc_train::prep::PrepWizardResult res;
    std::string werr;
    const bool ok = args.answers_path.empty()
                        ? rvc_train::prep::run_prep_wizard(std::cin, std::cout, &res, &werr)
                        : rvc_train::prep::run_prep_wizard_from_file(args.answers_path, std::cout,
                                                                     &res, &werr);
    if (!ok) {
        std::printf("[rvc-prep] 向导未完成: %s\n", werr.c_str());
        return 1;
    }
    args.input = res.input;
    args.exp = res.exp_dir;
    args.hubert = res.hubert;
    args.rmvpe = res.rmvpe;
    args.opt.version = res.target == 2 ? "v2" : "v1";
    args.opt.target_sr = res.target == 2 ? 48000 : 40000;
    if (res.device >= 0) {
        args.device_set = true;
        args.device_id = res.device;
    }
    const int rc = run_prep(args);
    if (rc != 0) {
        return rc;
    }
    if (!args.exp.empty()) {
        std::string summary;
        std::string verr;
        if (!rvc_train::prep::validate_cache(args.exp, &summary, &verr)) {
            std::printf("[rvc-prep] 缓存自校验失败: %s\n", verr.c_str());
            return 1;
        }
        std::printf("[rvc-prep] 预处理完成，缓存自校验通过: %s\n", summary.c_str());
        std::printf("[rvc-prep] 下一步：运行 rvc-train --interactive 开始训练"
                    "（数据入口选 1，缓存目录填 %s）\n",
                    args.exp.c_str());
    }
    return 0;
}

// M8-3：双击启动（本进程独占控制台）时，退出前等一次回车，避免窗口闪退看不到输出。
// 从已有终端（cmd/PowerShell/脚本）启动时控制台被多个进程共享，不暂停（不阻塞脚本）。
static int pause_if_sole_console(int rc) {
    DWORD pids[2] = {0, 0};
    if (GetConsoleProcessList(pids, 2) == 1) {
        std::printf("\n按回车键退出...");
        std::fflush(stdout);
        std::string line;
        std::getline(std::cin, line);
    }
    return rc;
}

int run_main(int argc, char** argv);
int main(int argc, char** argv) {
    return pause_if_sole_console(run_main(argc, argv));
}

int run_main(int argc, char** argv) {
    SetConsoleOutputCP(65001);
    SetConsoleCP(65001);

    // Q4：与 RVC-GGML 主程序一致，在初始化 ggml 后端之前设置 Intel 驱动规避开关
    // （RVC-GGML src\main.cpp:912-923 为标准做法）：
    //   - 整数点积路径在量化模型多次图执行时可能产生 NaN/零输出 → 默认禁用；
    //   - Vulkan fp16 计算路径显著放大误差（P4 实测 SNR 60dB→33dB）→ 默认禁用。
    // 保留 RVC_VK_ENABLE_INTDOT / RVC_VK_ENABLE_F16 显式恢复语义。
    if (!env_present("RVC_VK_ENABLE_INTDOT")) {
        _putenv_s("GGML_VK_DISABLE_INTEGER_DOT_PRODUCT", "1");
    }
    if (!env_present("RVC_VK_ENABLE_F16")) {
        _putenv_s("GGML_VK_DISABLE_F16", "1");
    }

    Args args;
    const int parse_rc = parse_args(argc, argv, &args);
    if (parse_rc != 0) {
        return parse_rc;
    }

    // M8-2：预处理向导自检（纯结构比较，不执行预处理）
    if (args.selftest_prep_wizard) {
        return rvc_train::prep::run_prep_wizard_selftest();
    }

    // M8-2/M8-3：交互式预处理向导（--interactive / --answers / 无参数双击进入；不影响命令行用法）
    if (args.interactive || argc <= 1) {
        return run_interactive_prep(args);
    }

    // M4-5a：索引自校验（独立模式，不需要 --input）
    if (!args.validate_index.empty()) {
        try {
            rvc::GgmlDevice dev(99);  // CPU 后端足够（RVC-GGML RetrievalIndex 只读张量）
            rvc_train::exporter::IndexSummary sum;
            std::string msg;
            if (!rvc_train::exporter::validate_index_gguf(args.validate_index, dev.backend(), nullptr,
                                                        0, &sum, &msg)) {
                std::printf("[rvc-prep] 索引校验失败: %s\n", msg.c_str());
                return 1;
            }
            std::printf("[rvc-prep] 索引校验通过: %s\n", msg.c_str());
            return 0;
        } catch (const std::exception& e) {
            std::printf("[rvc-prep] 索引校验失败: %s\n", e.what());
            return 1;
        }
    }

    // M4-5b：索引搜索 dump（G10 对拍）
    if (!args.search_index.empty()) {
        return run_search_index(args);
    }

    // M4-5a：索引构建模式
    if (args.index_mode) {
        return run_index(args);
    }

    // M4-4：缓存自校验（G9；独立模式，不需要 --input）
    if (!args.validate_cache.empty()) {
        std::string summary;
        std::string verr;
        if (!rvc_train::prep::validate_cache(args.validate_cache, &summary, &verr)) {
            std::printf("[rvc-prep] 缓存校验失败: %s\n", verr.c_str());
            return 1;
        }
        std::printf("[rvc-prep] 缓存校验通过: %s\n", summary.c_str());
        return 0;
    }

    if (args.has_work()) {
        return run_prep(args);
    }

    print_devices();
    if (args.list_only) {
        return 0;
    }

    if (!args.selftest_m0) {
        // M8-3：M0 冒烟改为显式 --selftest（无参数已进交互向导）
        std::printf("[rvc-prep] 未指定模式。提示:\n"
                    "  rvc-prep（无参数或 --interactive） 交互式预处理向导\n"
                    "  rvc-prep --selftest                M0 设备/图计算自检\n"
                    "  rvc-prep --input <dir|wav> --sr 40000 --version v1 [--exp DIR]\n"
                    "  详见 docs/使用指南.md\n");
        return 2;
    }

    if (!args.device_set) {
        args.device_id = rvc::list_gpu_devices().empty() ? 99 : 0;
    }

    const bool ok = run_selftest(args.device_id);
    if (ok) {
        std::printf("[rvc-prep] M0 自检通过\n");
    } else {
        std::printf("[rvc-prep] M0 自检失败\n");
    }
    return ok ? 0 : 1;
}
