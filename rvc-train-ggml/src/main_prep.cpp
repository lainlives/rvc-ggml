// rvc-prep - RVC training data preprocessing program
//
// Responsibilities (implemented in M4):
//   raw audio → slicing/resampling/high-pass → 16k / f0(RMVPE) / HuBERT features / linear-spectrogram cache
//   reuses RVC-GGML's rvc_core (ggml Vulkan); zero Python dependency during training runs.
//
// Current stage (after M5-0): M4 fully complete (5a MVP + 5b full IVF, see `handover.md` §4.8 K/L);
//   This exe's preprocessing/cache/index commands stay stable; the M5-1 data loader only adds cache **read-side** APIs (output format unchanged).
//   Index: `--index` defaults to full IVF (permutation + optional MiniBatchKMeans + official nlist formula + IVF k-means + list grouping;
//   `--index-mvp` falls back to the M4-5a single list); `--search-index` search dumps are also provided for G10 cross-checks.
//   `--debug-dump` dumps load/hp/gt/16k/f0/feat/spec/slices/chunks for official golden cross-checks (G1–G9;
//   indexes are additionally validated by tools/verify_index_gguf.py, G10).
//
// Usage:
//   rvc-prep (no arguments / --interactive)                 interactive preprocessing wizard (M8; double-click to launch)
//   rvc-prep --list-devices
//   rvc-prep --selftest [--device N]                   device self-test (M0 smoke test; explicit entry since M8-3)
//   rvc-prep --input <dir|wav> --sr 40000 --version v1 [--debug-dump DIR] [--exp DIR] [options]
//   rvc-prep --load-raw <case.load.f32.raw> --load-raw-sr 40000 [--debug-dump DIR]
//   rvc-prep --validate-cache <exp_dir>                cache self-validation (G9)
//   rvc-prep --index --features <dir|file.npy> --out <index.gguf>   full IVF index (M4-5b)
//   rvc-prep --validate-index <index.gguf>             index self-validation (M4-5a/5b)
//   rvc-prep --search-index <index.gguf> --queries <q.npy> --search-out <hits.raw>
//                                                      search dump (for G10 cross-checks)
//
// Exit codes: 0 = success; 1 = failure; 2 = argument error.

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

#ifdef _WIN32
#include <windows.h>
#endif

#include <ggml.h>

#include "common/json.h"
#include "common/platform.h"
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

void print_devices() {
    const std::vector<rvc::GpuDeviceInfo> devs = rvc::list_gpu_devices();
    std::printf("[rvc-prep] ggml GPU device count: %zu\n", devs.size());
    for (const rvc::GpuDeviceInfo& d : devs) {
        std::printf("  [%d] %s | %s\n", d.index, d.name.c_str(), d.description.c_str());
    }
}

// Run a minimal ggml graph on the selected device: c = a + b; validate the result
bool run_selftest(int device_id) {
    rvc::GgmlDevice dev(device_id);
    std::printf("[rvc-prep] backend: %s (device_id=%d)\n", dev.name().c_str(), device_id);

    // Graph/intermediates are managed by GgmlGraphRunner (ctx metadata 4 MiB, at most 64 nodes)
    rvc::GgmlGraphRunner runner(dev.backend(), static_cast<size_t>(4) << 20, 64);
    ggml_context* ctx = runner.ctx();

    ggml_tensor* a = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4);
    ggml_tensor* b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4);
    ggml_tensor* c = ggml_add(ctx, a, b);
    ggml_set_name(a, "a");
    ggml_set_name(b, "b");
    ggml_set_name(c, "c");

    runner.expand(c);  // expand the compute graph
    if (!runner.allocate()) {
        std::printf("[rvc-prep] compute buffer allocation failed\n");
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
    std::printf("[rvc-prep] a+b = [%.1f, %.1f, %.1f, %.1f] (expected [11, 22, 33, 44])\n",
                cv[0], cv[1], cv[2], cv[3]);
    return ok;
}

void print_usage() {
    std::printf(
        "Usage:\n"
        "  rvc-prep (no arguments / --interactive)  interactive preprocessing wizard (M8)\n"
        "  rvc-prep --list-devices\n"
        "  rvc-prep --selftest                 M0 device/graph-compute self-test\n"
        "  rvc-prep [--device N]\n"
        "  rvc-prep --interactive [--answers FILE]      # M8-2 interactive preprocessing wizard\n"
        "  rvc-prep --input <dir|wav> --sr 40000 --version v1 [--debug-dump DIR] [--exp DIR] [--limit N]\n"
        "  rvc-prep --load-raw <file.f32.raw> --load-raw-sr 40000 [--debug-dump DIR]\n"
        "  rvc-prep --validate-cache <exp_dir>\n"
        "  rvc-prep --index --features <dir|file.npy> --out <index.gguf>   # M4-5b full IVF\n"
        "      [--exp <cache>] also usable as the feature source\n"
        "      [--index-mvp]            fall back to M4-5a single list (full scan)\n"
        "      [--nprobe N]             write index.nprobe (default 1, official default)\n"
        "      [--seed N]               permutation/k-means seed (default 1234, reproducible)\n"
        "      [--kmeans-iters N]       IVF k-means iteration cap (default 25)\n"
        "      [--mini-centroids N]     MiniBatchKMeans centroid count when N>200000 (default 10000)\n"
        "      [--mini-epochs N]        MiniBatchKMeans epoch cap (default 10)\n"
        "      [--no-permute]           debug: skip feature-row permutation\n"
        "  rvc-prep --validate-index <index.gguf>\n"
        "  rvc-prep --search-index <index.gguf> --queries <file.npy> --search-out <hits.raw> [--k 8]\n"
        "Options: --exp DIR (training cache output / index feature input) --has-f0 0|1 --per 3.7 --overlap 0.3\n"
        "      --no-16k --no-f0 --no-feat --no-spec\n"
        "      --rmvpe FILE (M4-2: RMVPE weights, enables f0 extraction)\n"
        "      --hubert FILE (M4-3: HuBERT weights, enables feature extraction; --version v1|v2 selects 256/768 dims)\n");
}

struct Args {
    bool list_only = false;
    bool device_set = false;
    int device_id = 0;

    // M8-2: interactive preprocessing wizard
    bool interactive = false;            // --interactive (also entered by double-click with no arguments)
    bool selftest_m0 = false;            // --selftest (M0 smoke test; explicit entry since M8-3)
    std::string answers_path;            // --answers <file> (non-interactive injection)
    bool selftest_prep_wizard = false;   // --selftest-prep-wizard

    std::string input;         // directory or single WAV
    std::string load_raw;      // injection mode
    int load_raw_sr = 0;       // 0 = use --sr
    std::string exp;             // M4-4 cache output directory
    std::string validate_cache;  // M4-4 --validate-cache
    std::string debug_dump;
    // M4-5a: index mode
    bool index_mode = false;         // --index
    std::string features;            // --features <dir|file.npy> (also supports --exp cache)
    std::string index_out;           // --out <index.gguf>
    std::string validate_index;      // --validate-index <index.gguf>
    // M4-5b: full IVF / search dump
    bool index_mvp = false;              // --index-mvp (fall back to M4-5a single list)
    uint32_t nprobe = 1;                 // --nprobe
    uint64_t seed = 1234;                // --seed
    int64_t kmeans_iters = 25;           // --kmeans-iters
    int64_t mini_centroids = 10000;      // --mini-centroids
    int64_t mini_epochs = 10;            // --mini-epochs
    bool no_permute = false;             // --no-permute
    std::string search_index;            // --search-index <index.gguf>
    std::string search_queries;          // --queries <file.npy>
    std::string search_out;              // --search-out <hits.raw>
    int search_k = 8;                    // --k (default 8, matches RVC-GGML pipeline.cpp:103)
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
            std::printf("[rvc-prep] argument %s is missing a value\n", flag);
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
            std::printf("[rvc-prep] unknown argument: %s\n", a);
            print_usage();
            return 2;
        }
    }
    return 0;
}

// Case name for injection mode: `<case>.load.f32.raw` → `<case>`
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
        std::printf("[rvc-prep] failure: --version supports only v1|v2 (current %s)\n", opt.version.c_str());
        return 2;
    }
    // M4-4: training cache output directory (samples/*.bin written by pipeline; index written after all files succeed)
    opt.cache_dir = args.exp;
    if (!args.exp.empty()) {
        std::printf("[rvc-prep] training cache output: %s\n", args.exp.c_str());
    }

    // M4-4: linear-spectrogram config (official config: n_fft=win=2048, hop=sr/100; v1-40k → 400)
    opt.spec.n_fft = 2048;
    opt.spec.win = 2048;
    opt.spec.hop = opt.target_sr / 100;
    opt.spec.sample_rate = opt.target_sr;
    if (args.no_spec) {
        opt.with_spec = false;
        std::printf("[rvc-prep] linear spectrogram: --no-spec, skipped\n");
    } else {
        std::printf("[rvc-prep] linear spectrogram: n_fft=%d hop=%d win=%d bins=%d sr=%d\n",
                    opt.spec.n_fft, opt.spec.hop, opt.spec.win, opt.spec.bins(),
                    opt.spec.sample_rate);
    }

    // M4-2/M4-3: f0 (RMVPE) and HuBERT features share the same GgmlDevice;
    // device/model/extractor/encoder lifetime spans the whole run_prep.
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
            std::printf("[rvc-prep] failure: create device (device_id=%d): %s\n", device_id, e.what());
            return false;
        }
        return true;
    };

    if (args.no_f0) {
        opt.with_f0 = false;
        std::printf("[rvc-prep] f0: --no-f0, skipped\n");
    } else if (args.rmvpe.empty()) {
        opt.with_f0 = false;
        std::printf("[rvc-prep] warning: --rmvpe not provided, skipping f0 extraction (G5/G6 will have no output)\n");
    } else {
        if (!ensure_device()) {
            return 1;
        }
        try {
            std::printf("[rvc-prep] f0 backend: %s (device_id=%d)\n", device->name().c_str(),
                        device_id);
            f0_model = std::make_unique<GgufModel>();
            f0_model->load(args.rmvpe, device->backend());
            f0_extractor =
                std::make_unique<rvc::rmvpe::RmvpeExtractor>(*f0_model, device->backend());
            opt.rmvpe = f0_extractor.get();
            opt.with_f0 = true;
        } catch (const std::exception& e) {
            std::printf("[rvc-prep] failure: load RMVPE model %s: %s\n", args.rmvpe.c_str(), e.what());
            return 1;
        }
    }

    // M4-3: HuBERT features (share the device with f0; dims selected by --version)
    if (args.no_feat) {
        opt.with_feat = false;
        std::printf("[rvc-prep] features: --no-feat, skipped\n");
    } else if (args.hubert.empty()) {
        opt.with_feat = false;
        std::printf("[rvc-prep] warning: --hubert not provided, skipping feature extraction (G7 will have no output)\n");
    } else {
        if (!ensure_device()) {
            return 1;
        }
        try {
            hubert_model = std::make_unique<GgufModel>();
            hubert_model->load(args.hubert, device->backend());
            // Q22: do not use rvc::hubert::HubertEncoder (its pad1d uninitialized zero tensor is
            // polluted by the gallocr reused buffer under repeated compute → silent drift); use this project's own executor, which zeroes pad before each compute.
            hubert_extractor = std::make_unique<rvc_train::prep::HubertFeatureExtractor>(
                *hubert_model, device->backend());
            opt.hubert = hubert_extractor.get();
            opt.with_feat = true;
            const rvc::hubert::Config& hcfg = hubert_extractor->config();
            std::printf(
                "[rvc-prep] HuBERT backend: %s (device_id=%d; %s; v1=%d dims / v2=%d dims)\n",
                device->name().c_str(), device_id, opt.version.c_str(), hcfg.final_dim,
                hcfg.hidden);
        } catch (const std::exception& e) {
            std::printf("[rvc-prep] failure: load HuBERT model %s: %s\n", args.hubert.c_str(),
                        e.what());
            return 1;
        }
    }

    rvc_train::prep::PrepFileResult result;
    std::string err;

    if (!args.load_raw.empty()) {
        // injection mode: skip decoding/resampling
        const int raw_sr = args.load_raw_sr > 0 ? args.load_raw_sr : opt.target_sr;
        rvc_train::prep::MonoAudio load;
        if (!rvc_train::prep::load_raw_f32_mono(args.load_raw, raw_sr, &load, &err)) {
            std::printf("[rvc-prep] failure: %s\n", err.c_str());
            return 1;
        }
        if (raw_sr != opt.target_sr) {
            load.samples =
                rvc_train::prep::resample_mono_double(load.samples, raw_sr, opt.target_sr);
            load.sample_rate = opt.target_sr;
            std::printf("[rvc-prep] injection mode: raw sr=%d resampled to %d\n", raw_sr, opt.target_sr);
        }
        if (!rvc_train::prep::run_prep_injected(load, case_from_load_raw(args.load_raw), opt,
                                                &result, &err)) {
            std::printf("[rvc-prep] failure: %s\n", err.c_str());
            return 1;
        }
        std::printf("[rvc-prep] injection done: case=%s load=%lld samples, slices=%zu chunks=%zu "
                    "f0=%.3fs feat=%lld frames spec=%lld frames\n",
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
                std::printf("[rvc-prep] failure: write cache index: %s\n",
                            result.samples.empty() ? "no valid samples" : cerr.c_str());
                return 1;
            }
            std::printf("[rvc-prep] cache index written: %s (%zu samples)\n", args.exp.c_str(),
                        result.samples.size());
        }
        return 0;
    }

    // Input file list (directory: sort by filename, only .wav/.WAV)
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
                std::printf("[rvc-prep] skipping non-WAV: %s\n", p.c_str());
            }
        }
    } else if (fs::is_regular_file(input_path, ec)) {
        files.push_back(input_path.string());
    } else {
        std::printf("[rvc-prep] input path does not exist: %s\n", args.input.c_str());
        return 1;
    }

    if (files.empty()) {
        std::printf("[rvc-prep] no processable WAV found\n");
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
            std::printf("[rvc-prep] failure: %s | %s\n", f.c_str(), e.c_str());
            all_ok = false;
            continue;
        }
        all_samples.insert(all_samples.end(), r.samples.begin(), r.samples.end());
        std::printf(
            "[rvc-prep] %s: load=%lld slices=%zu chunks=%zu (valid %zu) gt=%.3fs 16k=%.3fs "
            "f0=%.3fs feat=%lld frames spec=%lld frames\n",
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
            std::printf("[rvc-prep] warning: failed files present, skipping cache index write (--exp %s)\n",
                        args.exp.c_str());
        } else if (all_samples.empty()) {
            std::printf("[rvc-prep] failure: no valid samples, cache index not written\n");
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
                std::printf("[rvc-prep] failure: write cache index: %s\n", cerr.c_str());
                return 1;
            }
            std::printf("[rvc-prep] cache index written: %s (%zu samples)\n", args.exp.c_str(),
                        all_samples.size());
        }
    }
    return all_ok ? 0 : 1;
}

// Read a raw F32 file (for cache samples/*.bin)
bool read_raw_f32_file(const std::string& path, std::vector<float>* out, std::string* err) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) {
        if (err) {
            *err = "cannot open: " + path;
        }
        return false;
    }
    const std::streamoff size = f.tellg();
    if (size < 0 || size % 4 != 0) {
        if (err) {
            *err = "invalid F32 raw size: " + path;
        }
        return false;
    }
    out->resize(static_cast<std::size_t>(size / 4));
    f.seekg(0, std::ios::beg);
    if (size > 0 && !f.read(reinterpret_cast<char*>(out->data()), size)) {
        if (err) {
            *err = "read failed: " + path;
        }
        return false;
    }
    return true;
}

// Index build (full IVF by default; `--index-mvp` falls back to MVP single list) -- feature source: --features <dir|file.npy> or --exp cache
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
            std::printf("[rvc-prep] failure: load features %s: %s\n", args.features.c_str(), err.c_str());
            return 1;
        }
        std::printf("[rvc-prep] features: %lld rows x %lld dims (%s)\n", static_cast<long long>(rows),
                    static_cast<long long>(dim), args.features.c_str());
    } else if (!args.exp.empty()) {
        rvc_train::common::JsonValue root;
        const std::string ds = (fs::path(args.exp) / "dataset.json").string();
        if (!rvc_train::common::json_read_file(ds, &root, &err)) {
            std::printf("[rvc-prep] failure: read cache index %s: %s\n", ds.c_str(), err.c_str());
            return 1;
        }
        const rvc_train::common::JsonValue* samples = root.find("samples");
        if (samples == nullptr || !samples->is_array() || samples->size() == 0) {
            std::printf("[rvc-prep] failure: %s has no samples\n", ds.c_str());
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
                std::printf("[rvc-prep] failure: %s\n", err.c_str());
                return 1;
            }
            if (part.size() % static_cast<std::size_t>(fdim) != 0) {
                std::printf("[rvc-prep] failure: %s element count not divisible by feat_dim=%lld\n", fp.c_str(),
                            static_cast<long long>(fdim));
                return 1;
            }
            if (dim == 0) {
                dim = fdim;
            } else if (dim != fdim) {
                std::printf("[rvc-prep] failure: feat_dim mismatch (%lld vs %lld)\n",
                            static_cast<long long>(dim), static_cast<long long>(fdim));
                return 1;
            }
            rows += static_cast<int64_t>(part.size()) / fdim;
            feats.insert(feats.end(), part.begin(), part.end());
            ++n_used;
        }
        if (n_used == 0 || feats.empty()) {
            std::printf("[rvc-prep] failure: cache contains no feat files (needs a cache produced with --with-feat)\n");
            return 1;
        }
        std::printf("[rvc-prep] cache features: %lld samples / %lld rows x %lld dims (%s)\n",
                    static_cast<long long>(n_used), static_cast<long long>(rows),
                    static_cast<long long>(dim), args.exp.c_str());
    } else {
        std::printf("[rvc-prep] --index needs --features <dir|file.npy> or --exp <cache_dir>\n");
        return 2;
    }

    if (args.opt.version == "v1" && dim != 256) {
        std::printf("[rvc-prep] failure: --version v1 requires 256-dim features (current %lld)\n",
                    static_cast<long long>(dim));
        return 1;
    }
    if (args.opt.version == "v2" && dim != 768) {
        std::printf("[rvc-prep] failure: --version v2 requires 768-dim features (current %lld)\n",
                    static_cast<long long>(dim));
        return 1;
    }
    if (args.index_out.empty()) {
        std::printf("[rvc-prep] --index needs --out <index.gguf>\n");
        return 2;
    }

    rvc_train::exporter::IndexSummary sum;
    if (args.index_mvp) {
        if (!rvc_train::exporter::build_write_index_gguf(args.index_out, feats.data(), rows, dim,
                                                         &sum, &err)) {
            std::printf("[rvc-prep] failure: write index %s: %s\n", args.index_out.c_str(), err.c_str());
            return 1;
        }
        std::printf("[rvc-prep] MVP index written: %s (dim=%lld nlist=%lld ntotal=%lld nprobe=%lld)\n",
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
            std::printf("[rvc-prep] failure: write full IVF index %s: %s\n", args.index_out.c_str(),
                        err.c_str());
            return 1;
        }
        std::printf(
            "[rvc-prep] IVF index written: %s (dim=%lld nlist=%lld ntotal=%lld nprobe=%lld"
            " list[min/mean/max]=%lld/%.1f/%lld empty=%lld)\n",
            args.index_out.c_str(), static_cast<long long>(sum.dim),
            static_cast<long long>(sum.nlist), static_cast<long long>(sum.ntotal),
            static_cast<long long>(sum.nprobe), static_cast<long long>(st.list_min),
            static_cast<double>(st.ntotal) / static_cast<double>(st.nlist > 0 ? st.nlist : 1),
            static_cast<long long>(st.list_max), static_cast<long long>(st.list_empty));
        if (st.reduced_centroids > 0) {
            std::printf("[rvc-prep] N>200000: MiniBatchKMeans reduced to %lld centroids\n",
                        static_cast<long long>(st.reduced_centroids));
        }
        std::printf("[rvc-prep] elapsed: k-means %.1fs / total %.1fs (seed=%llu)\n",
                    st.kmeans_seconds, st.total_seconds,
                    static_cast<unsigned long long>(args.seed));
    }

    // C++ self-check: CPU backend + RVC-GGML public read API (rvc::RetrievalIndex), plus a top-1 self-query with the first row
    try {
        rvc::GgmlDevice dev(99);
        rvc_train::exporter::IndexSummary vs;
        std::string msg;
        if (!rvc_train::exporter::validate_index_gguf(args.index_out, dev.backend(), feats.data(),
                                                    dim, &vs, &msg)) {
            std::printf("[rvc-prep] failure: index self-check: %s\n", msg.c_str());
            return 1;
        }
        std::printf("[rvc-prep] index self-check passed: %s\n", msg.c_str());
    } catch (const std::exception& e) {
        std::printf("[rvc-prep] failure: index self-check exception: %s\n", e.what());
        return 1;
    }
    return 0;
}

// M4-5b: search dump (G10 cross-check) -- read index.gguf + queries.npy, output top-k per query
// File format (little-endian): header = int64 qrows / int64 k / int64 dim;
//   per query: int64 count + count x {float score, int64 position}
int run_search_index(const Args& args) {
    if (args.search_queries.empty() || args.search_out.empty()) {
        std::printf("[rvc-prep] --search-index needs --queries <file.npy> and --search-out <hits.raw>\n");
        return 2;
    }
    std::vector<float> q;
    int64_t qrows = 0;
    int64_t qdim = 0;
    std::string err;
    if (!rvc_train::exporter::load_npy_f32(args.search_queries, &q, &qrows, &qdim, &err)) {
        std::printf("[rvc-prep] failure: load queries %s: %s\n", args.search_queries.c_str(),
                    err.c_str());
        return 1;
    }
    try {
        rvc::GgmlDevice dev(99);  // CPU suffices (host-only search after RetrievalIndex load)
        rvc::RetrievalIndex idx;
        idx.load(args.search_index, dev.backend());
        if (!idx.loaded()) {
            std::printf("[rvc-prep] failure: empty index %s\n", args.search_index.c_str());
            return 1;
        }
        if (qdim != idx.dim()) {
            std::printf("[rvc-prep] failure: queries dim=%lld does not match index dim=%lld\n",
                        static_cast<long long>(qdim), static_cast<long long>(idx.dim()));
            return 1;
        }
        std::ofstream f(args.search_out, std::ios::binary);
        if (!f) {
            std::printf("[rvc-prep] failure: cannot write %s\n", args.search_out.c_str());
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
            std::printf("[rvc-prep] failure: error while writing %s\n", args.search_out.c_str());
            return 1;
        }
        std::printf(
            "[rvc-prep] search dump done: %s (queries=%lld k=%d dim=%lld; index nlist=%lld ntotal=%lld)\n",
            args.search_out.c_str(), static_cast<long long>(qrows), args.search_k,
            static_cast<long long>(qdim), static_cast<long long>(idx.nlist()),
            static_cast<long long>(idx.ntotal()));
        return 0;
    } catch (const std::exception& e) {
        std::printf("[rvc-prep] failure: search dump exception: %s\n", e.what());
        return 1;
    }
}

} // namespace

// M8-2: interactive preprocessing wizard entry -- collect parameters, run the pipeline equivalent to --input, then self-validate and report.
int run_interactive_prep(Args& args) {
    rvc_train::prep::PrepWizardResult res;
    std::string werr;
    const bool ok = args.answers_path.empty()
                        ? rvc_train::prep::run_prep_wizard(std::cin, std::cout, &res, &werr)
                        : rvc_train::prep::run_prep_wizard_from_file(args.answers_path, std::cout,
                                                                     &res, &werr);
    if (!ok) {
        std::printf("[rvc-prep] wizard not completed: %s\n", werr.c_str());
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
            std::printf("[rvc-prep] cache self-validation failed: %s\n", verr.c_str());
            return 1;
        }
        std::printf("[rvc-prep] preprocessing complete, cache self-validation passed: %s\n", summary.c_str());
        std::printf("[rvc-prep] next step: run rvc-train --interactive to start training"
                    " (choose data entry 1, fill in cache directory %s)\n",
                    args.exp.c_str());
    }
    return 0;
}

// M8-3: when launched by double-click (this process owns the console), wait for one Enter before exit so the window does not flash-close.
// When launched from an existing terminal (cmd/PowerShell/script), the console is shared by multiple processes, so do not pause (does not block scripts).
static int pause_if_sole_console(int rc) {
#ifdef _WIN32
    // Pause only when this process owns the console (double-click launch), so scripts are not blocked.
    DWORD pids[2] = {0, 0};
    if (GetConsoleProcessList(pids, 2) == 1) {
        std::printf("\nPress Enter to exit...");
        std::fflush(stdout);
        std::string line;
        std::getline(std::cin, line);
    }
#endif
    return rc;
}

int run_main(int argc, char** argv);
int main(int argc, char** argv) {
    return pause_if_sole_console(run_main(argc, argv));
}

int run_main(int argc, char** argv) {
#ifdef _WIN32
    SetConsoleOutputCP(65001);
    SetConsoleCP(65001);
#endif

    // Q4: same as the RVC-GGML main program, set Intel driver workaround switches before initializing the ggml backend
    // (RVC-GGML src\main.cpp:912-923 is the standard practice):
    //   - the integer dot product path may produce NaN/zero output on repeated graph executions of quantized models → disabled by default;
    //   - the Vulkan fp16 compute path significantly amplifies error (P4 measured SNR 60dB→33dB) → disabled by default.
    // RVC_VK_ENABLE_INTDOT / RVC_VK_ENABLE_F16 keep their explicit restore semantics.
    if (!rvc_train::platform::env_present("RVC_VK_ENABLE_INTDOT")) {
        rvc_train::platform::env_set("GGML_VK_DISABLE_INTEGER_DOT_PRODUCT", "1");
    }
    if (!rvc_train::platform::env_present("RVC_VK_ENABLE_F16")) {
        rvc_train::platform::env_set("GGML_VK_DISABLE_F16", "1");
    }

    Args args;
    const int parse_rc = parse_args(argc, argv, &args);
    if (parse_rc != 0) {
        return parse_rc;
    }

    // M8-2: preprocessing wizard self-test (structural comparison only, no preprocessing run)
    if (args.selftest_prep_wizard) {
        return rvc_train::prep::run_prep_wizard_selftest();
    }

    // M8-2/M8-3: interactive preprocessing wizard (--interactive / --answers / double-click with no arguments; command-line usage unaffected)
    if (args.interactive || argc <= 1) {
        return run_interactive_prep(args);
    }

    // M4-5a: index self-validation (standalone mode, --input not needed)
    if (!args.validate_index.empty()) {
        try {
            rvc::GgmlDevice dev(99);  // CPU backend suffices (RVC-GGML RetrievalIndex reads tensors only)
            rvc_train::exporter::IndexSummary sum;
            std::string msg;
            if (!rvc_train::exporter::validate_index_gguf(args.validate_index, dev.backend(), nullptr,
                                                        0, &sum, &msg)) {
                std::printf("[rvc-prep] index validation failed: %s\n", msg.c_str());
                return 1;
            }
            std::printf("[rvc-prep] index validation passed: %s\n", msg.c_str());
            return 0;
        } catch (const std::exception& e) {
            std::printf("[rvc-prep] index validation failed: %s\n", e.what());
            return 1;
        }
    }

    // M4-5b: index search dump (G10 cross-check)
    if (!args.search_index.empty()) {
        return run_search_index(args);
    }

    // M4-5a: index build mode
    if (args.index_mode) {
        return run_index(args);
    }

    // M4-4: cache self-validation (G9; standalone mode, --input not needed)
    if (!args.validate_cache.empty()) {
        std::string summary;
        std::string verr;
        if (!rvc_train::prep::validate_cache(args.validate_cache, &summary, &verr)) {
            std::printf("[rvc-prep] cache validation failed: %s\n", verr.c_str());
            return 1;
        }
        std::printf("[rvc-prep] cache validation passed: %s\n", summary.c_str());
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
        // M8-3: M0 smoke test is now explicit via --selftest (no arguments enters the interactive wizard)
        std::printf("[rvc-prep] no mode specified. Hints:\n"
                    "  rvc-prep (no arguments or --interactive) interactive preprocessing wizard\n"
                    "  rvc-prep --selftest                M0 device/graph-compute self-test\n"
                    "  rvc-prep --input <dir|wav> --sr 40000 --version v1 [--exp DIR]\n"
                    "  see docs/usage_guide.md for details\n");
        return 2;
    }

    if (!args.device_set) {
        args.device_id = rvc::list_gpu_devices().empty() ? 99 : 0;
    }

    const bool ok = run_selftest(args.device_id);
    if (ok) {
        std::printf("[rvc-prep] M0 self-test passed\n");
    } else {
        std::printf("[rvc-prep] M0 self-test failed\n");
    }
    return ok ? 0 : 1;
}
