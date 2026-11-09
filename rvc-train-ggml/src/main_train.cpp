// rvc-train - RVC training program (M1 load + M2 forward + M3 backward/loss/dual-graph cross-check)
//
// Responsibilities (M2-M3 implemented; the training loop belongs to M5):
//   Assemble the RVC model (enc_p/enc_q/flow/dec/discriminator) -> dual-graph G/D + AdamW training -> export inference GGUF
//   Links traincpp (Vulkan training; CPU is reference/debug only).
//
// Currently supported modes:
//   --list-devices                list traincpp devices only
//   --device N                    select device N (default: prefer GPU -> IGPU -> CPU)
//   --validate-gguf <file>        validate a file with the train.cpp GGUF structure self-check
//   --load-gguf <file>            load training-state GGUF (rvc.train.*): create tensors by name -> allocate ->
//                                 load all -> per-tensor CRC32 -> negative tests (missing/shape mismatch)
//   --dump-hashes <file>          with --load-gguf: write a per-tensor CRC manifest
//                                 (name ne0 ne1 ne2 ne3 nbytes crc32), for comparison by verify_train_gguf.py
//   --selftest-forward            model skeleton/forward cross-check (M2): --weights <train.gguf> [--golden <dir>] [--module <name>]
//                                 without --module: M2-1 skeleton self-test (build params + check_expected + manifest summary)
//                                 with --module: run the runtime golden per-module cross-check
//                                 (emb_g/wn/resblock1/dec/flow/enc_p/enc_q/synth/all; disc requires D weights)
//   --selftest-backward           backward cross-check (M3-1, fixed cotangent VJP):
//                                 --weights <train.gguf> --golden <m3 dir> [--module <name>] [--gradcheck SPEC]
//                                 SPEC = "param-name[:element-index];..." or "all" (first/middle element of every param)
//   --selftest-loss               loss/frontend cross-check (M3-2, official 81eed5e training semantics):
//                                 --golden <m3 dir> [--module <stft|mel|...|all>] (no --weights needed)
//   --selftest-train-step         dual-graph G/D + one AdamW step cross-check (M3-3):
//                                 --weights <G train.gguf> --d-weights <D train.gguf> --golden <m3 dir>
//   --selftest-train-loop         training-loop 10-step trajectory cross-check (M5-2, single-ctx TrainStep):
//                                 --weights <G> --d-weights <D> --golden <m5loop dir>
//   --selftest-trainer            training-loop self-test (M5-3, Trainer + checkpoint save->load round trip):
//                                 --weights <G> --d-weights <D> --exp <cache> --out <dir> --steps N
//                                 [--bucket T] [--batch N] [--disc-len N] [--config <json> may supply a v2 cache]
//   --train <config.json>         full training (M5-3: loop/log/checkpoint/ExponentialLR; M7: run section/scratch):
//                                 [--exp <cache>] [--weights <G>|--pretrain-g <G>] [--d-weights <D>|--pretrain-d <D>]
//                                 [--out <dir>] [--scratch]
//                                 [--device N] [--steps N] [--bucket T] [--batch N] [--epochs N]
//                                 [--disc-len N] [--resume <ckpt>] [--export-run-json <path>]
//   --interactive                 M7 interactive training wizard (finetune / from scratch; after confirmation writes
//                                 <out>/train_config.json and starts training immediately):
//                                 [--config <base.json>] [--answers <file>] [--out DIR] [--device N]
//                                 [--exp DIR] [--weights G] [--d-weights D] [--steps N] ...
//   --selftest-wizard             M7-1 wizard equivalence self-test (G-7.1): built-in answers config vs hand-written JSON
//   --selftest-scratch            M7-2 from-scratch spec self-test (G-7.2): spec vs M1 hashes + generated GGUF validation
//                                 [--config <json>] [--out DIR] [--g-hashes <f>] [--d-hashes <f>]
//   --selftest-data               data loader cross-check (M5-1, host-only, no device/weights needed):
//                                 --golden <m5 dir> --exp <M4 cache dir>
//   --spike-train-ctx             single-ctx shared-param dual-graph spike (M5-2):
//                                 --weights <G> --d-weights <D> --exp <cache> [--bucket T] [--steps N]
//                                 (G-5.2a: two buffer_alloc_graph_tensors + measured memory/per-step loss)
//   --selftest                    device enumeration + graph compute self-test (M0 smoke; explicit entry since M8-3)
//   no arguments / --interactive        M8 interactive training wizard (double-click enters the wizard; waits for Enter before exit so the window does not flash-close)
//
// Exit codes: 0 = success; 1 = failure; 2 = argument error.

#include "traincpp/traincpp.h"

#include "cli/train_wizard.h"
#include "common/platform.h"
#include "export/mem_to_train_gguf.h"
#include "export/model_gguf.h"

#include "model/config.h"
#include "model/model.h"
#include "model/selftest.h"
#include "model/weights.h"
#include "train/dataset_selftest.h"
#include "train/grad_accum_selftest.h"
#include "train/loss_selftest.h"
#include "train/resume_selftest.h"
#include "train/scratch_selftest.h"
#include "train/train_step.h"
#include "train/train_step_selftest.h"
#include "train/trainer.h"
#include "train/trainer_selftest.h"

#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#ifdef _WIN32
// For console output code page switching (same as main_prep.cpp); NOMINMAX avoids the windows.h min/max macros
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

using namespace traincpp;

namespace {

// ---------------------------------------------------------------- CRC32 (compatible with zlib.crc32)

uint32_t crc32_update(uint32_t crc, const uint8_t* p, size_t n) {
    static uint32_t table[256];
    static bool     init = false;
    if (!init) {
        for (uint32_t i = 0; i < 256; ++i) {
            uint32_t c = i;
            for (int k = 0; k < 8; ++k) {
                c = (c & 1u) ? (0xEDB88320u ^ (c >> 1)) : (c >> 1);
            }
            table[i] = c;
        }
        init = true;
    }
    for (size_t i = 0; i < n; ++i) {
        crc = table[(crc ^ p[i]) & 0xFFu] ^ (crc >> 8);
    }
    return crc;
}

uint32_t crc32_tensor(const Tensor* t) {
    const int64_t total = tensor_nelements(t) * 4;  // F32
    std::vector<uint8_t> buf((size_t) (total < (1 << 20) ? total : (1 << 20)));
    uint32_t crc = 0xFFFFFFFFu;
    int64_t offset = 0;
    while (offset < total) {
        const size_t chunk = (size_t) ((total - offset) < (int64_t) buf.size() ? (total - offset)
                                                                               : (int64_t) buf.size());
        tensor_get(t, buf.data(), (size_t) offset, chunk);
        crc = crc32_update(crc, buf.data(), chunk);
        offset += (int64_t) chunk;
    }
    return crc ^ 0xFFFFFFFFu;
}

// ---------------------------------------------------------------- Devices

const char* device_type_name(DeviceType t) {
    switch (t) {
        case DeviceType::CPU:   return "CPU";
        case DeviceType::GPU:   return "GPU";
        case DeviceType::IGPU:  return "IGPU";
        case DeviceType::ACCEL: return "ACCEL";
    }
    return "?";
}

void print_devices() {
    const size_t n = device_count();
    std::printf("[rvc-train] traincpp device count: %zu\n", n);
    for (size_t i = 0; i < n; ++i) {
        const Device* d = device_get(i);
        const DeviceProps& p = d->props();
        std::printf("  [%zu] %s | %s | type=%s | free=%.1f MiB | total=%.1f MiB | unified=%d\n",
                    i, p.name.c_str(), p.description.c_str(), device_type_name(p.type),
                    (double) p.memory_free / (1024.0 * 1024.0),
                    (double) p.memory_total / (1024.0 * 1024.0),
                    p.unified_memory ? 1 : 0);
    }
}

Device* pick_device(int device_index, bool device_set) {
    if (device_set) {
        if (device_index < 0 || (size_t) device_index >= device_count()) {
            std::printf("[rvc-train] device index out of range: %d (of %zu)\n", device_index, device_count());
            return nullptr;
        }
        return device_get((size_t) device_index);
    }
    Device* dev = device_by_type(DeviceType::GPU);
    if (dev == nullptr) {
        dev = device_by_type(DeviceType::IGPU);
    }
    if (dev == nullptr) {
        dev = device_cpu();
    }
    return dev;
}

std::string join_path_simple(const std::string& dir, const std::string& name) {
    if (dir.empty()) {
        return name;
    }
    const char last = dir.back();
    if (last == '/' || last == '\\') {
        return dir + name;
    }
    return dir + rvc_train::platform::path_sep() + name;
}

// Create a Trainer from the config and run the full loop (shared by --train and --interactive).
// Device selection: explicit CLI --device first, then config.run.device (>=0), otherwise auto.
int run_train_from_config(const rvc_train::train::TrainerConfig& cfg, int device_index,
                          bool device_set) {
    const bool use_device = device_set || cfg.device >= 0;
    const int idx = device_set ? device_index : (int) cfg.device;
    Device* dev = pick_device(idx, use_device);
    if (dev == nullptr) {
        std::printf("[rvc-train] no usable training device\n");
        return 1;
    }
    std::string terr;
    rvc_train::train::Trainer* trainer = rvc_train::train::Trainer::create(cfg, dev, &terr);
    if (trainer == nullptr) {
        std::printf("[rvc-train] Trainer creation failed: %s\n", terr.c_str());
        return 1;
    }
    const bool ok = trainer->run(&terr);
    delete trainer;
    if (!ok) {
        std::printf("[rvc-train] training failed: %s\n", terr.c_str());
        return 1;
    }
    return 0;
}

// M7-2: for from-scratch training, generate a randomly initialized training-state GGUF per model_cfg and fill in weights_g/weights_d.
bool ensure_scratch_weights(rvc_train::train::TrainerConfig& cfg, std::string* err) {
    if (!cfg.scratch) {
        return true;
    }
    rvc_train::model::ModelConfig g = cfg.model_cfg;
    g.kind = "G";
    rvc_train::model::ModelConfig d = cfg.model_cfg;
    d.kind = "D";
    const std::string dir = join_path_simple(cfg.out_dir, "scratch");
    const std::string gp = join_path_simple(dir, "g.train.gguf");
    const std::string dp = join_path_simple(dir, "d.train.gguf");
    if (!rvc_train::exporter::generate_scratch_g(g, (uint64_t) cfg.seed, gp, err)) {
        return false;
    }
    if (!rvc_train::exporter::generate_scratch_d(d, (uint64_t) cfg.seed + 1, dp, err)) {
        return false;
    }
    cfg.weights_g = gp;
    cfg.weights_d = dp;
    std::printf("[rvc-train] scratch init generated: %s / %s\n", gp.c_str(), dp.c_str());
    return true;
}

// ---------------------------------------------------------------- M0 smoke: c = a + b

bool run_selftest(Device* dev) {
    if (dev == nullptr) {
        std::printf("[rvc-train] no usable device\n");
        return false;
    }
    const DeviceProps& p = dev->props();
    std::printf("[rvc-train] backend: %s (%s)\n", p.name.c_str(), device_type_name(p.type));

    Context* ctx = context_new(1 << 20);
    if (ctx == nullptr) {
        std::printf("[rvc-train] context_new failed\n");
        return false;
    }

    Tensor* a = new_tensor_1d(ctx, TYPE_F32, 4);
    Tensor* b = new_tensor_1d(ctx, TYPE_F32, 4);
    Tensor* c = add(ctx, a, b);
    std::snprintf(a->name, MAX_NAME, "a");
    std::snprintf(b->name, MAX_NAME, "b");
    std::snprintf(c->name, MAX_NAME, "c");

    Graph* g = graph_new(ctx);
    graph_build_forward_expand(ctx, g, c);

    Buffer* buf = buffer_alloc_ctx_tensors(ctx, dev->default_buffer_type());
    if (buf == nullptr) {
        std::printf("[rvc-train] compute buffer allocation failed\n");
        graph_free(g);
        context_free(ctx);
        return false;
    }

    const float av[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    const float bv[4] = {10.0f, 20.0f, 30.0f, 40.0f};
    tensor_set(a, av, 0, sizeof(av));
    tensor_set(b, bv, 0, sizeof(bv));

    const Tensor* unsupported = graph_first_unsupported(g, dev);
    if (unsupported != nullptr) {
        std::printf("[rvc-train] device does not support node: %s\n", unsupported->name);
        buffer_free(buf);
        graph_free(g);
        context_free(ctx);
        return false;
    }

    if (!dev->graph_compute(g)) {
        std::printf("[rvc-train] graph_compute failed\n");
        buffer_free(buf);
        graph_free(g);
        context_free(ctx);
        return false;
    }

    float cv[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    tensor_get(c, cv, 0, sizeof(cv));

    const float expect[4] = {11.0f, 22.0f, 33.0f, 44.0f};
    bool ok = true;
    for (int i = 0; i < 4; ++i) {
        if (std::fabs(cv[i] - expect[i]) > 1e-6f) {
            ok = false;
        }
    }
    std::printf("[rvc-train] a+b = [%.1f, %.1f, %.1f, %.1f] (expected [11, 22, 33, 44])\n",
                cv[0], cv[1], cv[2], cv[3]);

    buffer_free(buf);
    graph_free(g);
    context_free(ctx);
    return ok;
}

// ---------------------------------------------------------------- training-state metadata printing (shared by M1/M2)

void print_train_meta(const rvc_train::model::TrainMeta& meta, const char* tag) {
    std::printf("[%s] kind=%s version=%s has_f0=%d form=%s dtype=%s tensors=%lld\n", tag,
                meta.kind.c_str(), meta.version.c_str(), meta.has_f0 ? 1 : 0, meta.form.c_str(),
                meta.out_dtype.c_str(), (long long) meta.tensor_count);
    std::printf("[%s] sr=%lld spec_channels=%lld filter=%lld hop=%lld win=%lld n_mel=%lld segment=%lld\n",
                tag, (long long) meta.sample_rate, (long long) meta.spec_channels,
                (long long) meta.filter_length, (long long) meta.hop_length,
                (long long) meta.win_length, (long long) meta.n_mel_channels,
                (long long) meta.segment_size);
    std::printf("[%s] inter=%lld hidden=%lld filter_ch=%lld heads=%lld layers=%lld gin=%lld spk=%lld up_init=%lld\n",
                tag, (long long) meta.inter_channels, (long long) meta.hidden_channels,
                (long long) meta.filter_channels, (long long) meta.n_heads,
                (long long) meta.n_layers, (long long) meta.gin_channels,
                (long long) meta.spk_embed_dim, (long long) meta.upsample_initial_channel);
    std::printf("[%s] upsample_rates=[", tag);
    for (size_t i = 0; i < meta.upsample_rates.size(); ++i) {
        std::printf("%s%d", i ? "," : "", meta.upsample_rates[i]);
    }
    std::printf("] kernels=[");
    for (size_t i = 0; i < meta.upsample_kernel_sizes.size(); ++i) {
        std::printf("%s%d", i ? "," : "", meta.upsample_kernel_sizes[i]);
    }
    std::printf("] d_periods=[");
    for (size_t i = 0; i < meta.d_periods.size(); ++i) {
        std::printf("%s%d", i ? "," : "", meta.d_periods[i]);
    }
    std::printf("]\n");
}

// ---------------------------------------------------------------- M1: GGUF structure validation

int cmd_validate_gguf(const std::string& path) {
    const bool ok = gguf_validate(path.c_str());
    if (ok) {
        std::printf("[validate-gguf] structure validation passed: %s\n", path.c_str());
    } else {
        std::printf("[validate-gguf] structure validation failed: %s\n", path.c_str());
    }
    return ok ? 0 : 1;
}

// ---------------------------------------------------------------- M1: training-state GGUF load self-test

int cmd_load_gguf(const std::string& path, Device* dev, const std::string& dump_path) {
    std::string err;
    rvc_train::model::TrainWeights* w = rvc_train::model::TrainWeights::open(path, &err);
    if (w == nullptr) {
        std::printf("[load-gguf] open failed: %s\n", err.c_str());
        return 1;
    }
    if (dev == nullptr) {
        std::printf("[load-gguf] no usable device\n");
        delete w;
        return 1;
    }

    rvc_train::model::TrainMeta meta;
    w->read_meta(&meta);
    std::printf("[load-gguf] file    = %s\n", path.c_str());
    print_train_meta(meta, "load-gguf");

    const DeviceProps& dp = dev->props();
    std::printf("[load-gguf] backend = %s (%s)\n", dp.name.c_str(), device_type_name(dp.type));

    const int64_t n = w->tensor_count();
    Context* ctx = context_new((size_t) (n + 64) * 1024 + (1 << 20));
    if (ctx == nullptr) {
        std::printf("[load-gguf] context_new failed\n");
        delete w;
        return 1;
    }

    std::vector<Tensor*> tensors((size_t) n, nullptr);
    for (int64_t i = 0; i < n; ++i) {
        const char* name = w->tensor_name(i);
        tensors[(size_t) i] = w->new_tensor(ctx, name);
        if (tensors[(size_t) i] == nullptr) {
            std::printf("[load-gguf] failed to create tensor: %s\n", name ? name : "(null)");
            context_free(ctx);
            delete w;
            return 1;
        }
    }
    // dummy tensor for negative tests (created before allocation so it has a data pointer)
    Tensor* dummy_missing = new_tensor_1d(ctx, TYPE_F32, 4);

    Buffer* buf = buffer_alloc_ctx_tensors(ctx, dev->default_buffer_type());
    if (buf == nullptr) {
        std::printf("[load-gguf] compute buffer allocation failed (%lld tensors)\n", (long long) n);
        context_free(ctx);
        delete w;
        return 1;
    }

    std::ofstream dump;
    if (!dump_path.empty()) {
        dump.open(dump_path, std::ios::trunc);
        if (!dump.is_open()) {
            std::printf("[load-gguf] cannot write dump file: %s\n", dump_path.c_str());
        }
    }

    int64_t failed = 0;
    for (int64_t i = 0; i < n; ++i) {
        const char* name = w->tensor_name(i);
        err.clear();
        if (!w->load(tensors[(size_t) i], name, &err)) {
            std::printf("[load-gguf] load failed: %s\n", err.c_str());
            ++failed;
            continue;
        }
        const uint32_t crc = crc32_tensor(tensors[(size_t) i]);
        if (dump.is_open()) {
            std::array<int64_t, 4> ne{1, 1, 1, 1};
            w->tensor_ne(name, &ne);
            const int64_t nbytes = tensor_nelements(tensors[(size_t) i]) * 4;
            char line[512];
            std::snprintf(line, sizeof(line), "%s %lld %lld %lld %lld %lld %08X\n", name,
                          (long long) ne[0], (long long) ne[1], (long long) ne[2], (long long) ne[3],
                          (long long) nbytes, crc);
            dump << line;
        }
    }
    if (dump.is_open()) {
        dump.close();
        std::printf("[load-gguf] per-tensor CRC manifest written: %s\n", dump_path.c_str());
    }

    // negative test 1: missing tensor
    bool neg_missing_ok = false;
    {
        std::string e;
        if (!w->load(dummy_missing, "__no_such_tensor__", &e) && !e.empty()) {
            neg_missing_ok = true;
            std::printf("[load-gguf] negative (missing tensor): passed (%s)\n", e.c_str());
        } else {
            std::printf("[load-gguf] negative (missing tensor): failed\n");
        }
    }

    // negative test 2: shape mismatch (load tensor j into tensor i with a different shape)
    bool neg_shape_ok = false;
    if (failed == 0) {
        for (int64_t i = 0; i < n && !neg_shape_ok; ++i) {
            std::array<int64_t, 4> nei{1, 1, 1, 1};
            w->tensor_ne(w->tensor_name(i), &nei);
            for (int64_t j = 0; j < n; ++j) {
                if (i == j) {
                    continue;
                }
                std::array<int64_t, 4> nej{1, 1, 1, 1};
                w->tensor_ne(w->tensor_name(j), &nej);
                if (nei == nej) {
                    continue;
                }
                std::string e;
                if (!w->load(tensors[(size_t) i], w->tensor_name(j), &e) && !e.empty()) {
                    neg_shape_ok = true;
                    std::printf("[load-gguf] negative (shape mismatch): passed (%s)\n", e.c_str());
                }
                break;
            }
        }
        if (!neg_shape_ok) {
            std::printf("[load-gguf] negative (shape mismatch): failed\n");
        }
    } else {
        std::printf("[load-gguf] negative (shape mismatch): skipped (load failures present)\n");
    }

    buffer_free(buf);
    context_free(ctx);
    delete w;

    const bool ok = (failed == 0) && neg_missing_ok && neg_shape_ok;
    std::printf("[load-gguf] loaded %lld/%lld, negative tests missing=%s shape=%s\n",
                (long long) (n - failed), (long long) n,
                neg_missing_ok ? "passed" : "failed", neg_shape_ok ? "passed" : "failed");
    if (ok) {
        std::printf("[load-gguf] M1 load self-test passed\n");
    } else {
        std::printf("[load-gguf] M1 load self-test failed\n");
    }
    return ok ? 0 : 1;
}

// ---------------------------------------------------------------- M2-1: skeleton self-test when --module is absent

// read manifest.json from the golden directory (M2-1 only prints a summary; full parsing arrives in M2-2)
void print_golden_manifest(const std::string& golden_dir) {
    std::string dir = golden_dir;
    while (!dir.empty() && (dir.back() == '/' || dir.back() == '\\')) {
        dir.pop_back();
    }
    const std::string path = dir + "/manifest.json";
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) {
        std::printf("[selftest-forward] golden manifest not found (allowed in M2-1): %s\n", path.c_str());
        return;
    }
    std::string content((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    f.close();
    const size_t kPreview = 300;
    std::printf("[selftest-forward] golden manifest: %s (%zu bytes)\n", path.c_str(), content.size());
    std::printf("[selftest-forward] preview: %s%s\n", content.substr(0, kPreview).c_str(),
                content.size() > kPreview ? " ..." : "");
    size_t n_module = 0;
    size_t pos = 0;
    while ((pos = content.find("\"module\"", pos)) != std::string::npos) {
        ++n_module;
        pos += 8;
    }
    std::printf("[selftest-forward] manifest module entries (rough count): %zu\n", n_module);
}

int cmd_selftest_forward(const std::string& weights_path, const std::string& golden_dir) {
    std::printf("[selftest-forward] M2-1 skeleton self-test: build params + check_expected + manifest summary (use --module for per-module cross-checks)\n");
    if (weights_path.empty()) {
        std::printf("[selftest-forward] missing --weights <train.gguf>\n");
        return 2;
    }

    std::string err;
    rvc_train::model::TrainWeights* w = rvc_train::model::TrainWeights::open(weights_path, &err);
    if (w == nullptr) {
        std::printf("[selftest-forward] open failed: %s\n", err.c_str());
        return 1;
    }

    rvc_train::model::TrainMeta meta;
    w->read_meta(&meta);
    std::printf("[selftest-forward] file = %s\n", weights_path.c_str());
    print_train_meta(meta, "selftest-forward");

    const rvc_train::model::ModelConfig cfg = rvc_train::model::ModelConfig::from_meta(meta);
    if (!cfg.validate(&err)) {
        std::printf("[selftest-forward] config validation failed: %s\n", err.c_str());
        delete w;
        return 1;
    }
    std::printf("[selftest-forward] config validation passed (kind=%s version=%s has_f0=%d hop=%lld z_frames=%lld)\n",
                cfg.kind.c_str(), cfg.version.c_str(), cfg.has_f0 ? 1 : 0,
                (long long) cfg.hop_product(), (long long) cfg.z_frames());

    const int64_t n = w->tensor_count();
    Context* ctx = context_new((size_t) (n + 64) * 1024 + (1 << 20));
    if (ctx == nullptr) {
        std::printf("[selftest-forward] context_new failed\n");
        delete w;
        return 1;
    }

    rvc_train::model::ModelParams params;
    if (!params.build_all(ctx, *w, &err)) {
        std::printf("[selftest-forward] parameter build failed: %s\n", err.c_str());
        context_free(ctx);
        delete w;
        return 1;
    }

    std::printf("[selftest-forward] parameter tensors: %lld (groups: ", (long long) params.count());
    const std::vector<std::string>& groups = params.group_names();
    for (size_t i = 0; i < groups.size(); ++i) {
        std::printf("%s%s=%lld", i ? " " : "", groups[i].c_str(),
                    (long long) params.group_count(groups[i]));
    }
    std::printf(")\n");

    std::vector<std::string> missing;
    std::vector<std::string> mismatched;
    w->check_expected(params.expected(), &missing, &mismatched);
    std::printf("[selftest-forward] check_expected: missing=%zu shape_mismatch=%zu\n", missing.size(),
                mismatched.size());
    for (size_t i = 0; i < missing.size() && i < 5; ++i) {
        std::printf("[selftest-forward]   missing: %s\n", missing[i].c_str());
    }
    for (size_t i = 0; i < mismatched.size() && i < 5; ++i) {
        std::printf("[selftest-forward]   shape mismatch: %s\n", mismatched[i].c_str());
    }

    // M1 rule re-check: emb_g rows == spk_embed_dim (the converter writes metadata from actual values; this only warns)
    const Tensor* emb_g = params.find("emb_g.weight");
    if (emb_g != nullptr && cfg.spk_embed_dim > 0 && emb_g->ne[1] != cfg.spk_embed_dim) {
        std::printf("[selftest-forward] warning: emb_g.weight rows=%lld does not match spk_embed_dim=%lld\n",
                    (long long) emb_g->ne[1], (long long) cfg.spk_embed_dim);
    }

    if (golden_dir.empty()) {
        std::printf("[selftest-forward] --golden not provided, skipping manifest summary (allowed in M2-1)\n");
    } else {
        print_golden_manifest(golden_dir);
    }

    const bool ok = missing.empty() && mismatched.empty();
    std::printf("[selftest-forward] %s\n", ok ? "skeleton ready" : "skeleton self-test failed");
    context_free(ctx);
    delete w;
    return ok ? 0 : 1;
}

// M6: training-state G GGUF -> RVC-GGML inference GGUF (host-only)
int cmd_export_gguf(const std::string& in_path, const std::string& out_path) {
    if (in_path.empty() || out_path.empty()) {
        std::printf("[export-gguf] requires --export-gguf <in.train.gguf> --out <out.gguf>\n");
        return 2;
    }
    rvc_train::exporter::ModelExportOptions opt;
    opt.in_path  = in_path;
    opt.out_path = out_path;
    std::string err;
    if (!rvc_train::exporter::export_inference_gguf(opt, &err)) {
        std::printf("[export-gguf] failed: %s\n", err.c_str());
        return 1;
    }
    return 0;
}

} // namespace

// M8-3: on double-click launch (this process owns the console), wait for one Enter before exiting so the window does not flash-close before the output is visible.
// when launched from an existing terminal (cmd/PowerShell/script) the console is shared by multiple processes, so do not pause (does not block scripts).
static int pause_if_sole_console(int rc) {
#ifdef _WIN32
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
    // Sources/strings are UTF-8 (/utf-8); switch the console output/input code pages to UTF-8 to avoid mojibake
    // (same as main_prep.cpp).
    SetConsoleOutputCP(65001);
    SetConsoleCP(65001);
#endif
    bool list_only = false;
    bool device_set = false;
    int  device_index = -1;
    std::string validate_path;
    std::string load_path;
    std::string dump_path;
    std::string weights_path;
    std::string d_weights_path;
    std::string golden_dir;
    std::string exp_dir;
    std::string module_name;
    std::string gradcheck_spec;
    std::string dump_grads_dir;
    bool selftest_forward  = false;
    bool selftest_backward = false;
    bool selftest_loss     = false;
    bool selftest_train_step = false;
    bool selftest_train_loop = false;
    bool selftest_data     = false;
    bool spike_train_ctx   = false;
    bool selftest_trainer  = false;
    bool selftest_grad_accum = false;
    bool selftest_resume   = false;
    bool train_mode        = false;
    std::string train_config_path;
    bool interactive_mode  = false;
    bool selftest_m0       = false;  // M8-3: only run the M0 smoke with an explicit --selftest (no arguments = double-click enters the wizard)
    bool selftest_wizard   = false;
    bool selftest_scratch  = false;
    bool scratch_flag      = false;
    std::string pretrain_g_path;    // --pretrain-g: equivalent to --weights (finetune semantics)
    std::string pretrain_d_path;    // --pretrain-d: equivalent to --d-weights
    std::string g_hashes_path;      // --g-hashes: scratch self-test G fixture
    std::string d_hashes_path;      // --d-hashes: scratch self-test D fixture
    std::string base_config_path;   // --config: wizard/scratch base config (optional)
    std::string answers_path;       // --answers: wizard answers file (non-interactive/test)
    std::string out_dir;
    std::string resume_path;
    std::string export_run_json;
    bool export_gguf_mode = false;
    std::string export_in_path;
    int64_t spike_steps    = 1;
    int64_t spike_bucket   = 384;
    bool steps_set         = false;
    bool bucket_set        = false;
    int64_t cli_batch      = 0;
    int64_t cli_epochs     = 0;
    int64_t cli_disc_len   = -1;
    int64_t cli_grad_accum = 0;
    int64_t cli_pre_steps  = 0;   // M7-5: pre-save steps for --selftest-resume
    int64_t cli_export_every  = -1;  // M7-4: -1=not specified
    int64_t cli_snapshot_keep = -1;  // M7-4: -1=not specified
    int64_t cli_save_interval = -1;  // M7-4: -1=not specified

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--list-devices") == 0) {
            list_only = true;
        } else if (std::strcmp(argv[i], "--device") == 0 && i + 1 < argc) {
            device_index = std::atoi(argv[++i]);
            device_set = true;
        } else if (std::strcmp(argv[i], "--validate-gguf") == 0 && i + 1 < argc) {
            validate_path = argv[++i];
        } else if (std::strcmp(argv[i], "--load-gguf") == 0 && i + 1 < argc) {
            load_path = argv[++i];
        } else if (std::strcmp(argv[i], "--dump-hashes") == 0 && i + 1 < argc) {
            dump_path = argv[++i];
        } else if (std::strcmp(argv[i], "--selftest-forward") == 0) {
            selftest_forward = true;
        } else if (std::strcmp(argv[i], "--selftest-backward") == 0) {
            selftest_backward = true;
        } else if (std::strcmp(argv[i], "--selftest-loss") == 0) {
            selftest_loss = true;
        } else if (std::strcmp(argv[i], "--selftest-train-step") == 0) {
            selftest_train_step = true;
        } else if (std::strcmp(argv[i], "--selftest-train-loop") == 0) {
            selftest_train_loop = true;
        } else if (std::strcmp(argv[i], "--selftest-data") == 0) {
            selftest_data = true;
        } else if (std::strcmp(argv[i], "--spike-train-ctx") == 0) {
            spike_train_ctx = true;
        } else if (std::strcmp(argv[i], "--selftest-trainer") == 0) {
            selftest_trainer = true;
        } else if (std::strcmp(argv[i], "--selftest-grad-accum") == 0) {
            selftest_grad_accum = true;
        } else if (std::strcmp(argv[i], "--selftest-resume") == 0) {
            selftest_resume = true;
        } else if (std::strcmp(argv[i], "--pre-steps") == 0 && i + 1 < argc) {
            cli_pre_steps = std::atoll(argv[++i]);
        } else if (std::strcmp(argv[i], "--grad-accum") == 0 && i + 1 < argc) {
            cli_grad_accum = std::atoll(argv[++i]);
        } else if (std::strcmp(argv[i], "--export-every") == 0 && i + 1 < argc) {
            cli_export_every = std::atoll(argv[++i]);
        } else if (std::strcmp(argv[i], "--snapshot-keep") == 0 && i + 1 < argc) {
            cli_snapshot_keep = std::atoll(argv[++i]);
        } else if (std::strcmp(argv[i], "--save-interval") == 0 && i + 1 < argc) {
            cli_save_interval = std::atoll(argv[++i]);
        } else if (std::strcmp(argv[i], "--train") == 0 && i + 1 < argc) {
            train_mode = true;
            train_config_path = argv[++i];
        } else if (std::strcmp(argv[i], "--interactive") == 0) {
            interactive_mode = true;
        } else if (std::strcmp(argv[i], "--selftest") == 0) {
            selftest_m0 = true;
        } else if (std::strcmp(argv[i], "--selftest-wizard") == 0) {
            selftest_wizard = true;
        } else if (std::strcmp(argv[i], "--selftest-scratch") == 0) {
            selftest_scratch = true;
        } else if (std::strcmp(argv[i], "--scratch") == 0) {
            scratch_flag = true;
        } else if (std::strcmp(argv[i], "--pretrain-g") == 0 && i + 1 < argc) {
            pretrain_g_path = argv[++i];
        } else if (std::strcmp(argv[i], "--pretrain-d") == 0 && i + 1 < argc) {
            pretrain_d_path = argv[++i];
        } else if (std::strcmp(argv[i], "--g-hashes") == 0 && i + 1 < argc) {
            g_hashes_path = argv[++i];
        } else if (std::strcmp(argv[i], "--d-hashes") == 0 && i + 1 < argc) {
            d_hashes_path = argv[++i];
        } else if (std::strcmp(argv[i], "--config") == 0 && i + 1 < argc) {
            base_config_path = argv[++i];
        } else if (std::strcmp(argv[i], "--answers") == 0 && i + 1 < argc) {
            answers_path = argv[++i];
        } else if (std::strcmp(argv[i], "--export-gguf") == 0 && i + 1 < argc) {
            export_gguf_mode = true;
            export_in_path = argv[++i];
        } else if (std::strcmp(argv[i], "--out") == 0 && i + 1 < argc) {
            out_dir = argv[++i];
        } else if (std::strcmp(argv[i], "--resume") == 0 && i + 1 < argc) {
            resume_path = argv[++i];
        } else if (std::strcmp(argv[i], "--export-run-json") == 0 && i + 1 < argc) {
            export_run_json = argv[++i];
        } else if (std::strcmp(argv[i], "--batch") == 0 && i + 1 < argc) {
            cli_batch = std::atoll(argv[++i]);
        } else if (std::strcmp(argv[i], "--epochs") == 0 && i + 1 < argc) {
            cli_epochs = std::atoll(argv[++i]);
        } else if (std::strcmp(argv[i], "--disc-len") == 0 && i + 1 < argc) {
            cli_disc_len = std::atoll(argv[++i]);
        } else if (std::strcmp(argv[i], "--steps") == 0 && i + 1 < argc) {
            spike_steps = std::atoll(argv[++i]);
            steps_set = true;
        } else if (std::strcmp(argv[i], "--bucket") == 0 && i + 1 < argc) {
            spike_bucket = std::atoll(argv[++i]);
            bucket_set = true;
        } else if (std::strcmp(argv[i], "--weights") == 0 && i + 1 < argc) {
            weights_path = argv[++i];
        } else if (std::strcmp(argv[i], "--d-weights") == 0 && i + 1 < argc) {
            d_weights_path = argv[++i];
        } else if (std::strcmp(argv[i], "--golden") == 0 && i + 1 < argc) {
            golden_dir = argv[++i];
        } else if (std::strcmp(argv[i], "--exp") == 0 && i + 1 < argc) {
            exp_dir = argv[++i];
        } else if (std::strcmp(argv[i], "--module") == 0 && i + 1 < argc) {
            module_name = argv[++i];
        } else if (std::strcmp(argv[i], "--gradcheck") == 0 && i + 1 < argc) {
            gradcheck_spec = argv[++i];
        } else if (std::strcmp(argv[i], "--dump-grads") == 0 && i + 1 < argc) {
            dump_grads_dir = argv[++i];
        } else {
            std::printf("[rvc-train] unknown argument: %s\n", argv[i]);
            std::printf("usage: rvc-train (no arguments = interactive wizard) [--list-devices] [--selftest] [--device N] [--validate-gguf FILE] "
                        "[--export-gguf FILE --out FILE] "
                        "[--load-gguf FILE [--dump-hashes FILE]] "
                        "[--selftest-forward [--weights FILE] [--golden DIR] [--module NAME]] "
                        "[--selftest-backward [--weights FILE] [--golden DIR] [--module NAME] "
                        "[--gradcheck SPEC] [--dump-grads DIR]] "
                        "[--selftest-loss [--golden DIR] [--module NAME]] "
                        "[--selftest-train-step [--weights FILE] [--d-weights FILE] [--golden DIR]] "
                        "[--selftest-train-loop [--weights FILE] [--d-weights FILE] [--golden DIR]] "
                        "[--selftest-data [--golden DIR] [--exp DIR]] "
                        "[--selftest-trainer [--weights FILE] [--d-weights FILE] [--exp DIR] "
                        "[--out DIR] --steps N [--bucket T] [--batch N] [--disc-len N] [--config FILE]] "
                        "[--selftest-grad-accum [--weights FILE] [--d-weights FILE] [--exp DIR] "
                        "[--bucket T] [--batch N] [--grad-accum K] [--disc-len N]] "
                        "[--selftest-resume [--weights FILE] [--d-weights FILE] [--exp DIR] --out DIR "
                        "[--steps N] [--pre-steps N] [--bucket T] [--batch N] [--disc-len N] [--config FILE]] "
                        "[--train CONFIG --exp DIR --weights FILE --d-weights FILE --out DIR "
                        "[--steps N] [--bucket T] [--batch N] [--epochs N] [--disc-len N] "
                        "[--grad-accum K] [--save-interval N] [--export-every N] "
                        "[--snapshot-keep N] [--resume FILE] [--export-run-json FILE]] "
                        "[--interactive [--config FILE] [--answers FILE] [--out DIR] [--device N] "
                        "[--exp DIR] [--weights FILE] [--d-weights FILE] [--steps N] ...] "
                        "[--selftest-wizard] [--selftest-scratch [--config FILE] [--out DIR] "
                        "[--g-hashes FILE] [--d-hashes FILE]] "
                        "[--train CONFIG [--scratch | --pretrain-g FILE --pretrain-d FILE] ...] "
                        "[--spike-train-ctx [--weights FILE] [--d-weights FILE] [--exp DIR] "
                        "[--bucket T] [--steps N]]\n");
            return 2;
        }
    }
    // M8-3: no arguments (double-click launch) = enter the interactive wizard; --answers alone is also treated as a wizard entry
    if (argc <= 1 || (!answers_path.empty() && !interactive_mode)) {
        interactive_mode = true;
    }
    if (!dump_path.empty() && load_path.empty()) {
        std::printf("[rvc-train] --dump-hashes requires --load-gguf\n");
        return 2;
    }
    const bool any_selftest = selftest_forward || selftest_backward || selftest_loss ||
                              selftest_train_step || selftest_train_loop || selftest_data ||
                              selftest_trainer || spike_train_ctx || selftest_grad_accum ||
                              selftest_resume;
    if (!golden_dir.empty() && !any_selftest) {
        std::printf("[rvc-train] --golden requires --selftest-forward/--selftest-backward/--selftest-loss/--selftest-train-step/--selftest-data\n");
        return 2;
    }
    const bool any_model_selftest = selftest_forward || selftest_backward || selftest_loss;
    if (!module_name.empty() && !any_model_selftest) {
        std::printf("[rvc-train] --module requires --selftest-forward/--selftest-backward/--selftest-loss\n");
        return 2;
    }
    if (!exp_dir.empty() && !selftest_data && !spike_train_ctx && !train_mode &&
        !selftest_trainer && !selftest_grad_accum && !interactive_mode && !selftest_resume) {
        std::printf("[rvc-train] --exp requires --selftest-data/--spike-train-ctx/--train/--selftest-trainer/--selftest-grad-accum/--interactive/--selftest-resume\n");
        return 2;
    }
    if (selftest_data && (golden_dir.empty() || exp_dir.empty())) {
        std::printf("[rvc-train] --selftest-data requires --golden <m5 dir> and --exp <M4 cache dir>\n");
        return 2;
    }
    if (!d_weights_path.empty() && !selftest_train_step && !selftest_train_loop &&
        !spike_train_ctx && !train_mode && !selftest_trainer && !selftest_grad_accum &&
        !interactive_mode && !selftest_resume) {
        std::printf("[rvc-train] --d-weights requires --selftest-train-step/--selftest-train-loop/"
                    "--spike-train-ctx/--train/--selftest-trainer/--selftest-grad-accum/--interactive/--selftest-resume\n");
        return 2;
    }
    if (selftest_train_loop && (weights_path.empty() || d_weights_path.empty() ||
                                golden_dir.empty())) {
        std::printf("[rvc-train] --selftest-train-loop requires --weights <G> --d-weights <D> "
                    "--golden <m5loop dir>\n");
        return 2;
    }
    if (spike_train_ctx && (weights_path.empty() || d_weights_path.empty() || exp_dir.empty())) {
        std::printf("[rvc-train] --spike-train-ctx requires --weights <G> --d-weights <D> --exp <cache>\n");
        return 2;
    }
    if (!out_dir.empty() && !train_mode && !selftest_trainer && !export_gguf_mode &&
        !interactive_mode && !selftest_scratch && !selftest_resume) {
        std::printf("[rvc-train] --out requires --train/--selftest-trainer/--export-gguf/--interactive/--selftest-scratch/--selftest-resume\n");
        return 2;
    }
    if (export_gguf_mode && out_dir.empty()) {
        std::printf("[rvc-train] --export-gguf requires --out <output inference GGUF>\n");
        return 2;
    }
    if ((!resume_path.empty() || !export_run_json.empty()) && !train_mode &&
        !interactive_mode) {
        std::printf("[rvc-train] --resume/--export-run-json requires --train/--interactive\n");
        return 2;
    }
    if (!answers_path.empty() && !interactive_mode) {
        std::printf("[rvc-train] --answers requires --interactive\n");
        return 2;
    }
    if (!base_config_path.empty() && !interactive_mode && !selftest_scratch && !selftest_trainer &&
        !selftest_resume) {
        std::printf("[rvc-train] --config requires --interactive/--selftest-scratch/--selftest-trainer/"
                    "--selftest-resume\n");
        return 2;
    }
    if ((!g_hashes_path.empty() || !d_hashes_path.empty()) && !selftest_scratch) {
        std::printf("[rvc-train] --g-hashes/--d-hashes requires --selftest-scratch\n");
        return 2;
    }
    if (scratch_flag && (!weights_path.empty() || !d_weights_path.empty())) {
        std::printf("[rvc-train] --scratch is mutually exclusive with --weights/--d-weights\n");
        return 2;
    }
    if (scratch_flag && (!pretrain_g_path.empty() || !pretrain_d_path.empty())) {
        std::printf("[rvc-train] --scratch is mutually exclusive with --pretrain-g/--pretrain-d\n");
        return 2;
    }
    if (scratch_flag && !train_mode && !interactive_mode) {
        std::printf("[rvc-train] --scratch requires --train <config>/--interactive\n");
        return 2;
    }
    if ((!pretrain_g_path.empty() || !pretrain_d_path.empty()) && !train_mode && !interactive_mode) {
        std::printf("[rvc-train] --pretrain-g/--pretrain-d requires --train/--interactive\n");
        return 2;
    }
    if (!pretrain_g_path.empty() && !weights_path.empty()) {
        std::printf("[rvc-train] --pretrain-g conflicts with --weights\n");
        return 2;
    }
    if (!pretrain_d_path.empty() && !d_weights_path.empty()) {
        std::printf("[rvc-train] --pretrain-d conflicts with --d-weights\n");
        return 2;
    }
    // --pretrain-g/-d = finetune-semantics aliases for --weights/--d-weights
    if (!pretrain_g_path.empty()) {
        weights_path = pretrain_g_path;
    }
    if (!pretrain_d_path.empty()) {
        d_weights_path = pretrain_d_path;
    }
    if (train_mode && train_config_path.empty()) {
        std::printf("[rvc-train] --train requires <config.json>\n");
        return 2;
    }
    // --train run.* (exp/out/pretrain_g/pretrain_d/device) can come from the config `run` section or the CLI;
    // non-empty CLI values override. Trainer::create reports missing required items explicitly.
    if (selftest_trainer && (weights_path.empty() || d_weights_path.empty() || exp_dir.empty() ||
                             out_dir.empty() || !steps_set)) {
        std::printf("[rvc-train] --selftest-trainer requires --weights <G> --d-weights <D> --exp <cache> "
                    "--out <dir> --steps N\n");
        return 2;
    }
    if (selftest_train_step && (weights_path.empty() || d_weights_path.empty())) {
        std::printf("[rvc-train] --selftest-train-step requires --weights <G> and --d-weights <D>\n");
        return 2;
    }
    if (selftest_train_step && golden_dir.empty()) {
        std::printf("[rvc-train] --selftest-train-step requires --golden <m3 dir>\n");
        return 2;
    }
    if (selftest_grad_accum && (weights_path.empty() || d_weights_path.empty() || exp_dir.empty())) {
        std::printf("[rvc-train] --selftest-grad-accum requires --weights <G> --d-weights <D> --exp <cache>\n");
        return 2;
    }
    if (selftest_resume && (weights_path.empty() || d_weights_path.empty() || exp_dir.empty() ||
                            out_dir.empty())) {
        std::printf("[rvc-train] --selftest-resume requires --weights <G> --d-weights <D> --exp <cache> "
                    "--out <dir> [--steps N] [--pre-steps N] [--bucket T]\n");
        return 2;
    }
    if (cli_pre_steps != 0 && !selftest_resume) {
        std::printf("[rvc-train] --pre-steps requires --selftest-resume\n");
        return 2;
    }
    if (cli_grad_accum != 0 && !train_mode && !interactive_mode && !selftest_grad_accum &&
        !selftest_trainer) {
        std::printf("[rvc-train] --grad-accum requires --train/--interactive/--selftest-grad-accum/--selftest-trainer\n");
        return 2;
    }
    if ((cli_export_every >= 0 || cli_snapshot_keep >= 0 || cli_save_interval >= 0) &&
        !train_mode && !interactive_mode) {
        std::printf("[rvc-train] --export-every/--snapshot-keep/--save-interval requires --train/--interactive\n");
        return 2;
    }
    if (!gradcheck_spec.empty() && !selftest_backward) {
        std::printf("[rvc-train] --gradcheck requires --selftest-backward\n");
        return 2;
    }
    if (!dump_grads_dir.empty() && !selftest_backward) {
        std::printf("[rvc-train] --dump-grads requires --selftest-backward\n");
        return 2;
    }

    if (selftest_wizard) {
        return rvc_train::train::run_train_wizard_selftest();
    }

    if (selftest_scratch) {
        const std::string cfgp = base_config_path.empty() ? "configs/v1-40k-f0.json" : base_config_path;
        // V2-2: when the hash manifest is omitted, run_scratch_selftest chooses by config version (v1->f0G/D40k; v2->f0G/D48k)
        const std::string od = out_dir.empty() ? "work/scratch_selftest" : out_dir;
        return rvc_train::train::run_scratch_selftest(cfgp, g_hashes_path, d_hashes_path, od);
    }

    if (export_gguf_mode) {
        return cmd_export_gguf(export_in_path, out_dir);
    }

    if (!validate_path.empty()) {
        return cmd_validate_gguf(validate_path);
    }

    if (!load_path.empty()) {
        Device* dev = pick_device(device_index, device_set);
        return cmd_load_gguf(load_path, dev, dump_path);
    }

    if (selftest_data) {
        return rvc_train::train::run_dataset_selftest(golden_dir, exp_dir);
    }

    if (spike_train_ctx) {
        Device* dev = pick_device(device_index, device_set);
        return rvc_train::train::run_single_ctx_spike(weights_path, d_weights_path, exp_dir, dev,
                                                      spike_steps, spike_bucket);
    }

    if (selftest_trainer) {
        Device* dev = pick_device(device_index, device_set);
        return rvc_train::train::run_trainer_selftest(
            weights_path, d_weights_path, exp_dir, out_dir, dev, spike_steps,
            bucket_set ? spike_bucket : 384, cli_batch, cli_disc_len,
            cli_grad_accum > 0 ? cli_grad_accum : 1, base_config_path);
    }

    if (selftest_grad_accum) {
        Device* dev = pick_device(device_index, device_set);
        return rvc_train::train::run_grad_accum_selftest(
            weights_path, d_weights_path, exp_dir, dev, bucket_set ? spike_bucket : 384, cli_batch,
            cli_grad_accum > 0 ? cli_grad_accum : 2, cli_disc_len);
    }

    if (selftest_resume) {
        Device* dev = pick_device(device_index, device_set);
        const int64_t total = steps_set ? spike_steps : 6;
        return rvc_train::train::run_resume_selftest(
            weights_path, d_weights_path, exp_dir, out_dir, dev, total, cli_pre_steps,
            bucket_set ? spike_bucket : 200, cli_batch,
            cli_disc_len >= 0 ? cli_disc_len : 1024, cli_grad_accum > 0 ? cli_grad_accum : 1,
            base_config_path);
    }

    if (train_mode) {
        rvc_train::train::TrainerConfig cfg;
        std::string terr;
        if (!rvc_train::train::trainer_config_from_json(train_config_path, &cfg, &terr)) {
            std::printf("[rvc-train] failed to read training config: %s\n", terr.c_str());
            return 1;
        }
        // Non-empty CLI values override the config `run` section (otherwise config values are kept)
        if (!weights_path.empty()) {
            cfg.weights_g = weights_path;
        }
        if (!d_weights_path.empty()) {
            cfg.weights_d = d_weights_path;
        }
        if (!exp_dir.empty()) {
            cfg.exp_dir = exp_dir;
        }
        if (!out_dir.empty()) {
            cfg.out_dir = out_dir;
        }
        if (!resume_path.empty()) {
            cfg.resume = resume_path;
        }
        if (!export_run_json.empty()) {
            cfg.export_run_json = export_run_json;
        }
        if (bucket_set) {
            cfg.bucket = spike_bucket;
        }
        if (cli_batch > 0) {
            cfg.batch_size = cli_batch;
        }
        if (cli_epochs > 0) {
            cfg.epochs = cli_epochs;
        }
        if (cli_disc_len >= 0) {
            cfg.disc_len = cli_disc_len;
        }
        if (cli_grad_accum > 0) {
            cfg.grad_accum = cli_grad_accum;
        }
        if (cli_export_every >= 0) {
            cfg.export_every = cli_export_every;
        }
        if (cli_snapshot_keep >= 0) {
            cfg.snapshot_keep = cli_snapshot_keep;
        }
        if (cli_save_interval >= 0) {
            cfg.save_interval = cli_save_interval;
        }
        if (steps_set) {
            cfg.max_steps = spike_steps;
        }
        if (device_set) {
            cfg.device = device_index;
        }
        if (scratch_flag) {
            cfg.scratch = true;
        }
        if (!ensure_scratch_weights(cfg, &terr)) {
            std::printf("[rvc-train] scratch init generation failed: %s\n", terr.c_str());
            return 1;
        }
        return run_train_from_config(cfg, device_index, device_set);
    }

    if (interactive_mode) {
        rvc_train::train::TrainerConfig cfg;
        std::string terr;
        if (!base_config_path.empty()) {
            if (!rvc_train::train::trainer_config_from_json(base_config_path, &cfg, &terr)) {
                std::printf("[rvc-train] failed to read base config: %s\n", terr.c_str());
                return 1;
            }
        }
        // CLI values serve as wizard defaults
        if (!exp_dir.empty())        cfg.exp_dir = exp_dir;
        if (!weights_path.empty())   cfg.weights_g = weights_path;
        if (!d_weights_path.empty()) cfg.weights_d = d_weights_path;
        if (!out_dir.empty())        cfg.out_dir = out_dir;
        if (!resume_path.empty())    cfg.resume = resume_path;
        if (!export_run_json.empty()) cfg.export_run_json = export_run_json;
        if (device_set)              cfg.device = device_index;
        if (bucket_set)              cfg.bucket = spike_bucket;
        if (cli_batch > 0)           cfg.batch_size = cli_batch;
        if (cli_epochs > 0)          cfg.epochs = cli_epochs;
        if (cli_disc_len >= 0)       cfg.disc_len = cli_disc_len;
        if (cli_grad_accum > 0)      cfg.grad_accum = cli_grad_accum;
        if (cli_export_every >= 0)   cfg.export_every = cli_export_every;
        if (cli_snapshot_keep >= 0)  cfg.snapshot_keep = cli_snapshot_keep;
        if (cli_save_interval >= 0)  cfg.save_interval = cli_save_interval;
        // M8-1a: without a base config the wizard default = the user-validated stable profile (Q31: B=2/T=300/d=4800 on a discrete GPU
        // stable for 70+ epochs; the default spec 384/1/12800 crashed after only 5 epochs in 7 hours). CLI overrides take precedence.
        if (base_config_path.empty()) {
            if (!bucket_set)      cfg.bucket = 300;
            if (cli_batch <= 0)   cfg.batch_size = 2;
            if (cli_disc_len < 0) cfg.disc_len = 4800;
        }

        const bool okw = answers_path.empty()
                             ? rvc_train::train::run_train_wizard(std::cin, std::cout, &cfg, &terr)
                             : rvc_train::train::run_train_wizard_from_file(answers_path, std::cout,
                                                                            &cfg, &terr);
        if (!okw) {
            std::printf("[rvc-train] wizard did not complete: %s\n", terr.c_str());
            return 1;
        }
        if (steps_set) {
            cfg.max_steps = spike_steps;
        }
        if (scratch_flag) {
            cfg.scratch = true;
        }
        const std::string cfg_path = join_path_simple(cfg.out_dir, "train_config.json");
        if (!rvc_train::train::trainer_config_write_json(cfg, cfg_path, &terr)) {
            std::printf("[rvc-train] failed to write training config: %s\n", terr.c_str());
            return 1;
        }
        std::printf("[rvc-train] wizard config written: %s\n", cfg_path.c_str());
        if (!ensure_scratch_weights(cfg, &terr)) {
            std::printf("[rvc-train] scratch init generation failed: %s\n", terr.c_str());
            return 1;
        }
        return run_train_from_config(cfg, device_index, device_set);
    }

    if (selftest_train_loop) {
        Device* dev = pick_device(device_index, device_set);
        return rvc_train::train::run_train_loop_selftest(weights_path, d_weights_path, golden_dir,
                                                         dev);
    }

    if (selftest_forward) {
        if (!module_name.empty()) {
            Device* dev = pick_device(device_index, device_set);
            return rvc_train::model::run_forward_selftest(weights_path, golden_dir, module_name, dev);
        }
        return cmd_selftest_forward(weights_path, golden_dir);
    }

    if (selftest_loss) {
        Device* dev = pick_device(device_index, device_set);
        const std::string m = module_name.empty() ? std::string("all") : module_name;
        return rvc_train::train::run_loss_selftest(golden_dir, m, dev);
    }

    if (selftest_train_step) {
        Device* dev = pick_device(device_index, device_set);
        return rvc_train::train::run_train_step_selftest(weights_path, d_weights_path, golden_dir,
                                                         dev);
    }

    if (selftest_backward) {
        Device* dev = pick_device(device_index, device_set);
        const std::string m = module_name.empty() ? std::string("all") : module_name;
        return rvc_train::model::run_backward_selftest(weights_path, golden_dir, m, gradcheck_spec,
                                                       dump_grads_dir, dev);
    }

    print_devices();
    if (list_only) {
        return 0;
    }
    if (!selftest_m0) {
        // M8-3: the M0 smoke now requires an explicit --selftest (no arguments already enters the interactive wizard)
        std::printf("[rvc-train] no mode specified. Hints:\n"
                    "  rvc-train (no arguments or --interactive) interactive training wizard\n"
                    "  rvc-train --selftest                M0 device/graph compute self-test\n"
                    "  rvc-train --train <config.json>     train from a config\n"
                    "  see docs/usage_guide.md\n");
        return 2;
    }

    Device* dev = pick_device(device_index, device_set);
    const bool ok = run_selftest(dev);
    if (ok) {
        std::printf("[rvc-train] M0 self-test passed\n");
    } else {
        std::printf("[rvc-train] M0 self-test failed\n");
    }
    return ok ? 0 : 1;
}
