// rvc-train - RVC 训练程序（M1 装载 + M2 前向 + M3 反向/损失/双图对拍）
//
// 职责（M2–M3 已实现；训练循环属 M5）：
//   RVC 模型装配（enc_p/enc_q/flow/dec/判别器）→ 双图 G/D + AdamW 训练 → 导出推理 GGUF
//   链接 traincpp（Vulkan 训练；CPU 仅供参考/调试）。
//
// 当前支持的模式：
//   --list-devices                仅列举 traincpp 设备
//   --device N                    选择第 N 个设备（默认：优先 GPU → IGPU → CPU）
//   --validate-gguf <file>        用 train.cpp 的 GGUF 结构自检校验文件
//   --load-gguf <file>            装载训练态 GGUF（rvc.train.*）：按名建张量 → 分配 →
//                                 装载全部 → 逐张量 CRC32 → 负向测试（缺失/形状不符）
//   --dump-hashes <file>          配合 --load-gguf：写逐张量 CRC 清单
//                                 （name ne0 ne1 ne2 ne3 nbytes crc32），供 verify_train_gguf.py 对比
//   --selftest-forward            模型骨架/前向对拍（M2）：--weights <train.gguf> [--golden <dir>] [--module <name>]
//                                 无 --module：M2-1 骨架自检（建参 + check_expected + manifest 摘要）
//                                 有 --module：调运行时 golden 逐模块对拍
//                                 （emb_g/wn/resblock1/dec/flow/enc_p/enc_q/synth/all；disc 需指定 D 权重）
//   --selftest-backward           反向对拍（M3-1，固定 cotangent VJP）：
//                                 --weights <train.gguf> --golden <m3 目录> [--module <name>] [--gradcheck SPEC]
//                                 SPEC = "参数名[:元素下标];..." 或 "all"（全部参数首/中元素）
//   --selftest-loss               损失/前端对拍（M3-2，官方 81eed5e 训练语义）：
//                                 --golden <m3 目录> [--module <stft|mel|...|all>]（无需 --weights）
//   --selftest-train-step         双图 G/D + 一步 AdamW 对拍（M3-3）：
//                                 --weights <G train.gguf> --d-weights <D train.gguf> --golden <m3 目录>
//   --selftest-train-loop         训练循环 10 步轨迹对拍（M5-2，单 ctx TrainStep）：
//                                 --weights <G> --d-weights <D> --golden <m5loop 目录>
//   --selftest-trainer            训练循环自检（M5-3，Trainer + checkpoint save→load 往返）：
//                                 --weights <G> --d-weights <D> --exp <缓存> --out <目录> --steps N
//                                 [--bucket T] [--batch N] [--disc-len N] [--config <json> 可给 v2 缓存]
//   --train <config.json>         正式训练（M5-3：循环/日志/checkpoint/ExponentialLR；M7：run 段/scratch）：
//                                 [--exp <缓存>] [--weights <G>|--pretrain-g <G>] [--d-weights <D>|--pretrain-d <D>]
//                                 [--out <目录>] [--scratch]
//                                 [--device N] [--steps N] [--bucket T] [--batch N] [--epochs N]
//                                 [--disc-len N] [--resume <ckpt>] [--export-run-json <path>]
//   --interactive                 M7 交互式训练向导（微调 / 从零；确认后落盘
//                                 <out>/train_config.json 并直接开始训练）：
//                                 [--config <base.json>] [--answers <file>] [--out DIR] [--device N]
//                                 [--exp DIR] [--weights G] [--d-weights D] [--steps N] ...
//   --selftest-wizard             M7-1 向导等价自检（G-7.1）：内置答案生成配置 vs 手写 JSON
//   --selftest-scratch            M7-2 从零规格自检（G-7.2）：规格 vs M1 hashes + 生成 GGUF 校验
//                                 [--config <json>] [--out DIR] [--g-hashes <f>] [--d-hashes <f>]
//   --selftest-data               数据加载器对拍（M5-1，纯主机，无需设备/权重）：
//                                 --golden <m5 目录> --exp <M4 缓存目录>
//   --spike-train-ctx             单 ctx 共享参数双图 spike（M5-2）：
//                                 --weights <G> --d-weights <D> --exp <缓存> [--bucket T] [--steps N]
//                                 （G-5.2a：两次 buffer_alloc_graph_tensors + 内存/逐步 loss 实测）
//   --selftest                    设备枚举 + 图计算自检（M0 冒烟；M8-3 起改为显式进入）
//   无参数 / --interactive        M8 交互式训练向导（双击启动即向导；退出前等回车防闪退）
//
// 退出码：0 = 成功；1 = 失败；2 = 参数错误。

#include "traincpp/traincpp.h"

#include "cli/train_wizard.h"
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
// 控制台输出代码页切换用（与 main_prep.cpp 一致）；NOMINMAX 避免 windows.h 的 min/max 宏
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

// ---------------------------------------------------------------- CRC32（与 zlib.crc32 兼容）

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

// ---------------------------------------------------------------- 设备

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
    std::printf("[rvc-train] traincpp 设备数: %zu\n", n);
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
            std::printf("[rvc-train] 设备下标越界: %d（共 %zu 个）\n", device_index, device_count());
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
    return dir + "\\" + name;
}

// 按配置创建 Trainer 并跑完整循环（--train 与 --interactive 共用）。
// 设备选择：显式 CLI --device 优先，其次 config.run.device（>=0），否则自动。
int run_train_from_config(const rvc_train::train::TrainerConfig& cfg, int device_index,
                          bool device_set) {
    const bool use_device = device_set || cfg.device >= 0;
    const int idx = device_set ? device_index : (int) cfg.device;
    Device* dev = pick_device(idx, use_device);
    if (dev == nullptr) {
        std::printf("[rvc-train] 无可用训练设备\n");
        return 1;
    }
    std::string terr;
    rvc_train::train::Trainer* trainer = rvc_train::train::Trainer::create(cfg, dev, &terr);
    if (trainer == nullptr) {
        std::printf("[rvc-train] Trainer 创建失败: %s\n", terr.c_str());
        return 1;
    }
    const bool ok = trainer->run(&terr);
    delete trainer;
    if (!ok) {
        std::printf("[rvc-train] 训练失败: %s\n", terr.c_str());
        return 1;
    }
    return 0;
}

// M7-2：从零训练时按 model_cfg 生成随机初始化训练态 GGUF，并回填 weights_g/weights_d。
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
    std::printf("[rvc-train] scratch 初始化已生成: %s / %s\n", gp.c_str(), dp.c_str());
    return true;
}

// ---------------------------------------------------------------- M0 冒烟：c = a + b

bool run_selftest(Device* dev) {
    if (dev == nullptr) {
        std::printf("[rvc-train] 无可用设备\n");
        return false;
    }
    const DeviceProps& p = dev->props();
    std::printf("[rvc-train] 后端: %s（%s）\n", p.name.c_str(), device_type_name(p.type));

    Context* ctx = context_new(1 << 20);
    if (ctx == nullptr) {
        std::printf("[rvc-train] context_new 失败\n");
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
        std::printf("[rvc-train] 计算缓冲分配失败\n");
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
        std::printf("[rvc-train] 设备不支持节点: %s\n", unsupported->name);
        buffer_free(buf);
        graph_free(g);
        context_free(ctx);
        return false;
    }

    if (!dev->graph_compute(g)) {
        std::printf("[rvc-train] graph_compute 失败\n");
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
    std::printf("[rvc-train] a+b = [%.1f, %.1f, %.1f, %.1f]（期望 [11, 22, 33, 44]）\n",
                cv[0], cv[1], cv[2], cv[3]);

    buffer_free(buf);
    graph_free(g);
    context_free(ctx);
    return ok;
}

// ---------------------------------------------------------------- 训练态元数据打印（M1/M2 共用）

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

// ---------------------------------------------------------------- M1：GGUF 结构校验

int cmd_validate_gguf(const std::string& path) {
    const bool ok = gguf_validate(path.c_str());
    if (ok) {
        std::printf("[validate-gguf] 结构校验通过: %s\n", path.c_str());
    } else {
        std::printf("[validate-gguf] 结构校验失败: %s\n", path.c_str());
    }
    return ok ? 0 : 1;
}

// ---------------------------------------------------------------- M1：训练态 GGUF 装载自检

int cmd_load_gguf(const std::string& path, Device* dev, const std::string& dump_path) {
    std::string err;
    rvc_train::model::TrainWeights* w = rvc_train::model::TrainWeights::open(path, &err);
    if (w == nullptr) {
        std::printf("[load-gguf] 打开失败: %s\n", err.c_str());
        return 1;
    }
    if (dev == nullptr) {
        std::printf("[load-gguf] 无可用设备\n");
        delete w;
        return 1;
    }

    rvc_train::model::TrainMeta meta;
    w->read_meta(&meta);
    std::printf("[load-gguf] file    = %s\n", path.c_str());
    print_train_meta(meta, "load-gguf");

    const DeviceProps& dp = dev->props();
    std::printf("[load-gguf] backend = %s（%s）\n", dp.name.c_str(), device_type_name(dp.type));

    const int64_t n = w->tensor_count();
    Context* ctx = context_new((size_t) (n + 64) * 1024 + (1 << 20));
    if (ctx == nullptr) {
        std::printf("[load-gguf] context_new 失败\n");
        delete w;
        return 1;
    }

    std::vector<Tensor*> tensors((size_t) n, nullptr);
    for (int64_t i = 0; i < n; ++i) {
        const char* name = w->tensor_name(i);
        tensors[(size_t) i] = w->new_tensor(ctx, name);
        if (tensors[(size_t) i] == nullptr) {
            std::printf("[load-gguf] 创建张量失败: %s\n", name ? name : "(null)");
            context_free(ctx);
            delete w;
            return 1;
        }
    }
    // 负向测试用哑张量（分配前创建，保证其有数据指针）
    Tensor* dummy_missing = new_tensor_1d(ctx, TYPE_F32, 4);

    Buffer* buf = buffer_alloc_ctx_tensors(ctx, dev->default_buffer_type());
    if (buf == nullptr) {
        std::printf("[load-gguf] 计算缓冲分配失败（张量数 %lld）\n", (long long) n);
        context_free(ctx);
        delete w;
        return 1;
    }

    std::ofstream dump;
    if (!dump_path.empty()) {
        dump.open(dump_path, std::ios::trunc);
        if (!dump.is_open()) {
            std::printf("[load-gguf] 无法写 dump 文件: %s\n", dump_path.c_str());
        }
    }

    int64_t failed = 0;
    for (int64_t i = 0; i < n; ++i) {
        const char* name = w->tensor_name(i);
        err.clear();
        if (!w->load(tensors[(size_t) i], name, &err)) {
            std::printf("[load-gguf] 装载失败: %s\n", err.c_str());
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
        std::printf("[load-gguf] 逐张量 CRC 清单已写: %s\n", dump_path.c_str());
    }

    // 负向测试 1：缺失张量
    bool neg_missing_ok = false;
    {
        std::string e;
        if (!w->load(dummy_missing, "__no_such_tensor__", &e) && !e.empty()) {
            neg_missing_ok = true;
            std::printf("[load-gguf] 负向(缺失张量): 通过（%s）\n", e.c_str());
        } else {
            std::printf("[load-gguf] 负向(缺失张量): 失败\n");
        }
    }

    // 负向测试 2：形状不符（把张量 j 装进形状不同的张量 i）
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
                    std::printf("[load-gguf] 负向(形状不符): 通过（%s）\n", e.c_str());
                }
                break;
            }
        }
        if (!neg_shape_ok) {
            std::printf("[load-gguf] 负向(形状不符): 失败\n");
        }
    } else {
        std::printf("[load-gguf] 负向(形状不符): 跳过（存在装载失败）\n");
    }

    buffer_free(buf);
    context_free(ctx);
    delete w;

    const bool ok = (failed == 0) && neg_missing_ok && neg_shape_ok;
    std::printf("[load-gguf] 装载 %lld/%lld，负向测试 缺失=%s 形状=%s\n",
                (long long) (n - failed), (long long) n,
                neg_missing_ok ? "通过" : "失败", neg_shape_ok ? "通过" : "失败");
    if (ok) {
        std::printf("[load-gguf] M1 装载自检通过\n");
    } else {
        std::printf("[load-gguf] M1 装载自检失败\n");
    }
    return ok ? 0 : 1;
}

// ---------------------------------------------------------------- M2-1：无 --module 时的骨架自检

// golden 目录的 manifest.json 读取（M2-1 仅打印摘要；正式解析在 M2-2 起接入）
void print_golden_manifest(const std::string& golden_dir) {
    std::string dir = golden_dir;
    while (!dir.empty() && (dir.back() == '/' || dir.back() == '\\')) {
        dir.pop_back();
    }
    const std::string path = dir + "/manifest.json";
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) {
        std::printf("[selftest-forward] golden manifest 不存在（M2-1 允许）: %s\n", path.c_str());
        return;
    }
    std::string content((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    f.close();
    const size_t kPreview = 300;
    std::printf("[selftest-forward] golden manifest: %s（%zu 字节）\n", path.c_str(), content.size());
    std::printf("[selftest-forward] 预览: %s%s\n", content.substr(0, kPreview).c_str(),
                content.size() > kPreview ? " ..." : "");
    size_t n_module = 0;
    size_t pos = 0;
    while ((pos = content.find("\"module\"", pos)) != std::string::npos) {
        ++n_module;
        pos += 8;
    }
    std::printf("[selftest-forward] manifest 模块条目（粗计）: %zu\n", n_module);
}

int cmd_selftest_forward(const std::string& weights_path, const std::string& golden_dir) {
    std::printf("[selftest-forward] M2-1 骨架自检：建参 + check_expected + manifest 摘要（逐模块对拍用 --module）\n");
    if (weights_path.empty()) {
        std::printf("[selftest-forward] 缺少 --weights <train.gguf>\n");
        return 2;
    }

    std::string err;
    rvc_train::model::TrainWeights* w = rvc_train::model::TrainWeights::open(weights_path, &err);
    if (w == nullptr) {
        std::printf("[selftest-forward] 打开失败: %s\n", err.c_str());
        return 1;
    }

    rvc_train::model::TrainMeta meta;
    w->read_meta(&meta);
    std::printf("[selftest-forward] file = %s\n", weights_path.c_str());
    print_train_meta(meta, "selftest-forward");

    const rvc_train::model::ModelConfig cfg = rvc_train::model::ModelConfig::from_meta(meta);
    if (!cfg.validate(&err)) {
        std::printf("[selftest-forward] 配置校验失败: %s\n", err.c_str());
        delete w;
        return 1;
    }
    std::printf("[selftest-forward] 配置校验通过（kind=%s version=%s has_f0=%d hop=%lld z_frames=%lld）\n",
                cfg.kind.c_str(), cfg.version.c_str(), cfg.has_f0 ? 1 : 0,
                (long long) cfg.hop_product(), (long long) cfg.z_frames());

    const int64_t n = w->tensor_count();
    Context* ctx = context_new((size_t) (n + 64) * 1024 + (1 << 20));
    if (ctx == nullptr) {
        std::printf("[selftest-forward] context_new 失败\n");
        delete w;
        return 1;
    }

    rvc_train::model::ModelParams params;
    if (!params.build_all(ctx, *w, &err)) {
        std::printf("[selftest-forward] 建参失败: %s\n", err.c_str());
        context_free(ctx);
        delete w;
        return 1;
    }

    std::printf("[selftest-forward] 参数张量: %lld（分组：", (long long) params.count());
    const std::vector<std::string>& groups = params.group_names();
    for (size_t i = 0; i < groups.size(); ++i) {
        std::printf("%s%s=%lld", i ? " " : "", groups[i].c_str(),
                    (long long) params.group_count(groups[i]));
    }
    std::printf("）\n");

    std::vector<std::string> missing;
    std::vector<std::string> mismatched;
    w->check_expected(params.expected(), &missing, &mismatched);
    std::printf("[selftest-forward] check_expected: 缺失=%zu 形状不符=%zu\n", missing.size(),
                mismatched.size());
    for (size_t i = 0; i < missing.size() && i < 5; ++i) {
        std::printf("[selftest-forward]   缺失: %s\n", missing[i].c_str());
    }
    for (size_t i = 0; i < mismatched.size() && i < 5; ++i) {
        std::printf("[selftest-forward]   形状不符: %s\n", mismatched[i].c_str());
    }

    // M1 规则复核：emb_g 行数 == spk_embed_dim（转换器按实际值写元数据，这里只提示）
    const Tensor* emb_g = params.find("emb_g.weight");
    if (emb_g != nullptr && cfg.spk_embed_dim > 0 && emb_g->ne[1] != cfg.spk_embed_dim) {
        std::printf("[selftest-forward] 警告: emb_g.weight 行数=%lld 与 spk_embed_dim=%lld 不一致\n",
                    (long long) emb_g->ne[1], (long long) cfg.spk_embed_dim);
    }

    if (golden_dir.empty()) {
        std::printf("[selftest-forward] 未提供 --golden，跳过 manifest 摘要（M2-1 允许）\n");
    } else {
        print_golden_manifest(golden_dir);
    }

    const bool ok = missing.empty() && mismatched.empty();
    std::printf("[selftest-forward] %s\n", ok ? "骨架就绪" : "骨架自检失败");
    context_free(ctx);
    delete w;
    return ok ? 0 : 1;
}

// M6：训练态 G GGUF → RVC-GGML 推理 GGUF（纯主机）
int cmd_export_gguf(const std::string& in_path, const std::string& out_path) {
    if (in_path.empty() || out_path.empty()) {
        std::printf("[export-gguf] 需要 --export-gguf <in.train.gguf> --out <out.gguf>\n");
        return 2;
    }
    rvc_train::exporter::ModelExportOptions opt;
    opt.in_path  = in_path;
    opt.out_path = out_path;
    std::string err;
    if (!rvc_train::exporter::export_inference_gguf(opt, &err)) {
        std::printf("[export-gguf] 失败: %s\n", err.c_str());
        return 1;
    }
    return 0;
}

} // namespace

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
#ifdef _WIN32
    // 源码/字符串为 UTF-8（/utf-8），把控制台输出/输入代码页切到 UTF-8，避免中文乱码
    // （与 main_prep.cpp 一致）。
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
    bool selftest_m0       = false;  // M8-3：显式 --selftest 才跑 M0 冒烟（无参数=双击进向导）
    bool selftest_wizard   = false;
    bool selftest_scratch  = false;
    bool scratch_flag      = false;
    std::string pretrain_g_path;    // --pretrain-g：等价 --weights（微调语义）
    std::string pretrain_d_path;    // --pretrain-d：等价 --d-weights
    std::string g_hashes_path;      // --g-hashes：scratch 自检 G 夹具
    std::string d_hashes_path;      // --d-hashes：scratch 自检 D 夹具
    std::string base_config_path;   // --config：向导/scratch 基础配置（可选）
    std::string answers_path;       // --answers：向导答案文件（非交互/测试）
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
    int64_t cli_pre_steps  = 0;   // M7-5：--selftest-resume 的存档前步数
    int64_t cli_export_every  = -1;  // M7-4：-1=未指定
    int64_t cli_snapshot_keep = -1;  // M7-4：-1=未指定
    int64_t cli_save_interval = -1;  // M7-4：-1=未指定

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
            std::printf("[rvc-train] 未知参数: %s\n", argv[i]);
            std::printf("用法: rvc-train（无参数=交互向导） [--list-devices] [--selftest] [--device N] [--validate-gguf FILE] "
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
    // M8-3：无参数（双击启动）= 进交互向导；单独给 --answers 亦视作向导入口
    if (argc <= 1 || (!answers_path.empty() && !interactive_mode)) {
        interactive_mode = true;
    }
    if (!dump_path.empty() && load_path.empty()) {
        std::printf("[rvc-train] --dump-hashes 需要配合 --load-gguf\n");
        return 2;
    }
    const bool any_selftest = selftest_forward || selftest_backward || selftest_loss ||
                              selftest_train_step || selftest_train_loop || selftest_data ||
                              selftest_trainer || spike_train_ctx || selftest_grad_accum ||
                              selftest_resume;
    if (!golden_dir.empty() && !any_selftest) {
        std::printf("[rvc-train] --golden 需要配合 --selftest-forward/--selftest-backward/--selftest-loss/--selftest-train-step/--selftest-data\n");
        return 2;
    }
    const bool any_model_selftest = selftest_forward || selftest_backward || selftest_loss;
    if (!module_name.empty() && !any_model_selftest) {
        std::printf("[rvc-train] --module 需要配合 --selftest-forward/--selftest-backward/--selftest-loss\n");
        return 2;
    }
    if (!exp_dir.empty() && !selftest_data && !spike_train_ctx && !train_mode &&
        !selftest_trainer && !selftest_grad_accum && !interactive_mode && !selftest_resume) {
        std::printf("[rvc-train] --exp 需要配合 --selftest-data/--spike-train-ctx/--train/--selftest-trainer/--selftest-grad-accum/--interactive/--selftest-resume\n");
        return 2;
    }
    if (selftest_data && (golden_dir.empty() || exp_dir.empty())) {
        std::printf("[rvc-train] --selftest-data 需要 --golden <m5 目录> 与 --exp <M4 缓存目录>\n");
        return 2;
    }
    if (!d_weights_path.empty() && !selftest_train_step && !selftest_train_loop &&
        !spike_train_ctx && !train_mode && !selftest_trainer && !selftest_grad_accum &&
        !interactive_mode && !selftest_resume) {
        std::printf("[rvc-train] --d-weights 需要配合 --selftest-train-step/--selftest-train-loop/"
                    "--spike-train-ctx/--train/--selftest-trainer/--selftest-grad-accum/--interactive/--selftest-resume\n");
        return 2;
    }
    if (selftest_train_loop && (weights_path.empty() || d_weights_path.empty() ||
                                golden_dir.empty())) {
        std::printf("[rvc-train] --selftest-train-loop 需要 --weights <G> --d-weights <D> "
                    "--golden <m5loop 目录>\n");
        return 2;
    }
    if (spike_train_ctx && (weights_path.empty() || d_weights_path.empty() || exp_dir.empty())) {
        std::printf("[rvc-train] --spike-train-ctx 需要 --weights <G> --d-weights <D> --exp <缓存>\n");
        return 2;
    }
    if (!out_dir.empty() && !train_mode && !selftest_trainer && !export_gguf_mode &&
        !interactive_mode && !selftest_scratch && !selftest_resume) {
        std::printf("[rvc-train] --out 需要配合 --train/--selftest-trainer/--export-gguf/--interactive/--selftest-scratch/--selftest-resume\n");
        return 2;
    }
    if (export_gguf_mode && out_dir.empty()) {
        std::printf("[rvc-train] --export-gguf 需要 --out <输出推理 GGUF>\n");
        return 2;
    }
    if ((!resume_path.empty() || !export_run_json.empty()) && !train_mode &&
        !interactive_mode) {
        std::printf("[rvc-train] --resume/--export-run-json 需要配合 --train/--interactive\n");
        return 2;
    }
    if (!answers_path.empty() && !interactive_mode) {
        std::printf("[rvc-train] --answers 需要配合 --interactive\n");
        return 2;
    }
    if (!base_config_path.empty() && !interactive_mode && !selftest_scratch && !selftest_trainer &&
        !selftest_resume) {
        std::printf("[rvc-train] --config 需要配合 --interactive/--selftest-scratch/--selftest-trainer/"
                    "--selftest-resume\n");
        return 2;
    }
    if ((!g_hashes_path.empty() || !d_hashes_path.empty()) && !selftest_scratch) {
        std::printf("[rvc-train] --g-hashes/--d-hashes 需要配合 --selftest-scratch\n");
        return 2;
    }
    if (scratch_flag && (!weights_path.empty() || !d_weights_path.empty())) {
        std::printf("[rvc-train] --scratch 与 --weights/--d-weights 互斥\n");
        return 2;
    }
    if (scratch_flag && (!pretrain_g_path.empty() || !pretrain_d_path.empty())) {
        std::printf("[rvc-train] --scratch 与 --pretrain-g/--pretrain-d 互斥\n");
        return 2;
    }
    if (scratch_flag && !train_mode && !interactive_mode) {
        std::printf("[rvc-train] --scratch 需要配合 --train <config>/--interactive\n");
        return 2;
    }
    if ((!pretrain_g_path.empty() || !pretrain_d_path.empty()) && !train_mode && !interactive_mode) {
        std::printf("[rvc-train] --pretrain-g/--pretrain-d 需要配合 --train/--interactive\n");
        return 2;
    }
    if (!pretrain_g_path.empty() && !weights_path.empty()) {
        std::printf("[rvc-train] --pretrain-g 与 --weights 冲突\n");
        return 2;
    }
    if (!pretrain_d_path.empty() && !d_weights_path.empty()) {
        std::printf("[rvc-train] --pretrain-d 与 --d-weights 冲突\n");
        return 2;
    }
    // --pretrain-g/-d = 微调语义的 --weights/--d-weights 别名
    if (!pretrain_g_path.empty()) {
        weights_path = pretrain_g_path;
    }
    if (!pretrain_d_path.empty()) {
        d_weights_path = pretrain_d_path;
    }
    if (train_mode && train_config_path.empty()) {
        std::printf("[rvc-train] --train 需要 <config.json>\n");
        return 2;
    }
    // --train 的 run.*（exp/out/pretrain_g/pretrain_d/device）可来自 config `run` 段或 CLI；
    // CLI 非空时覆盖。缺失必填项由 Trainer::create 给出明确错误。
    if (selftest_trainer && (weights_path.empty() || d_weights_path.empty() || exp_dir.empty() ||
                             out_dir.empty() || !steps_set)) {
        std::printf("[rvc-train] --selftest-trainer 需要 --weights <G> --d-weights <D> --exp <缓存> "
                    "--out <目录> --steps N\n");
        return 2;
    }
    if (selftest_train_step && (weights_path.empty() || d_weights_path.empty())) {
        std::printf("[rvc-train] --selftest-train-step 需要 --weights <G> 与 --d-weights <D>\n");
        return 2;
    }
    if (selftest_train_step && golden_dir.empty()) {
        std::printf("[rvc-train] --selftest-train-step 需要 --golden <m3 目录>\n");
        return 2;
    }
    if (selftest_grad_accum && (weights_path.empty() || d_weights_path.empty() || exp_dir.empty())) {
        std::printf("[rvc-train] --selftest-grad-accum 需要 --weights <G> --d-weights <D> --exp <缓存>\n");
        return 2;
    }
    if (selftest_resume && (weights_path.empty() || d_weights_path.empty() || exp_dir.empty() ||
                            out_dir.empty())) {
        std::printf("[rvc-train] --selftest-resume 需要 --weights <G> --d-weights <D> --exp <缓存> "
                    "--out <目录> [--steps N] [--pre-steps N] [--bucket T]\n");
        return 2;
    }
    if (cli_pre_steps != 0 && !selftest_resume) {
        std::printf("[rvc-train] --pre-steps 需要配合 --selftest-resume\n");
        return 2;
    }
    if (cli_grad_accum != 0 && !train_mode && !interactive_mode && !selftest_grad_accum &&
        !selftest_trainer) {
        std::printf("[rvc-train] --grad-accum 需要配合 --train/--interactive/--selftest-grad-accum/--selftest-trainer\n");
        return 2;
    }
    if ((cli_export_every >= 0 || cli_snapshot_keep >= 0 || cli_save_interval >= 0) &&
        !train_mode && !interactive_mode) {
        std::printf("[rvc-train] --export-every/--snapshot-keep/--save-interval 需要配合 --train/--interactive\n");
        return 2;
    }
    if (!gradcheck_spec.empty() && !selftest_backward) {
        std::printf("[rvc-train] --gradcheck 需要配合 --selftest-backward\n");
        return 2;
    }
    if (!dump_grads_dir.empty() && !selftest_backward) {
        std::printf("[rvc-train] --dump-grads 需要配合 --selftest-backward\n");
        return 2;
    }

    if (selftest_wizard) {
        return rvc_train::train::run_train_wizard_selftest();
    }

    if (selftest_scratch) {
        const std::string cfgp = base_config_path.empty() ? "configs/v1-40k-f0.json" : base_config_path;
        // V2-2：hash 清单缺省时由 run_scratch_selftest 按 config 版本选择（v1→f0G/D40k；v2→f0G/D48k）
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
            std::printf("[rvc-train] 读取训练配置失败: %s\n", terr.c_str());
            return 1;
        }
        // CLI 非空时覆盖 config `run` 段（否则保留 config 值）
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
            std::printf("[rvc-train] 生成 scratch 初始化失败: %s\n", terr.c_str());
            return 1;
        }
        return run_train_from_config(cfg, device_index, device_set);
    }

    if (interactive_mode) {
        rvc_train::train::TrainerConfig cfg;
        std::string terr;
        if (!base_config_path.empty()) {
            if (!rvc_train::train::trainer_config_from_json(base_config_path, &cfg, &terr)) {
                std::printf("[rvc-train] 读取基础配置失败: %s\n", terr.c_str());
                return 1;
            }
        }
        // CLI 值作为向导默认值
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
        // M8-1a：无 base config 时向导默认 = 用户验证稳定档（Q31：B=2/T=300/d=4800 独显
        // 70+ epoch 稳定；默认定稿 384/1/12800 实测 7 小时仅 5 epoch 后崩溃）。CLI 覆盖优先。
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
            std::printf("[rvc-train] 向导未完成: %s\n", terr.c_str());
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
            std::printf("[rvc-train] 写出训练配置失败: %s\n", terr.c_str());
            return 1;
        }
        std::printf("[rvc-train] 向导配置已写出: %s\n", cfg_path.c_str());
        if (!ensure_scratch_weights(cfg, &terr)) {
            std::printf("[rvc-train] 生成 scratch 初始化失败: %s\n", terr.c_str());
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
        // M8-3：M0 冒烟改为显式 --selftest（无参数已进交互向导）
        std::printf("[rvc-train] 未指定模式。提示:\n"
                    "  rvc-train（无参数或 --interactive） 交互式训练向导\n"
                    "  rvc-train --selftest                M0 设备/图计算自检\n"
                    "  rvc-train --train <config.json>     按配置训练\n"
                    "  详见 docs/使用指南.md\n");
        return 2;
    }

    Device* dev = pick_device(device_index, device_set);
    const bool ok = run_selftest(dev);
    if (ok) {
        std::printf("[rvc-train] M0 自检通过\n");
    } else {
        std::printf("[rvc-train] M0 自检失败\n");
    }
    return ok ? 0 : 1;
}
