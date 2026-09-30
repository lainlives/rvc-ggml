// RVC-Train-GGML - 交互式训练向导实现（src/cli/train_wizard.cpp）
//
// 见 train_wizard.h 与 `交接说明` §4.11/§4.13。支持微调与从零（M7-2）两条分支。
// M8-1a：设备选择列表化（traincpp device API）+ 大配置 Q31 警告。
// M8-1b：数据入口双模式——已有缓存目录 / 原始音频目录（spawn rvc-prep.exe 子进程预处理）。
#include "cli/train_wizard.h"

#include "prep/cache.h"
#include "traincpp/traincpp.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <system_error>

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

namespace rvc_train {
namespace train {
namespace {

std::string trim(const std::string& s) {
    size_t b = 0;
    size_t e = s.size();
    while (b < e && (s[b] == ' ' || s[b] == '\t' || s[b] == '\r' || s[b] == '\n')) {
        ++b;
    }
    while (e > b && (s[e - 1] == ' ' || s[e - 1] == '\t' || s[e - 1] == '\r' || s[e - 1] == '\n')) {
        --e;
    }
    return s.substr(b, e - b);
}

// 读取一行答案；流结束返回 false。
bool read_answer(std::istream& in, std::string* out) {
    if (!std::getline(in, *out)) {
        return false;
    }
    *out = trim(*out);
    return true;
}

// 打印提示并读一行：空行/EOF 采用默认值。
std::string ask(const std::string& label, const std::string& def, std::istream& in,
                std::ostream& out, bool* eof) {
    out << label;
    if (!def.empty()) {
        out << " [" << def << "]";
    }
    out << ": ";
    out.flush();
    std::string line;
    if (!read_answer(in, &line)) {
        *eof = true;
        out << "\n";
        return def;
    }
    return line.empty() ? def : line;
}

std::string ask_nonempty(const std::string& label, const std::string& def, std::istream& in,
                         std::ostream& out, bool* eof) {
    std::string cur = def;
    for (;;) {
        const std::string s = ask(label, cur, in, out, eof);
        if (!s.empty()) {
            return s;
        }
        if (*eof) {
            return s;
        }
        out << "  不能为空，请重试。\n";
    }
}

std::string ask_existing_file(const std::string& label, const std::string& def, std::istream& in,
                              std::ostream& out, bool* eof) {
    std::string cur = def;
    for (;;) {
        const std::string s = ask(label, cur, in, out, eof);
        std::error_code ec;
        if (!s.empty() && std::filesystem::is_regular_file(s, ec)) {
            return s;
        }
        if (*eof) {
            return s;
        }
        out << "  文件不存在: " << s << "，请重试。\n";
        cur = s;
    }
}

std::string ask_cache_dir(const std::string& label, const std::string& def, std::istream& in,
                          std::ostream& out, bool* eof) {
    std::string cur = def;
    for (;;) {
        const std::string s = ask(label, cur, in, out, eof);
        std::error_code ec;
        const std::filesystem::path p(s);
        if (!s.empty() && std::filesystem::is_directory(p, ec) &&
            std::filesystem::is_regular_file(p / "dataset.json", ec)) {
            return s;
        }
        if (*eof) {
            return s;
        }
        out << "  缓存目录无效（需存在且含 dataset.json）: " << s << "，请重试。\n";
        cur = s;
    }
}

// M8-1b：已存在的目录（输入音频目录等）。
std::string ask_existing_dir(const std::string& label, const std::string& def, std::istream& in,
                             std::ostream& out, bool* eof) {
    std::string cur = def;
    for (;;) {
        const std::string s = ask(label, cur, in, out, eof);
        std::error_code ec;
        if (!s.empty() && std::filesystem::is_directory(s, ec)) {
            return s;
        }
        if (*eof) {
            return s;
        }
        out << "  目录不存在: " << s << "，请重试。\n";
        cur = s;
    }
}

// M8-1b：可选文件——空行表示留空（如 RMVPE 留空则跳过 f0）；非空必须是已存在文件。
std::string ask_optional_file(const std::string& label, const std::string& def, std::istream& in,
                              std::ostream& out, bool* eof) {
    std::string cur = def;
    for (;;) {
        const std::string s = ask(label, cur, in, out, eof);
        if (s.empty() || *eof) {
            return s;
        }
        std::error_code ec;
        if (std::filesystem::is_regular_file(s, ec)) {
            return s;
        }
        out << "  文件不存在: " << s << "，请重试（留空请直接回车）。\n";
        cur = s;
    }
}

int64_t ask_int(const std::string& label, int64_t def, int64_t min_v, int64_t max_v,
                std::istream& in, std::ostream& out, bool* eof) {
    int64_t cur = def;
    for (;;) {
        const std::string s = ask(label, std::to_string((long long) cur), in, out, eof);
        if (!s.empty()) {
            char* end = nullptr;
            const long long v = std::strtoll(s.c_str(), &end, 10);
            if (end != s.c_str() && *end == '\0' && v >= min_v && v <= max_v) {
                return (int64_t) v;
            }
        } else if (*eof) {
            return cur;
        } else {
            return cur;  // 空行采用默认（已展示）
        }
        if (*eof) {
            return cur;
        }
        out << "  请输入 " << min_v << ".." << max_v << " 范围内的整数，请重试。\n";
        cur = def;
    }
}

float ask_float(const std::string& label, float def, std::istream& in, std::ostream& out,
                bool* eof) {
    char defbuf[64];
    std::snprintf(defbuf, sizeof(defbuf), "%.*g", 9, (double) def);
    float cur = def;
    for (;;) {
        const std::string s = ask(label, defbuf, in, out, eof);
        if (!s.empty()) {
            char* end = nullptr;
            const double v = std::strtod(s.c_str(), &end);
            if (end != s.c_str() && *end == '\0' && v > 0.0) {
                return (float) v;
            }
        } else {
            return cur;
        }
        if (*eof) {
            return cur;
        }
        out << "  请输入 > 0 的数值，请重试。\n";
    }
}

void print_summary(const TrainerConfig& c, std::ostream& out) {
    out << "\n---- 配置摘要 ----\n";
    out << "  数据缓存   : " << c.exp_dir << "\n";
    if (c.scratch) {
        out << "  模型来源   : 从零训练（按配置随机初始化 G/D）\n";
    } else {
        out << "  预训练 G   : " << c.weights_g << "\n";
        out << "  预训练 D   : " << c.weights_d << "\n";
    }
    out << "  输出目录   : " << c.out_dir << "\n";
    out << "  epochs     : " << (long long) c.epochs << "\n";
    out << "  batch_size : " << (long long) c.batch_size << "\n";
    out << "  bucket     : " << (long long) c.bucket << "\n";
    out << "  disc_len   : " << (long long) c.disc_len << "（0=整段 segment_size）\n";
    out << "  learning_rate: " << c.lr << "\n";
    out << "  seed       : " << (long long) c.seed << "\n";
    out << "  device     : " << (long long) c.device << "（-1=自动）\n";
    if (!c.resume.empty()) {
        out << "  续训自     : " << c.resume << "\n";
    }
    out << "------------------\n";
}

bool confirm_yes(const std::string& yn) {
    return yn == "y" || yn == "Y" || yn == "yes" || yn == "YES" || yn == "是";
}

// M8-1b：本进程 exe 所在目录（用于定位同目录的 rvc-prep.exe）。
std::string current_exe_dir() {
    wchar_t buf[MAX_PATH];
    const DWORD n = GetModuleFileNameW(nullptr, buf, MAX_PATH);
    if (n == 0 || n >= MAX_PATH) {
        return {};
    }
    std::error_code ec;
    const std::filesystem::path p(buf);
    const std::filesystem::path dir = p.parent_path();
    return std::filesystem::is_directory(dir, ec) ? dir.string() : std::string();
}

// M8-1b：目录 + 文件名拼接（Windows 风格）。
std::string join_simple(const std::string& dir, const std::string& name) {
    if (dir.empty()) {
        return name;
    }
    const char last = dir.back();
    if (last == '/' || last == '\\') {
        return dir + name;
    }
    return dir + "\\" + name;
}

// M8-1b：定位 rvc-prep.exe——优先本 exe 同目录，其次工作目录；找不到返回空。
std::string find_rvc_prep_exe(std::ostream& out) {
    std::error_code ec;
    const std::string candidates[2] = {
        join_simple(current_exe_dir(), "rvc-prep.exe"),
        std::string("rvc-prep.exe"),
    };
    for (const std::string& c : candidates) {
        if (std::filesystem::is_regular_file(c, ec)) {
            return c;
        }
    }
    out << "  未找到 rvc-prep.exe（已尝试本程序同目录与当前目录）。\n";
    return {};
}

// M8-1b：拼接引号包裹的命令片段。
std::string quoted_arg(const std::string& s) {
    return "\"" + s + "\"";
}

// M8-1c：在输出目录找最新可续训 checkpoint（train*.ckpt.gguf 且旁挂 .meta.json 存在）。
// 按文件修改时间取最新；没有返回空。
std::string find_latest_checkpoint(const std::string& out_dir) {
    std::error_code ec;
    const std::filesystem::path dir(out_dir);
    if (out_dir.empty() || !std::filesystem::is_directory(dir, ec)) {
        return {};
    }
    std::filesystem::path best;
    std::filesystem::file_time_type best_time{};
    for (const std::filesystem::directory_entry& e : std::filesystem::directory_iterator(dir, ec)) {
        if (!e.is_regular_file(ec)) {
            continue;
        }
        const std::filesystem::path p = e.path();
        const std::string name = p.filename().string();
        if (name.rfind("train", 0) != 0 ||
            name.rfind(".ckpt.gguf") != name.size() - std::string(".ckpt.gguf").size()) {
            continue;
        }
        // meta 旁挂命名与 resume_meta_path_for 一致：<ckpt>.gguf → <ckpt>.meta.json
        const std::string meta_name = name.substr(0, name.size() - std::string(".gguf").size()) +
                                      ".meta.json";
        if (!std::filesystem::is_regular_file(dir / meta_name, ec)) {
            continue;
        }
        const std::filesystem::file_time_type t = e.last_write_time(ec);
        if (best.empty() || t > best_time) {
            best = p;
            best_time = t;
        }
    }
    return best.empty() ? std::string() : best.string();
}

// M8-1b：spawn rvc-prep.exe 子进程执行预处理（两个 Vulkan 后端分属不同进程，符合隔离规则）。
// 走 CreateProcessW（避免 cmd 的引号解析问题，且原生支持中文路径）；继承控制台实时输出。
// 退出码 0 且缓存 dataset.json 出现才视为成功。
bool run_prep_subprocess(const std::string& exe, const std::string& input, int sr,
                         const std::string& version, const std::string& rmvpe,
                         const std::string& hubert, const std::string& exp_dir,
                         std::string* err) {
    std::string cmd = quoted_arg(exe) + " --input " + quoted_arg(input) + " --sr " +
                      std::to_string(sr) + " --version " + version + " --exp " +
                      quoted_arg(exp_dir);
    if (!rmvpe.empty()) {
        cmd += " --rmvpe " + quoted_arg(rmvpe);
    }
    if (!hubert.empty()) {
        cmd += " --hubert " + quoted_arg(hubert);
    }
    std::printf("[rvc-train] 正在运行预处理（可能耗时较长）:\n  %s\n", cmd.c_str());
    std::fflush(stdout);

    // UTF-8 → UTF-16（命令行含中文路径时必须）
    const int wlen = MultiByteToWideChar(CP_UTF8, 0, cmd.c_str(), -1, nullptr, 0);
    if (wlen <= 0) {
        if (err != nullptr) {
            *err = "命令行转宽字符失败";
        }
        return false;
    }
    std::wstring wcmd((size_t) wlen, L'\0');
    MultiByteToWideChar(CP_UTF8, 0, cmd.c_str(), -1, wcmd.data(), wlen);

    STARTUPINFOW si{};
    si.cb = sizeof(si);
    PROCESS_INFORMATION pi{};
    if (!CreateProcessW(nullptr, wcmd.data(), nullptr, nullptr, FALSE, 0, nullptr, nullptr, &si,
                        &pi)) {
        if (err != nullptr) {
            *err = "启动 rvc-prep.exe 失败（GetLastError=" + std::to_string(GetLastError()) + "）";
        }
        return false;
    }
    WaitForSingleObject(pi.hProcess, INFINITE);
    DWORD exit_code = 1;
    GetExitCodeProcess(pi.hProcess, &exit_code);
    CloseHandle(pi.hThread);
    CloseHandle(pi.hProcess);
    if (exit_code != 0) {
        if (err != nullptr) {
            *err = "rvc-prep 退出码 " + std::to_string(exit_code) + "（预处理失败，详见上方日志）";
        }
        return false;
    }
    std::error_code ec;
    if (!std::filesystem::is_regular_file(std::filesystem::path(exp_dir) / "dataset.json", ec)) {
        if (err != nullptr) {
            *err = "预处理命令成功但未生成 " + exp_dir + "\\dataset.json";
        }
        return false;
    }
    return true;
}

const char* wizard_device_type_name(traincpp::DeviceType t) {
    using traincpp::DeviceType;
    switch (t) {
        case DeviceType::CPU:   return "CPU";
        case DeviceType::GPU:   return "GPU";
        case DeviceType::IGPU:  return "IGPU";
        case DeviceType::ACCEL: return "ACCEL";
    }
    return "?";
}

// M8-1a：向导内打印 traincpp 设备列表（含显存/统一内存），供设备选择参考。
void print_wizard_devices(std::ostream& out) {
    const size_t n = traincpp::device_count();
    out << "\n可用训练设备（traincpp）:\n";
    if (n == 0) {
        out << "  （未枚举到任何设备）\n";
        return;
    }
    for (size_t i = 0; i < n; ++i) {
        const traincpp::Device* d = traincpp::device_get(i);
        const traincpp::DeviceProps& p = d->props();
        out << "  [" << i << "] " << p.name << " | " << p.description << " | "
            << wizard_device_type_name(p.type) << " | 空闲 "
            << (double) p.memory_free / (1024.0 * 1024.0) << " MiB / 共 "
            << (double) p.memory_total / (1024.0 * 1024.0) << " MiB"
            << (p.unified_memory ? " | 统一内存" : "") << "\n";
    }
    out << "  -1 = 自动（GPU → IGPU → CPU）\n";
}

// M8-1a：Q31 口径——大 bucket/大 disc_len 的实测代价警告（用户独显 7 小时仅 5 epoch 后崩溃）。
void warn_large_config(const TrainerConfig& c, std::ostream& out) {
    if (c.bucket > 384 || c.disc_len > 6400) {
        out << "  警告: 大 bucket/disc_len 配置代价极高——独显实测 B=1/T=384/disc=12800 "
               "7 小时仅 5 epoch 后崩溃。\n";
        out << "        推荐 bucket<=300、disc_len<=4800（同机实测 70+ epoch 稳定）。\n";
    }
}

} // namespace

bool run_train_wizard(std::istream& in, std::ostream& out, TrainerConfig* cfg, std::string* err) {
    if (cfg == nullptr) {
        if (err != nullptr) {
            *err = "run_train_wizard: cfg 为空";
        }
        return false;
    }
    bool eof = false;
    out << "==== RVC 训练向导（微调 / 从零；M8）====\n";
    out << "直接回车采用 [方括号] 中的默认值。\n\n";

    // M8-1b：数据入口双模式。
    const int64_t entry = ask_int("数据入口（1=已有缓存目录 2=原始音频目录，先预处理）", 1, 1, 2,
                                  in, out, &eof);
    if (entry == 2) {
        const std::string input_dir =
            ask_existing_dir("原始音频目录（wav 文件）", "", in, out, &eof);
        const int64_t tgt = ask_int("预处理目标（1=v1-40k / 2=v2-48k）", 1, 1, 2, in, out, &eof);
        const char* version = tgt == 2 ? "v2" : "v1";
        const int sr = tgt == 2 ? 48000 : 40000;
        const std::string rmvpe =
            ask_optional_file("RMVPE 权重路径（回车跳过 f0；训练建议提供）", "", in, out, &eof);
        const std::string hubert =
            ask_existing_file("HuBERT 权重路径（特征提取，训练必需）", "", in, out, &eof);
        const std::string exp_def = tgt == 2 ? std::string("work/cache_auto_48k")
                                             : std::string("work/cache_auto_40k");
        const std::string exp_out =
            ask_nonempty("预处理缓存输出目录", exp_def, in, out, &eof);
        if (eof) {
            if (err != nullptr) {
                *err = "向导：输入流结束";
            }
            return false;
        }
        const std::string exe = find_rvc_prep_exe(out);
        if (exe.empty()) {
            if (err != nullptr) {
                *err = "未找到 rvc-prep.exe；请手动执行（或将其放到本程序同目录后重试）:\n"
                       "  rvc-prep --input \"" + input_dir + "\" --sr " + std::to_string(sr) +
                       " --version " + version + " --exp \"" + exp_out + "\"" +
                       (rmvpe.empty() ? "" : " --rmvpe \"" + rmvpe + "\"") +
                       " --hubert \"" + hubert + "\"";
            }
            return false;
        }
        std::string perr;
        if (!run_prep_subprocess(exe, input_dir, sr, version, rmvpe, hubert, exp_out, &perr)) {
            if (err != nullptr) {
                *err = perr;
            }
            return false;
        }
        out << "预处理完成: " << exp_out << "\n\n";
        cfg->exp_dir = exp_out;
    } else {
        cfg->exp_dir = ask_cache_dir("数据缓存目录（rvc-prep --exp 产物）", cfg->exp_dir, in, out,
                                     &eof);
    }

    const int64_t src = ask_int("模型来源（1=预训练微调 2=从零训练）", cfg->scratch ? 2 : 1, 1, 2, in,
                                out, &eof);
    if (src == 2) {
        cfg->scratch = true;
    } else {
        cfg->scratch = false;
        cfg->weights_g = ask_existing_file("预训练生成器 G（训练态 GGUF）", cfg->weights_g, in, out, &eof);
        cfg->weights_d = ask_existing_file("预训练判别器 D（训练态 GGUF）", cfg->weights_d, in, out, &eof);
    }

    const std::string out_def = cfg->out_dir.empty() ? std::string("work/train_out") : cfg->out_dir;
    cfg->out_dir = ask_nonempty("输出目录（日志/checkpoint/导出）", out_def, in, out, &eof);

    // M8-1c：续训入口——输出目录已有 checkpoint（含 meta）时询问是否续训。
    if (cfg->resume.empty()) {
        const std::string ckpt = find_latest_checkpoint(cfg->out_dir);
        if (!ckpt.empty()) {
            out << "  检测到已有 checkpoint: " << ckpt << "\n";
            const std::string yn = ask("是否从该 checkpoint 续训? (y/N)", "n", in, out, &eof);
            if (confirm_yes(yn)) {
                cfg->resume = ckpt;
                out << "  注意: 续训要求 batch/bucket/seed/grad_accum/lr 与原训练一致，"
                       "且数据缓存未增删（不一致将被拒绝）。\n";
            }
        }
    }

    cfg->epochs = ask_int("训练 epoch 数", cfg->epochs > 0 ? cfg->epochs : 1, 1, 1000000000, in,
                          out, &eof);
    cfg->batch_size = ask_int("batch_size", cfg->batch_size > 0 ? cfg->batch_size : 1, 1, 1024, in,
                              out, &eof);
    const int64_t bucket_def = cfg->bucket > 0 ? cfg->bucket : cfg->segment_frames;
    cfg->bucket = ask_int("bucket（图 T）", bucket_def, cfg->segment_frames, 1000000, in, out, &eof);
    cfg->disc_len = ask_int("disc_len（0=整段 segment_size）", cfg->disc_len, 0, 100000000, in, out,
                            &eof);
    warn_large_config(*cfg, out);
    cfg->lr = ask_float("learning_rate", cfg->lr, in, out, &eof);
    cfg->seed = ask_int("随机种子", cfg->seed, 0, 2147483647, in, out, &eof);

    print_wizard_devices(out);
    const int64_t dev_max = traincpp::device_count() > 0
                                ? (int64_t) traincpp::device_count() - 1
                                : -1;
    cfg->device = ask_int("训练设备下标（-1=自动）", cfg->device, -1, dev_max > 1024 ? 1024 : dev_max,
                          in, out, &eof);

    // M8-1c：高级参数（默认折叠，直接回车跳过）。
    const std::string adv = ask("配置高级参数? (y/N)", "n", in, out, &eof);
    if (confirm_yes(adv)) {
        cfg->grad_accum = ask_int("grad_accum（梯度累积步数，1=关）", cfg->grad_accum > 0
                                                                      ? cfg->grad_accum
                                                                      : 1,
                                  1, 64, in, out, &eof);
        cfg->save_interval = ask_int("save_interval（每 N epoch 存 checkpoint，0=仅收尾）",
                                     cfg->save_interval, 0, 1000000, in, out, &eof);
        cfg->export_every = ask_int("export_every（每 N epoch 导出推理 GGUF，0=关）",
                                    cfg->export_every, 0, 1000000, in, out, &eof);
        cfg->snapshot_keep = ask_int("snapshot_keep（epoch 快照保留数，0=全部保留）",
                                     cfg->snapshot_keep, 0, 10000, in, out, &eof);
        cfg->log_interval = ask_int("log_interval（每 N 优化步打日志）",
                                    cfg->log_interval > 0 ? cfg->log_interval : 50, 1, 1000000,
                                    in, out, &eof);
    }

    // M8-1c：确认前校验增强——缓存版本/样本数/预计步数摘要 + 与模型配置的一致性提示。
    if (!cfg->exp_dir.empty()) {
        prep::CacheDataset ds;
        std::string cerr2;
        if (load_cache_index(cfg->exp_dir, &ds, &cerr2)) {
            const int64_t n = (int64_t) ds.samples.size();
            out << "\n  缓存信息: " << n << " 样本 | 版本 " << ds.meta.version << " | sr "
                << ds.meta.target_sr << "\n";
            const int64_t batch = cfg->batch_size > 0 ? cfg->batch_size : 1;
            const int64_t accum = cfg->grad_accum > 0 ? cfg->grad_accum : 1;
            const int64_t micro = n / batch;      // 尾部丢弃
            const int64_t steps = micro / accum;  // Trainer 以优化步计
            out << "  预计: 约 " << steps << " 优化步/epoch（batch=" << batch << "×accum="
                << accum << "；超 bucket 的样本会被跳过，实际可能更少）\n";
            if (ds.meta.version != cfg->model_cfg.version ||
                ds.meta.target_sr != cfg->model_cfg.sample_rate) {
                out << "  警告: 缓存版本/采样率（" << ds.meta.version << "/"
                    << ds.meta.target_sr << "）与模型配置（" << cfg->model_cfg.version << "/"
                    << cfg->model_cfg.sample_rate << "）不一致——训练将被预检拒绝，"
                    "请使用匹配的权重与 --config。\n";
            }
        } else {
            out << "  注意: 读取缓存索引失败（" << cerr2 << "），训练启动时仍会再次校验。\n";
        }
    }

    print_summary(*cfg, out);
    const std::string yn = ask("确认并开始训练? (y/N)", "n", in, out, &eof);
    if (!confirm_yes(yn)) {
        if (err != nullptr) {
            *err = "用户取消";
        }
        return false;
    }

    if (cfg->exp_dir.empty() || cfg->out_dir.empty() ||
        (!cfg->scratch && (cfg->weights_g.empty() || cfg->weights_d.empty()))) {
        if (err != nullptr) {
            *err = "向导：关键路径为空（exp/pretrain_g/pretrain_d/out）";
        }
        return false;
    }
    return true;
}

bool run_train_wizard_from_file(const std::string& answers_path, std::ostream& out,
                                TrainerConfig* cfg, std::string* err) {
    std::ifstream f(answers_path);
    if (!f.is_open()) {
        if (err != nullptr) {
            *err = "无法打开答案文件: " + answers_path;
        }
        return false;
    }
    return run_train_wizard(f, out, cfg, err);
}

int run_train_wizard_selftest() {
    // 固定答案（顺序同向导提问）：数据入口(1=已有缓存) / exp / 模型来源(1=微调) / G / D / out /
    // epochs / batch / bucket / disc_len / lr / seed / device / 高级参数(n) / confirm
    static const char* kAnswers =
        "1\n"
        "work/cache_v1\n"
        "1\n"
        "work/models/f0G40k.train.gguf\n"
        "work/models/f0D40k.train.gguf\n"
        "work/wizard_selftest\n"
        "3\n"
        "1\n"
        "384\n"
        "1024\n"
        "0.0001\n"
        "1234\n"
        "1\n"
        "n\n"
        "y\n";

    // "手写 JSON"黄金配置：与上面答案应产生完全等价的 TrainerConfig（其余字段取结构体默认值）。
    static const char* kGoldenJson = R"json({
  "run": {
    "exp": "work/cache_v1",
    "out": "work/wizard_selftest",
    "pretrain_g": "work/models/f0G40k.train.gguf",
    "pretrain_d": "work/models/f0D40k.train.gguf",
    "device": 1,
    "resume": "",
    "export_every": 0,
    "export_run_json": ""
  },
  "train": {
    "batch_size": 1,
    "learning_rate": 0.0001,
    "betas": [0.8, 0.99],
    "eps": 1e-09,
    "weight_decay": 0.01,
    "lr_decay": 0.999875,
    "c_mel": 45,
    "c_kl": 1,
    "epochs": 3,
    "max_steps": 0,
    "segment_size": 12800,
    "bucket": 384,
    "disc_len": 1024,
    "log_interval": 50,
    "save_interval": 0,
    "seed": 1234,
    "grad_accum": 1,
    "buckets": []
  },
  "data": {
    "sampling_rate": 40000,
    "filter_length": 2048,
    "hop_length": 400,
    "win_length": 2048,
    "n_mel_channels": 125,
    "mel_fmin": 0,
    "mel_fmax": null
  },
  "model": { "inter_channels": 192 }
})json";

    std::printf("[wizard-selftest] ==== 向导等价自检（G-7.1）====\n");
    TrainerConfig wiz;
    std::istringstream in(kAnswers);
    std::string err;
    if (!run_train_wizard(in, std::cout, &wiz, &err)) {
        std::printf("[wizard-selftest] FAIL: 向导生成失败: %s\n", err.c_str());
        return 1;
    }
    TrainerConfig gold;
    if (!trainer_config_from_json_text(kGoldenJson, &gold, &err)) {
        std::printf("[wizard-selftest] FAIL: 手写 JSON 解析失败: %s\n", err.c_str());
        return 1;
    }
    std::string diff;
    if (!trainer_config_equivalent(wiz, gold, &diff)) {
        std::printf("[wizard-selftest] FAIL: 向导配置与手写 JSON 不等价 —— %s\n", diff.c_str());
        return 1;
    }
    std::printf("[wizard-selftest] PASS: 向导生成配置与手写 JSON 逐字段等价\n");
    return 0;
}

} // namespace train
} // namespace rvc_train
