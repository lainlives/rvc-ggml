// RVC-Train-GGML - M8-2 rvc-prep 交互式预处理向导实现（src/cli/prep_wizard.cpp）
//
// 见 prep_wizard.h 与 `交接说明` §4.13。本文件属于 rvc-prep（可调用 rvc_core 公共 API
// rvc::list_gpu_devices 做设备列表）；不链接 traincpp。
#include "cli/prep_wizard.h"

#include "engine.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <system_error>

namespace rvc_train {
namespace prep {
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

bool read_answer(std::istream& in, std::string* out) {
    if (!std::getline(in, *out)) {
        return false;
    }
    *out = trim(*out);
    return true;
}

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

// 目录或单个 wav 文件均可（与 CLI --input 一致）。
std::string ask_input_path(const std::string& label, const std::string& def, std::istream& in,
                           std::ostream& out, bool* eof) {
    std::string cur = def;
    for (;;) {
        const std::string s = ask(label, cur, in, out, eof);
        std::error_code ec;
        const std::filesystem::path p(s);
        if (!s.empty() &&
            (std::filesystem::is_directory(p, ec) || std::filesystem::is_regular_file(p, ec))) {
            return s;
        }
        if (*eof) {
            return s;
        }
        out << "  路径不存在: " << s << "，请重试。\n";
        cur = s;
    }
}

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
        } else {
            return cur;
        }
        if (*eof) {
            return cur;
        }
        out << "  请输入 " << min_v << ".." << max_v << " 范围内的整数，请重试。\n";
        cur = def;
    }
}

bool confirm_yes(const std::string& yn) {
    return yn == "y" || yn == "Y" || yn == "yes" || yn == "YES" || yn == "是";
}

void print_gpu_devices(std::ostream& out) {
    const std::vector<rvc::GpuDeviceInfo> gpus = rvc::list_gpu_devices();
    out << "\n可用计算设备（ggml）:\n";
    if (gpus.empty()) {
        out << "  （未枚举到 GPU，将回退 CPU=99）\n";
    }
    for (const rvc::GpuDeviceInfo& g : gpus) {
        out << "  [" << g.index << "] " << g.name << " | " << g.description << "\n";
    }
    out << "  99 = CPU（软件回退，仅调试）\n";
    out << "  -1 = 自动（有 GPU 用 [0]，否则 CPU）\n";
}

} // namespace

bool run_prep_wizard(std::istream& in, std::ostream& out, PrepWizardResult* res, std::string* err) {
    if (res == nullptr) {
        if (err != nullptr) {
            *err = "run_prep_wizard: res 为空";
        }
        return false;
    }
    bool eof = false;
    out << "==== RVC 预处理向导（原始音频 -> 训练缓存；M8）====\n";
    out << "直接回车采用 [方括号] 中的默认值。处理完成后请在 rvc-train 中开始训练。\n\n";

    res->input = ask_input_path("输入音频目录（或单个 wav 文件）", res->input, in, out, &eof);
    res->target = (int) ask_int("预处理目标（1=v1-40k / 2=v2-48k）", res->target, 1, 2, in, out,
                                &eof);
    res->rmvpe = ask_optional_file("RMVPE 权重路径（回车跳过 f0；训练建议提供）", res->rmvpe, in,
                                   out, &eof);
    res->hubert = ask_optional_file("HuBERT 权重路径（回车跳过特征提取；训练必需）", res->hubert,
                                    in, out, &eof);
    if (!res->hubert.empty() && !res->rmvpe.empty()) {
        out << "  （f0 + 特征都将提取，缓存可直接用于训练）\n";
    } else if (!res->hubert.empty()) {
        out << "  警告: 未提供 RMVPE，将跳过 f0——该缓存暂不能用于 f0 模型训练。\n";
    } else {
        out << "  警告: 未提供 HuBERT，将跳过特征提取——该缓存不能用于训练。\n";
    }
    const std::string exp_def = res->target == 2 ? std::string("work/cache_auto_48k")
                                                 : std::string("work/cache_auto_40k");
    res->exp_dir = ask_nonempty("训练缓存输出目录", exp_def, in, out, &eof);

    print_gpu_devices(out);
    res->device = (int) ask_int("计算设备下标（-1=自动，99=CPU）", res->device, -1, 99, in, out,
                                &eof);

    out << "\n---- 预处理摘要 ----\n";
    out << "  输入       : " << res->input << "\n";
    out << "  目标       : " << (res->target == 2 ? "v2-48k（sr 48000 / 768 维特征）"
                                                  : "v1-40k（sr 40000 / 256 维特征）")
        << "\n";
    out << "  RMVPE      : " << (res->rmvpe.empty() ? "（跳过 f0）" : res->rmvpe) << "\n";
    out << "  HuBERT     : " << (res->hubert.empty() ? "（跳过特征提取）" : res->hubert) << "\n";
    out << "  缓存输出   : " << res->exp_dir << "\n";
    out << "  设备       : " << res->device << "（-1=自动）\n";
    out << "--------------------\n";

    const std::string yn = ask("确认并开始预处理? (y/N)", "n", in, out, &eof);
    if (!confirm_yes(yn)) {
        if (err != nullptr) {
            *err = "用户取消";
        }
        return false;
    }

    if (res->input.empty() || res->exp_dir.empty()) {
        if (err != nullptr) {
            *err = "向导：输入/缓存输出为空";
        }
        return false;
    }
    return true;
}

bool run_prep_wizard_from_file(const std::string& answers_path, std::ostream& out,
                               PrepWizardResult* res, std::string* err) {
    std::ifstream f(answers_path);
    if (!f.is_open()) {
        if (err != nullptr) {
            *err = "无法打开答案文件: " + answers_path;
        }
        return false;
    }
    return run_prep_wizard(f, out, res, err);
}

int run_prep_wizard_selftest() {
    // 固定答案（顺序同向导提问）：input / target / rmvpe / hubert / exp / device / confirm
    static const char* kAnswers =
        ".\n"
        "2\n"
        "\n"
        "\n"
        "work/prep_wiz_selftest\n"
        "-1\n"
        "y\n";

    std::printf("[prep-wizard-selftest] ==== 预处理向导自检（M8-2）====\n");
    PrepWizardResult res;
    std::istringstream in(kAnswers);
    std::string err;
    if (!run_prep_wizard(in, std::cout, &res, &err)) {
        std::printf("[prep-wizard-selftest] FAIL: 向导未完成: %s\n", err.c_str());
        return 1;
    }
    int fails = 0;
    auto check = [&](bool ok, const char* what) {
        if (!ok) {
            std::printf("[prep-wizard-selftest] FAIL: %s\n", what);
            ++fails;
        }
    };
    check(res.input == ".", "input 应为 .");
    check(res.target == 2, "target 应为 2（v2-48k）");
    check(res.rmvpe.empty(), "rmvpe 应为空");
    check(res.hubert.empty(), "hubert 应为空");
    check(res.exp_dir == "work/prep_wiz_selftest", "exp_dir 应为 work/prep_wiz_selftest");
    check(res.device == -1, "device 应为 -1（自动）");
    if (fails == 0) {
        std::printf("[prep-wizard-selftest] PASS: 向导答案逐字段符合预期\n");
        return 0;
    }
    return 1;
}

} // namespace prep
} // namespace rvc_train
