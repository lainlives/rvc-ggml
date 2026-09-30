// RVC-Train-GGML - M8-2 rvc-prep 交互式预处理向导（src/cli/prep_wizard.h）
//
// 职责：以"运行程序后导向式输入"的方式收集预处理参数（输入音频/目标版本/RMVPE/HuBERT/
// 缓存输出/设备），由 main_prep.cpp 在确认后执行与 --input 等价的预处理管线，并在完成后
// 自动 --validate-cache 自校验、报告样本数与"下一步"提示。
//
// 设计要点（与 train_wizard 一致）：
//   - 可注入输入/输出流，便于用固定答案做自检与批处理（--answers）；
//   - 中文提示 + 默认值（直接回车采用默认）；输入非法时重试；EOF 时采用默认继续；
//   - 仅显式 --interactive / --answers 进入，不影响既有命令行用法。
#pragma once

#include <iosfwd>
#include <string>

namespace rvc_train {
namespace prep {

struct PrepWizardResult {
    std::string input;    // 音频目录或单个 wav
    int target = 1;       // 1 = v1-40k（sr 40000）；2 = v2-48k（sr 48000）
    std::string rmvpe;    // 空 = 跳过 f0
    std::string hubert;   // 空 = 跳过特征提取（训练必需，向导会提示）
    std::string exp_dir;  // 训练缓存输出目录
    int device = -1;      // -1 = 自动（GPU → CPU 99）；0..n-1 = ggml 设备；99 = CPU
};

// 交互式向导：从 in 逐行读取答案、向 out 打印提示，填充 *res。
// 用户拒绝确认（y/N 输入 n）时返回 false 并写 err。
bool run_prep_wizard(std::istream& in, std::ostream& out, PrepWizardResult* res, std::string* err);

// 从答案文件读取（非交互；answers_path 每行一个答案，顺序同向导提问）。
bool run_prep_wizard_from_file(const std::string& answers_path, std::ostream& out,
                               PrepWizardResult* res, std::string* err);

// 自检：内置答案驱动向导，与预期 PrepWizardResult 逐字段比较。0 = PASS，1 = FAIL。
int run_prep_wizard_selftest();

} // namespace prep
} // namespace rvc_train
