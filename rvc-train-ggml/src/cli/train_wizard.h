// RVC-Train-GGML - M7 交互式训练向导（src/cli/train_wizard.h）
//
// 职责：以"运行程序后导向式输入"的方式收集训练参数，填充 TrainerConfig；
// 由 main_train.cpp 在确认后落盘 <out>/train_config.json 并直接开始训练。
//
// 语义基准：`交接说明` §4.11 C。**微调（载入预训练训练态 GGUF）与从零（--scratch，M7-2 起）
// 两条分支**均支持。
//
// 设计要点：
//   - 可注入输入/输出流（run_train_wizard），便于用固定答案做等价自检与批处理；
//   - 中文提示 + 默认值（直接回车采用默认）；输入非法时重试；输入流结束（EOF）时采用默认继续。
#pragma once

#include <iosfwd>
#include <string>

#include "train/trainer.h"

namespace rvc_train {
namespace train {

// 交互式向导：从 in 逐行读取答案、向 out 打印提示，填充 *cfg。
// cfg 可先由 base config 初始化（提供 train/data/model 默认值）；向导只覆盖其询问的字段。
// 用户拒绝确认（y/N 输入 n）或参数无法解析时返回 false 并写 err。
bool run_train_wizard(std::istream& in, std::ostream& out, TrainerConfig* cfg, std::string* err);

// 从答案文件读取（非交互；answers_path 每行一个答案，顺序同向导提问）。提示仍打印到 out。
bool run_train_wizard_from_file(const std::string& answers_path, std::ostream& out,
                                TrainerConfig* cfg, std::string* err);

// G-7.1 等价自检：内置一组合法答案驱动向导生成配置，与内置的"手写 JSON"黄金配置
// 逐字段比较（trainer_config_equivalent）。返回 0 = PASS，1 = FAIL。
int run_train_wizard_selftest();

} // namespace train
} // namespace rvc_train
