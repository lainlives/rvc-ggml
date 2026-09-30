// RVC-Train-GGML - M5-1 数据加载器对拍入口（src/train/dataset_selftest.h）
#pragma once

#include <string>

namespace rvc_train {
namespace train {

// 对拍 M5-1 数据加载器（纯主机，无需设备）：
//   --golden <dir>（manifest.dataset.json + raw） --exp <M4 缓存目录>
// 返回 0 = 全部通过；1 = 失败；2 = 参数/环境错误。
int run_dataset_selftest(const std::string& golden_dir, const std::string& exp_dir);

} // namespace train
} // namespace rvc_train
