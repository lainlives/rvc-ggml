// RVC-Train-GGML - M7-3 梯度累积等价自检（src/train/grad_accum_selftest.h）
//
// 门 G-7.3（用户拍板：梯度级等价为主）：
//   同权重、同数据下，A 臂（batch=B, grad_accum=K，跑 run_step_accum）与
//   B 臂（batch=B×K, grad_accum=1，跑 run_step）的**参数梯度**应等价
//   （累积图内损失按 1/K 缩放 → 累积梯度 == 全批量均值梯度）。
// 小规模 Vulkan/CPU 均可；本机用 Vulkan 小档（bucket=384/B=1/K=2/disc=1024）。
#pragma once

#include <cstdint>
#include <string>

namespace traincpp {
class Device;
}

namespace rvc_train {
namespace train {

// weights_g/weights_d = 训练态 GGUF；exp_dir = M4 缓存。
// bucket = 图 T；batch = 每 micro 的样本数 B（>=1）；grad_accum = 累积步数 K（>=2）；
// disc_len = D 输入长度（<=0 = 官方整段）。返回 0 = 门通过；1 = 失败；2 = 参数/环境错误。
int run_grad_accum_selftest(const std::string& weights_g, const std::string& weights_d,
                            const std::string& exp_dir, traincpp::Device* dev, int64_t bucket,
                            int64_t batch, int64_t grad_accum, int64_t disc_len);

} // namespace train
} // namespace rvc_train
