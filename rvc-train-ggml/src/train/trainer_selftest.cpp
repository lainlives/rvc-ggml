// RVC-Train-GGML - M5-3 训练循环自检实现（src/train/trainer_selftest.cpp）
//
// 门 G-5.3：N 步 loss 有限；checkpoint save→load 往返（参数/step/lr/RNG）一致。
// 强验证：save 后再跑一步（A），载入后同位置再跑一步（B），要求 6 loss 逐位一致。
#include "train/trainer_selftest.h"

#include "train/trainer.h"

#include <cmath>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

namespace rvc_train {
namespace train {

namespace {

bool finite6(const TrainStepLosses& L) {
    return std::isfinite(L.loss_disc) && std::isfinite(L.loss_gen_all) &&
           std::isfinite(L.loss_gen) && std::isfinite(L.loss_fm) &&
           std::isfinite(L.loss_mel) && std::isfinite(L.loss_kl);
}

double max_abs6(const TrainStepLosses& a, const TrainStepLosses& b) {
    double m = 0.0;
    m = std::max(m, std::fabs((double) a.loss_disc - (double) b.loss_disc));
    m = std::max(m, std::fabs((double) a.loss_gen_all - (double) b.loss_gen_all));
    m = std::max(m, std::fabs((double) a.loss_gen - (double) b.loss_gen));
    m = std::max(m, std::fabs((double) a.loss_fm - (double) b.loss_fm));
    m = std::max(m, std::fabs((double) a.loss_mel - (double) b.loss_mel));
    m = std::max(m, std::fabs((double) a.loss_kl - (double) b.loss_kl));
    return m;
}

TrainerConfig make_selftest_config(const std::string& weights_g, const std::string& weights_d,
                                   const std::string& exp_dir, const std::string& out_dir,
                                   int64_t steps, int64_t bucket, int64_t batch, int64_t disc_len,
                                   int64_t grad_accum, const std::string& config_path) {
    TrainerConfig cfg;
    // V2-6：非空时以配置文件为基底（train/data/model 段），随后覆盖测试参数。
    if (!config_path.empty()) {
        std::string cerr;
        if (!trainer_config_from_json(config_path, &cfg, &cerr)) {
            std::printf("[trainer-selftest] 读取配置失败（回退内置默认）: %s\n", cerr.c_str());
            cfg = TrainerConfig{};
        }
    }
    cfg.weights_g = weights_g;
    cfg.weights_d = weights_d;
    cfg.exp_dir   = exp_dir;
    cfg.out_dir   = out_dir;
    cfg.batch_size    = batch > 0 ? batch : 1;
    cfg.grad_accum    = grad_accum > 0 ? grad_accum : 1;
    cfg.epochs        = 2;
    cfg.max_steps     = 0;
    cfg.bucket        = bucket > 0 ? bucket : 384;
    cfg.disc_len      = disc_len > 0 ? disc_len : 0;
    cfg.log_interval  = 1;
    cfg.save_interval = 0;
    cfg.seed          = 1234;
    (void) steps;
    return cfg;
}

} // namespace

int run_trainer_selftest(const std::string& weights_g, const std::string& weights_d,
                         const std::string& exp_dir, const std::string& out_dir,
                         traincpp::Device* dev, int64_t steps, int64_t bucket, int64_t batch,
                         int64_t disc_len, int64_t grad_accum, const std::string& config_path) {
    if (dev == nullptr) {
        std::printf("[trainer-selftest] 无可用设备\n");
        return 2;
    }
    if (steps <= 0) {
        std::printf("[trainer-selftest] --steps 必须 >= 1\n");
        return 2;
    }
    std::printf("[trainer-selftest] 设备=%s steps=%lld bucket=%lld batch=%lld disc_len=%lld "
                "grad_accum=%lld\n",
                dev->props().name.c_str(), (long long) steps, (long long) bucket,
                (long long) batch, (long long) disc_len, (long long) (grad_accum > 0 ? grad_accum : 1));

    const TrainerConfig cfg = make_selftest_config(weights_g, weights_d, exp_dir, out_dir, steps,
                                                   bucket, batch, disc_len, grad_accum, config_path);
    std::string err;
    int failed = 0;
    int64_t checked = 0;

    // ---------------- A：训练 steps 步 ----------------
    std::unique_ptr<Trainer> a(Trainer::create(cfg, dev, &err));
    if (a == nullptr) {
        std::printf("[trainer-selftest] A 创建失败: %s\n", err.c_str());
        return 1;
    }
    {
        const TrainStepMemory m = a->train_step()->memory();
        std::printf("[trainer-selftest] A：samples=%lld skipped=%lld steps/epoch=%lld "
                    "buf_g=%.1f buf_d=%.1f MiB\n",
                    (long long) a->trainable_samples(), (long long) a->skipped_samples(),
                    (long long) a->steps_per_epoch(),
                    (double) m.buf_g_bytes / (1024.0 * 1024.0),
                    (double) m.buf_d_bytes / (1024.0 * 1024.0));
    }
    for (int64_t i = 0; i < steps; ++i) {
        if (!a->run_one_step(&err)) {
            std::printf("[trainer-selftest] A step %lld 失败: %s\n", (long long) (i + 1), err.c_str());
            return 1;
        }
        const TrainStepLosses& L = a->last_losses();
        const bool fin = finite6(L);
        ++checked;
        if (!fin) {
            ++failed;
        }
        std::printf("[trainer-selftest] A step %lld lr_g=%.6e loss_disc=%.6f loss_gen_all=%.6f "
                    "loss_gen=%.6f loss_fm=%.6f loss_mel=%.6f loss_kl=%.6f %s\n",
                    (long long) (i + 1), (double) a->lr_g(), (double) L.loss_disc,
                    (double) L.loss_gen_all, (double) L.loss_gen, (double) L.loss_fm,
                    (double) L.loss_mel, (double) L.loss_kl, fin ? "有限 OK" : "[FAIL] 非有限");
    }

    // ---------------- 保存 + A 侧 verify + 存档点状态 ----------------
    const std::string ckpt = a->checkpoint_path();
    if (!a->save_checkpoint(ckpt, &err)) {
        std::printf("[trainer-selftest] [FAIL] %s\n", err.c_str());
        return 1;
    }
    if (!a->verify_checkpoint_memory(ckpt, &err)) {
        std::printf("[trainer-selftest] [FAIL] A 侧 checkpoint_verify: %s\n", err.c_str());
        ++failed;
    } else {
        std::printf("[trainer-selftest] A 侧 checkpoint_verify OK（参数/step/lr/RNG 与文件一致）\n");
    }
    const int64_t step_save = a->step_count();
    const float lr_save_g   = a->lr_g();
    const float lr_save_d   = a->lr_d();
    const uint64_t rng_save = a->rng_state();

    // A 存档后再跑一步（与 B 载入后的下一步对比）
    if (!a->run_one_step(&err)) {
        std::printf("[trainer-selftest] A 存档后单步失败: %s\n", err.c_str());
        return 1;
    }
    const TrainStepLosses loss_next_a = a->last_losses();
    const int64_t step_after_a   = a->step_count();
    const float lr_after_a_g     = a->lr_g();
    const uint64_t rng_after_a   = a->rng_state();
    a.reset();

    // ---------------- B：全新 Trainer 载入 checkpoint ----------------
    TrainerConfig cfg_b = cfg;
    cfg_b.resume = ckpt;
    std::unique_ptr<Trainer> b(Trainer::create(cfg_b, dev, &err));
    if (b == nullptr) {
        std::printf("[trainer-selftest] [FAIL] B 创建/载入失败: %s\n", err.c_str());
        return 1;
    }
    std::printf("[trainer-selftest] B：载入后 step=%lld epoch=%lld lr_g=%.6e lr_d=%.6e "
                "rng=0x%016llx（期望 step=%lld lr_g=%.6e lr_d=%.6e rng=0x%016llx）\n",
                (long long) b->step_count(), (long long) b->epoch_index(), (double) b->lr_g(),
                (double) b->lr_d(), (unsigned long long) b->rng_state(), (long long) step_save,
                (double) lr_save_g, (double) lr_save_d, (unsigned long long) rng_save);
    if (b->step_count() != step_save || b->lr_g() != lr_save_g || b->lr_d() != lr_save_d ||
        b->rng_state() != rng_save) {
        std::printf("[trainer-selftest] [FAIL] 载入状态与存档点不一致\n");
        ++failed;
    } else {
        std::printf("[trainer-selftest] 载入状态（step/lr/RNG）一致 OK\n");
    }
    if (!b->verify_checkpoint_memory(ckpt, &err)) {
        std::printf("[trainer-selftest] [FAIL] B 侧 checkpoint_verify: %s\n", err.c_str());
        ++failed;
    } else {
        std::printf("[trainer-selftest] B 侧 checkpoint_verify OK（参数/优化器状态逐元素一致）\n");
    }

    // B 跑存档后的同一步
    if (!b->run_one_step(&err)) {
        std::printf("[trainer-selftest] B 续跑单步失败: %s\n", err.c_str());
        return 1;
    }
    const TrainStepLosses loss_next_b = b->last_losses();
    const double dmax = max_abs6(loss_next_a, loss_next_b);
    std::printf("[trainer-selftest] 下一步 6 loss 对比：max_abs=%.3e（A step %lld → B step %lld）\n",
                dmax, (long long) step_after_a, (long long) b->step_count());
    if (dmax != 0.0) {
        std::printf("[trainer-selftest] [FAIL] 续训下一步 loss 与 A 不一致\n");
        ++failed;
    } else {
        std::printf("[trainer-selftest] 续训下一步 loss 逐位一致 OK\n");
    }
    if (b->step_count() != step_after_a || b->lr_g() != lr_after_a_g ||
        b->rng_state() != rng_after_a) {
        std::printf("[trainer-selftest] [FAIL] 续训后 step/lr/RNG 与 A 不一致\n");
        ++failed;
    } else {
        std::printf("[trainer-selftest] 续训后 step/lr/RNG 一致 OK\n");
    }

    std::printf("[trainer-selftest] %s（loss 有限检查 %lld/%lld；失败项 %d）\n",
                failed == 0 ? "通过" : "失败", (long long) (checked - failed),
                (long long) checked, failed);
    return failed == 0 ? 0 : 1;
}

} // namespace train
} // namespace rvc_train
