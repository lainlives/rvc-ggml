// RVC-Train-GGML - M7-5 续训自检实现（src/train/resume_selftest.cpp）
//
// 门 G-7.5：跨 epoch 的续训与不中断训练逐位一致；换配置/样本集明确报错。
#include "train/resume_selftest.h"

#include "train/resume_meta.h"
#include "train/trainer.h"

#include <algorithm>
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

struct StepState {
    int64_t  step = 0;
    int64_t  epoch = 0;
    int64_t  batch_pos = 0;
    float    lr_g = 0.0f;
    float    lr_d = 0.0f;
    uint64_t rng = 0;
};

StepState snapshot(const Trainer* t) {
    StepState s;
    s.step      = t->step_count();
    s.epoch     = t->epoch_index();
    s.batch_pos = t->batch_pos();
    s.lr_g      = t->lr_g();
    s.lr_d      = t->lr_d();
    s.rng       = t->rng_state();
    return s;
}

void print_state(const char* tag, const StepState& s) {
    std::printf("[resume-selftest] %s step=%lld epoch=%lld batch_pos=%lld lr_g=%.6e lr_d=%.6e "
                "rng=0x%016llx\n",
                tag, (long long) s.step, (long long) s.epoch, (long long) s.batch_pos,
                (double) s.lr_g, (double) s.lr_d, (unsigned long long) s.rng);
}

bool state_eq(const StepState& a, const StepState& b) {
    return a.step == b.step && a.epoch == b.epoch && a.batch_pos == b.batch_pos &&
           a.lr_g == b.lr_g && a.lr_d == b.lr_d && a.rng == b.rng;
}

TrainerConfig make_base_config(const std::string& weights_g, const std::string& weights_d,
                               const std::string& exp_dir, const std::string& out_dir,
                               int64_t bucket, int64_t batch, int64_t disc_len, int64_t grad_accum,
                               const std::string& config_path) {
    TrainerConfig cfg;
    // V2-6：非空时以配置文件为基底（train/data/model 段），随后覆盖测试参数。
    if (!config_path.empty()) {
        std::string cerr;
        if (!trainer_config_from_json(config_path, &cfg, &cerr)) {
            std::printf("[resume-selftest] 读取配置失败（回退内置默认）: %s\n", cerr.c_str());
            cfg = TrainerConfig{};
        }
    }
    cfg.weights_g     = weights_g;
    cfg.weights_d     = weights_d;
    cfg.exp_dir       = exp_dir;
    cfg.out_dir       = out_dir;
    cfg.batch_size    = batch > 0 ? batch : 1;
    cfg.grad_accum    = grad_accum > 0 ? grad_accum : 1;
    cfg.epochs        = 1000000;  // 仅由 total_steps 控制停止
    cfg.max_steps     = 0;
    cfg.bucket        = bucket > 0 ? bucket : 200;
    cfg.disc_len      = disc_len;
    cfg.log_interval  = 1;
    cfg.save_interval = 0;
    cfg.export_every  = 0;
    cfg.seed          = 1234;
    return cfg;
}

} // namespace

int run_resume_selftest(const std::string& weights_g, const std::string& weights_d,
                        const std::string& exp_dir, const std::string& out_dir,
                        traincpp::Device* dev, int64_t total_steps, int64_t pre_steps,
                        int64_t bucket, int64_t batch, int64_t disc_len, int64_t grad_accum,
                        const std::string& config_path) {
    if (dev == nullptr) {
        std::printf("[resume-selftest] 无可用设备\n");
        return 2;
    }
    if (total_steps < 2) {
        std::printf("[resume-selftest] --steps 必须 >= 2\n");
        return 2;
    }
    if (pre_steps <= 0 || pre_steps >= total_steps) {
        pre_steps = total_steps / 2;
        if (pre_steps <= 0) {
            pre_steps = 1;
        }
        std::printf("[resume-selftest] pre_steps 非法，自动取 %lld\n", (long long) pre_steps);
    }
    const int64_t bkt = bucket > 0 ? bucket : 200;
    const int64_t acc = grad_accum > 0 ? grad_accum : 1;
    std::printf("[resume-selftest] 设备=%s steps=%lld pre=%lld bucket=%lld batch=%lld disc_len=%lld "
                "grad_accum=%lld\n",
                dev->props().name.c_str(), (long long) total_steps, (long long) pre_steps,
                (long long) bkt, (long long) (batch > 0 ? batch : 1), (long long) disc_len,
                (long long) acc);

    const TrainerConfig base =
        make_base_config(weights_g, weights_d, exp_dir, out_dir, bkt, batch, disc_len, acc,
                         config_path);
    std::string err;
    int failed = 0;

    // ---------------- A：不中断参考（total_steps 步） ----------------
    std::unique_ptr<Trainer> a(Trainer::create(base, dev, &err));
    if (a == nullptr) {
        std::printf("[resume-selftest] A 创建失败: %s\n", err.c_str());
        return 1;
    }
    std::printf("[resume-selftest] A：samples=%lld skipped=%lld steps/epoch=%lld\n",
                (long long) a->trainable_samples(), (long long) a->skipped_samples(),
                (long long) a->steps_per_epoch());
    if (a->steps_per_epoch() <= 1) {
        std::printf("[resume-selftest] [WARN] steps/epoch=%lld，建议用小桶（如 --bucket 200）让跨 "
                    "epoch 更明显\n",
                    (long long) a->steps_per_epoch());
    }
    std::vector<TrainStepLosses> loss_a((size_t) total_steps);
    std::vector<StepState> state_a((size_t) total_steps);
    for (int64_t i = 0; i < total_steps; ++i) {
        if (!a->run_one_step(&err)) {
            std::printf("[resume-selftest] A step %lld 失败: %s\n", (long long) (i + 1), err.c_str());
            return 1;
        }
        loss_a[(size_t) i] = a->last_losses();
        state_a[(size_t) i] = snapshot(a.get());
        if (!finite6(loss_a[(size_t) i])) {
            std::printf("[resume-selftest] [FAIL] A step %lld loss 非有限\n", (long long) (i + 1));
            ++failed;
        }
    }
    a.reset();
    print_state("A 末态", state_a[(size_t) (total_steps - 1)]);

    // ---------------- B1：跑到 pre_steps 并保存 ----------------
    std::unique_ptr<Trainer> b1(Trainer::create(base, dev, &err));
    if (b1 == nullptr) {
        std::printf("[resume-selftest] [FAIL] B1 创建失败: %s\n", err.c_str());
        return 1;
    }
    for (int64_t i = 0; i < pre_steps; ++i) {
        if (!b1->run_one_step(&err)) {
            std::printf("[resume-selftest] B1 step %lld 失败: %s\n", (long long) (i + 1), err.c_str());
            return 1;
        }
    }
    const StepState state_pre = snapshot(b1.get());
    const StepState expect_pre = state_a[(size_t) (pre_steps - 1)];
    if (!state_eq(state_pre, expect_pre)) {
        std::printf("[resume-selftest] [FAIL] B1 存档点状态与 A 不一致\n");
        print_state("  B1", state_pre);
        print_state("  A ", expect_pre);
        ++failed;
    } else {
        std::printf("[resume-selftest] B1 存档点状态与 A 一致 OK\n");
    }
    const std::string ckpt = b1->checkpoint_path();
    if (!b1->save_checkpoint(ckpt, &err)) {
        std::printf("[resume-selftest] [FAIL] 保存 checkpoint 失败: %s\n", err.c_str());
        return 1;
    }
    b1.reset();

    // ---------------- B2：从 checkpoint 续跑到 total_steps ----------------
    TrainerConfig cfg_b2 = base;
    cfg_b2.resume = ckpt;
    std::unique_ptr<Trainer> b2(Trainer::create(cfg_b2, dev, &err));
    if (b2 == nullptr) {
        std::printf("[resume-selftest] [FAIL] B2 续训创建失败: %s\n", err.c_str());
        return 1;
    }
    const StepState state_load = snapshot(b2.get());
    if (!state_eq(state_load, expect_pre)) {
        std::printf("[resume-selftest] [FAIL] B2 载入状态与 A 存档点不一致\n");
        print_state("  载入", state_load);
        print_state("  A  ", expect_pre);
        ++failed;
    } else {
        std::printf("[resume-selftest] B2 载入状态（epoch/batch_pos/step/lr/RNG）与 A 一致 OK\n");
    }
    const int64_t remain = total_steps - pre_steps;
    double worst = 0.0;
    int64_t mism = 0;
    for (int64_t k = 0; k < remain; ++k) {
        if (!b2->run_one_step(&err)) {
            std::printf("[resume-selftest] B2 step %lld 失败: %s\n", (long long) (k + 1), err.c_str());
            return 1;
        }
        const TrainStepLosses& lb = b2->last_losses();
        const TrainStepLosses& la = loss_a[(size_t) (pre_steps + k)];
        const double d = max_abs6(lb, la);
        worst = std::max(worst, d);
        const StepState sb = snapshot(b2.get());
        if (d != 0.0 || !state_eq(sb, state_a[(size_t) (pre_steps + k)])) {
            ++mism;
            std::printf("[resume-selftest] [FAIL] 续训 step %lld loss max_abs=%.3e 或状态不一致\n",
                        (long long) (pre_steps + k + 1), d);
        }
    }
    std::printf("[resume-selftest] 续训段 %lld 步：loss max_abs=%.3e，不一致步=%lld\n",
                (long long) remain, worst, (long long) mism);
    if (mism != 0) {
        ++failed;
    } else {
        std::printf("[resume-selftest] 续训段 loss/状态逐位一致 OK\n");
    }
    b2.reset();

    // ---------------- 负向：换配置必须明确报错 ----------------
    struct Neg { const char* what; TrainerConfig cfg; };
    std::vector<Neg> negs;
    {
        TrainerConfig c = base; c.batch_size = base.batch_size == 2 ? 1 : 2; c.resume = ckpt;
        negs.push_back({"batch_size 改动", c});
    }
    {
        TrainerConfig c = base; c.bucket = base.bucket + 100; c.resume = ckpt;
        negs.push_back({"bucket 改动", c});
    }
    {
        TrainerConfig c = base; c.seed = base.seed + 1; c.resume = ckpt;
        negs.push_back({"seed 改动", c});
    }
    {
        TrainerConfig c = base; c.grad_accum = base.grad_accum + 1; c.resume = ckpt;
        negs.push_back({"grad_accum 改动", c});
    }
    {
        TrainerConfig c = base; c.lr = base.lr * 2.0f; c.resume = ckpt;
        negs.push_back({"learning_rate 改动", c});
    }
    for (const Neg& n : negs) {
        std::string nerr;
        std::unique_ptr<Trainer> t(Trainer::create(n.cfg, dev, &nerr));
        if (t != nullptr) {
            std::printf("[resume-selftest] [FAIL] %s：续训未报错（应拒绝）\n", n.what);
            ++failed;
        } else {
            std::printf("[resume-selftest] 负向 OK（%s 明确报错）：%s\n", n.what, nerr.c_str());
        }
    }

    // ---------------- 负向：篡改样本指纹必须报错 ----------------
    {
        const std::string mp = resume_meta_path_for(ckpt);
        ResumeMeta m;
        std::string merr;
        if (!resume_meta_read(mp, &m, &merr)) {
            std::printf("[resume-selftest] [FAIL] 读取 meta 失败: %s\n", merr.c_str());
            ++failed;
        } else {
            const ResumeMeta orig = m;
            m.sample_count += 1;
            if (!resume_meta_write(m, mp, &merr)) {
                std::printf("[resume-selftest] [FAIL] 篡改写 meta 失败: %s\n", merr.c_str());
                ++failed;
            } else {
                TrainerConfig c = base;
                c.resume = ckpt;
                std::string nerr;
                std::unique_ptr<Trainer> t(Trainer::create(c, dev, &nerr));
                if (t != nullptr) {
                    std::printf("[resume-selftest] [FAIL] 样本指纹篡改：续训未报错（应拒绝）\n");
                    ++failed;
                } else {
                    std::printf("[resume-selftest] 负向 OK（样本指纹改动明确报错）：%s\n", nerr.c_str());
                }
            }
            // 还原 meta（便于重复运行）
            std::string rerr;
            if (!resume_meta_write(orig, mp, &rerr)) {
                std::printf("[resume-selftest] [WARN] 还原 meta 失败: %s\n", rerr.c_str());
            }
        }
    }

    std::printf("[resume-selftest] %s（失败项 %d）\n", failed == 0 ? "通过" : "失败", failed);
    return failed == 0 ? 0 : 1;
}

} // namespace train
} // namespace rvc_train
