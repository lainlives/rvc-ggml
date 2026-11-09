// RVC-Train-GGML - M5-3 training loop self-test implementation (src/train/trainer_selftest.cpp)
//
// Gate G-5.3: N steps with finite loss; checkpoint save->load round trip (parameters/step/lr/RNG) consistent.
// Strong verification: after save run one more step (A); after load run one step at the same position (B); the 6 losses must be bit-identical.
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
    // V2-6: when non-empty, use the config file as the base (train/data/model sections), then override test parameters.
    if (!config_path.empty()) {
        std::string cerr;
        if (!trainer_config_from_json(config_path, &cfg, &cerr)) {
            std::printf("[trainer-selftest] failed to read config (falling back to built-in defaults): %s\n", cerr.c_str());
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
        std::printf("[trainer-selftest] no usable device\n");
        return 2;
    }
    if (steps <= 0) {
        std::printf("[trainer-selftest] --steps must be >= 1\n");
        return 2;
    }
    std::printf("[trainer-selftest] device=%s steps=%lld bucket=%lld batch=%lld disc_len=%lld "
                "grad_accum=%lld\n",
                dev->props().name.c_str(), (long long) steps, (long long) bucket,
                (long long) batch, (long long) disc_len, (long long) (grad_accum > 0 ? grad_accum : 1));

    const TrainerConfig cfg = make_selftest_config(weights_g, weights_d, exp_dir, out_dir, steps,
                                                   bucket, batch, disc_len, grad_accum, config_path);
    std::string err;
    int failed = 0;
    int64_t checked = 0;

    // ---------------- A: train steps steps ----------------
    std::unique_ptr<Trainer> a(Trainer::create(cfg, dev, &err));
    if (a == nullptr) {
        std::printf("[trainer-selftest] A create failed: %s\n", err.c_str());
        return 1;
    }
    {
        const TrainStepMemory m = a->train_step()->memory();
        std::printf("[trainer-selftest] A: samples=%lld skipped=%lld steps/epoch=%lld "
                    "buf_g=%.1f buf_d=%.1f MiB\n",
                    (long long) a->trainable_samples(), (long long) a->skipped_samples(),
                    (long long) a->steps_per_epoch(),
                    (double) m.buf_g_bytes / (1024.0 * 1024.0),
                    (double) m.buf_d_bytes / (1024.0 * 1024.0));
    }
    for (int64_t i = 0; i < steps; ++i) {
        if (!a->run_one_step(&err)) {
            std::printf("[trainer-selftest] A step %lld failed: %s\n", (long long) (i + 1), err.c_str());
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
                    (double) L.loss_mel, (double) L.loss_kl, fin ? "finite OK" : "[FAIL] not finite");
    }

    // ---------------- save + verify on A's side + save-point state ----------------
    const std::string ckpt = a->checkpoint_path();
    if (!a->save_checkpoint(ckpt, &err)) {
        std::printf("[trainer-selftest] [FAIL] %s\n", err.c_str());
        return 1;
    }
    if (!a->verify_checkpoint_memory(ckpt, &err)) {
        std::printf("[trainer-selftest] [FAIL] checkpoint_verify on A's side: %s\n", err.c_str());
        ++failed;
    } else {
        std::printf("[trainer-selftest] checkpoint_verify on A's side OK (parameters/step/lr/RNG match the file)\n");
    }
    const int64_t step_save = a->step_count();
    const float lr_save_g   = a->lr_g();
    const float lr_save_d   = a->lr_d();
    const uint64_t rng_save = a->rng_state();

    // After A's save, run one more step (to compare against B's first step after load)
    if (!a->run_one_step(&err)) {
        std::printf("[trainer-selftest] A single step after save failed: %s\n", err.c_str());
        return 1;
    }
    const TrainStepLosses loss_next_a = a->last_losses();
    const int64_t step_after_a   = a->step_count();
    const float lr_after_a_g     = a->lr_g();
    const uint64_t rng_after_a   = a->rng_state();
    a.reset();

    // ---------------- B: fresh Trainer loads the checkpoint ----------------
    TrainerConfig cfg_b = cfg;
    cfg_b.resume = ckpt;
    std::unique_ptr<Trainer> b(Trainer::create(cfg_b, dev, &err));
    if (b == nullptr) {
        std::printf("[trainer-selftest] [FAIL] B create/load failed: %s\n", err.c_str());
        return 1;
    }
    std::printf("[trainer-selftest] B: after load step=%lld epoch=%lld lr_g=%.6e lr_d=%.6e "
                "rng=0x%016llx (expected step=%lld lr_g=%.6e lr_d=%.6e rng=0x%016llx)\n",
                (long long) b->step_count(), (long long) b->epoch_index(), (double) b->lr_g(),
                (double) b->lr_d(), (unsigned long long) b->rng_state(), (long long) step_save,
                (double) lr_save_g, (double) lr_save_d, (unsigned long long) rng_save);
    if (b->step_count() != step_save || b->lr_g() != lr_save_g || b->lr_d() != lr_save_d ||
        b->rng_state() != rng_save) {
        std::printf("[trainer-selftest] [FAIL] loaded state does not match the save point\n");
        ++failed;
    } else {
        std::printf("[trainer-selftest] loaded state (step/lr/RNG) matches OK\n");
    }
    if (!b->verify_checkpoint_memory(ckpt, &err)) {
        std::printf("[trainer-selftest] [FAIL] checkpoint_verify on B's side: %s\n", err.c_str());
        ++failed;
    } else {
        std::printf("[trainer-selftest] checkpoint_verify on B's side OK (parameters/optimizer state match element by element)\n");
    }

    // B runs the same step that followed the save
    if (!b->run_one_step(&err)) {
        std::printf("[trainer-selftest] B single resumed step failed: %s\n", err.c_str());
        return 1;
    }
    const TrainStepLosses loss_next_b = b->last_losses();
    const double dmax = max_abs6(loss_next_a, loss_next_b);
    std::printf("[trainer-selftest] next-step 6-loss comparison: max_abs=%.3e (A step %lld -> B step %lld)\n",
                dmax, (long long) step_after_a, (long long) b->step_count());
    if (dmax != 0.0) {
        std::printf("[trainer-selftest] [FAIL] next resumed-step loss does not match A\n");
        ++failed;
    } else {
        std::printf("[trainer-selftest] next resumed-step loss bit-identical OK\n");
    }
    if (b->step_count() != step_after_a || b->lr_g() != lr_after_a_g ||
        b->rng_state() != rng_after_a) {
        std::printf("[trainer-selftest] [FAIL] step/lr/RNG after resume do not match A\n");
        ++failed;
    } else {
        std::printf("[trainer-selftest] step/lr/RNG after resume match OK\n");
    }

    std::printf("[trainer-selftest] %s (finite-loss checks %lld/%lld; failures %d)\n",
                failed == 0 ? "PASSED" : "FAILED", (long long) (checked - failed),
                (long long) checked, failed);
    return failed == 0 ? 0 : 1;
}

} // namespace train
} // namespace rvc_train
