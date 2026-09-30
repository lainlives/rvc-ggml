// RVC-Train-GGML - M7-2 scratch 规格对拍自检实现（src/train/scratch_selftest.cpp）
#include "train/scratch_selftest.h"

#include <array>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "export/mem_to_train_gguf.h"
#include "model/config.h"
#include "model/scratch.h"
#include "model/weights.h"
#include "train/trainer.h"

#include "traincpp/traincpp.h"

namespace rvc_train {
namespace train {

namespace {

std::string join(const std::string& dir, const std::string& name) {
    if (dir.empty()) {
        return name;
    }
    const char last = dir.back();
    return (last == '/' || last == '\\') ? dir + name : dir + "\\" + name;
}

bool parse_hashes(const std::string& path,
                  std::unordered_map<std::string, std::array<int64_t, 4>>* out, std::string* err) {
    std::ifstream f(path);
    if (!f.is_open()) {
        if (err != nullptr) {
            *err = "无法打开哈希清单: " + path;
        }
        return false;
    }
    out->clear();
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty()) {
            continue;
        }
        std::istringstream ss(line);
        std::string name;
        int64_t ne[4] = {1, 1, 1, 1};
        if (!(ss >> name >> ne[0] >> ne[1] >> ne[2] >> ne[3])) {
            continue;
        }
        (*out)[name] = {ne[0], ne[1], ne[2], ne[3]};
    }
    return true;
}

int compare_specs(const std::string& tag, const std::vector<model::TensorSpec>& specs,
                  const std::string& hash_path) {
    std::unordered_map<std::string, std::array<int64_t, 4>> want;
    std::string err;
    if (!parse_hashes(hash_path, &want, &err)) {
        std::printf("[scratch-selftest] %s FAIL: %s\n", tag.c_str(), err.c_str());
        return 1;
    }
    std::unordered_map<std::string, std::array<int64_t, 4>> got;
    for (const model::TensorSpec& s : specs) {
        got[s.name] = s.ne;
    }
    int mism = 0;
    int missing = 0;   // 期望有、生成没有
    int extra = 0;     // 生成有、期望没有
    for (const auto& kv : want) {
        const auto it = got.find(kv.first);
        if (it == got.end()) {
            if (missing < 5) {
                std::printf("[scratch-selftest] %s 缺失: %s\n", tag.c_str(), kv.first.c_str());
            }
            ++missing;
        } else if (it->second != kv.second) {
            if (mism < 5) {
                std::printf("[scratch-selftest] %s 形状不符: %s 期望 [%lld,%lld,%lld,%lld] 实得 "
                            "[%lld,%lld,%lld,%lld]\n",
                            tag.c_str(), kv.first.c_str(),
                            (long long) kv.second[0], (long long) kv.second[1],
                            (long long) kv.second[2], (long long) kv.second[3],
                            (long long) it->second[0], (long long) it->second[1],
                            (long long) it->second[2], (long long) it->second[3]);
            }
            ++mism;
        }
    }
    for (const auto& kv : got) {
        if (want.find(kv.first) == want.end()) {
            if (extra < 5) {
                std::printf("[scratch-selftest] %s 多余: %s\n", tag.c_str(), kv.first.c_str());
            }
            ++extra;
        }
    }
    if (missing == 0 && extra == 0 && mism == 0) {
        std::printf("[scratch-selftest] %s PASS: %zu 张量名称+形状与夹具一致\n", tag.c_str(),
                    specs.size());
        return 0;
    }
    std::printf("[scratch-selftest] %s FAIL: 缺失=%d 多余=%d 形状不符=%d（共 %zu）\n", tag.c_str(),
                missing, extra, mism, specs.size());
    return 1;
}

} // namespace

int run_scratch_selftest(const std::string& config_path, const std::string& g_hashes,
                         const std::string& d_hashes, const std::string& out_dir) {
    std::printf("[scratch-selftest] ==== 从零规格对拍（G-7.2）====\n");
    TrainerConfig cfg;
    std::string err;
    if (!trainer_config_from_json(config_path, &cfg, &err)) {
        std::printf("[scratch-selftest] FAIL: 读取配置失败: %s\n", err.c_str());
        return 1;
    }
    // V2-2：hash 清单缺省时按 config 版本选择（v1→f0G/D40k；v2→f0G/D48k），避免硬编码 v1。
    std::string gh = g_hashes;
    std::string dh = d_hashes;
    const bool is_v2 = cfg.model_cfg.is_v2();
    if (gh.empty()) {
        gh = is_v2 ? "work/models/f0G48k.hashes.txt" : "work/models/f0G40k.hashes.txt";
    }
    if (dh.empty()) {
        dh = is_v2 ? "work/models/f0D48k.hashes.txt" : "work/models/f0D40k.hashes.txt";
    }
    model::ModelConfig g = cfg.model_cfg;
    g.kind = "G";
    model::ModelConfig d = cfg.model_cfg;
    d.kind = "D";
    if (!g.validate(&err)) {
        std::printf("[scratch-selftest] FAIL: G 配置非法: %s\n", err.c_str());
        return 1;
    }
    if (!d.validate(&err)) {
        std::printf("[scratch-selftest] FAIL: D 配置非法: %s\n", err.c_str());
        return 1;
    }

    std::vector<model::TensorSpec> gs;
    std::vector<model::TensorSpec> ds;
    if (!model::build_g_specs(g, &gs, &err) || !model::build_d_specs(d, &ds, &err)) {
        std::printf("[scratch-selftest] FAIL: 生成规格失败: %s\n", err.c_str());
        return 1;
    }

    int fails = 0;
    fails += compare_specs("G", gs, gh);
    fails += compare_specs("D", ds, dh);

    // 生成 GGUF + 结构/元数据自检
    const std::string gp = join(out_dir, "scratch_g.train.gguf");
    const std::string dp = join(out_dir, "scratch_d.train.gguf");
    if (!exporter::generate_scratch_g(g, cfg.seed, gp, &err) ||
        !exporter::generate_scratch_d(d, cfg.seed + 1, dp, &err)) {
        std::printf("[scratch-selftest] FAIL: 生成 GGUF 失败: %s\n", err.c_str());
        return 1;
    }
    std::printf("[scratch-selftest] 已生成: %s / %s\n", gp.c_str(), dp.c_str());
    if (!traincpp::gguf_validate(gp.c_str()) || !traincpp::gguf_validate(dp.c_str())) {
        std::printf("[scratch-selftest] FAIL: 生成的 GGUF 结构自检失败\n");
        ++fails;
    }
    for (const std::string& p : {gp, dp}) {
        model::TrainWeights* w = model::TrainWeights::open(p, &err);
        if (w == nullptr) {
            std::printf("[scratch-selftest] FAIL: 打开 %s 失败: %s\n", p.c_str(), err.c_str());
            ++fails;
            continue;
        }
        model::TrainMeta meta;
        w->read_meta(&meta);
        model::ModelConfig mc = model::ModelConfig::from_meta(meta);
        if (!mc.validate(&err)) {
            std::printf("[scratch-selftest] FAIL: %s 元数据配置非法: %s\n", p.c_str(), err.c_str());
            ++fails;
        } else {
            std::printf("[scratch-selftest] %s 元数据 OK（kind=%s version=%s tensors=%lld）\n",
                        p.c_str(), meta.kind.c_str(), meta.version.c_str(),
                        (long long) meta.tensor_count);
        }
        delete w;
    }

    if (fails == 0) {
        std::printf("[scratch-selftest] PASS\n");
        return 0;
    }
    std::printf("[scratch-selftest] FAIL: %d 项\n", fails);
    return 1;
}

} // namespace train
} // namespace rvc_train
