// pipeline.cpp -- M4-1..M4-4 preprocessing core pipeline implementation (f0/feat/spec/cache + debug dump)

#include "prep/pipeline.h"

#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <system_error>

#include "prep/f0.h"
#include "prep/feat.h"
#include "prep/highpass.h"
#include "prep/norm.h"

namespace fs = std::filesystem;

namespace rvc_train::prep {
namespace {

constexpr double kButterCutoffHz = 48.0;
constexpr int kButterOrder = 5;

std::string join_path(const std::string& dir, const std::string& name) {
    return (fs::path(dir) / name).string();
}

bool ensure_dir(const std::string& dir, std::string* err) {
    std::error_code ec;
    fs::create_directories(dir, ec);
    if (ec) {
        if (err) {
            *err = "failed to create directory: " + dir + " (" + ec.message() + ")";
        }
        return false;
    }
    return true;
}

bool write_text(const std::string& path, const std::string& text, std::string* err) {
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    if (!f) {
        if (err) {
            *err = "cannot write: " + path;
        }
        return false;
    }
    f << text;
    return static_cast<bool>(f);
}

std::string fmt_hex(double v) {
    std::ostringstream ss;
    ss << std::setprecision(17) << v;
    return ss.str();
}

// Run the core pipeline (load already ready); dumping driven by opt.debug_dump_dir, caching by opt.cache_dir
bool run_prep_core(const MonoAudio& load, const std::string& case_name,
                   const std::string& source_name, const PrepOptions& opt, PrepFileResult* out,
                   std::string* err) {
    out->case_name = case_name;
    out->load_samples = static_cast<int64_t>(load.samples.size());

    const fs::path case_dir = fs::path(opt.debug_dump_dir) / case_name;
    const bool dump = !opt.debug_dump_dir.empty();
    if (dump && !ensure_dir(case_dir.string(), err)) {
        return false;
    }
    const bool cache = !opt.cache_dir.empty();
    if (cache && !ensure_dir((fs::path(opt.cache_dir) / "samples").string(), err)) {
        return false;
    }
    auto cache_path = [&](const std::string& id, const char* kind) {
        return (fs::path(opt.cache_dir) / cache_sample_relpath(id, kind)).string();
    };

    // M4-2: compute and dump f0 (RMVPE) for valid chunks; file naming matches golden
    auto handle_f0 = [&](const NormChunk& nc, int64_t idx1, ChunkInfo* info,
                         CacheSampleInfo* si) -> bool {
        if (!opt.with_f0 || opt.rmvpe == nullptr || !nc.valid) {
            return true;
        }
        const F0Result fr = compute_f0_rmvpe(nc.audio16k, *opt.rmvpe);
        info->f0_valid = fr.valid;
        info->f0_frames = static_cast<int64_t>(fr.raw.size());
        si->f0_frames = info->f0_frames;
        out->total_f0_frames += info->f0_frames;
        if (cache && fr.valid) {
            if (!write_raw_f32(cache_path(si->id, "f0hz"), fr.f0_hz, err) ||
                !write_raw_i32(cache_path(si->id, "f0bin"), fr.f0_bin, err)) {
                return false;
            }
            si->has_f0 = true;
        }
        if (dump) {
            const std::string tag = std::to_string(idx1);
            if (!write_raw_f32((case_dir / (case_name + ".f0raw." + tag + ".f32.raw")).string(),
                               fr.raw, err)) {
                return false;
            }
            if (fr.valid) {
                if (!write_raw_f32((case_dir / (case_name + ".f0." + tag + ".f32.raw")).string(),
                                   fr.f0_hz, err)) {
                    return false;
                }
                if (!write_raw_i32((case_dir / (case_name + ".f0bin." + tag + ".i32.raw")).string(),
                                   fr.f0_bin, err)) {
                    return false;
                }
            }
        }
        return true;
    };

    // M4-3: compute and dump HuBERT features for valid chunks (file naming matches golden; dims decided by version)
    auto handle_feat = [&](const NormChunk& nc, int64_t idx1, ChunkInfo* info,
                           CacheSampleInfo* si) -> bool {
        if (!opt.with_feat || opt.hubert == nullptr || !nc.valid || nc.audio16k.empty()) {
            return true;
        }
        const FeatResult fr =
            compute_hubert_feat(nc.audio16k, *opt.hubert, version_is_v2(opt.version));
        info->feat_valid = fr.frames > 0;
        info->feat_frames = fr.frames;
        info->feat_dim = fr.dim;
        si->feat_frames = fr.frames;
        si->feat_dim = fr.dim;
        out->total_feat_frames += fr.frames;
        if (cache && fr.frames > 0 && !fr.values.empty()) {
            if (!write_raw_f32(cache_path(si->id, "feat"), fr.values, err)) {
                return false;
            }
            si->has_feat = true;
        }
        if (dump) {
            const std::string tag = std::to_string(idx1);
            if (!write_raw_f32((case_dir / (case_name + ".feat." + tag + ".f32.raw")).string(),
                               fr.values, err)) {
                return false;
            }
        }
        return true;
    };

    // M4-4: linear spectrogram for valid chunks (input = 0_gt F32; official spectrogram_torch semantics, see spec.h)
    auto handle_spec = [&](const NormChunk& nc, int64_t idx1, ChunkInfo* info,
                           CacheSampleInfo* si) -> bool {
        if (!opt.with_spec || !nc.valid || nc.gt.empty()) {
            return true;
        }
        SpecResult sr;
        if (!compute_linear_spec(nc.gt.data(), static_cast<int64_t>(nc.gt.size()), opt.spec, &sr,
                                 err)) {
            return false;
        }
        info->spec_valid = sr.frames > 0;
        info->spec_frames = sr.frames;
        info->spec_bins = sr.bins;
        si->spec_frames = sr.frames;
        si->spec_bins = sr.bins;
        out->total_spec_frames += sr.frames;
        if (cache && !sr.values.empty()) {
            if (!write_raw_f32(cache_path(si->id, "spec"), sr.values, err)) {
                return false;
            }
            si->has_spec = true;
        }
        if (dump) {
            const std::string tag = std::to_string(idx1);
            if (!write_raw_f32((case_dir / (case_name + ".spec." + tag + ".f32.raw")).string(),
                               sr.values, err)) {
                return false;
            }
        }
        return true;
    };

    // ---------- 48Hz high-pass (causal lfilter, double) ----------
    std::vector<double> b;
    std::vector<double> a;
    butter_highpass_ba(kButterOrder, kButterCutoffHz, static_cast<double>(opt.target_sr), &b, &a);
    const std::vector<double> hp = lfilter(b, a, load.samples);

    if (dump) {
        if (!write_raw_f32((case_dir / (case_name + ".hp.f32.raw")).string(), to_f32(hp), err)) {
            return false;
        }
        std::ostringstream coeffs;
        coeffs << "# rvc-prep butter coeffs v1\n";
        coeffs << "order " << kButterOrder << " cutoff_hz " << kButterCutoffHz
               << " fs " << opt.target_sr << "\n";
        coeffs << "b";
        for (const double v : b) {
            coeffs << " " << fmt_hex(v);
        }
        coeffs << "\na";
        for (const double v : a) {
            coeffs << " " << fmt_hex(v);
        }
        coeffs << "\n";
        if (!write_text((case_dir / "coeffs.txt").string(), coeffs.str(), err)) {
            return false;
        }
    }

    // ---------- Slicer2 ----------
    const Slicer slicer(opt.target_sr, opt.slicer);
    out->slices = slicer.slice(hp);

    if (dump) {
        const SlicerDerived& d = slicer.derived();
        std::ostringstream sp;
        sp << "# rvc-prep slicer params v1\n";
        sp << "hop " << d.hop_size << " win " << d.win_size << " min_length " << d.min_length
           << " min_interval " << d.min_interval << " max_sil_kept " << d.max_sil_kept
           << " threshold " << fmt_hex(d.threshold) << "\n";
        sp << "# sil_tags";
        for (const auto& t : slicer.last_sil_tags()) {
            sp << " (" << t.first << "," << t.second << ")";
        }
        sp << "\n";
        sp << "# events\n" << slicer.last_debug();
        if (!write_text((case_dir / "slicer.txt").string(), sp.str(), err)) {
            return false;
        }
        const std::vector<double>& rms = slicer.last_rms();
        std::ostringstream rp;
        rp << "# rvc-prep rms v1\n";
        for (const double v : rms) {
            rp << fmt_hex(v) << "\n";
        }
        if (!write_text((case_dir / "rms.txt").string(), rp.str(), err)) {
            return false;
        }

        std::ostringstream ss;
        ss << "# rvc-prep slices v1\n# begin_sample end_sample\n";
        for (const SliceRange& r : out->slices) {
            ss << r.begin_sample << " " << r.end_sample << "\n";
        }
        if (!write_text((case_dir / "slices.txt").string(), ss.str(), err)) {
            return false;
        }
    }

    // ---------- 3.7s chunking + peak normalization ----------
    std::ostringstream chunks;
    chunks << "# rvc-prep chunks v1\n# index slice start_in_slice n_samples valid\n";

    const int64_t step = static_cast<int64_t>(
        static_cast<double>(opt.target_sr) * (opt.per - opt.overlap));  // int(...) truncation
    const double tail_seconds = opt.per + opt.overlap;
    const int64_t take = static_cast<int64_t>(static_cast<double>(opt.target_sr) * opt.per);

    int idx1 = 0;
    std::vector<double> tmp_audio;  // official line 109 uses the tmp_audio of its last assignment
    for (std::size_t si = 0; si < out->slices.size(); ++si) {
        const SliceRange& r = out->slices[si];
        const int64_t len = r.end_sample - r.begin_sample;
        int64_t i = 0;
        while (true) {
            const int64_t start = step * i;
            ++i;
            const int64_t remain = len > start ? len - start : 0;
            const bool has_tail = static_cast<double>(remain) > tail_seconds * static_cast<double>(opt.target_sr);
            const int64_t count = has_tail ? take : remain;
            const int64_t hp_begin = r.begin_sample + start;  // sample offset relative to the hp waveform
            tmp_audio.assign(hp.begin() + static_cast<std::ptrdiff_t>(hp_begin),
                             hp.begin() + static_cast<std::ptrdiff_t>(hp_begin + count));

            if (has_tail) {
                const NormChunk nc = normalize_chunk(tmp_audio, opt.target_sr, !opt.no_16k);
                ChunkInfo info;
                info.index = idx1;
                info.slice_index = static_cast<int>(si);
                info.start_in_slice = start;
                info.n_samples = count;
                info.valid = nc.valid;
                CacheSampleInfo csi;
                csi.id = case_name + "_" + std::to_string(idx1);
                csi.source = source_name;
                csi.chunk = idx1;
                if (nc.valid) {
                    csi.gt_samples = static_cast<int64_t>(nc.gt.size());
                    csi.frames16k = static_cast<int64_t>(nc.audio16k.size());
                    csi.has_16k = !opt.no_16k && !nc.audio16k.empty();
                }
                if (!handle_f0(nc, idx1, &info, &csi) || !handle_feat(nc, idx1, &info, &csi) ||
                    !handle_spec(nc, idx1, &info, &csi)) {
                    return false;
                }
                out->chunks.push_back(info);
                if (nc.valid) {
                    if (cache) {
                        if (!write_raw_f32(cache_path(csi.id, "gt"), nc.gt, err)) {
                            return false;
                        }
                        if (csi.has_16k &&
                            !write_raw_f32(cache_path(csi.id, "16k"), nc.audio16k, err)) {
                            return false;
                        }
                    }
                    out->samples.push_back(csi);
                }
                if (dump) {
                    if (nc.valid) {
                        if (!write_raw_f32((case_dir / (case_name + ".gt." + std::to_string(idx1) +
                                                      ".f32.raw"))
                                               .string(),
                                           nc.gt, err)) {
                            return false;
                        }
                        out->total_gt_samples += static_cast<int64_t>(nc.gt.size());
                        if (!opt.no_16k) {
                            if (!write_raw_f32((case_dir / (case_name + ".16k." + std::to_string(idx1) +
                                                           ".f32.raw"))
                                                   .string(),
                                               nc.audio16k, err)) {
                                return false;
                            }
                            out->total_16k_samples += static_cast<int64_t>(nc.audio16k.size());
                        }
                    }
                } else if (nc.valid) {
                    out->total_gt_samples += static_cast<int64_t>(nc.gt.size());
                    out->total_16k_samples += static_cast<int64_t>(nc.audio16k.size());
                }
                chunks << idx1 << " " << si << " " << start << " " << count << " "
                       << (nc.valid ? 1 : 0) << "\n";
                ++idx1;
            } else {
                ++idx1;
                break;
            }
        }
    }

    // official line 109: write the last chunk's tail after the for loop ends (with the current idx1)
    if (!out->slices.empty()) {
        const NormChunk nc = normalize_chunk(tmp_audio, opt.target_sr, !opt.no_16k);
        ChunkInfo info;
        info.index = idx1;
        info.slice_index = static_cast<int>(out->slices.size()) - 1;
        info.start_in_slice = -1;  // official does not record this here (final tail chunk)
        info.n_samples = static_cast<int64_t>(tmp_audio.size());
        info.valid = nc.valid;
        CacheSampleInfo csi;
        csi.id = case_name + "_" + std::to_string(idx1);
        csi.source = source_name;
        csi.chunk = idx1;
        if (nc.valid) {
            csi.gt_samples = static_cast<int64_t>(nc.gt.size());
            csi.frames16k = static_cast<int64_t>(nc.audio16k.size());
            csi.has_16k = !opt.no_16k && !nc.audio16k.empty();
        }
        if (!handle_f0(nc, idx1, &info, &csi) || !handle_feat(nc, idx1, &info, &csi) ||
            !handle_spec(nc, idx1, &info, &csi)) {
            return false;
        }
        out->chunks.push_back(info);
        if (nc.valid) {
            if (cache) {
                if (!write_raw_f32(cache_path(csi.id, "gt"), nc.gt, err)) {
                    return false;
                }
                if (csi.has_16k && !write_raw_f32(cache_path(csi.id, "16k"), nc.audio16k, err)) {
                    return false;
                }
            }
            out->samples.push_back(csi);
        }
        if (dump && nc.valid) {
            if (!write_raw_f32((case_dir / (case_name + ".gt." + std::to_string(idx1) + ".f32.raw"))
                                   .string(),
                               nc.gt, err)) {
                return false;
            }
            out->total_gt_samples += static_cast<int64_t>(nc.gt.size());
            if (!opt.no_16k) {
                if (!write_raw_f32((case_dir / (case_name + ".16k." + std::to_string(idx1) +
                                               ".f32.raw"))
                                       .string(),
                                   nc.audio16k, err)) {
                    return false;
                }
                out->total_16k_samples += static_cast<int64_t>(nc.audio16k.size());
            }
        } else if (!dump && nc.valid) {
            out->total_gt_samples += static_cast<int64_t>(nc.gt.size());
            out->total_16k_samples += static_cast<int64_t>(nc.audio16k.size());
        }
        chunks << idx1 << " " << (out->slices.size() - 1) << " -1 " << info.n_samples << " "
               << (nc.valid ? 1 : 0) << "\n";
    }

    if (dump && !write_text((case_dir / "chunks.txt").string(), chunks.str(), err)) {
        return false;
    }
    if (err) {
        err->clear();
    }
    return true;
}

} // namespace

bool run_prep_wav(const std::string& wav_path, const std::string& case_name,
                  const PrepOptions& opt, PrepFileResult* out, std::string* err) {
    MonoAudio load;
    if (!load_wav_mono(wav_path, opt.target_sr, &load, err)) {
        return false;
    }

    if (!opt.debug_dump_dir.empty()) {
        const fs::path case_dir = fs::path(opt.debug_dump_dir) / case_name;
        if (!ensure_dir(case_dir.string(), err)) {
            return false;
        }
        if (!write_raw_f32((case_dir / (case_name + ".load.f32.raw")).string(), to_f32(load.samples),
                           err)) {
            return false;
        }
    }
    return run_prep_core(load, case_name, fs::path(wav_path).filename().string(), opt, out, err);
}

bool run_prep_injected(const MonoAudio& load, const std::string& case_name,
                       const PrepOptions& opt, PrepFileResult* out, std::string* err) {
    return run_prep_core(load, case_name, "", opt, out, err);
}

} // namespace rvc_train::prep
