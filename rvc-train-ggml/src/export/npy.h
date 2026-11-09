// npy.h -- M4-5a: minimal .npy reader (F32, C order, little-endian)
//
// Accepts only `<f4` / `=f4` / `|f4` (4-byte float) with `fortran_order=False`; 1D is treated as [n,1].
// Used by index building to read official preprocessed features (soyo 3_feature768/*.npy or total_fea.npy).

#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace rvc_train::exporter {

// Single file: out is row-major [rows, cols]; cols=1 for 1D
bool load_npy_f32(const std::string& path, std::vector<float>* out, int64_t* rows, int64_t* cols,
                  std::string* err);

// Concatenates all *.npy in a directory (sorted by filename) by rows; every file must have the same cols
bool load_npy_dir_f32(const std::string& dir, std::vector<float>* out, int64_t* rows,
                      int64_t* cols, std::string* err);

} // namespace rvc_train::exporter
