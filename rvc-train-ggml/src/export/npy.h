// npy.h —— M4-5a：最小 .npy 读取器（F32、C 序、小端）
//
// 只接受 `<f4` / `=f4` / `|f4`（4 字节浮点）且 `fortran_order=False`；1D 视作 [n,1]。
// 用于索引构建读取官方预处理特征（soyo 3_feature768/*.npy 或 total_fea.npy）。

#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace rvc_train::exporter {

// 单文件：out 为行主序 [rows, cols]；1D 时 cols=1
bool load_npy_f32(const std::string& path, std::vector<float>* out, int64_t* rows, int64_t* cols,
                  std::string* err);

// 目录下全部 *.npy（文件名排序）按行拼接；各文件 cols 必须一致
bool load_npy_dir_f32(const std::string& dir, std::vector<float>* out, int64_t* rows,
                      int64_t* cols, std::string* err);

} // namespace rvc_train::exporter
