# RVC-Train-GGML —— 基于 train.cpp 的 RVC 训练项目
#
# 架构与模块说明：见 docs/架构设计.md、docs/使用指南.md
# 构建：powershell -ExecutionPolicy Bypass -File scripts\dev_build.ps1
#
# 依赖策略（monorepo 内相对路径，可用 CMake 变量覆盖指向外部副本）：
#   - train.cpp（traincpp） 默认 ../third_party/train.cpp -> rvc-train.exe 链接 traincpp::traincpp（Vulkan/CPU）
#   - RVC-GGML（rvc_core）  默认 ../rvc-ggml             -> rvc-prep.exe  链接 rvc_core（ggml Vulkan）
#   - ggml                 由 RVC-GGML 解析（官方 v0.23.0 + 自动补丁，或本地 RVC_GGML_DIR）
#
# 注意：两个 Vulkan 后端（ggml 与 traincpp）不进同一进程（集成指南 R28），
#       因此拆成两个 exe：rvc-prep 不链 traincpp；rvc-train 不链 ggml/rvc_core。

cmake_minimum_required(VERSION 3.20)

project(RVC-Train-GGML VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(MSVC)
    # /utf-8 全局（源码含中文注释）；/W4 只加在本项目自己的目标上，
    # 避免把 ggml 等依赖的既有警告（其按 /W3 维护）刷进构建输出
    add_compile_options(/utf-8)
endif()

# 统一输出目录（多配置生成器见下）
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
foreach(_cfg DEBUG RELEASE RELWITHDEBINFO MINSIZEREL)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_${_cfg} ${CMAKE_BINARY_DIR}/bin)
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_${_cfg} ${CMAKE_BINARY_DIR}/lib)
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_${_cfg} ${CMAKE_BINARY_DIR}/lib)
endforeach()

# ------------------------------------------------------------------ 依赖路径
set(RVCTRAIN_TRAINCPP_DIR "" CACHE PATH "train.cpp 源码目录（默认 ../third_party/train.cpp）")
set(RVCTRAIN_RVCGGML_DIR  "" CACHE PATH "RVC-GGML 源码目录（默认 ../rvc-ggml）")
set(RVCTRAIN_GGML_DIR     "" CACHE PATH "ggml 源码目录（可选，转发给 RVC-GGML；默认由 RVC-GGML 解析）")

if(NOT RVCTRAIN_TRAINCPP_DIR)
    set(RVCTRAIN_TRAINCPP_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../third_party/train.cpp")
endif()
if(NOT RVCTRAIN_RVCGGML_DIR)
    set(RVCTRAIN_RVCGGML_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../rvc-ggml")
endif()

if(NOT EXISTS "${RVCTRAIN_TRAINCPP_DIR}/CMakeLists.txt")
    message(FATAL_ERROR "未找到 train.cpp：${RVCTRAIN_TRAINCPP_DIR}（用 -DRVCTRAIN_TRAINCPP_DIR=... 指定）")
endif()
if(NOT EXISTS "${RVCTRAIN_RVCGGML_DIR}/CMakeLists.txt")
    message(FATAL_ERROR "未找到 RVC-GGML：${RVCTRAIN_RVCGGML_DIR}（用 -DRVCTRAIN_RVCGGML_DIR=... 指定）")
endif()

# ------------------------------------------------------------------ train.cpp（traincpp）
# 关闭 traincpp 的测试/示例/安装，只取库本体
set(TRC_BUILD_TESTS    OFF CACHE BOOL "RVC-Train-GGML：关闭 traincpp 测试" FORCE)
set(TRC_BUILD_EXAMPLES OFF CACHE BOOL "RVC-Train-GGML：关闭 traincpp 示例" FORCE)
set(TRC_INSTALL        OFF CACHE BOOL "RVC-Train-GGML：关闭 traincpp 安装规则" FORCE)
add_subdirectory("${RVCTRAIN_TRAINCPP_DIR}" "${CMAKE_BINARY_DIR}/_deps/traincpp")

# ------------------------------------------------------------------ RVC-GGML（rvc_core + ggml）
if(RVCTRAIN_GGML_DIR)
    set(RVC_GGML_DIR "${RVCTRAIN_GGML_DIR}" CACHE PATH "ggml 源码目录（由 RVC-Train-GGML 转发）" FORCE)
endif()
add_subdirectory("${RVCTRAIN_RVCGGML_DIR}" "${CMAKE_BINARY_DIR}/_deps/rvc-ggml")

# ------------------------------------------------------------------ rvc-prep（预处理，ggml Vulkan）
add_executable(rvc-prep
    src/main_prep.cpp
    src/cli/prep_wizard.cpp
    src/common/json.cpp
    src/export/gguf_writer.cpp
    src/export/index_gguf.cpp
    src/export/kmeans.cpp
    src/export/npy.cpp
    src/prep/cache.cpp
    src/prep/f0.cpp
    src/prep/feat.cpp
    src/prep/highpass.cpp
    src/prep/load.cpp
    src/prep/norm.cpp
    src/prep/pipeline.cpp
    src/prep/resample.cpp
    src/prep/slicer.cpp
    src/prep/spec.cpp
)
target_include_directories(rvc-prep PRIVATE src)
target_link_libraries(rvc-prep PRIVATE rvc_core)
if(MSVC)
    target_compile_options(rvc-prep PRIVATE /W4)
    # 与 RVC-GGML/ggml 保持一致的静态 CRT（否则 MT/MD 混链会 LNK2038）
    set_property(TARGET rvc-prep PROPERTY MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
endif()

# ------------------------------------------------------------------ rvc-train（训练，traincpp Vulkan）
add_executable(rvc-train
    src/main_train.cpp
    src/common/json.cpp
    src/export/mem_to_train_gguf.cpp
    src/export/model_gguf.cpp
    src/data/dataset.cpp
    src/model/weights.cpp
    src/model/config.cpp
    src/model/model.cpp
    src/model/scratch.cpp
    src/model/emb.cpp
    src/model/wn.cpp
    src/model/resblock.cpp
    src/model/sinegen.cpp
    src/model/dec.cpp
    src/model/flow.cpp
    src/model/enc_p.cpp
    src/model/enc_q.cpp
    src/model/discriminator.cpp
    src/model/synthesizer.cpp
    src/model/selftest.cpp
    src/prep/cache_read.cpp
    src/train/stft_mel.cpp
    src/train/losses.cpp
    src/train/train_step.cpp         
    src/train/grad_accum_selftest.cpp
    src/train/loss_selftest.cpp      
    src/train/dataset_selftest.cpp
    src/train/train_step_selftest.cpp
    src/train/trainer.cpp
    src/train/trainer_selftest.cpp
    src/train/resume_meta.cpp
    src/train/resume_selftest.cpp
    src/train/scratch_selftest.cpp
    src/cli/train_wizard.cpp
)
target_include_directories(rvc-train PRIVATE src)
target_link_libraries(rvc-train PRIVATE traincpp::traincpp)
if(MSVC)
    target_compile_options(rvc-train PRIVATE /W4)
endif()

message(STATUS "RVC-Train-GGML：train.cpp = ${RVCTRAIN_TRAINCPP_DIR}")
message(STATUS "RVC-Train-GGML：RVC-GGML  = ${RVCTRAIN_RVCGGML_DIR}")
