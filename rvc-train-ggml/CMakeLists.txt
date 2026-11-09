# RVC-Train-GGML -- RVC training project built on train.cpp
#
# Architecture and module docs: see docs/architecture.md, docs/usage_guide.md
# Build: powershell -ExecutionPolicy Bypass -File scripts\dev_build.ps1
#
# Dependency strategy (relative paths inside the monorepo; CMake variables can override to point at external copies):
#   - train.cpp (traincpp) defaults to ../third_party/train.cpp -> rvc-train.exe links traincpp::traincpp (Vulkan/CPU)
#   - RVC-GGML (rvc_core)  defaults to ../rvc-ggml             -> rvc-prep.exe  links rvc_core (ggml Vulkan)
#   - ggml                 resolved by RVC-GGML (official v0.23.0 + automatic patches, or local RVC_GGML_DIR)
#
# Note: the two Vulkan backends (ggml and traincpp) do not share one process (integration guide R28),
#       hence split into two exes: rvc-prep does not link traincpp; rvc-train does not link ggml/rvc_core.

cmake_minimum_required(VERSION 3.20)

project(RVC-Train-GGML VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(MSVC)
    # /utf-8 globally (sources contain UTF-8 text); /W4 is added only to this project's own targets,
    # to avoid flooding the build output with existing warnings from ggml and other dependencies (maintained at /W3)
    add_compile_options(/utf-8)
endif()

# Unified output directory (multi-config generators below)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
foreach(_cfg DEBUG RELEASE RELWITHDEBINFO MINSIZEREL)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_${_cfg} ${CMAKE_BINARY_DIR}/bin)
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_${_cfg} ${CMAKE_BINARY_DIR}/lib)
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_${_cfg} ${CMAKE_BINARY_DIR}/lib)
endforeach()

# ------------------------------------------------------------------ Dependency paths
set(RVCTRAIN_TRAINCPP_DIR "" CACHE PATH "train.cpp source directory (default: ../third_party/train.cpp)")
set(RVCTRAIN_RVCGGML_DIR  "" CACHE PATH "RVC-GGML source directory (default: ../rvc-ggml)")
set(RVCTRAIN_GGML_DIR     "" CACHE PATH "ggml source directory (optional, forwarded to RVC-GGML; by default resolved by RVC-GGML)")

if(NOT RVCTRAIN_TRAINCPP_DIR)
    set(RVCTRAIN_TRAINCPP_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../third_party/train.cpp")
endif()
if(NOT RVCTRAIN_RVCGGML_DIR)
    set(RVCTRAIN_RVCGGML_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../rvc-ggml")
endif()

if(NOT EXISTS "${RVCTRAIN_TRAINCPP_DIR}/CMakeLists.txt")
    message(FATAL_ERROR "train.cpp not found: ${RVCTRAIN_TRAINCPP_DIR} (specify with -DRVCTRAIN_TRAINCPP_DIR=...)")
endif()
if(NOT EXISTS "${RVCTRAIN_RVCGGML_DIR}/CMakeLists.txt")
    message(FATAL_ERROR "RVC-GGML not found: ${RVCTRAIN_RVCGGML_DIR} (specify with -DRVCTRAIN_RVCGGML_DIR=...)")
endif()

# ------------------------------------------------------------------ train.cpp (traincpp)
# Disable traincpp tests/examples/install; keep only the library itself
set(TRC_BUILD_TESTS    OFF CACHE BOOL "RVC-Train-GGML: disable traincpp tests" FORCE)
set(TRC_BUILD_EXAMPLES OFF CACHE BOOL "RVC-Train-GGML: disable traincpp examples" FORCE)
set(TRC_INSTALL        OFF CACHE BOOL "RVC-Train-GGML: disable traincpp install rules" FORCE)
add_subdirectory("${RVCTRAIN_TRAINCPP_DIR}" "${CMAKE_BINARY_DIR}/_deps/traincpp")

# ------------------------------------------------------------------ RVC-GGML (rvc_core + ggml)
if(RVCTRAIN_GGML_DIR)
    set(RVC_GGML_DIR "${RVCTRAIN_GGML_DIR}" CACHE PATH "ggml source directory (forwarded by RVC-Train-GGML)" FORCE)
endif()
add_subdirectory("${RVCTRAIN_RVCGGML_DIR}" "${CMAKE_BINARY_DIR}/_deps/rvc-ggml")

# ------------------------------------------------------------------ rvc-prep (preprocessing, ggml Vulkan)
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
    # Static CRT consistent with RVC-GGML/ggml (otherwise mixing MT/MD causes LNK2038)
    set_property(TARGET rvc-prep PROPERTY MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
endif()

# ------------------------------------------------------------------ rvc-train (training, traincpp Vulkan)
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
if(NOT MSVC AND TARGET traincpp_cpu AND TARGET traincpp_core)
    # GNU ld is single-pass and does not rescan archives: traincpp_core's backend
    # registry references cpu_device()/vulkan_device() from the backend libs while
    # those libs depend on traincpp_core (an archive cycle). MSVC LINK rescans
    # libraries, GNU ld (Linux/MinGW) needs an explicit RESCAN link group.
    if(TARGET traincpp_vulkan)
        target_link_libraries(rvc-train PRIVATE "$<LINK_GROUP:RESCAN,traincpp_cpu,traincpp_vulkan,traincpp_core>")
    else()
        target_link_libraries(rvc-train PRIVATE "$<LINK_GROUP:RESCAN,traincpp_cpu,traincpp_core>")
    endif()
else()
    target_link_libraries(rvc-train PRIVATE traincpp::traincpp)
endif()
if(MSVC)
    target_compile_options(rvc-train PRIVATE /W4)
endif()

message(STATUS "RVC-Train-GGML: train.cpp = ${RVCTRAIN_TRAINCPP_DIR}")
message(STATUS "RVC-Train-GGML: RVC-GGML  = ${RVCTRAIN_RVCGGML_DIR}")
