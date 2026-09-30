cmake_minimum_required(VERSION 3.20)

project(RVC-GGML VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(MSVC)
    add_compile_options(/utf-8)
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
endif()

# ================================================================
# 选项
# ================================================================
option(RVC_VULKAN "使用 ggml Vulkan 后端" ON)
option(RVC_CUDA   "使用 ggml CUDA 后端"   OFF)
set(RVC_GGML_DIR "" CACHE PATH "本地 ggml 源码目录（优先级最高）")

# ================================================================
# ggml 补丁自动应用
# 官方 ggml v0.23.0 缺少 Vulkan matmul N 维切分补丁，缺失时：
# 长音频（单块 >43.7s）+ 48kHz 角色模型会触发 maxComputeWorkGroupCount 断言
# 见 patches/ggml-vulkan-matmul-n-split.patch 与 patches/README.md
# ================================================================
function(rvc_apply_ggml_patch ggml_source_dir)
    set(RVC_PATCH_FILE "${CMAKE_CURRENT_SOURCE_DIR}/patches/ggml-vulkan-matmul-n-split.patch")
    if(NOT EXISTS "${RVC_PATCH_FILE}")
        message(WARNING "[rvc] 未找到补丁文件 ${RVC_PATCH_FILE}，跳过自动打补丁。")
        return()
    endif()
    # 幂等：源码已含补丁标记则不重复应用
    if(NOT EXISTS "${ggml_source_dir}/src/ggml-vulkan/ggml-vulkan.cpp")
        message(WARNING
            "[rvc] 在 ${ggml_source_dir} 中未找到 src/ggml-vulkan/ggml-vulkan.cpp，跳过补丁应用。")
        return()
    endif()
    file(READ "${ggml_source_dir}/src/ggml-vulkan/ggml-vulkan.cpp" _ggml_vk_src)
    string(FIND "${_ggml_vk_src}" "ggml_vk_mul_mat_q_f16_m_split" _pos)
    if(NOT _pos EQUAL -1)
        message(STATUS "[deps] ggml 已包含 Vulkan matmul N 维切分补丁，无需重复应用")
        return()
    endif()
    find_package(Git QUIET)
    if(NOT Git_FOUND)
        message(WARNING
            "[rvc] 未找到 git，无法自动应用补丁。"
            "请手动将 patches/ggml-vulkan-matmul-n-split.patch 应用到 ${ggml_source_dir} 后重新构建，"
            "否则长音频 + 48kHz 模型功能受限（见 patches/README.md）。")
        return()
    endif()
    # git apply 在仓库子目录内运行时按仓库根解析路径，且会把目录外的目标"静默跳过"；
    # 构建目录通常位于本仓库内，因此先探测仓库根，再以 --directory=<相对前缀> 从根应用
    execute_process(
        COMMAND "${GIT_EXECUTABLE}" -C "${ggml_source_dir}" rev-parse --show-toplevel
        OUTPUT_VARIABLE _git_toplevel
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE _git_top_result
        ERROR_QUIET
    )
    set(_rvc_patch_cwd "${ggml_source_dir}")
    set(_rvc_patch_prefix "")
    if(_git_top_result EQUAL 0 AND _git_toplevel)
        get_filename_component(_git_toplevel "${_git_toplevel}" ABSOLUTE)
        file(RELATIVE_PATH _rvc_rel_root "${_git_toplevel}" "${ggml_source_dir}")
        set(_rvc_patch_cwd "${_git_toplevel}")
        set(_rvc_patch_prefix "--directory=${_rvc_rel_root}")
    endif()
    message(STATUS "[deps] 正在为 ggml 应用补丁: ${RVC_PATCH_FILE}")
    execute_process(
        COMMAND "${GIT_EXECUTABLE}" apply ${_rvc_patch_prefix} --whitespace=nowarn "${RVC_PATCH_FILE}"
        WORKING_DIRECTORY "${_rvc_patch_cwd}"
        RESULT_VARIABLE _patch_result
        ERROR_VARIABLE _patch_error
    )
    if(NOT _patch_result EQUAL 0)
        message(FATAL_ERROR
            "[rvc] ggml 补丁应用失败（exit=${_patch_result}）：${_patch_error}\n"
            "请手动应用 ${RVC_PATCH_FILE} 到 ${ggml_source_dir}，见 patches/README.md。")
    endif()
    # 应用后必须验证真实生效（git apply 在某些情形会静默跳过）
    file(READ "${ggml_source_dir}/src/ggml-vulkan/ggml-vulkan.cpp" _ggml_vk_after)
    string(FIND "${_ggml_vk_after}" "ggml_vk_mul_mat_q_f16_m_split" _pos_after)
    if(_pos_after EQUAL -1)
        message(FATAL_ERROR
            "[rvc] 补丁命令执行成功但 ggml-vulkan.cpp 未发生变化（git apply 被跳过）。\n"
            "请手动应用 ${RVC_PATCH_FILE} 到 ${ggml_source_dir}，见 patches/README.md。")
    endif()
    message(STATUS "[deps] ggml Vulkan matmul N 维切分补丁应用成功")
endfunction()

# ================================================================
# ggml 依赖解析
# 优先级: RVC_GGML_DIR -> find_package(ggml) -> 兄弟目录 ../ggml-master / ../ggml
#         -> FetchContent(官方 v0.23.0，Vulkan 启用时自动应用本项目补丁)
# ================================================================
if(NOT TARGET ggml)
    set(GGML_VULKAN ${RVC_VULKAN} CACHE BOOL "ggml: use Vulkan" FORCE)
    set(GGML_CUDA   ${RVC_CUDA}   CACHE BOOL "ggml: use CUDA"   FORCE)
    set(BUILD_SHARED_LIBS OFF CACHE BOOL "静态链接 ggml" FORCE)

    if(RVC_GGML_DIR AND EXISTS "${RVC_GGML_DIR}/CMakeLists.txt")
        message(STATUS "[deps] 使用本地 ggml: ${RVC_GGML_DIR}")
        add_subdirectory("${RVC_GGML_DIR}" "${CMAKE_BINARY_DIR}/ggml" EXCLUDE_FROM_ALL)
    else()
        find_package(ggml QUIET CONFIG)
        if(ggml_FOUND)
            message(STATUS "[deps] 使用已安装的 ggml 包")
        elseif(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../ggml-master/CMakeLists.txt")
            message(STATUS "[deps] 使用兄弟目录 ggml: ${CMAKE_CURRENT_SOURCE_DIR}/../ggml-master")
            add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/../ggml-master" "${CMAKE_BINARY_DIR}/ggml" EXCLUDE_FROM_ALL)
        elseif(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../ggml/CMakeLists.txt")
            message(STATUS "[deps] 使用兄弟目录 ggml: ${CMAKE_CURRENT_SOURCE_DIR}/../ggml")
            add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/../ggml" "${CMAKE_BINARY_DIR}/ggml" EXCLUDE_FROM_ALL)
        else()
            message(STATUS "[deps] 本地未找到 ggml，从 codeload 拉取官方 v0.23.0")
            # 有意使用 Populate（需在 add_subdirectory 前打补丁），关闭 CMP0169 弃用提示
            if(POLICY CMP0169)
                cmake_policy(SET CMP0169 OLD)
            endif()
            include(FetchContent)
            FetchContent_Declare(ggml
                URL https://codeload.github.com/ggml-org/ggml/tar.gz/refs/tags/v0.23.0
                DOWNLOAD_EXTRACT_TIMESTAMP ON
            )
            FetchContent_GetProperties(ggml)
            if(NOT ggml_POPULATED)
                FetchContent_Populate(ggml)
                if(RVC_VULKAN)
                    rvc_apply_ggml_patch("${ggml_SOURCE_DIR}")
                endif()
                add_subdirectory("${ggml_SOURCE_DIR}" "${ggml_BINARY_DIR}" EXCLUDE_FROM_ALL)
            endif()
        endif()
    endif()
endif()

# ================================================================
# 检查 ggml 源码是否包含本项目必需的 Vulkan matmul N 维切分补丁
# 见 patches/ggml-vulkan-matmul-n-split.patch
# 缺补丁时：长音频 + 48kHz 角色模型会在运行期触发 maxComputeWorkGroupCount 断言
# ================================================================
if(RVC_VULKAN)
    set(RVC_GGML_VK_FILE "")
    foreach(_rvc_dir "${RVC_GGML_DIR}"
                     "${CMAKE_CURRENT_SOURCE_DIR}/../ggml-master"
                     "${CMAKE_CURRENT_SOURCE_DIR}/../ggml"
                     "${CMAKE_BINARY_DIR}/_deps/ggml-src")
        if(EXISTS "${_rvc_dir}/src/ggml-vulkan/ggml-vulkan.cpp")
            set(RVC_GGML_VK_FILE "${_rvc_dir}/src/ggml-vulkan/ggml-vulkan.cpp")
            break()
        endif()
    endforeach()
    if(RVC_GGML_VK_FILE)
        file(READ "${RVC_GGML_VK_FILE}" _rvc_ggml_vk_src)
        string(FIND "${_rvc_ggml_vk_src}" "ggml_vk_mul_mat_q_f16_m_split" _rvc_nsplit_pos)
        if(_rvc_nsplit_pos EQUAL -1)
            message(WARNING
                "[rvc] 当前 ggml 源码缺少 Vulkan matmul N 维切分补丁：长音频（单块 >43.7s）与 48kHz 角色模型"
                "可能触发 maxComputeWorkGroupCount 断言。请应用 patches/ggml-vulkan-matmul-n-split.patch 后重新构建。")
        endif()
    else()
        message(WARNING
            "[rvc] 未找到 ggml 源码（ggml-vulkan.cpp），无法确认 N 维切分补丁是否存在；"
            "长音频 + 48kHz 模型功能可能受影响，见 patches/README.md。")
    endif()
endif()

# ================================================================
# 共享库（推理核心）
# ================================================================
set(RVC_CORE_SOURCES
    src/utils.cpp
    src/engine.cpp
    src/audio/audio_io.cpp
    src/audio/dsp.cpp
    src/audio/mel.cpp
    src/gguf/gguf_model.cpp
    src/hubert/hubert.cpp
    src/rmvpe/rmvpe.cpp
    src/vits/modules.cpp
    src/vits/text_encoder.cpp
    src/vits/flow.cpp
    src/vits/generator.cpp
    src/vits/synthesizer.cpp
    src/rvc/sinegen.cpp
    src/rvc/retrieval.cpp
    src/rvc/pipeline.cpp
)

add_library(rvc_core STATIC ${RVC_CORE_SOURCES})
target_include_directories(rvc_core PUBLIC include src)
target_link_libraries(rvc_core PUBLIC ggml)

if(MSVC)
    target_compile_options(rvc_core PRIVATE /W3)
endif()

# ================================================================
# 主程序
# ================================================================
add_executable(RVC-GGML src/main.cpp)
target_link_libraries(RVC-GGML PRIVATE rvc_core)

if(MSVC)
    target_compile_options(RVC-GGML PRIVATE /W3)
endif()


