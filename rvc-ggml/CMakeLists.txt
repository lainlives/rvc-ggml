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
# Options
# ================================================================
option(RVC_VULKAN "Use the ggml Vulkan backend" ON)
option(RVC_CUDA   "Use the ggml CUDA backend"   OFF)
set(RVC_GGML_DIR "" CACHE PATH "Local ggml source directory (highest priority)")

# ================================================================
# Automatic ggml patch application
# Upstream ggml v0.23.0 lacks the Vulkan matmul N-dimension split patch; without it:
# long audio (single chunk >43.7s) + a 48kHz character model will trigger the maxComputeWorkGroupCount assertion
# see patches/ggml-vulkan-matmul-n-split.patch and patches/README.md
# ================================================================
function(rvc_apply_ggml_patch ggml_source_dir)
    set(RVC_PATCH_FILE "${CMAKE_CURRENT_SOURCE_DIR}/patches/ggml-vulkan-matmul-n-split.patch")
    if(NOT EXISTS "${RVC_PATCH_FILE}")
        message(WARNING "[rvc] Patch file not found: ${RVC_PATCH_FILE}, skipping automatic patching.")
        return()
    endif()
    # Idempotent: skip re-application if the source already contains the patch marker
    if(NOT EXISTS "${ggml_source_dir}/src/ggml-vulkan/ggml-vulkan.cpp")
        message(WARNING
            "[rvc] src/ggml-vulkan/ggml-vulkan.cpp not found under ${ggml_source_dir}, skipping patch application.")
        return()
    endif()
    file(READ "${ggml_source_dir}/src/ggml-vulkan/ggml-vulkan.cpp" _ggml_vk_src)
    string(FIND "${_ggml_vk_src}" "ggml_vk_mul_mat_q_f16_m_split" _pos)
    if(NOT _pos EQUAL -1)
        message(STATUS "[deps] ggml already contains the Vulkan matmul N-dimension split patch, no need to re-apply")
        return()
    endif()
    find_package(Git QUIET)
    if(NOT Git_FOUND)
        message(WARNING
            "[rvc] git not found, cannot apply the patch automatically. "
            "Please manually apply patches/ggml-vulkan-matmul-n-split.patch to ${ggml_source_dir} and rebuild, "
            "otherwise long-audio + 48kHz model functionality is limited (see patches/README.md).")
        return()
    endif()
    # When git apply runs inside a repository subdirectory it resolves paths relative to the repository root, and "silently skips" targets outside that directory;
    # the build directory usually lives inside this repository, so probe the repository root first, then apply from the root with --directory=<relative prefix>
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
    message(STATUS "[deps] Applying patch to ggml: ${RVC_PATCH_FILE}")
    execute_process(
        COMMAND "${GIT_EXECUTABLE}" apply ${_rvc_patch_prefix} --whitespace=nowarn "${RVC_PATCH_FILE}"
        WORKING_DIRECTORY "${_rvc_patch_cwd}"
        RESULT_VARIABLE _patch_result
        ERROR_VARIABLE _patch_error
    )
    if(NOT _patch_result EQUAL 0)
        message(FATAL_ERROR
            "[rvc] ggml patch application failed (exit=${_patch_result}): ${_patch_error}\n"
            "Please manually apply ${RVC_PATCH_FILE} to ${ggml_source_dir}, see patches/README.md.")
    endif()
    # After applying, verify it actually took effect (git apply silently skips in some cases)
    file(READ "${ggml_source_dir}/src/ggml-vulkan/ggml-vulkan.cpp" _ggml_vk_after)
    string(FIND "${_ggml_vk_after}" "ggml_vk_mul_mat_q_f16_m_split" _pos_after)
    if(_pos_after EQUAL -1)
        message(FATAL_ERROR
            "[rvc] Patch command succeeded but ggml-vulkan.cpp did not change (git apply was skipped).\n"
            "Please manually apply ${RVC_PATCH_FILE} to ${ggml_source_dir}, see patches/README.md.")
    endif()
    message(STATUS "[deps] ggml Vulkan matmul N-dimension split patch applied successfully")
endfunction()

# ================================================================
# ggml dependency resolution
# priority: RVC_GGML_DIR -> find_package(ggml) -> sibling directories ../ggml-master / ../ggml
#         -> FetchContent (upstream v0.23.0; the project patch is applied automatically when Vulkan is enabled)
# ================================================================
if(NOT TARGET ggml)
    set(GGML_VULKAN ${RVC_VULKAN} CACHE BOOL "ggml: use Vulkan" FORCE)
    set(GGML_CUDA   ${RVC_CUDA}   CACHE BOOL "ggml: use CUDA"   FORCE)
    set(BUILD_SHARED_LIBS OFF CACHE BOOL "statically link ggml" FORCE)

    if(RVC_GGML_DIR AND EXISTS "${RVC_GGML_DIR}/CMakeLists.txt")
        message(STATUS "[deps] Using local ggml: ${RVC_GGML_DIR}")
        add_subdirectory("${RVC_GGML_DIR}" "${CMAKE_BINARY_DIR}/ggml" EXCLUDE_FROM_ALL)
    else()
        find_package(ggml QUIET CONFIG)
        if(ggml_FOUND)
            message(STATUS "[deps] Using the installed ggml package")
        elseif(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../ggml-master/CMakeLists.txt")
            message(STATUS "[deps] Using sibling directory ggml: ${CMAKE_CURRENT_SOURCE_DIR}/../ggml-master")
            add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/../ggml-master" "${CMAKE_BINARY_DIR}/ggml" EXCLUDE_FROM_ALL)
        elseif(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../ggml/CMakeLists.txt")
            message(STATUS "[deps] Using sibling directory ggml: ${CMAKE_CURRENT_SOURCE_DIR}/../ggml")
            add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/../ggml" "${CMAKE_BINARY_DIR}/ggml" EXCLUDE_FROM_ALL)
        else()
            message(STATUS "[deps] ggml not found locally; fetching upstream v0.23.0 from codeload")
            # Intentionally using Populate (the patch must be applied before add_subdirectory); silence the CMP0169 deprecation notice
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
# Check whether the ggml source contains the Vulkan matmul N-dimension split patch required by this project
# see patches/ggml-vulkan-matmul-n-split.patch
# Without the patch: long audio + a 48kHz character model will trigger the maxComputeWorkGroupCount assertion at runtime
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
                "[rvc] The current ggml source is missing the Vulkan matmul N-dimension split patch: long audio (single chunk >43.7s) and 48kHz character models "
                "may trigger the maxComputeWorkGroupCount assertion. Please apply patches/ggml-vulkan-matmul-n-split.patch and rebuild.")
        endif()
    else()
        message(WARNING
            "[rvc] ggml source not found (ggml-vulkan.cpp); cannot confirm whether the N-dimension split patch is present; "
            "long-audio + 48kHz model functionality may be affected, see patches/README.md.")
    endif()
endif()

# ================================================================
# Shared library (inference core)
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
# Main executable
# ================================================================
add_executable(RVC-GGML src/main.cpp)
target_link_libraries(RVC-GGML PRIVATE rvc_core)

if(MSVC)
    target_compile_options(RVC-GGML PRIVATE /W3)
endif()


